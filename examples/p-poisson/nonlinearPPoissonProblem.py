import scipy.sparse
import numpy as np

import dolfin as dl
from petsc4py import PETSc
import hippylib as hp
from minimization import NewtonFwdSolver


class NonlinearPPossionForm(object):
    def __init__(self, p, f, ds_r):
        """TODO: Docstring for __init__.

        :param p: the exponent
        :param f: the forcing term
        :param ds_r: measure of the boundary the Robin condition applies

        """
        self.p = p
        self.f = f
        self.ds_r = ds_r

        # Regularization term
        self.eps = dl.Constant(1e-8)

    def energy_functional(self, u, m):
        """Energy functional

        :param u: state variable
        :param m: parameter variable (here the Neumann data)
        :returns: energy functional in UFL

        """
        grad_u = dl.nabla_grad(u)
        etah = dl.inner(grad_u, grad_u) + self.eps

        if self.f is None:
            return (
                1.0 / self.p * etah ** (0.5 * self.p) * dl.dx
                + 0.5 * dl.exp(m) * u * u * self.ds_r
            )
        else:
            return (
                1.0 / self.p * etah ** (0.5 * self.p) * dl.dx
                + 0.5 * dl.exp(m) * u * u * self.ds_r
                - self.f * u * dl.dx
            )

    def variational_form(self, u, m, p):
        """Variational form

        :param u: state variable
        :param m: parameter variable
        :param p: adjoint variable
        :returns: TODO

        """
        return dl.derivative(self.energy_functional(u, m), u, p)


class EnergyFunctionalPDEVariationalProblem(hp.PDEVariationalProblem):
    def __init__(self, Vh, energyform, bc, bc0):
        """TODO: Docstring for __init__.

        :param function: TODO
        :returns: TODO

        """
        super().__init__(Vh, energyform.variational_form, bc, bc0)

        self.energy_fun = energyform.energy_functional

        self.fwd_solver = NewtonFwdSolver()

        self.qoi = None
        self.cal_qoi = False

        # original parameter space (defined on the coarse mesh)
        self.Vh0 = Vh[hp.PARAMETER]

        # parameter space is mapped to the refined mesh
        self.Vh = Vh.copy()
        self.Vh[hp.PARAMETER] = dl.FunctionSpace(self.Vh[hp.STATE].mesh(), "CG", 1)

        self._construct_parameter_projection()

    def _construct_parameter_projection(self):
        V0 = self.Vh0
        V1 = self.Vh[hp.PARAMETER]

        dim0 = V0.dim()
        dim1 = V1.dim()
        M = np.zeros((dim1, dim0))

        tree = V0.mesh().bounding_box_tree()
        dof2coordinates = V1.tabulate_dof_coordinates()
        for i in range(dim1):
            if dl.near(dof2coordinates[i][2], 0.0):
                px = dl.Point(dof2coordinates[i])
                cell = tree.compute_collisions(px)[0]
                dof_V0 = V0.dofmap().cell_dofs(cell)[0]

                M[i, dof_V0] = 1

        Msp = scipy.sparse.csr_matrix(M)
        self.M = dl.Matrix(
            dl.PETScMatrix(
                PETSc.Mat().createAIJWithArrays(
                    Msp.shape, (Msp.indptr, Msp.indices, Msp.data)
                )
            )
        )

    def parameter_projection(self, m0: dl.Vector, outfmt="vector"):
        """Project vector from Vh0 to Vh[hl.PARAMETER]

        :param outfmt: 'vector' or 'function'
        :type outfmt:
        :param m0: input vector
        :type m0: dl.Vector
        :return: output vector
        :rtype:
        """
        y = self.M * m0
        if outfmt == "function":
            return hp.vector2Function(y, self.Vh[hp.PARAMETER])
        elif outfmt == "vector":
            return y
        else:
            raise NotImplementedError("Given return format is not available.")

    def transmult_M(self, m1: dl.Vector, m0: dl.Vector):
        """Apply :math:`\\mathbf{M}^T` to vector

        :param m1: input vector in Vh[hl.PARAMETER] space
        :type m1: dl.Vector
        :param m0: output vector in Vh0 space
        :type m0: dl.Vector
        """
        self.M.transpmult(m1, m0)


    def solveFwd(self, state, x):
        """TODO: Solve the nonlinear forward problem using Newton's method

        :param function: TODO
        :returns: TODO

        """
        if self.solver is None:
            self.solver = self._createLUSolver()

        u = hp.vector2Function(x[hp.STATE], self.Vh[hp.STATE])
        m = hp.vector2Function(x[hp.PARAMETER], self.Vh[hp.PARAMETER])

        F = self.energy_fun(u, m)

        uvec, reason = self.fwd_solver.solve(F, u, self.bc, self.bc0)

        if not self.fwd_solver.converged:
            print("Newton did not converged", reason)

        state.zero()
        state.axpy(1.0, uvec.vector())

        if self.cal_qoi:
            self.qoi.update_tracer(state)

class MixedDimensionModel(hp.Model):
    """
    This class contains the full description of the inverse problem.
    As inputs it takes a :code:`PDEProblem object`, a :code:`Prior` object, and a :code:`Misfit` object.

    In the following we will denote with

        - :code:`u` the state variable
        - :code:`m` the (model) parameter variable
        - :code:`p` the adjoint variable

    """

    def __init__(
        self,
        problem: EnergyFunctionalPDEVariationalProblem,
        prior,
        misfit,
    ):
        """
        Create a model given:

            - problem: the description of the forward/adjoint problem and all the sensitivities
            - prior: the prior component of the cost functional
            - misfit: the misfit componenent of the cost functional
        """
        self.problem = problem
        self.prior = prior
        self.misfit = misfit
        self.gauss_newton_approx = False

        self.n_fwd_solve = 0
        self.n_adj_solve = 0
        self.n_inc_solve = 0

    def generate_vector(self, component="ALL"):
        """
        By default, return the list :code:`[u,m,p]` where:
        
            - :code:`u` is any object that describes the state variable
            - :code:`m` is a :code:`dolfin.Vector` object that describes the parameter variable. \
            (Needs to support linear algebra operations)
            - :code:`p` is any object that describes the adjoint variable
        
        If :code:`component = STATE` return only :code:`u`
            
        If :code:`component = PARAMETER` return only :code:`m`
            
        If :code:`component = ADJOINT` return only :code:`p`
        """
        if component == "ALL":
            x = [
                self.problem.generate_state(),
                dl.Function(self.problem.Vh0).vector(),
                self.problem.generate_state(),
            ]
        elif component == hp.STATE:
            x = self.problem.generate_state()
        elif component == hp.PARAMETER:
            x = dl.Function(self.problem.Vh0).vector()
        elif component == hp.ADJOINT:
            x = self.problem.generate_state()

        return x

    def solveFwd(self, out, x):
        """
        Solve the (possibly non-linear) forward problem.

        Parameters:

            - :code:`out`: is the solution of the forward problem (i.e. the state) (Output parameters)
            - :code:`x = [u,m,p]` provides

                1) the parameter variable :code:`m` for the solution of the forward problem
                2) the initial guess :code:`u` if the forward problem is non-linear

                .. note:: :code:`p` is not accessed.
        """
        self.n_fwd_solve = self.n_fwd_solve + 1
        m1 = self.problem.parameter_projection(x[hp.PARAMETER])
        self.problem.solveFwd(out, [x[hp.STATE], m1, x[hp.ADJOINT]])

    def solveAdj(self, out, x):
        """
        Solve the linear adjoint problem.

        Parameters:

            - :code:`out`: is the solution of the adjoint problem (i.e. the adjoint :code:`p`) (Output parameter)
            - :code:`x = [u, m, p]` provides

                1) the parameter variable :code:`m` for assembling the adjoint operator
                2) the state variable :code:`u` for assembling the adjoint right hand side

                .. note:: :code:`p` is not accessed
        """
        self.n_adj_solve = self.n_adj_solve + 1
        rhs = self.problem.generate_state()
        m1 = self.problem.parameter_projection(x[hp.PARAMETER])
        x1 = [x[hp.STATE], m1, x[hp.ADJOINT]]
        self.misfit.grad(hp.STATE, x1, rhs)
        rhs *= -1.0
        self.problem.solveAdj(out, x1, rhs)

    def evalGradientParameter(self, x, mg, misfit_only=False):
        """
        Evaluate the gradient for the variational parameter equation at the point :code:`x=[u,m,p]`.

        Parameters:

            - :code:`x = [u,m,p]` the point at which to evaluate the gradient.
            - :code:`mg` the variational gradient :math:`(g, mtest)`, mtest being a test function in the parameter space \
            (Output parameter)
        
        Returns the norm of the gradient in the correct inner product :math:`g_norm = sqrt(g,g)`
        """
        tmp = self.generate_vector(hp.PARAMETER)

        m1 = self.problem.parameter_projection(x[hp.PARAMETER])
        mg1 = self.problem.generate_parameter()
        self.problem.evalGradientParameter([x[hp.STATE], m1, x[hp.ADJOINT]], mg1)
        self.problem.transmult_M(mg1, mg)

        self.misfit.grad(hp.PARAMETER, x, tmp)
        mg.axpy(1.0, tmp)
        if not misfit_only:
            self.prior.grad(x[hp.PARAMETER], tmp)
            mg.axpy(1.0, tmp)

        self.prior.Msolver.solve(tmp, mg)
        # self.prior.Msolver.solve(tmp, mg)
        return np.sqrt(mg.inner(mg))

    def setPointForHessianEvaluations(self, x, gauss_newton_approx=False):
        """
        Specify the point :code:`x = [u,m,p]` at which the Hessian operator (or the Gauss-Newton approximation)
        needs to be evaluated.

        Parameters:

            - :code:`x = [u,m,p]`: the point at which the Hessian or its Gauss-Newton approximation needs to be evaluated.
            - :code:`gauss_newton_approx (bool)`: whether to use Gauss-Newton approximation (default: use Newton)

        .. note:: This routine should either:

            - simply store a copy of x and evaluate action of blocks of the Hessian on the fly
            - or partially precompute the block of the hessian (if feasible)
        """
        self.gauss_newton_approx = gauss_newton_approx

        m1 = self.problem.parameter_projection(x[hp.PARAMETER])
        x1 = [x[hp.STATE], m1, x[hp.ADJOINT]]
        self.problem.setLinearizationPoint(x1, self.gauss_newton_approx)

        self.misfit.setLinearizationPoint(x, self.gauss_newton_approx)
        if hasattr(self.prior, "setLinearizationPoint"):
            self.prior.setLinearizationPoint(x[hp.PARAMETER], self.gauss_newton_approx)

    def applyC(self, dm, out):
        """
        Apply the :math:`C` block of the Hessian to a (incremental) parameter variable, i.e.
        :code:`out` = :math:`C dm`

        Parameters:

            - :code:`dm` the (incremental) parameter variable
            - :code:`out` the action of the :math:`C` block on :code:`dm`

        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        dm1 = self.problem.parameter_projection(dm)
        self.problem.apply_ij(hp.ADJOINT, hp.PARAMETER, dm1, out)

    def applyCt(self, dp, out):
        """
        Apply the transpose of the :math:`C` block of the Hessian to a (incremental) adjoint variable.
        :code:`out` = :math:`C^t dp`

        Parameters:

            - :code:`dp` the (incremental) adjoint variable
            - :code:`out` the action of the :math:`C^T` block on :code:`dp`

        ..note:: This routine assumes that :code:`out` has the correct shape.
        """
        out1 = self.problem.generate_parameter()
        self.problem.apply_ij(hp.PARAMETER, hp.ADJOINT, dp, out1)
        self.problem.transmult_M(out1, out)

    def applyWum(self, dm, out):
        """
        Apply the :math:`W_{um}` block of the Hessian to a (incremental) parameter variable.
        :code:`out` = :math:`W_{um} dm`

        Parameters:

            - :code:`dm` the (incremental) parameter variable
            - :code:`out` the action of the :math:`W_{um}` block on :code:`du`

        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        if self.gauss_newton_approx:
            out.zero()
        else:
            dm1 = self.problem.parameter_projection(dm)
            self.problem.apply_ij(hp.STATE, hp.PARAMETER, dm1, out)
            tmp = self.generate_vector(hp.STATE)
            self.misfit.apply_ij(hp.STATE, hp.PARAMETER, dm, tmp)
            out.axpy(1.0, tmp)

    def applyWmu(self, du, out):
        """
        Apply the :math:`W_{mu}` block of the Hessian to a (incremental) state variable.
        :code:`out` = :math:`W_{mu} du`

        Parameters:

            - :code:`du` the (incremental) state variable
            - :code:`out` the action of the :math:`W_{mu}` block on :code:`du`

        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        if self.gauss_newton_approx:
            out.zero()
        else:
            out1 = self.problem.generate_parameter()
            self.problem.apply_ij(hp.PARAMETER, hp.STATE, du, out1)
            self.problem.transmult_M(out1, out)

            tmp = self.generate_vector(hp.PARAMETER)
            self.misfit.apply_ij(hp.PARAMETER, hp.STATE, du, tmp)
            out.axpy(1.0, tmp)

    def applyWmm(self, dm, out):
        """
        Apply the :math:`W_{mm}` block of the Hessian to a (incremental) parameter variable.
        :code:`out` = :math:`W_{mm} dm`

        Parameters:

            - :code:`dm` the (incremental) parameter variable
            - :code:`out` the action of the :math:`W_{mm}` on block :code:`dm`

        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        if self.gauss_newton_approx:
            out.zero()
        else:
            dm1 = self.problem.parameter_projection(dm)
            out1 = self.problem.generate_parameter()
            self.problem.apply_ij(hp.PARAMETER, hp.PARAMETER, dm1, out1)
            self.problem.transmult_M(out1, out)

            tmp = self.generate_vector(hp.PARAMETER)
            self.misfit.apply_ij(hp.PARAMETER, hp.PARAMETER, dm, tmp)
            out.axpy(1.0, tmp)


