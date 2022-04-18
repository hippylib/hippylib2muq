import os
import math

import yaml
import h5py
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt

import dolfin as dl
from petsc4py import PETSc

import hippylib as hl
import muq.Modeling as mm
import muq.SamplingAlgorithms as ms
import hippylib2muq as hm


# Energy functional of the forward problem
def pde_varf(u, m, p):
    return (
        dl.exp(m) * dl.inner(dl.nabla_grad(u), dl.nabla_grad(p)) * dl.dx - f * p * dl.dx
    )


# Bounday of the problem domain
def boundary(x, on_boundary):
    return on_boundary


# Ground truth parameter field
class ExactParameter(dl.UserExpression):
    def eval(self, value, x):
        if (x[0] >= 0.125 and x[0] <= 0.375) and (x[1] >= 0.125 and x[1] <= 0.375):
            value[0] = -2.3025850929940455
        elif (x[0] >= 0.625 and x[0] <= 0.875) and (x[1] >= 0.625 and x[1] <= 0.875):
            value[0] = 2.302585092994046
        else:
            value[0] = 0.0


class PDEVariationalProblem(hl.PDEVariationalProblem):
    def __init__(self, Vh, varf_handler, bc, bc0):
        super().__init__(Vh, varf_handler, bc, bc0, is_fwd_linear=True)

        # original parameter space (defined on the coarse mesh)
        self.Vh0 = Vh[hl.PARAMETER]

        # parameter space is mapped to the refined mesh
        self.Vh = Vh.copy()
        self.Vh[hl.PARAMETER] = dl.FunctionSpace(self.Vh[hl.STATE].mesh(), "DG", 0)

        self._construct_parameter_projection()

    # Construct projection matrix from V0 to V1 (from the coarse to the refined).
    # Only consider DG = 0 space.
    def _construct_parameter_projection(self):
        V0 = self.Vh0
        V1 = self.Vh[hl.PARAMETER]
        assert (
            V0.ufl_element().is_cellwise_constant()
        ), "Parameter function space should be DG with order 0."

        dim0 = V0.dim()
        dim1 = V1.dim()
        M = np.zeros((dim1, dim0))

        tree = V0.mesh().bounding_box_tree()
        dof2coordinates = V1.tabulate_dof_coordinates()
        for i in range(dim1):
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
            return hl.vector2Function(y, self.Vh[hl.PARAMETER])
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


class PointwiseStateObservation(hl.PointwiseStateObservation):
    def __init__(self, h, Vh, obs_points, noise_variance=None):
        self._construct_obs_matrix(h, Vh, obs_points)
        self.d = dl.Vector(self.B.mpi_comm())
        self.d.init(obs_points.shape[0])
        self.Bu = dl.Vector(self.B.mpi_comm())
        self.B.init_vector(self.Bu, 0)
        self.noise_variance = noise_variance

    def _construct_obs_matrix(self, h, Vh, obs_points):
        # Define characteristic function of unit square
        def heaviside(x):
            if x < 0:
                return 0
            else:
                return 1

        def S(x, y):
            return (
                heaviside(x)
                * heaviside(y)
                * (1 - heaviside(x - h))
                * (1 - heaviside(y - h))
            )

        # Define tent function
        def phi(x, y):
            return (
                (x + h) * (y + h) * S(x + h, y + h)
                + (h - x) * (h - y) * S(x, y)
                + (x + h) * (h - y) * S(x + h, y)
                + (h - x) * (y + h) * S(x, y + h)
            ) / h ** 2

        B = np.zeros((obs_points.shape[0], Vh.dim()))
        dof2coord = Vh.tabulate_dof_coordinates()
        for i in range(obs_points.shape[0]):
            xp = obs_points[i]
            for j in range(Vh.dim()):
                bx = xp - dof2coord[j]
                B[i, j] = phi(bx[0], bx[1])

        Bsp = scipy.sparse.csr_matrix(B)
        self.B = dl.Matrix(
            dl.PETScMatrix(
                PETSc.Mat().createAIJWithArrays(
                    Bsp.shape, (Bsp.indptr, Bsp.indices, Bsp.data)
                )
            )
        )


class Prior:
    def __init__(self, Vh, variance, mean=None):
        self.variance = variance

        u = dl.TrialFunction(Vh)
        v = dl.TestFunction(Vh)
        I = dl.assemble(dl.inner(u, v) * dl.dx)
        I.zero()
        I.ident_zeros()
        self.R = I / self.variance
        self.Rsolver = hl.PETScLUSolver(Vh.mesh().mpi_comm())
        self.Rsolver.set_operator(self.R)

        self.mean = mean
        if self.mean is None:
            self.mean = dl.Vector(self.R.mpi_comm())
            self.init_vector(self.mean, 0)

    def cost(self, m):
        d = self.mean.copy()
        d.axpy(-1.0, m)
        Rd = dl.Vector(self.R.mpi_comm())
        self.init_vector(Rd, 0)
        self.R.mult(d, Rd)
        return 0.5 * Rd.inner(d)

    def init_vector(self, x, ind):
        if ind == "noise":
            self.R.init_vector(x, 1)
            return

        self.R.init_vector(x, ind)

    def grad(self, m, out):
        d = m.copy()
        d.axpy(-1.0, self.mean)
        self.R.mult(d, out)

    def sample(self, noise, s, add_mean=True):
        rhs = noise
        self.Rsolver.solve(s, rhs)

        if add_mean:
            s.axpy(1.0, self.mean)


class Model(hl.Model):
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
        problem: PDEVariationalProblem,
        prior: Prior,
        misfit: PointwiseStateObservation,
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
                self.problem.generate_parameter(),
                self.problem.generate_state(),
            ]
        elif component == hl.STATE:
            x = self.problem.generate_state()
        elif component == hl.PARAMETER:
            x = dl.Function(self.problem.Vh0).vector()
        elif component == hl.ADJOINT:
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
        m = self.problem.parameter_projection(x[hl.PARAMETER])
        self.problem.solveFwd(out, [x[hl.STATE], m, x[hl.ADJOINT]])

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
        m1 = self.problem.parameter_projection(x[hl.PARAMETER])
        x1 = [x[hl.STATE], m1, x[hl.ADJOINT]]
        self.misfit.grad(hl.STATE, x1, rhs)
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
        tmp = self.generate_vector(hl.PARAMETER)

        m1 = self.problem.parameter_projection(x[hl.PARAMETER])
        mg1 = self.problem.generate_parameter()
        self.problem.evalGradientParameter([x[hl.STATE], m1, x[hl.ADJOINT]], mg1)
        self.problem.transmult_M(mg1, mg)

        self.misfit.grad(hl.PARAMETER, x, tmp)
        mg.axpy(1.0, tmp)
        if not misfit_only:
            self.prior.grad(x[hl.PARAMETER], tmp)
            mg.axpy(1.0, tmp)

        # self.prior.Msolver.solve(tmp, mg)
        return math.sqrt(mg.inner(mg))

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

        m1 = self.problem.parameter_projection(x[hl.PARAMETER])
        x1 = [x[hl.STATE], m1, x[hl.ADJOINT]]
        self.problem.setLinearizationPoint(x1, self.gauss_newton_approx)

        self.misfit.setLinearizationPoint(x, self.gauss_newton_approx)
        if hasattr(self.prior, "setLinearizationPoint"):
            self.prior.setLinearizationPoint(x[hl.PARAMETER], self.gauss_newton_approx)

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
        self.problem.apply_ij(hl.ADJOINT, hl.PARAMETER, dm1, out)

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
        self.problem.apply_ij(hl.PARAMETER, hl.ADJOINT, dp, out1)
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
            self.problem.apply_ij(hl.STATE, hl.PARAMETER, dm1, out)
            tmp = self.generate_vector(hl.STATE)
            self.misfit.apply_ij(hl.STATE, hl.PARAMETER, dm, tmp)
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
            self.problem.apply_ij(hl.PARAMETER, hl.STATE, du, out1)
            self.problem.transmult_M(out1, out)

            tmp = self.generate_vector(hl.PARAMETER)
            self.misfit.apply_ij(hl.PARAMETER, hl.STATE, du, tmp)
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
            self.problem.apply_ij(hl.PARAMETER, hl.PARAMETER, dm1, out1)
            self.problem.transmult_M(out1, out)

            tmp = self.generate_vector(hl.PARAMETER)
            self.misfit.apply_ij(hl.PARAMETER, hl.PARAMETER, dm, tmp)
            out.axpy(1.0, tmp)


def get_fwdsol(msample="true"):
    if msample == "true":
        m = ExactParameter()
        m = dl.interpolate(m, Vh[hl.PARAMETER])
    else:
        return

    uvec = pde.generate_state()
    m1 = pde.parameter_projection(m.vector())
    pde.solveFwd(uvec, [uvec, m1, None])
    u = dl.Function(Vh[hl.STATE], uvec)

    dl.File(results_path + "m.pvd") << m
    dl.File(results_path + "u.pvd") << u


def setup_modpiece(prior, loglikelihood_model, method, dim):
    # a place holder ModPiece for the parameters
    idparam = mm.IdentityOperator(dim)

    # log Gaussian Prior ModPiece
    gaussprior = prior
    log_gaussprior = gaussprior.AsDensity()

    # parameter to log likelihood Modpiece
    param2likelihood = loglikelihood_model

    # log target ModPiece
    log_target = mm.DensityProduct(2)

    workgraph = mm.WorkGraph()

    # Identity operator for the parameters
    workgraph.AddNode(idparam, "Identity")

    # Prior model
    workgraph.AddNode(log_gaussprior, "Prior")

    # Likelihood model
    workgraph.AddNode(param2likelihood, "Param2Like")

    # Posterior
    workgraph.AddNode(log_target, "Target")

    workgraph.AddEdge("Identity", 0, "Prior", 0)
    workgraph.AddEdge("Prior", 0, "Target", 0)

    workgraph.AddEdge("Identity", 0, "Param2Like", 0)
    workgraph.AddEdge("Param2Like", 0, "Target", 1)

    # Enable caching
    if method not in ("pcn", "hpcn", ""):
        log_gaussprior.EnableCache()
        param2likelihood.EnableCache()

    # Construct the problem
    post_dens = workgraph.CreateModPiece("Target")
    problem = ms.SamplingProblem(post_dens)

    return problem


def generate_MHoptions(nburnin, nsamples, beta=0.0, tau=0.0):
    opts = dict()
    opts["BurnIn"] = nburnin
    opts["NumSamples"] = nsamples + nburnin
    opts["PrintLevel"] = 2
    opts["Beta"] = beta
    opts["StepSize"] = tau
    return opts


def setup_proposal(propName, options, problem, propGaussian):
    if propName == "pcn":
        proposal = ms.CrankNicolsonProposal(options, problem, propGaussian)
    elif propName == "mala":
        proposal = ms.MALAProposal(options, problem, propGaussian)
    else:
        raise NotImplementedError("Proposal is not availabe.")

    return proposal


def setup_kernel(kernName, options, problem, proposal):
    if kernName == "mh":
        kernel = ms.MHKernel(options, problem, proposal)
    else:
        raise NotImplementedError("Kernel is not availabe.")

    return kernel


def run_MCMC(options, kernel, startpoint):
    # Construct the MCMC sampler
    sampler = ms.SingleChainMCMC(options, [kernel])

    # Run the MCMC sampler
    samples = sampler.Run([startpoint])

    if "AcceptanceRate" in dir(kernel):
        return samples, kernel.AcceptanceRate(), sampler.TotalTime()
    elif "AcceptanceRates" in dir(kernel):
        return samples, kernel.AcceptanceRates(), sampler.TotalTime()


def generate_starting():
    noise = dl.Vector()
    nu.init_vector(noise, "noise")
    hl.parRandom.normal(1.0, noise)
    pr_s = model.generate_vector(hl.PARAMETER)
    post_s = model.generate_vector(hl.PARAMETER)
    nu.sample(noise, pr_s, post_s, add_mean=True)
    x0 = hm.dlVector2npArray(post_s)
    return x0


def generate_MCMCsamples(mcmc_parameters, fname=None):
    problem = setup_modpiece(
        mcmc_parameters["prior"],
        mcmc_parameters["likelihood_model"],
        mcmc_parameters["method"],
        Vh[hl.PARAMETER].dim(),
    )
    opts = generate_MHoptions(
        mcmc_parameters["burnin"],
        mcmc_parameters["nsamples"],
        beta=mcmc_parameters["beta"],
    )
    prop = setup_proposal(
        mcmc_parameters["proposal_name"],
        opts,
        problem,
        mcmc_parameters["proposal_gauss"],
    )
    kern = setup_kernel("mh", opts, problem, prop)

    m0 = generate_starting()
    samps, acceptrate, etime = run_MCMC(opts, kern, m0)
    if fname is not None:
        with h5py.File(
            os.path.dirname(os.path.realpath(__file__)) + "/" + fname + ".h5", "w"
        ) as fid:
            fid["/"].attrs["Beta"] = mcmc_parameters["beta"]
            fid["/samples"] = samps.AsMatrix()
            fid["/samples"].attrs["AR"] = acceptrate
            fid["/samples"].attrs["etime"] = etime
            # covariance = samps.Covariance()
            # fid["/covariance"] = covariance


#
# Set up miscellaneous things
#

# path to save all the results
results_path = os.path.dirname(os.path.realpath(__file__)) + "/results/"
# read observations
z_hat = np.loadtxt(
    os.path.dirname(os.path.realpath(__file__)) + "/obs.dat", delimiter=","
)
# read configuration parameters for mcmc
with open(os.path.dirname(os.path.realpath(__file__)) + "/" + "mcmc.yaml") as fid:
    inargs = yaml.full_load(fid)


#
# Set up the mesh and finite element spaces
#
nx = 32
nx_par = 8
mesh = dl.UnitSquareMesh.create(nx, nx, dl.CellType.Type.quadrilateral)
mesh_coarse = dl.UnitSquareMesh.create(nx_par, nx_par, dl.CellType.Type.quadrilateral)

Vh1 = dl.FunctionSpace(mesh, "CG", 1)
Vh0 = dl.FunctionSpace(mesh_coarse, "DG", 0)
Vh = [Vh1, Vh0, Vh1]

print(
    "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(
        Vh[hl.STATE].dim(), Vh[hl.PARAMETER].dim(), Vh[hl.ADJOINT].dim()
    )
)

#
# Set up the forward problem
#
u0 = dl.Constant(0.0)
bc = dl.DirichletBC(Vh[hl.STATE], u0, boundary)
f = dl.Constant(10.0)

pde = PDEVariationalProblem(Vh, pde_varf, bc, bc)

# solve for the exact state field
# get_fwdsol()

#
# Set up the prior
#
prior_std = 2
prior_variance = prior_std * prior_std
prior = Prior(Vh[hl.PARAMETER], prior_variance)

#
# Set up the misfit
#
targets = np.zeros((169, 2))
k = 0
for i in range(1, 14):
    xp = i / (13.0 + 1.0)
    for j in range(1, 14):
        yp = j / (13.0 + 1.0)

        targets[k, 0] = xp
        targets[k, 1] = yp
        k += 1

misfit = PointwiseStateObservation(1.0 / nx, Vh[hl.STATE], targets)
misfit.d.set_local(z_hat)
noise_std_dev = 0.05
misfit.noise_variance = noise_std_dev * noise_std_dev

#
# Set up the model
#
model = Model(pde, prior, misfit)

#
# Compute the MAP point
#
m = prior.mean.copy()
solver = hl.ReducedSpaceNewtonCG(model)
solver.parameters["rel_tolerance"] = 1e-8
solver.parameters["abs_tolerance"] = 1e-12
solver.parameters["max_iter"] = 25
solver.parameters["GN_iter"] = 5
solver.parameters["globalization"] = "LS"
solver.parameters["LS"]["c_armijo"] = 1e-4

x = solver.solve([None, m, None])

if solver.converged:
    print("\nConverged in ", solver.it, " iterations.")
else:
    print("\nNot Converged")

print("Termination reason: ", solver.termination_reasons[solver.reason])
print("Final gradient norm: ", solver.final_grad_norm)
print("Final cost: ", solver.final_cost)


map = dl.Function(Vh[hl.PARAMETER], m)
dl.File(results_path + "map.pvd") << map


#
# Construct the low-rank based Laplace approximation of the posterior
#
model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
Hmisfit = hl.ReducedHessian(model, misfit_only=True)
k = 60
p = 20

Omega = hl.MultiVector(x[hl.PARAMETER], k + p)
hl.parRandom.normal(1.0, Omega)
lmbda, V = hl.doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k)

nu = hl.GaussianLRPosterior(prior, lmbda, V)
nu.mean = x[hl.PARAMETER]

plt.plot(range(0, k), lmbda, "b*", range(0, k + 1), np.ones(k + 1), "-r")
plt.yscale("log")
plt.xlabel("number")
plt.ylabel("eigenvalue")
plt.show()

#
#  Set up ModPieces for implementing MCMC methods
#
mcmc_parameters = {}
mcmc_parameters["method"] = inargs["method"]
mcmc_parameters["nsamples"] = inargs["nsamples"]
mcmc_parameters["burnin"] = int(mcmc_parameters["nsamples"] * 0.05)
mcmc_parameters["beta"] = inargs["beta"]
mcmc_parameters["tau"] = 0.0
if mcmc_parameters["method"] == "hpcn":
    mcmc_parameters["proposal_name"] = "pcn"
else:
    raise NotImplementedError()
mcmc_parameters["proposal_gauss"] = hm.LAPosteriorGaussian(nu)
mcmc_parameters["prior"] = mm.Gaussian(
    prior.mean[:], prior.R.array(), mm.Gaussian.Mode.Precision
)
mcmc_parameters["likelihood_model"] = hm.Param2LogLikelihood(model)
generate_MCMCsamples(mcmc_parameters, fname=inargs["fname"])
