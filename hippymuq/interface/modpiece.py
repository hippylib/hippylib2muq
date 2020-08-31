"""
ModPiece interfacing hippylib and muq
"""

import numpy as np
import dolfin as df
import hippylib as hl
import pymuqModeling as mm
from ..utility.conversion import dfVector2npArray, const_dfVector, npArray2dfVector


class Param2LogLikelihood(mm.PyModPiece):

    """Docstring for Param2LogLikelihood.

    This class implements a mapping from parameter to Loglikelihood.
    It assumes that state, parameter, adjoint are 1D variables.
    """
    def __init__(self, model):
        self.model = model

        self.u = self.model.generate_vector(component=hl.STATE)
        self.m = self.model.generate_vector(component=hl.PARAMETER)
        self.p = self.model.generate_vector(component=hl.ADJOINT)
        self.gm = self.model.generate_vector(component=hl.PARAMETER)
        self.adjrhs = self.model.generate_vector(component=hl.STATE)

        self.hess0 = self.model.generate_vector(component=hl.PARAMETER)
        self.hess1 = self.model.generate_vector(component=hl.PARAMETER)

        # number of finite element coefficient of parameter
        self.npar = self.model.problem.Vh[hl.PARAMETER].dim()

        mm.PyModPiece.__init__(self, [self.npar], [1])

    def EvaluateImpl(self, inputs):
        npArray2dfVector(inputs[0], self.m)
        x = [self.u, self.m, None]

        # Solve for state variable
        self.model.solveFwd(x[hl.STATE], x)

        # Compute the cost of misfit
        y = self.model.misfit.cost(x)
        self.outputs = [np.array([-y])]

    def JacobianImpl(self, outDimWrt, inDimWrt, inputs):
        """
        outDimWrt = inDimWrt = 0 in this case
        """
        npArray2dfVector(inputs[0], self.m)
        x = [self.u, self.m, self.p]

        # Solve for state variable
        self.model.solveFwd(x[hl.STATE], x)

        # Solve for adjoint variable
        self.model.solveAdj(x[hl.ADJOINT], x)

        # Evaluate Jacobian
        self.model.problem.evalGradientParameter(x, self.gm)
        self.gm *= -1

        # Jacobian here works as vector,
        # but we leave it as 2d array as originally intened
        y = np.zeros((1, self.npar))
        y[0, :] = dfVector2npArray(self.gm)
        self.jacobian = y

    def GradientImpl(self, outDimWrt, inDimWrt, inputs, sens):
        """
        Apply :math:`J^T sens`.
        """
        self.Jacobian(outDimWrt, inDimWrt, inputs)
        self.gradient = self.jacobian[0, :] * sens[0]

    def ApplyJacobianImpl(self, outDimWrt, inDimWrt, inputs, vec):
        """
        Apply :math:`J vec`
        """
        self.Jacobian(outDimWrt, inDimWrt, inputs)
        self.jacobianAction = self.jacobian.dot(vec)

    def ApplyHessianImpl(self, outWrt, inWrt1, inWrt2, inputs, sens, vec):
        assert inWrt1 == 0 and inWrt2 == 0

        npArray2dfVector(inputs[0], self.m)
        x = [self.u, self.m, self.p]

        # Solve for state and adjoint variables
        self.model.solveFwd(x[hl.STATE], x)

        self.model.misfit.grad(hl.STATE, x, self.adjrhs)
        self.adjrhs *= -sens[0]
        self.model.problem.solveAdj(x[hl.ADJOINT], x, self.adjrhs)

        # Hessian apply
        self.model.setPointForHessianEvaluations(x)
        HessApply = hl.ReducedHessian(self.model, misfit_only=True)

        npArray2dfVector(vec, self.hess0)
        HessApply.mult(self.hess0, self.hess1)
        self.hess1 *= -1.0

        self.hessAction = dfVector2npArray(self.hess1)


class Param2obs(mm.PyModPiece):

    """Interfacing class between mm.PyModPiece and hl.PDEProblem.

    It assumes that state, parameter, adjoint are scalar variables.
    """
    def __init__(self, model):
        self.model = model

        self.u = self.model.generate_vector(component=hl.STATE)
        self.m = self.model.generate_vector(component=hl.PARAMETER)
        self.p = self.model.generate_vector(component=hl.ADJOINT)
        self.adjrhs = self.model.generate_vector(component=hl.STATE)
        self.gm = self.model.generate_vector(component=hl.PARAMETER)

        self.Bu = df.Vector(self.model.misfit.B.mpi_comm())
        self.model.misfit.B.init_vector(self.Bu, 0)
        self.sens = df.Vector(self.model.misfit.B.mpi_comm())
        self.model.misfit.B.init_vector(self.sens, 0)

        self.hess0 = self.model.generate_vector(component=hl.PARAMETER)
        self.hess1 = self.model.generate_vector(component=hl.PARAMETER)

        # number of finite element coefficient of parameter
        self.npar = self.model.problem.Vh[hl.PARAMETER].dim()

        # number of observation points
        self.nobs = self.model.misfit.d.size()

        mm.PyModPiece.__init__(self, [self.npar], [self.nobs])

    def EvaluateImpl(self, inputs):
        """
        :param inputs: parameters
        :param outputs: observations
        """
        npArray2dfVector(inputs[0], self.m)
        x = [self.u, self.m, None]

        # Solve for state variable
        self.model.solveFwd(x[hl.STATE], x)

        # Apply B operator to state variable
        self.model.misfit.B.mult(x[hl.STATE], self.Bu)

        y = dfVector2npArray(self.Bu)

        self.outputs = [y]

    def GradientImpl(self, outDimWrt, inDimWrt, inputs, sens):
        npArray2dfVector(inputs[0], self.m)
        x = [self.u, self.m, self.p]

        # Solve for state and adjoint variables
        self._solves_stateadj(x, sens)

        # Evaluate Gradient
        self.model.problem.evalGradientParameter(x, self.gm)

        self.gradient = dfVector2npArray(self.gm)

    def ApplyHessianImpl(self, outWrt, inWrt1, inWrt2, inputs, sens, vec):
        assert inWrt1 == 0 and inWrt2 == 0

        npArray2dfVector(inputs[0], self.m)
        x = [self.u, self.m, self.p]

        # Solve for state and adjoint variables
        self._solves_stateadj(x, sens)

        # Hessian apply
        self.model.setPointForHessianEvaluations(x)
        HessApply = hl.ReducedHessian(self.model, misfit_only=True)

        npArray2dfVector(vec, self.hess0)
        HessApply.mult(self.hess0, self.hess1)
        self.hess1 *= -1.0

        self.hessAction = dfVector2npArray(self.hess1)

    def _solves_stateadj(self, x, sens):
        """
        Solves the state and adjoint problems given parameters and sensitivity
        """
        # Solve for state variable
        self.model.solveFwd(x[hl.STATE], x)

        # Solve for adjoint variable
        npArray2dfVector(sens, self.sens)
        self.model.misfit.B.transpmult(self.sens, self.adjrhs)
        self.adjrhs *= -1
        self.model.problem.solveAdj(x[hl.ADJOINT], x, self.adjrhs)


# class Param2state(mm.PyModPiece):
#     def __init__(self, model):
#         self.model = model

#         self.u = self.model.generate_vector(component=hl.STATE)
#         self.m = self.model.generate_vector(component=hl.PARAMETER)
#         self.p = self.model.generate_vector(component=hl.ADJOINT)
#         self.adjrhs = self.model.generate_vector(component=hl.STATE)
#         self.gm = self.model.generate_vector(component=hl.PARAMETER)

#         # number of finite element coefficient of parameter
#         self.npar = self.model.problem.Vh[hl.PARAMETER].dim()

#         # degrees of freedom of state
#         self.nstate = self.model.problem.Vh[hl.STATE].dim()

#         mm.PyModPiece.__init__(self, [self.npar], [self.nstate])

#     def EvaluateImpl(self, inputs):
#         """
#         :param inputs: parameters
#         :param outputs: states
#         """
#         npArray2dfVector(inputs[0], self.m)
#         x = [self.u, self.m, None]

#         # Solve for state variable
#         self.model.solveFwd(x[hl.STATE], x)

#         y = dfVector2npArray(x[hl.STATE],)
#         self.outputs = [y]

#     def GradientImpl(self, outDimWrt, inDimWrt, inputs, sens):
#         npArray2dfVector(inputs[0], self.m)
#         x = [self.u, self.m, self.p]

#         # Solve for state variable
#         self.model.solveFwd(x[hl.STATE], x)

#         # Solve for adjoint variable
#         npArray2dfVector(sens, self.adjrhs)
#         self.adjrhs *= -1
#         self.model.problem.solveAdj(x[hl.ADJOINT], x, self.adjrhs)

#         # Evaluate Gradient
#         self.model.problem.evalGradientParameter(x, self.gm)
#         self.gradient = dfVector2npArray(self.gm)

# class State2obs(mm.PyModPiece):
#     def __init__(self, model):
#         self.model = model

#         self.u = self.model.generate_vector(component=hl.STATE)
#         self.Bu = df.Vector(self.model.misfit.B.mpi_comm())
#         self.adjrhs = df.Vector(self.model.misfit.B.mpi_comm())
#         self.model.misfit.B.init_vector(self.adjrhs, 0)
#         self.grad = self.model.generate_vector(component=hl.STATE)

#         # degrees of freedom of state
#         self.nstate = self.model.problem.Vh[hl.STATE].dim()

#         # number of observation points
#         self.nobs = self.model.misfit.d.size()

#         mm.PyModPiece.__init__(self, [self.nstate], [self.nobs])

#     def EvaluateImpl(self, inputs):
#         """
#         :param inputs: parameters
#         :param outputs: observations
#         """
#         npArray2dfVector(inputs[0], self.u)

#         # Apply B operator to state variable
#         self.model.misfit.B.mult(self.u, self.Bu)

#         y = dfVector2npArray(self.Bu)
#         self.outputs = [y]
      
#     def GradientImpl(self, outDimWrt, inDimWrt, inputs, sens):
#         # Solve for adjoint variable
#         npArray2dfVector(sens, self.adjrhs)

#         # Evaluate Gradient
#         self.model.misfit.B.transpmult(self.adjrhs, self.grad)

#         self.gradient = dfVector2npArray(self.grad)
       

class LogBiLaplaceGaussian(mm.PyModPiece):

    """A mm.PyModPiece class for evaluating the log prior"""

    def __init__(self, prior):
        """TODO: to be defined. """
        self.prior = prior

        self.m = const_dfVector(self.prior.A, 1)
        self.help = const_dfVector(self.prior.A, 0)

        self.npar = self.m.local_size()

        mm.PyModPiece.__init__(self, [self.npar], [1])

    def EvaluateImpl(self, inputs):
        """TODO: Docstring for EvaluateImpl.

        :param inputs: TODO
        :returns: TODO

        """
        npArray2dfVector(inputs[0], self.m)
        self.m.axpy(-1, self.prior.mean)
        self.prior.R.mult(self.m, self.help)

        self.outputs = [np.array([-0.5 * self.m.inner(self.help)])]



