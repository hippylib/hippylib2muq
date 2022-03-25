#  hIPPYlib-MUQ interface for large-scale Bayesian inverse problems
#  Copyright (c) 2019-2020, The University of Texas at Austin,
#  University of California--Merced, Washington University in St. Louis,
#  The United States Army Corps of Engineers, Massachusetts Institute of Technology

#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
This module provides a set of wrappers that bind some ``hippylib`` functionalities
such that they can be used by ``muq``.

Please refer to ModPiece_ for the detailes of member functions defined here.

.. _ModPiece: https://mituq.bitbucket.io/classmuq_1_1Modeling_1_1ModPiece.html
"""
import numpy as np
import dolfin as dl
import hippylib as hp
import muq.Modeling as mm
from ..utility.conversion import dlVector2npArray, const_dlVector, npArray2dlVector


class Param2LogLikelihood(mm.PyModPiece):
    """Parameter to log-likelihood map

    This class implements mapping from parameter to log-likelihood.
    """

    def __init__(self, model):
        """
        :param hippylib::Model model: a ``hipplib::Model`` instance
        """
        self.model = model

        self.u = self.model.generate_vector(component=hp.STATE)
        self.m = self.model.generate_vector(component=hp.PARAMETER)
        self.p = self.model.generate_vector(component=hp.ADJOINT)
        self.gm = self.model.generate_vector(component=hp.PARAMETER)
        self.adjrhs = self.model.generate_vector(component=hp.STATE)

        self.hess0 = self.model.generate_vector(component=hp.PARAMETER)
        self.hess1 = self.model.generate_vector(component=hp.PARAMETER)

        # number of finite element coefficient of parameter
        self.npar = self.m.size()

        mm.PyModPiece.__init__(self, [self.npar], [1])

    def EvaluateImpl(self, inputs):
        """
        Evaluate the log-likelihood for given ``inputs``.

        :param numpy::ndarray inputs: parameter values
        """
        npArray2dlVector(inputs[0], self.m)
        x = [self.u, self.m, None]

        # Solve for state variable
        self.model.solveFwd(x[hp.STATE], x)

        # Compute the cost of misfit
        y = self.model.misfit.cost(x)
        self.outputs = [np.array([-y])]

    def JacobianImpl(self, outDimWrt, inDimWrt, inputs):
        """
        Compute the Jacobian for given ``inputs``.

        :param int outDimWrt: output dimension; should be 0
        :param int inDimWrt: input dimension; should be 0
        :param numpy::ndarray inputs: parameter values
        """
        npArray2dlVector(inputs[0], self.m)
        x = [self.u, self.m, self.p]

        # Solve for state variable
        self.model.solveFwd(x[hp.STATE], x)

        # Solve for adjoint variable
        self.model.solveAdj(x[hp.ADJOINT], x)

        # Evaluate Jacobian
        self.model.problem.evalGradientParameter(x, self.gm)
        self.gm *= -1

        # Jacobian here works as vector,
        # but we leave it as 2d array as originally intened
        y = np.zeros((1, self.npar))
        y[0, :] = dlVector2npArray(self.gm)
        self.jacobian = y

    def GradientImpl(self, outDimWrt, inDimWrt, inputs, sens):
        """
        Compute gradient; apply the transpose of Jacobian to ``sens`` for given
        ``inputs``.

        :param int outDimWrt: output dimension; should be 0
        :param int inDimWrt: input dimension; should be 0
        :param numpy::ndarray inputs: parameter values
        :param numpy::ndarray sens: input vector the transpose of Jacobian
                                    applies to
        """
        self.Jacobian(outDimWrt, inDimWrt, inputs)
        self.gradient = self.jacobian[0, :] * sens[0]

    def ApplyJacobianImpl(self, outDimWrt, inDimWrt, inputs, vec):
        """
        Apply Jacobian to ``vec`` for given ``inputs``.

        :param int outDimWrt: output dimension; should be 0
        :param int inDimWrt: input dimension; should be 0
        :param numpy::ndarray inputs: parameter values
        :param numpy::ndarray vec: input vector Jacobian applies to
        """
        self.Jacobian(outDimWrt, inDimWrt, inputs)
        self.jacobianAction = self.jacobian.dot(vec)

    def ApplyHessianImpl(self, outWrt, inWrt1, inWrt2, inputs, sens, vec):
        """
        Apply Hessian to ``vec`` for given ``sens`` and ``inputs``.

        :param int outWrt: output dimension; should be 0
        :param int inWrt1: input dimension; should be 0
        :param int inWrt2: input dimension; should be 0
        :param numpy::ndarray inputs: parameter values
        :param numpy::ndarray sens: sensitivity values
        :param numpy::ndarray vec: input vector Hessian applies to
        """
        assert inWrt1 == 0 and inWrt2 == 0

        npArray2dlVector(inputs[0], self.m)
        x = [self.u, self.m, self.p]

        # Solve for state and adjoint variables
        self.model.solveFwd(x[hp.STATE], x)

        self.model.misfit.grad(hp.STATE, x, self.adjrhs)
        self.adjrhs *= -sens[0]
        self.model.problem.solveAdj(x[hp.ADJOINT], x, self.adjrhs)

        # Hessian apply
        self.model.setPointForHessianEvaluations(x)
        HessApply = hp.ReducedHessian(self.model, misfit_only=True)

        npArray2dlVector(vec, self.hess0)
        HessApply.mult(self.hess0, self.hess1)
        self.hess1 *= -1.0

        self.hessAction = dlVector2npArray(self.hess1)


class Param2obs(mm.PyModPiece):
    """Parameter to observable map

    This class implements mapping from parameter to observations.
    """

    def __init__(self, model):
        """
        :param hippylib::Model model: a ``hippylib::Model`` instance
        """
        self.model = model

        self.u = self.model.generate_vector(component=hp.STATE)
        self.m = self.model.generate_vector(component=hp.PARAMETER)
        self.p = self.model.generate_vector(component=hp.ADJOINT)
        self.adjrhs = self.model.generate_vector(component=hp.STATE)
        self.gm = self.model.generate_vector(component=hp.PARAMETER)

        self.Bu = dl.Vector(self.model.misfit.B.mpi_comm())
        self.model.misfit.B.init_vector(self.Bu, 0)
        self.sens = dl.Vector(self.model.misfit.B.mpi_comm())
        self.model.misfit.B.init_vector(self.sens, 0)

        self.hess0 = self.model.generate_vector(component=hp.PARAMETER)
        self.hess1 = self.model.generate_vector(component=hp.PARAMETER)

        # number of finite element coefficient of parameter
        self.npar = self.model.problem.Vh[hp.PARAMETER].dim()

        # number of observation points
        self.nobs = self.model.misfit.d.size()

        mm.PyModPiece.__init__(self, [self.npar], [self.nobs])

    def EvaluateImpl(self, inputs):
        """
        Evaluate the observations for given ``inputs``.

        :param numpy::ndarray inputs: parameter values
        """
        npArray2dlVector(inputs[0], self.m)
        x = [self.u, self.m, None]

        # Solve for state variable
        self.model.solveFwd(x[hp.STATE], x)

        # Apply B operator to state variable
        self.model.misfit.B.mult(x[hp.STATE], self.Bu)

        y = dlVector2npArray(self.Bu)

        self.outputs = [y]

    def GradientImpl(self, outDimWrt, inDimWrt, inputs, sens):
        """
        Compute gradient; apply the transpose of Jacobian to ``sens``.

        :param int outDimWrt: output dimension; should be 0
        :param int inDimWrt: input dimension; should be 0
        :param numpy::ndarray inputs: parameter values
        :param numpy::ndarray sens: input vector the transpose of Jacobian
                                    applies to
        """
        npArray2dlVector(inputs[0], self.m)
        x = [self.u, self.m, self.p]

        # Solve for state and adjoint variables
        self._solves_stateadj(x, sens)

        # Evaluate Gradient
        self.model.problem.evalGradientParameter(x, self.gm)

        self.gradient = dlVector2npArray(self.gm)

    def _solves_stateadj(self, x, sens):
        """
        Solves the state and adjoint problems given parameters and sensitivity.

        :param numpy::ndarray x: parameter values
        :param numpy::ndarray sens: sensitivity
        """
        """
        """
        # Solve for state variable
        self.model.solveFwd(x[hp.STATE], x)

        # Solve for adjoint variable
        npArray2dlVector(sens, self.sens)
        self.model.misfit.B.transpmult(self.sens, self.adjrhs)
        self.adjrhs *= -1
        self.model.problem.solveAdj(x[hp.ADJOINT], x, self.adjrhs)


class LogBiLaplaceGaussian(mm.PyModPiece):
    """Log-bi-Laplace prior

    This class evaluates log of the bi-Laplacian prior.
    """

    def __init__(self, prior):
        """
        :param hippylib::BiLaplacianPrior prior: ``hippylib::BiLaplacianPrior``
                                                 instance
        """
        self.prior = prior

        self.m = const_dlVector(self.prior.A, 1)
        self.help = const_dlVector(self.prior.A, 0)

        self.npar = self.m.local_size()

        mm.PyModPiece.__init__(self, [self.npar], [1])

    def EvaluateImpl(self, inputs):
        """
        Evaluate the log of bi-Laplacian prior.

        :param numpy::ndarray inputs: input vector
        """
        npArray2dlVector(inputs[0], self.m)
        self.m.axpy(-1, self.prior.mean)
        self.prior.R.mult(self.m, self.help)

        self.outputs = [np.array([-0.5 * self.m.inner(self.help)])]

