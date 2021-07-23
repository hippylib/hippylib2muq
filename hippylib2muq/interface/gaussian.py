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
This module provides a set of wrappers that expose some functions of
``hippylib`` (the prior Gaussian distributions and the low-rank based Laplace
approximation to the posterior distribution) to ``muq``.
"""
import numpy as np
import dolfin as dl
import hippylib as hp
import muq.Modeling as mm

from ..utility.conversion import dlVector2npArray, const_dlVector, npArray2dlVector


class LaplaceGaussian(mm.PyGaussianBase):
    """ The prior Gaussian distribution with Laplacian-like covariance operator

    An interface class between ``hippylib::LaplaceGaussian`` and \
    ``muq::GaussianBase``.
    This class is appropriate for 1D (parameter) problems.
    """

    def __init__(self, hp_prior, use_zero_mean=False):
        """
        :param hippylib::LaplacianPrior hp_prior: a hippylib class instance
        :param bool use_zero_mean: if True, mean = 0
        """
        if use_zero_mean:
            mean = np.zeros(hp_prior.mean.size())
        else:
            mean = dlVector2npArray(hp_prior.mean)
        mm.PyGaussianBase.__init__(self, mean)

        self.prior = hp_prior

        self.xa0 = const_dlVector(self.prior.R, 0)
        self.xa1 = const_dlVector(self.prior.R, 1)
        self.noise = const_dlVector(self.prior.sqrtR, 1)
        self.sample = const_dlVector(self.prior.R, 1)

    def ApplyCovariance(self, x):
        """
        Apply the covariance matrix to ``x``.

        :param numpy::ndarray x: input vector
        """
        if x.ndim == 1:
            # Convert `x` to dolfin.Vector
            npArray2dlVector(x, self.xa0)

            # Solve
            nit = self.prior.Rsolver.solve(self.xa1, self.xa0)

            return dlVector2npArray(self.xa1)
        else:
            ymat = np.zeros(x.shape)
            for i in range(x.shape[1]):
                npArray2dlVector(x[:, i], self.xa0)
                nit = self.prior.Rsolver.solve(self.xa1, self.xa0)
                ymat[:, i] = dlVector2npArray(self.xa1)

            return ymat

    def ApplyPrecision(self, x):
        """
        Apply the precision matrix to ``x``.

        :param numpy::ndarray x: input vector
        """
        if x.ndim == 1:
            # Convert `x` to dolfin.Vector
            npArray2dlVector(x, self.xa1)

            # Apply R
            self.prior.R.mult(self.xa1, self.xa0)

            return dlVector2npArray(self.xa0)
        else:
            ymat = np.zeros(x.shape)
            for i in range(x.shape):
                npArray2dlVector(x[:, i], self.xa1)
                self.prior.R.mult(self.xa1, self.xa0)
                ymat[:, i] = dlVector2npArray(self.xa0)

            return ymat

    def SampleImpl(self, inputs):
        """
        Draw a sample from the prior distribution.
        This is an overloaded function of ``muq::PyGaussianBase``.
        The argument ``inputs`` is not used, but should be given when
        ``SampleImpl`` is called.

        :param numpy::ndarray inputs: input vector
        """
        hp.parRandom.normal(1.0, self.noise)

        self.prior.sample(self.noise, self.sample)

        x = dlVector2npArray(self.sample)
        return x


class BiLaplaceGaussian(mm.PyGaussianBase):
    """ The prior Gaussian distribution with Laplacian-like covariance operator

    A class interfacing between ``hippylib::BiLaplacianPrior`` and \
    ``muq::GaussianBase``.
    """

    def __init__(self, hp_prior, use_zero_mean=False):
        """
        :param hippylib::BiLaplacianPrior hp_prior: a hippylib class instance
        :param bool use_zero_mean: if True, mean = 0
        """
        if use_zero_mean:
            mean = np.zeros(hp_prior.mean.size())
        else:
            mean = dlVector2npArray(hp_prior.mean)
        mm.PyGaussianBase.__init__(self, mean)

        self.prior = hp_prior

        # temporary dl.Vector for ApplyCovariance
        self.vecc = const_dlVector(self.prior.A, 1)

        # for ApplyPrecision
        self.vecr = const_dlVector(self.prior.A, 0)

        # for ApplyCovSqrt
        self.vecc1 = const_dlVector(self.prior.sqrtM, 0)
        self.vecc2 = const_dlVector(self.prior.A, 1)

        # for self.ApplyPrecSqrt
        self.vecr1 = const_dlVector(self.prior.sqrtM, 0)
        self.vecr2 = const_dlVector(self.prior.M, 1)
        self.vecr3 = const_dlVector(self.prior.A, 0)

        # for noise
        self.noise = dl.Vector(self.prior.sqrtM.mpi_comm())
        self.prior.init_vector(self.noise, "noise")

        # for working vector
        self.xa0 = const_dlVector(self.prior.A, 0)
        self.xa1 = const_dlVector(self.prior.A, 1)
        self.xsqm1 = const_dlVector(self.prior.sqrtM, 1)

    def ApplyCovariance(self, x):
        """
        Apply the covariance matrix to ``x``.

        :param numpy::ndarray x: input vector
        """
        if x.ndim == 1:
            # Convert `x` to dolfin.Vector
            npArray2dlVector(x, self.xa0)

            # Solve
            nit = self.prior.Rsolver.solve(self.vecc, self.xa0)

            return dlVector2npArray(self.vecc)
        else:
            ymat = np.zeros(x.shape)
            for i in range(x.shape[1]):
                npArray2dlVector(x[:, i], self.xa0)
                nit = self.prior.Rsolver.solve(self.vecc, self.xa0)
                ymat[:, i] = dlVector2npArray(self.vecc)

            return ymat

    def ApplyPrecision(self, x):
        """
        Apply the precision matrix to ``x``.

        :param numpy::ndarray x: input vector
        """
        if x.ndim == 1:
            # Convert `x` to dolfin.Vector
            npArray2dlVector(x, self.xa1)

            # Apply R
            self.prior.R.mult(self.xa1, self.vecr)

            return dlVector2npArray(self.vecr)
        else:
            ymat = np.zeros(x.shape)
            for i in range(x.shape[1]):
                npArray2dlVector(x[:, i], self.xa1)
                self.prior.R.mult(self.xa1, self.vecr)
                ymat[:, i] = dlVector2npArray(self.vecr)

            return ymat

    def ApplyCovSqrt(self, x):
        """
        Apply the square root of covariance matrix to ``x``.

        :param numpy::ndarray x: input vector
        """
        if x.ndim == 1:
            # Convert `x` to dolfin.Vector
            npArray2dlVector(x, self.xsqm1)

            # Apply sqrtM to dl_x
            self.prior.sqrtM.mult(self.xsqm1, self.vecc1)

            # Solve
            self.prior.Asolver.solve(self.vecc2, self.vecc1)

            return dlVector2npArray(self.vecc2)
        else:
            ymat = np.zeros(x.shape)
            for i in range(x.shape[1]):
                npArray2dlVector(x[:, i], self.xsqm1)
                self.prior.sqrtM.mult(self.xsqm1, self.vecc1)
                self.prior.Asolver.solve(self.vecc2, self.vecc1)

                ymat[:, i] = dlVector2npArray(self.vecc2)

            return ymat

    def ApplyPrecSqrt(self, x):
        """
        Apply the square root of precision matrix to ``x``.

        :param numpy::ndarray x: input vector
        """
        if x.ndim == 1:
            # Convert `x` to dolfin.Vector
            npArray2dlVector(x, self.xsqm1)

            # Apply sqrtM to dl_x
            self.prior.sqrtM.mult(self.xsqm1, self.vecr1)

            # Solve M temppc1 = z
            self.prior.M.mult(self.vecr1, self.vecr2)

            # Apply A
            self.prior.A.mult(self.vecr2, self.vecr3)

            return dlVector2npArray(self.vecr3)
        else:
            ymat = np.zeros(x.shape)
            for i in range(x.shape[1]):
                npArray2dlVector(x[:, i], self.xsqm1)
                self.prior.sqrtM.mult(self.xsqm1, self.vecr1)
                self.prior.M.mult(self.vecr1, self.vecr2)
                self.prior.A.mult(self.vecr2, self.vecr3)
                ymat[:, i] = dlVector2npArray(self.vecr3)

            return ymat

    def SampleImpl(self, inputs):
        """
        Draw a sample from the prior distribution.
        This is an overloaded function of ``muq::PyGaussianBase``
        The argument ``inputs`` is not used, but should be given when
        ``SampleImpl`` is called.

        :param numpy::ndarray inputs: input vector
        """
        hp.parRandom.normal(1.0, self.noise)
        x = dlVector2npArray(self.noise)
        return self.GetMean() + self.ApplyCovSqrt(x)


class LAPosteriorGaussian(mm.PyGaussianBase):
    """ Low-rank based Laplace approximation to the posterior distribution

    A class interfacing between ``hippylib::GaussianLRPosterior`` and \
    ``muq:PyGaussianBase``.
    """

    def __init__(self, lapost, use_zero_mean=False):
        """
        :param hippylib::GaussianLRPosterior lapost: a hippylib class instance
        :param bool use_zero_mean: if True, mean = 0
        """
        self.lapost = lapost
        if use_zero_mean:
            mean = np.zeros(lapost.mean.size())
        else:
            mean = dlVector2npArray(lapost.mean)
        mm.PyGaussianBase.__init__(self, mean)

        self.noise = dl.Vector()
        self.lapost.prior.init_vector(self.noise, "noise")

        self.help0 = dl.Vector(self.lapost.Hlr.help.mpi_comm())
        self.lapost.init_vector(self.help0, 0)

        self.help1 = dl.Vector(self.lapost.Hlr.help.mpi_comm())
        self.lapost.init_vector(self.help1, 1)

        self.wprior = dl.Vector()
        self.lapost.init_vector(self.wprior, 1)
        self.wpost = dl.Vector()
        self.lapost.init_vector(self.wpost, 1)

    def ApplyPrecision(self, x):
        """
        Apply the precision matrix to ``x``.

        :param numpy::ndarray x: input vector
        """
        if x.ndim == 1:
            npArray2dlVector(x, self.help1)
            self.lapost.Hlr.mult(self.help1, self.help0)

            return dlVector2npArray(self.help0)
        else:
            ymat = np.zeros(x.shape)
            for i in ragne(x.shape[1]):
                npArray2dlVector(x[:, i], self.help1)
                self.lapost.Hlr.mult(self.help1, self.help0)
                ymat[:, i] = dlVector2npArray(self.help0)

            return ymat

    def ApplyCovariance(self, x):
        """
        Apply the covariance matrix to ``x``.

        :param numpy::ndarray x: input vector
        """
        if x.ndim == 1:
            npArray2dlVector(x, self.help0)
            self.lapost.Hlr.solve(self.help1, self.help0)

            return dlVector2npArray(self.help1)
        else:
            ymat = np.zeros(x.shape)
            for i in range(x.shape[1]):
                npArray2dlVector(x[:, i], self.help0)
                self.lapost.Hlr.solve(self.help1, self.help0)

                ymat[:, i] = dlVector2npArray(self.help1)

            return ymat

    def ApplyCovSqrt(self, x):
        """
        Apply the square root of covariance matrix to ``x``.

        :param numpy::ndarray x: input vector
        """
        if x.ndim == 1:
            npArray2dlVector(x, self.noise)
            self.lapost.sample(self.noise, self.wprior, self.wpost, add_mean=False)

            return dlVector2npArray(self.wpost)
        else:
            ymat = np.zeros(x.shape)
            for i in range(x.shape[1]):
                npArray2dlVector(x[:, i], self.noise)
                self.lapost.sample(self.noise, self.wprior, self.wpost, add_mean=False)
                ymat[:, i] = dlVector2npArray(self.wpost)

            return ymat

    def SampleImpl(self, inputs):
        """
        Draw a sample from the approximated posterior distribution.
        This is an overloaded function of ``muq::PyGaussianBase``.

        :param numpy::ndarray inputs: input vector
        """
        hp.parRandom.normal(1.0, self.noise)
        x = dlVector2npArray(self.noise)
        return self.GetMean() + self.ApplyCovSqrt(x)
