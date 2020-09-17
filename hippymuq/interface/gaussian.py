import numpy as np
import dolfin as df
import hippylib as hl
import pymuqModeling as mm

from ..utility.conversion import dfVector2npArray, const_dfVector, npArray2dfVector

class LaplaceGaussian(mm.PyGaussianBase):
    """
    An interface class between ``hippylib::LaplaceGaussian`` and \
    ``muq::GaussianBase``
    """
    def __init__(self, hl_prior, use_zero_mean=False):
        """
        :param hl_prior hippylib::BiLaplacianPrior: a hippylib class instance
        :param use_zero_mean bool: if True, mean = 0
        """
        if use_zero_mean:
            mean = np.zeros(hl_prior.mean.size())
        else:
            mean = dfVector2npArray(hl_prior.mean)
        mm.PyGaussianBase.__init__(self, mean)

        self.prior = hl_prior

        self.xa0 = const_dfVector(self.prior.R, 0)
        self.xa1 = const_dfVector(self.prior.R, 1)
        self.noises = const_dfVector(self.prior.sqrtR, 1)
        self.sample = const_dfVector(self.prior.R, 1)

    def ApplyCovariance(self, x):
        """
        Apply the covariance matrix.

        :param x numpy::ndarray: input vector
        """
        if x.ndim == 1:
            # Convert `x` to dolfin.Vector
            npArray2dfVector(x, self.xa0)

            # Solve
            nit = self.prior.Rsolver.solve(self.xa1, self.xa0)

            return dfVector2npArray(self.xa1)
        else:
            nrow = x.shape[0]
            ncol = x.shape[1]
            xt = x.T
            yarr = np.zeros((ncol, nrow))
            for i in range(ncol):
                xi = xt[i, :]

                # Convert `x` to dolfin.Vector
                npArray2dfVector(xi, self.xa0)

                # Solve
                nit = self.prior.Rsolver.solve(self.xa1, self.xa0)

                yarr[i, :] = dfVector2npArray(self.xa1)

            return yarr.T

    def ApplyPrecision(self, x):
        """
        Apply the precision matrix

        :param x numpy::ndarray: input vector
        """
        if x.ndim == 1:
            # Convert `x` to dolfin.Vector
            npArray2dfVector(x, self.xa1)

            # Apply R
            self.prior.R.mult(self.xa1, self.xa0)

            return dfVector2npArray(self.xa0)
        else:
            nrow = x.shape[0]
            ncol = x.shape[1]
            xt = x.T
            yarr = np.zeros((ncol, nrow))
            for i in range(ncol):
                xi = xt[i, :]

                # Convert `x` to dolfin.Vector
                npArray2dfVector(xi, self.xa1)

                # Apply R
                self.prior.R.mult(self.xa1, self.xa0)

                yarr[i, :] = dfVector2npArray(self.xa0)

            return yarr.T

    def SampleImpl(self, inputs):
        """
        Draw a sample.
        This is an overladed function of ``muq::PyGaussianBase``

        :param inputs numpy::ndarray: input vector
        """
        hl.parRandom.normal(1., self.noise)

        self.prior.sample(self.noise, self.sample)

        x = dfVector2npArray(self.sample)
        return x

class BiLaplaceGaussian(mm.PyGaussianBase):
    """
    A class interfacing between ``hippylib::BiLaplacianPrior`` and \
    ``muq::GaussianBase``
    """

    def __init__(self, hl_prior, use_zero_mean=False):
        """
        :param hl_prior hippylib::BiLaplacianPrior: a hippylib class instance
        :param use_zero_mean bool: if True, mean = 0
        """
        if use_zero_mean:
            mean = np.zeros(hl_prior.mean.size())
        else:
            mean = dfVector2npArray(hl_prior.mean)
        mm.PyGaussianBase.__init__(self, mean)

        self.prior = hl_prior

        # temporary df.Vector for ApplyCovariance
        self.vecc = const_dfVector(self.prior.A, 1)

        # for ApplyPrecision
        self.vecr = const_dfVector(self.prior.A, 0)

        # for ApplyCovSqrt
        self.vecc1 = const_dfVector(self.prior.sqrtM, 0)
        self.vecc2 = const_dfVector(self.prior.A, 1)

        # for self.ApplyPrecSqrt
        self.vecr1 = const_dfVector(self.prior.sqrtM, 0)
        self.vecr2 = const_dfVector(self.prior.M, 1)
        self.vecr3 = const_dfVector(self.prior.A, 0)

        # for noise
        self.noise = df.Vector(self.prior.sqrtM.mpi_comm())
        self.prior.init_vector(self.noise, "noise")

        # for working vector
        self.xa0 = const_dfVector(self.prior.A, 0)
        self.xa1 = const_dfVector(self.prior.A, 1)
        self.xsqm1 = const_dfVector(self.prior.sqrtM, 1)

    def ApplyCovariance(self, x):
        """
        Apply the covariance matrix.

        :param x numpy::ndarray: input vector
        """
        if x.ndim == 1:
            # Convert `x` to dolfin.Vector
            npArray2dfVector(x, self.xa0)

            # Solve
            nit = self.prior.Rsolver.solve(self.vecc, self.xa0)

            return dfVector2npArray(self.vecc)
        else:
            nrow = x.shape[0]
            ncol = x.shape[1]
            xt = x.T
            yarr = np.zeros((ncol, nrow))
            for i in range(ncol):
                xi = xt[i, :]

                # Convert `x` to dolfin.Vector
                npArray2dfVector(xi, self.xa0)

                # Solve
                nit = self.prior.Rsolver.solve(self.vecc, self.xa0)

                yarr[i, :] = dfVector2npArray(self.vecc)

            return yarr.T

    def ApplyPrecision(self, x):
        """
        Apply the precision matrix

        :param x numpy::ndarray: input vector
        """
        if x.ndim == 1:
            # Convert `x` to dolfin.Vector
            npArray2dfVector(x, self.xa1)

            # Apply R
            self.prior.R.mult(self.xa1, self.vecr)

            return dfVector2npArray(self.vecr)
        else:
            nrow = x.shape[0]
            ncol = x.shape[1]
            xt = x.T
            yarr = np.zeros((ncol, nrow))
            for i in range(ncol):
                xi = xt[i, :]

                # Convert `x` to dolfin.Vector
                npArray2dfVector(xi, self.xa1)

                # Apply R
                self.prior.R.mult(self.xa1, self.vecr)

                yarr[i, :] = dfVector2npArray(self.vecr)

            return yarr.T

    def ApplyCovSqrt(self, x):
        """
        Apply the square root of covariance matrix.

        :param x numpy::ndarray: input vector
        """
        if x.ndim == 1:

            # Convert `x` to dolfin.Vector
            npArray2dfVector(x, self.xsqm1)

            # Apply sqrtM to df_x
            self.prior.sqrtM.mult(self.xsqm1, self.vecc1)

            # Solve
            self.prior.Asolver.solve(self.vecc2, self.vecc1)

            return dfVector2npArray(self.vecc2)
        else:
            nrow = x.shape[0]
            ncol = x.shape[1]
            xt = x.T
            yarr = np.zeros((ncol, nrow))
            for i in range(ncol):
                xi = xt[i, :]

                # Convert `x` to dolfin.Vector
                npArray2dfVector(xi, self.xsqm1)

                # Apply sqrtM to df_x
                self.prior.sqrtM.mult(self.xsqm1, self.vecc1)

                # Solve
                self.prior.Asolver.solve(self.vecc2, self.vecc1)

                yarr[i, :] = dfVector2npArray(self.vecc2)

            return yarr.T

    def ApplyPrecSqrt(self, x):
        """
        Apply the square root of precision matrix.

        :param x numpy::ndarray: input vector
        """
        if x.ndim == 1:
            # Convert `x` to dolfin.Vector
            npArray2dfVector(x, self.xsqm1)

            # Apply sqrtM to df_x
            self.prior.sqrtM.mult(self.xsqm1, self.vecr1)

            # Solve M temppc1 = z
            self.prior.M.mult(self.vecr1, self.vecr2)

            # Apply A
            self.prior.A.mult(self.vecr2, self.vecr3)

            return dfVector2npArray(self.vecr3)
        else:
            nrow = x.shape[0]
            ncol = x.shape[1]
            xt = x.T
            yarr = np.zeros((ncol, nrow))
            for i in range(ncol):
                xi = xt[i, :]

                # Convert `x` to dolfin.Vector
                npArray2dfVector(xi, self.xsqm1)

                # Apply sqrtM to df_x
                self.prior.sqrtM.mult(self.xsqm1, self.vecr1)

                # Solve M temppc1 = z
                self.prior.M.mult(self.vecr1, self.vecr2)

                # Apply A
                self.prior.A.mult(self.vecr2, self.vecr3)

                yarr[i, :] = dfVector2npArray(self.vecr3)

            return yarr.T


    def SampleImpl(self, inputs):
        """
        Draw a sample.
        This is an overladed function of ``muq::PyGaussianBase``

        :param inputs numpy::ndarray: input vector
        """
        hl.parRandom.normal(1., self.noise)
        x = dfVector2npArray(self.noise)
        return self.GetMean() + self.ApplyCovSqrt(x)


class LAPosteriorGaussian(mm.PyGaussianBase):
    """
    A class interfacing between ``hippylib::GaussianLRPosterior`` and 
    ``muq:PyGaussianBase``
    """
    def __init__(self, lapost, use_zero_mean=False):
        """
        :param lapost hippylib::GaussianLRPosterior: a hippylib class instance
        :param use_zero_mean bool: if True, mean = 0
        """
        self.lapost = lapost
        if use_zero_mean:
            mean = np.zeros(lapost.mean.size())
        else:
            mean = dfVector2npArray(lapost.mean)
        mm.PyGaussianBase.__init__(self, mean)

        self.noise = df.Vector(self.lapost.prior.sqrtM.mpi_comm())
        self.lapost.prior.init_vector(self.noise, "noise")

        self.help0 = df.Vector(self.lapost.Hlr.help.mpi_comm())
        self.lapost.init_vector(self.help0, 0)

        self.help1 = df.Vector(self.lapost.Hlr.help.mpi_comm())
        self.lapost.init_vector(self.help1, 1)

        self.wprior = const_dfVector(self.lapost.prior.A, 1)
        self.wpost = const_dfVector(self.lapost.prior.A, 1)

    def ApplyPrecision(self, x):
        """
        Apply the precision matrix.

        :param x numpy::ndarray: input vector
        """
        npArray2dfVector(x, self.help1)
        self.lapost.Hlr.mult(self.help1, self.help0)

        return dfVector2npArray(self.help0)

    def ApplyCovariance(self, x):
        """
        Apply the covariance matrix.

        :param x numpy::ndarray: input vector
        """
        npArray2dfVector(x, self.help0)
        self.lapost.Hlr.solve(self.help1, self.help0)

        return dfVector2npArray(self.help1)

    def ApplyCovSqrt(self, x):
        """
        Apply the square root of covariance matrix.

        :param x numpy::ndarray: input vector
        """
        npArray2dfVector(x, self.noise)
        self.lapost.sample(self.noise, self.wprior, self.wpost, add_mean=False)

        return dfVector2npArray(self.wpost)

    def SampleImpl(self, inputs):
        """
        Draw a sample.
        This is an overladed function of ``muq::PyGaussianBase``

        :param inputs numpy::ndarray: input vector
        """
        hl.parRandom.normal(1., self.noise)
        x = dfVector2npArray(self.noise)
        return self.GetMean() + self.ApplyCovSqrt(x)
