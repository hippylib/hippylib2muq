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
This module provides a convergence diagnostic for samples drawn from MCMC methods.
"""
import math
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


class MultPSRF(object):
    """Computing the Multivariate Potential Scale Reduction Factor

    This class is to compute the Multivariate Potential Scale Reduction Factor 
    (MPSRF) described in [Brooks1998]_.
    Note that MPSRF is the square-root version, i.e., :math:`\\hat{R}^p` where 
    :math:`\\hat{R}^p` is defined by Equation (4.1) in [Brooks1998]_.

    .. [Brooks1998] Brooks and Gelman, 1998, General Methods for 
                   Monitoring Convergence of Iterative Simulations.
    """

    def __init__(self, ndof, nsamps, nchain):
        """
        :param int ndof: dimension of the parameter
        :param int nsamps: number of samples
        :param int nchain: number of MCMC chains
        """
        self.ndof = ndof
        self.nsamps = nsamps
        self.nchain = nchain
        self.ct = 0
        self.W = np.zeros((self.ndof, self.ndof))
        self.Bn = np.zeros((self.ndof, self.ndof))
        self.withinmean = np.zeros((self.nchain, self.ndof))
        self.mean = np.zeros(self.ndof)
        self.mpsrf = None

    def update_W(self, samps):
        """
        Update the within-sequence varance matrix W for a chain ``samps``.

        :param numpy:ndarray samps: a sequence of samples generated
        """
        wmean = samps.mean(axis=1)
        self.withinmean[self.ct, :] = wmean

        work1 = samps - wmean[:, None]
        self.W += np.matmul(work1, work1.T)

        self.ct += 1

    def compute_mpsrf(self):
        """
        Compute MPSRF.
        """
        assert self.ct == self.nchain, "Not all the chains are passed to update_W"

        self.W /= (self.nchain * (self.nsamps - 1))

        self.mean = self.withinmean.mean(axis=0)
        work1 = self.withinmean - self.mean
        self.Bn = np.matmul(work1.T, work1)
        self.Bn /= (self.nchain - 1)

        eigvals = scipy.linalg.eigh(self.Bn, self.W, eigvals_only=True,
                eigvals=(self.ndof-1, self.ndof-1))

        lambda1 = eigvals[0]
        mpsrf = (self.nsamps - 1.)/self.nsamps +\
                (self.nchain + 1.)/self.nchain * lambda1

        self.mpsrf = math.sqrt(mpsrf)

        return self.mpsrf

    def print_result(self):
        """
        Print the description and the result of MCMC chains and its diagnostic.
        """
        assert self.ct == self.nchain, "Not all the chains are passed to update_W"

        print("Number of chains: {0:>21d}".format(self.nchain))
        print("Number of samples in each chain: {0:>6d}".format(self.nsamps))
        print("="*40)
        print("MPSRF: {0:>10.3f}".format(self.mpsrf))


class PSRF(object):

    """Computing the Potential Scale Reduction Factor and the effective sample size

    This class is to compute the Potential Scale Reduction Factor (PSRF) and
    the effective sample size (ESS) as described in [Brooks1998]_ and [Gelman2014]_.
    Note that PSRF is the square-root version of :math:`\\hat{R}` where 
    :math:`\\hat{R}` is defined by Equation (1.1) defined in [Brooks1998].

    .. [Gelman2014] Gelman et al., 2014, Bayesian Data Analysis, pp 286-287.
    """

    def __init__(self, nsamps, nchain, calEss=False, max_lag=None):
        """
        :param int nsamps: number of samples
        :param int nchain: number of MCMC chains
        :param bool calEss: if True, ESS is calculated
        :param int max_lag: maximum of time lag for computing the autocorrelation 
                            function
        """
        self.nsamps = nsamps
        self.nchain = nchain
        self.ct = 0
        self.W = 0
        self.Bn = 0
        self.withinmean = np.zeros(self.nchain)

        self.mean = None
        self.variance = None
        self.psrf = None
        self.ess = None

        if calEss:
            self.calEss = calEss
            if max_lag is None:
                self.max_lag = self.nsamps // 2
            else:
                self.max_lag = max_lag

            self.variogram_chain = np.zeros((self.nchain, self.max_lag))

    def update_W(self, sample):
        """
        Update the within-sequence varance W for a chain ``samps``.

        :param numpy:ndarray samps: a sequence of samples generated
        """
        wmean = sample.mean()
        self.withinmean[self.ct] = wmean

        work1 = sample - wmean
        self.W += np.dot(work1, work1)

        if self.calEss:
            for k in range(self.max_lag):
                t = k + 1
                work2 = sample[t:] - sample[:-t]
                self.variogram_chain[self.ct, k] =\
                    np.dot(work2, work2) / (self.nsamps - t)

        self.ct += 1

    def compute_PsrfEss(self, plot_acorr=False, write_acorr=False, fname=None):
        """
        Compute PSRF and ESS

        :param bool plot_acorr: if True, plot the autocorrelation function
        :param bool write_acorr: if True, write the autocorrelation function to
                                 a file
        :param string fname: file name for the autocorrelation function result
        """
        assert self.ct == self.nchain, "Not all the chains are passed to update_W"

        self.W /= (self.nchain * (self.nsamps - 1))

        self.mean = self.withinmean.mean()

        work1 = self.withinmean - self.mean
        self.Bn = np.dot(work1, work1)
        self.Bn /= (self.nchain - 1)

        self.variance = (self.nsamps - 1.)/self.nsamps * self.W +\
                        (self.nchain + 1.)/self.nchain * self.Bn
        self.psrf = math.sqrt(self.variance / self.W)

        if self.calEss:
            Vt = self.variogram_chain.mean(axis=0)
            autocorrelation = 1.0 - Vt / (2.0*self.variance)

            two_sum = [sum(autocorrelation[k:k+2]) for k in range(1, autocorrelation.size, 2)]
            first_negative_id = next((i for i, x in enumerate(two_sum) if x < 0.0), len(two_sum))
            acorr_sum = autocorrelation[0] + sum(two_sum[:first_negative_id])
            if first_negative_id*2 == self.max_lag:
                print('first_negative_id*2:', first_negative_id*2)

            iacf = 1.0 + 2.0 * acorr_sum
            self.ess = self.nsamps*self.nchain / iacf

            if plot_acorr:
                plt.plot([i+1 for i in range(self.max_lag)], autocorrelation)
                plt.show()

            if write_acorr:
                assert fname is not None, "file name is not specified"

                with open(fname, 'w') as f:
                    f.write('lag autocorrelation\n')
                    lag = np.arange(0, self.max_lag+1)
                    np.savetxt(f, np.transpose([lag, np.hstack((1.0,autocorrelation))]), fmt='%.4e')

            return self.psrf, self.ess
        else:
            return self.psrf

    def print_result(self):
        """
        Print the description and the result of MCMC chains and its diagnostic.
        """
        assert self.ct == self.nchain, "Not all the chains are passed to update_W"

        print("Number of chains: {0:>21d}".format(self.nchain))
        print("Number of samples in each chain: {0:>6d}".format(self.nsamps))
        print("="*40)
        print("Sample mean averaged over all chains: {0:>11.4E}".format(self.mean))
        print("Within-sequence variance W: {0:>21.4E}".format(self.W))
        print("Between-sequence variance B: {0:>20.4E}".format(self.Bn*self.nsamps))
        print("Pooled variance estimate: {0:>23.4E}".format(self.variance))
        print("")
        print("PSRF: {0:>10.4f}".format(self.psrf))
        print("ESS : {0:>10.4f}".format(self.ess))
        print("Standard error: {0:5.4f}".format(math.sqrt(self.variance/self.ess)))

