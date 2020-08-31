import math
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


class MultPSRF(object):

    """Computing the Multivariate Potential Scale Reduction Factor and effective sample size

    Return the square-root of :math:`\\hat{R}^p`

    Referece:
    [1] Brooks and Gelman 1998, General Methods for Monitoring Convergence of Iterative Simulations
    """

    def __init__(self, ndof, nsamps, nchain):
        """TODO: to be defined. """
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
        """TODO: Docstring for update_chain.

        :param samps: TODO
        :returns: TODO

        """
        wmean = samps.mean(axis=1)
        self.withinmean[self.ct, :] = wmean

        work1 = samps - wmean[:, None]
        self.W += np.matmul(work1, work1.T)

        self.ct += 1

    def compute_mpsrf(self):
        """TODO: Docstring for compute_mpsrf.
        :returns: TODO

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
        assert self.ct == self.nchain, "Not all the chains are passed to update_W"

        print("Number of chains: {0:>21d}".format(self.nchain))
        print("Number of samples in each chain: {0:>6d}".format(self.nsamps))
        print("="*40)
        print("MPSRF: {0:>10.3f}".format(self.mpsrf))


class PSRF(object):

    """Computing the Potential Scale Reduction Factor and effective sample size

    Return the square-root of :math:`\\hat{R}`

    Referece:
    [1] Brooks and Gelman 1998, General Methods for Monitoring Convergence of Iterative Simulations
    [2] Gelman et al. 2014, pp 286-287, Bayesian Data Analysis 
    """

    def __init__(self, nsamps, nchain, calEss=False, max_lag=None):
        """TODO: to be defined. """
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
        """TODO: Docstring for update_W.

        :param sample: TODO
        :returns: TODO

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
        """TODO: Docstring for compute_psrf.

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

