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
This module contains some functions related to the quantity of interest.
"""
import hippylib as hp

def cal_qoiTracer(pde, qoi, muq_samps):
    """
    This function is for tracing the quantity of interest.

    :param hippylib:PDEProblem pde: a hippylib:PDEProblem instance
    :param qoi: the quantity of interest; it should contain the member function
                named as ``eval`` which evaluates the value of qoi
    :param muq_samps: samples generated from ``muq`` sampler
    """
    samps_mat = muq_samps.AsMatrix()
    nums = samps_mat.shape[1]
    tracer = hp.QoiTracer(nums)

    ct = 0
    u = pde.generate_state()
    m = pde.generate_parameter()
    while ct < nums:
        m.set_local(samps_mat[:,ct])
        x = [u, m, None]
        pde.solveFwd(u, x)
        q = qoi.eval([u, m])
        tracer.append(ct, q)
        ct += 1
    return tracer


def track_qoiTracer(pde, qoi, method_list, max_lag=None):
    """
    This function computes the autocorrelation function and the effective sample
    size of the quantity of interest.

    :param hippylib:PDEProblem pde: a hippylib:PDEProblem instance
    :param qoi: the quantity of interest; it should contain the member function
    :param dictionary method_list: a dictionary containing MCMC methods descriptions
                                   with samples generated from muq sampler
    :param int max_lag: maximum of time lag for computing the autocorrelation 
                        function
    """
    qoi_dataset = dict()
    for mName, method in method_list.items():
        qoi_data = dict()
        samps = method['Samples']

        # Compute QOI
        tracer = cal_qoiTracer(pde, qoi, samps)

        # Estimate IAT
        iact, lags, acorrs = hp.integratedAutocorrelationTime(tracer.data, max_lag=max_lag)

        # Estimate ESS
        ess = samps.size() / iact

        # Save computed results
        qoi_data['qoi'] = tracer.data
        qoi_data['iact'] = iact
        qoi_data['ess'] = ess

        qoi_dataset[mName] = qoi_data
    return qoi_dataset
