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
import numpy as np
import dolfin as dl
import hippylib as hp


class Hessian_avg:
    def __init__(self, R):
        """
        Incrementally update the expected Hessian H using its local spectral
        decomposition:
        :math:`H \\sim B X LAMBDA X^{T} B^{T}`

        :param: R : the pre-conditioning matrix of the generalized eigenvalue
                    problem :math:`H x = lambda B x`
        """

        self.m = 0
        self.lmbda = None
        self.lmbdaHist = None
        self.X = None
        self.k = None
        self.Z = None

        self.B = R

    def initialize(self, l, V):
        self.lmbda = l
        self.lmbdaHist = l
        self.X = V
        self.k = V.nvec()

        self.Z = hp.MultiVector(V[0], 2 * self.k)

        self.m = 1

    def update(self, l, V):
        """
        Update the expected Hessian with local eigenvalues and eigenvectors of
        the generalized eigenvalue problem

        :param l: local eigenvalues
        :param V: local eigenvectors
        """
        if self.m == 0:
            self.initialize(l, V)
            return

        self.Z.zero()
        for i in range(2 * self.k):
            if i < self.k:
                self.Z[i].axpy(1.0, self.X[i])
            else:
                j = i - self.k
                self.Z[i].axpy(1.0, V[j])

        d = np.concatenate((self.m * self.lmbda, l))
        DD = np.diag(d) / (self.m + 1)

        _, R = self.Z.Borthogonalize(self.B)

        A = np.matmul(R, np.matmul(DD, R.T))
        sigma, W = np.linalg.eigh(A)

        sort_perm = sigma.argsort()
        sort_perm = sort_perm[::-1]
        self.lmbda = sigma[sort_perm[: self.k]]
        self.lmbdaHist = np.vstack((self.lmbdaHist, l))
        W = W[:, sort_perm[: self.k]]

        hp.MvDSmatMult(self.Z, W, self.X)

        self.m += 1


def average_H(model, nu, k, p, nsample, wrt="prior"):
    """
    Average the Hessian over the domain specified by `wrt`

    :param model: the inverse problem model
    :param nu: Laplace approximation of the posterior
    :param k: the size of the local generalized eigenvalue problem
    :param p: the number of over sampling for the randomized method
    :param nsample: the number of samples for the average
    :param wrt: when wrt = "prior", the average is over the prior distribution.
                when wrt = "LA", the average is over the Laplace posterior.
    """
    z = dl.Vector()
    nu.init_vector(z, "noise")

    sample = dl.Function(model.problem.Vh[hp.PARAMETER]).vector()
    if wrt == "LA":
        sample_prior = dl.Function(model.problem.Vh[hp.PARAMETER]).vector()

    xi = model.generate_vector()

    Havg = Hessian_avg(model.prior.R)
    for i in range(nsample):
        if i % 100 == 0:
            print("Calculating averaged Hessian: ", i)
        hp.parRandom.normal(1.0, z)

        if wrt == "prior":
            model.prior.sample(z, sample)
        elif wrt == "LA":
            nu.sample(z, sample_prior, sample)

        xi[hp.PARAMETER] = sample
        model.solveFwd(xi[hp.STATE], xi)
        model.solveAdj(xi[hp.ADJOINT], xi)

        model.setPointForHessianEvaluations(xi, gauss_newton_approx=True)
        H = hp.ReducedHessian(model, misfit_only=True)

        Om = hp.MultiVector(xi[hp.PARAMETER], k + p)
        hp.parRandom.normal(1.0, Om)
        l, U = hp.doublePassG(H, model.prior.R, model.prior.Rsolver, Om, k)

        Havg.update(l, U)

    return Havg

