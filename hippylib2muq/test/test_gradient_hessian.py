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

import unittest

import muq.Modeling as mm
import dolfin as dl
import hippylib as hp

import hippylib2muq as hm

import numpy as np


def u_boundary(x, on_boundary):
    return on_boundary and ( x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

class TestGradientHessian(unittest.TestCase):
    def setUp(self):
        mesh = dl.UnitSquareMesh(10,10)
        
        Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
        Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
        self.Vh = [Vh2, Vh1, Vh2]
        
        f = dl.Constant(0.0)
        
        u_bdr = dl.Expression("x[1]", degree=1)
        u_bdr0 = dl.Constant(0.0)
        bc = dl.DirichletBC(self.Vh[hp.STATE], u_bdr, u_boundary)
        bc0 = dl.DirichletBC(self.Vh[hp.STATE], u_bdr0, u_boundary)
        
        def pde_varf(u,m,p):
            return dl.exp(m)*dl.inner(dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx - f*p*dl.dx
        pde = hp.PDEVariationalProblem(self.Vh, pde_varf, bc, bc0, is_fwd_linear=True)
        
        #
        #  prior
        #
        gamma = .1
        delta = .5
        
        prior = hp.BiLaplacianPrior(self.Vh[hp.PARAMETER], gamma, delta, robin_bc=True)
        
        #
        #  likelihood
        #
        ntargets  = 10
        rel_noise = 0.005
        targets   = np.random.uniform(0.05, 0.95, [ntargets, 2])
        
        misfit = hp.PointwiseStateObservation(self.Vh[hp.STATE], targets)
        
        utrue = pde.generate_state()
        mtrue = dl.interpolate(dl.Expression("x[0] + x[1]", degree=1), 
                               self.Vh[hp.PARAMETER]).vector()
        
        x = [utrue, mtrue, None]
        pde.solveFwd(x[hp.STATE], x)
        misfit.B.mult(x[hp.STATE], misfit.d)
        MAX = misfit.d.norm("linf")
        noise_std_dev = rel_noise * MAX
        hp.parRandom.normal_perturb(noise_std_dev, misfit.d)
        misfit.noise_variance = noise_std_dev*noise_std_dev
        
        self.model = hp.Model(pde, prior, misfit)
        
        #
        # parameter to log likelihood Modpiece
        #
        self.param2loglikelihood = hm.Param2LogLikelihood(self.model)

    def test_param2loglikelihood(self):
        m0 = dl.interpolate(dl.Expression("sin(x[0])", degree=5), 
                            self.Vh[hp.PARAMETER]).vector()
        h = self.model.generate_vector(hp.PARAMETER)
        hp.parRandom.normal(1., h)
        sens = np.ones(1)

        grad = self.param2loglikelihood.Gradient(0,0,[m0],sens)
        grad_h = np.dot(grad, h)
        gradfd = self.param2loglikelihood.GradientByFD(0,0,[m0],sens)
        gradfd_h = np.dot(gradfd, h)

        err = abs(grad_h - gradfd_h)

        sum_gradh = 0.5*(grad_h + gradfd_h)
        if abs(sum_gradh) > 0.0:
            rel_err = err / abs(sum_gradh)
        else:
            rel_err = err

        Hh = self.param2loglikelihood.ApplyHessian(0,0,0,[m0],sens,h)
        Hhfd = self.param2loglikelihood.ApplyHessianByFD(0,0,0,[m0],sens,h)

        err_Hh = abs(Hh - Hhfd)
  
        sum_Hh = 0.5*(Hh + Hhfd)
        den = np.linalg.norm(sum_Hh)
        if den > 0.0:
            rel_err_Hh = np.linalg.norm(err_Hh, ord=np.inf) / den
        else:
            rel_err_Hh = np.linalg.norm(err_Hh, ord=np.inf)

        self.assertAlmostEqual(rel_err, 0.0, places=2)
        self.assertAlmostEqual(rel_err_Hh, 0.0, places=2)

if __name__ == '__main__':
    unittest.main()
