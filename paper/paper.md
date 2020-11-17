---
title: 'hIPPYlib-MUQ: Scalable Markov Chain Monte Carlo Sampling Methods for Large-scale Bayesian Inverse Problems Governed by PDEs'
tags:
  - Python
  - Inverse problems
  - PDE-constrained Optimization
  - Low-rank approximation
  - Bayesian inference
  - Uncertainty quantification
  - MCMC sampling
authors:
  - name: Ki-Tae Kim
    affiliation: 1
  - name: Umberto Villa
    affiliation: 2
  - name: Matthew Parno
    affiliation: 3
  - name: Noemi Petra
    affiliation: 1
  - name: Youssef Marzouk
    affiliation: 4
  - name: Omar Ghattas
    affiliation: 5
affiliations:
  - name: Applied Mathematics, School of Natural Sciences, University of California, Merced
    index: 1
  - name: Department of Electrical & Systems Engineering, Washington University in St. Louis
    index: 2
  - name: Cold Regions Research & Engineering Laboratory, The United States Army Corps of Engineers
    index: 3
  - name: Department of Aeronautics and Astronautics, Massachusetts Institute of Technology
    index: 4
  - name: Oden Institute for Computational Engineering and Sciences, Department of Mechanical Engineering, and Department of Geological Sciences, The University of Texas at Austin
    index: 5
date:
bibliography: paper.bib
---

# Summary

Inverse problems arise in all areas of science, engineering, technology, and
medicine and are often governed by complex physics-based mathematical models.
These models are often subject to considerable uncertainties stemming from
unknown or uncertain inputs (e.g., coefficient fields, constitutive laws,
source terms, geometries, initial and/or boundary conditions) as well as from
noisy and limited observations. While many of these input parameters cannot be
directly observed, they can be inferred indirectly from observations via an
inverse problem. Bayesian inversion provides a framework for integration of
data with complex physics-based models to quantify and reduce uncertainties in
model predictions [@Kaipio2006]. Bayesian inversion with complex forward models
faces several computational challenges.  First, characterizing the posterior
distribution of the parameters of interest or subsequent predictions often
requires repeated evaluations of large-scale partial differential equation
(PDE) models.  Second, the posterior distribution often has a complex structure
stemming from nonlinear parameter-to-observable maps and heterogeneous sources
of data.  Third, the parameters often are fields, which when discretized lead
to very high-dimensional posteriors.

Our objective is to create a robust and scalable software framework to tackle
large-scale PDE-constrained Bayesian inverse problems across a wide range of
science and engineering fields.  `hIPPYlib-MUQ` is a Python interface between
two open source software packages, `hIPPYlib` and `MUQ`, which have
complementary capabilities.  [`hIPPYlib`](https://hippylib.github.io)
[@Villa2018; @VillaPetraGhattas20] is an extensible software package aimed at
solving deterministic and linearized Bayesian inverse problems governed by
PDEs.  Based on FEniCS [@LoggWells10; @LoggMardalGarth12; @LangtangenLogg17]
for the solution of forward PDE problems and on PETSc [@petsc-web-page] for
scalable and parallel linear algebra operations and solvers, `hIPPYlib`
implements globalized inexact Newton-conjugate gradient methods, adjoint-based
computation of gradients and Hessian actions, low-rank approximation of
Hessians, and sampling from large-scale Gaussian fields; see
@VillaPetraGhattas20 for the details.  [`MUQ`](http://muq.mit.edu/)
[@Parno2014] is a collection of tools for solving uncertainty quantification
problems.  `MUQ` provides a suite of powerful uncertainty quantification
algorithms including Markov chain Monte Carlo (MCMC) methods [@Parno2018],
transport maps [@Marzouk2016], polynomial chaos expansions [@Conrad2013],
Karhunen-Loeve expansions, and Gaussian process modeling [@GPML2005;
@Hartikainen2010].  `MUQ` also provides a framework for easily combining
statistical and physical models in a way that supports the efficient
computation of gradients, Jacobians, and Hessian-vector products.

`hIPPYlib-MUQ` integrates these two libraries into a unique software framework,
allowing users to implement state-of-the-art Bayesian inversion algorithms for
PDE models in a seamless way.  In this framework, `hIPPYlib` is used to define
the forward model, the prior, and the likelihood, to compute the maximum a
posteriori (MAP) point, and to construct a Gaussian (Laplace) approximation of
the posterior distribution based on approximations of the posterior covariance
as a low-rank update of the prior [@Bui-ThanhGhattasMartinEtAl13].  `MUQ` is
employed to exploit advanced MCMC methods to fully characterize the posterior
distribution in non-Gaussain/nonlinear settings.  `hIPPYlib-MUQ` offers a set
of wrappers that encapsulate the functionality of `hIPPYlib` in a way that
various features of `hIPPYlib` can be accessed by `MUQ`.  A key aspect of
`hIPPYlib-MUQ` is that it enables the use of MCMC methods enriched by the
Hessian of the log likelihood [@Cui2016], which is crucial for efficient and
scalable exploration of the posterior distribution for large-scale Bayesian
inverse problems. For example, the Laplace approximation of the posterior with
the low-rank factorization of the Hessian can be invoked to generate
high-quality proposals for MCMC methods, thereby significantly enhancing
sampling efficiency [@PetraMartinStadlerEtAl14].

`hIPPYlib-MUQ` also provides convergence diagnostics for MCMC samples: the potential
scale reduction factor and its extension to multivariate parameter cases [@Brooks98],
the autocorrelation function, and the effective sample size.


`hIPPYlib-MUQ` is designed for general large-scale Bayesian inverse problems,
not only for research and application in diverse fields, but also for
educational purposes.

The source code for `hIPPYlib-MUQ` has been archived to Zenodo with the linked DOI [10.5281/zenodo.4270739](https://doi.org/10.5281/zenodo.4270739).


# Acknowledgements

This work was supported by the U.S. National Science Foundation (NSF),
Software Infrastructure for Sustained Innovation (SI2: SSE & SSI) Program under
grants ACI-1550487, ACI-1550593, and ACI-1550547.
In addition, the authors acknowledge computing time on the Multi-Environment
Computer for Exploration and Discovery (MERCED) cluster at UC Merced, which was
funded by NSF grant ACI-1429783.
<!-- TODO: ask Noemi if there is missing -->

# References
