---
title: 'hIPPYlib-MUQ: An Extensible Software Framework for Large-scale Bayesian Inverse Problems'
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
  - name: Youssef Marzouk
    affiliation: 4
  - name: Omar Ghattas
    affiliation: 5
  - name: Noemi Petra
    affiliation: 1
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

Inverse problems arise in all areas of science, engineering,
technology, and medicine and are often governed by complex
physics-based mathematical models. These models are subject to
considerable uncertainties stemming from unknown or uncertain inputs
(e.g., coefficient fields, constitutive laws, source terms,
geometries, initial and/or boundary conditions) as well as from noisy
and limited observations. While many of these input parameters cannot
be directly observed, they can be inferred from observations. Bayesian
inversion facilitates the integration of data with complex
physics-based models to quantify and reduce uncertainties in model
predictions. The algorithmic developments for Bayesian inversion are
subject to several computational challenges. For instance,
characterizing the posterior distribution of the parameters of
interest or predictions inevitably requires repeated evaluations of
(possibly) large-scale and complex forward models governed by
differential equations. In addition, the posterior distribution has a
complex structure stemming from the presence of possibly nonlinear
forward models and heterogeneous sources of data (e.g.,
[@Bui-ThanhGhattasMartinEtAl13], [@PetraMartinStadlerEtAl14]).

Our objective is to create a robust and scalable software framework to
tackle such complex large-scale partial differential differential
(PDEs)-constrained Bayesian inverse problems across a wide range of
science and engineering disciplines.  `hIPPYlib-MUQ` is a Python
interface between two open source softwares, `hIPPYlib` and `MUQ`,
which have complementary capabilities.
[`hIPPYlib`](https://hippylib.github.io) [@VillaPetraGhattas18] is an
extensible software package aimed at solving deterministic and
linearized Bayesian inverse problems governed by PDEs.  Based on
FEniCS [@LoggWells10], [@LoggMardalGarth12], [@LangtangenLogg17] for
the solution of forward PDE problems and on PETSc [@petsc-web-page]
for scalable and parallel linear algebra operations and solvers,
`hIPPYlib` implements the globalized inexact Newton-conjugate gradient
algorithm, low-rank approximation of the Hessian, and sampling from
large-scale Gaussian fields, see. [@Villa2018] for the details.
[`MUQ`](http://muq.mit.edu/) is a collection of tools for solving
uncertainty quantification problems.  `MUQ` provides an easy way to
combine many model components (physical and statistical models) into a
single sophisticated model and a spectrum of powerful uncertainty
quantification algorithms such as Markov chain Monte Carlo (MCMC)
methods, polynomial chaos expansions, Karhunen-Loeve expansions, and
Gaussian process regression.  <!-- TODO: give some reference for MUQ
-->

`hIPPYlib-MUQ` integrates these two libraries into a unique software framework, 
allowing users to implement the state-of-the-art Bayesian inversion algorithms 
in a seamless way. 
In this framework, `hIPPYlib` is used to define the forward model, the prior,
and the likelihood, to compute the maximum a posteriori (MAP) point, and to
construct the low-rank based Laplace approximation to the posterior distribution.
`MUQ` is employed to exploit advanced MCMC methods to fully characterize the
posterior distribution.
`hIPPYlib-MUQ` offers a set of wrappers that encapsulate the functionality of
`hIPPYlib` in a way that various features of `hIPPYlib` can be accessed by `MUQ`.
A key aspect of `hIPPYlib-MUQ` is that it enables the use of MCMC methods enriched
by the Hessian of the log likelihood, which is crucial for efficient and scalable 
exploration of the posterior distribution for large-scale Bayesian inverse problems.
Namely, the Laplace approximation of the posterior with the low-rank factorization
of the Hessian can be invoked to generate high-quality proposals for MCMC methods, 
thereby significantly enhancing sampling efficiency.

`hIPPYlib-MUQ` also provides convergence diagnostics for MCMC samples: the potential 
scale reduction factor and its extension to multivariate parameter cases [@Brooks98], 
the autocorrelation function and the effective sample size.


`hIPPYlib-MUQ` is designed to be used for general large-scale Bayesian inverse problems, 
allowing not only for use in many diverse research and application fields, but
also for educational purpose.

The source code for `hIPPYlib-MUQ` has been archived to Zenodo with the linked DOI.
<!-- TODO: upload source cod to Zenodo -->

# Acknowledgements

This work was supported by the U.S. National Science Foundation (NSF), 
Software Infrastructure for Sustained Innovation (SI2: SSE & SSI) Program under 
grants ACI-1550487, ACI-1550593, and ACI-1550547. 
In addition, the authors acknowledge computing time on the Multi-Environment 
Computer for Exploration and Discovery (MERCED) cluster at UC Merced, which was 
funded by NSF grant ACI-1429783.
<!-- TODO: ask Noemi if there is missing -->

# References 
