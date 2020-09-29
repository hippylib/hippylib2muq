---
title: 'MUQ-hIPPYlib: A Extensible Software Framework for Large-scale Bayesian Inverse Problems'
tags:
  - Python
  - Inverse problems
  - Low-rank approximation
  - Bayesian inference
  - Uncertainty quantification
  - MCMC sampling
authors:
  - name: Ki-Tae Kim
    affiliation: 1
  - name: Noemi Petra
    affiliation: 1
  - name: Umberto Villa
    affiliation: 2
  - name: Matthew Parno
    affiliation: 3
  - name: Omar Ghattas
    affiliation: 4
  - name: Youssef Marzouk
    affiliation: 5
affiliations:
  - name: Applied Mathematics, School of Natural Sciences, University of California, Merced
    index: 1
  - name: Department of Electrical & Systems Engineering, Washington University in St. Louis
    index: 2
  - name: Cold Regions Research & Engineering Laboratory, The United States Army Corps of Engineers
    index: 3
  - name: Oden Institute for Computational Engineering and Sciences, Department of Mechanical Engineering, and Department of Geological Sciences, The University of Texas at Austin
    index: 4
  - name: Department of Aeronautics and Astronautics, Massachusetts Institute of Technology
    index: 5
date: 
bibliography: paper.bib
---

# Summary

Inverse problems arise in many branches of scientific and engineering applications.
The aim of inverse problems is to infer unobservable parameters of interest, 
given noisy observational data and model that links the parameters and the data.
It is also important to quantify the uncertainty inherent in the data, the model 
and the inferred parameters.
The Bayesian approach to inverse problems enables these tasks in a principled way 
by recasting the problem as a problem of statistical inference of uncertain parameters, 
see, e.g., [@Bui-ThanhGhattasMartinEtAl13], [@PetraMartinStadlerEtAl14] for 
a Bayesian inference framework for infinite dimensional PDE-constrained inverse problems.

Our objective is to create a robust and scalable software framework to tackle 
complex large-scale PDE-constrained Bayesian inverse problems across a wide range 
of science and engineering disciplines. 
`hIPPYlib-MUQ` is a Python interface between two open source softwares, `hIPPYlib` and `MUQ`, 
which have complementary capabilities. 
[`hIPPYlib`](https://hippylib.github.io) [@VillaPetraGhattas18] is an extensible 
software package aimed at solving deterministic and linearized Bayesian inverse 
problems governed by PDEs.
Based on FEniCS [@LoggWells10], [@LoggMardalGarth12], [@LangtangenLogg17] for 
the solution of forward PDE problems and on PETSc [@petsc-web-page] for scalable 
and parallel linear algebra operations and solvers, `hIPPYlib` provides a robust 
implementation of large scale optimization algorithms, randomized algorithms for
matrix factorization, the Laplace approximation of the posterior, and scalable
sampling of Gaussian fields, see. [@Villa2018] for the details.
[`MUQ`](http://muq.mit.edu/) is a collection of tools for solving uncertainty 
quantification problems. 
`MUQ` offers an easy way to combine many model components into a single sophisticated
model and performs various advanced uncertainty quantification algorithms like 
Markov chain Monte Carlo (MCMC) methods, polynomial chaos expansions, 
Karhunen-Loeve expansions, Gaussian process regression, etc.

`hIPPYlib-MUQ` integrates these two libraries into a unique software framework, 
allowing users to implement the state-of-the-art Bayesian inversion algorithms 
in a seamless way. 
`hIPPYlib-MUQ` enables the full exploration of the posterior modeled by using `hIPPYlib`
through MCMC methods provided by `MUQ`, leading to efficient and scalable estimations
of the posterior probability distribution or some quantity of interest.

The integration is made as simple as possible so that any future developments in
both `hIPPYlib` and `MUQ` can be easily reflected.
Simply, an abstract model interface from `MUQ` is used to incorporate 
various features of `hIPPYlib`.

`hIPPYlib-MUQ` is designed to be used in general Bayesian inverse problems, 
allowing for use in many diverse research and application fields.

The source code for `hIPPYlib-MUQ` has been archived to Zenodo with the linked DOI.


# Acknowledgements

`hIPPYlib-MUQ` development is supported by the U.S. National Science Foundation (NSF), 
Software Infrastructure for Sustained Innovation (SI2-SSI) Program under grants 
ACI-15505470.

# References 
