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
`hIPPYlib-MUQ` is an interface between two open source softwares, `hIPPYlib` and `MUQ`, 
which have complementary abilities. 
[`hIPPYlib`](https://hippylib.github.io) [@VillaPetraGhattas18] is an extensible 
software package aimed at solving deterministic and linearized Bayesian inverse 
problems governed by PDEs. 
`hIPPYlib` provides a robust implementation of various algorithms like the inexact 
Newton-conjugate gradient method equipped with line search algorithms, randomized generalized 
eigensolvers, the Laplace approximation of the posterior, and scalable sampling of 
Gaussian fields.
[`MUQ`](http://muq.mit.edu/) is a collection of tools for the solution of 
uncertainty quantification problems. 
`MUQ` contains modeling tools like constructing statistical models, sampling algorithms like
various advanced Markov chain Monte Carlo (MCMC) methods, and function 
approximation tools such as Gaussian processes.

`hIPPYlib-MUQ` integrates these two libraries into a unique software framework, 
allowing users to implement the state-of-the-art Bayesian inference algorithms 
in a seamless way. 
<!-- `hIPPYlib-MUQ` is designed to be used in many different research and application -->
<!-- fields. -->

The source code for `hIPPYlib-MUQ` has been archived to Zenodo with the linked DOI


# Acknowledgements

`hIPPYlib-MUQ` development is supported

# References 
