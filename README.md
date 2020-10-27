```
 _     ___ ____  ______   ___ _ _           __  __ _   _  ___  
| |__ |_ _|  _ \|  _ \ \ / / (_) |__       |  \/  | | | |/ _ \ 
| '_ \ | || |_) | |_) \ V /| | | '_ \ _____| |\/| | | | | | | |
| | | || ||  __/|  __/ | | | | | |_) |_____| |  | | |_| | |_| |
|_| |_|___|_|   |_|    |_| |_|_|_.__/      |_|  |_|\___/ \__\_\
                                                               
```

> An Extensible Software Framework for Large-scale Bayesian Inverse Problems

## Overview

`hIPPYlib-MUQ` is a Python interface between two open source softwares, `hIPPYlib` 
and `MUQ`, which have complementary capabilities. [`hIPPYlib`](https://hippylib.github.io) is an extensible 
software package aimed at solving deterministic and linearized Bayesian inverse 
problems governed by PDEs.
[`MUQ`](http://muq.mit.edu/) is a collection of tools for solving uncertainty quantification problems. 
`hIPPYlib-MUQ` integrates these two libraries into a unique software framework, 
allowing users to implement the state-of-the-art Bayesian inversion algorithms 
in a seamless way. 

To get started, we recommend to follow the interactive tutorials in `tutorial`
folder.

## Installation

`hIPPYlib-MUQ` builds on [`hIPPYlib`](https://github.com/hippylib/hippylib)
version 3.0.0 with [FEniCS](https://fenicsproject.org/) version 2019.1 and
[`MUQ`](https://bitbucket.org/mituq/muq2/src/master/) version 0.2.0.

Additional dependencies are 

- jupyter, matplotlib (for tutorial notebooks)
- seaborn, statsmodels (for postprocessing)

We highly recommend to use our prebuilt Docker image, which is the easiest way
to run `hIPPYlib-MUQ`. The docker image with the installation of all the
dependencies is available here.

With [Docker](https://www.docker.com/) installed on your system, type: 
```
docker run -ti --rm ktkimyu/hippylib2muq
```
Then, `hIPPYlib-MUQ` is available within the generated Docker container.

If you want to run interactive notebooks directly, please type

``` 
docker run -ti --rm -v $(pwd):/home/fenics/hippylib2muq/tutorial \ 
           -p 127.0.0.1:8888:8888 ktkimyu/hippylib2muq 'jupyter-notebook --ip=0.0.0.0' 
``` 
The notebook will be available at the following address in your web-browser.

## Documentation

A complete API documentation of `hIPPYlib-MUQ` is available here.

## Authors

- Ki-Tae Kim, University of California, Merced
- Umberto Villa, Washington University in St. Louis
- Matthew Parno, The United States Army Corps of Engineers 
- Noemi Petra, University of California, Merced
- Youssef Marzouk, Massachusetts Institute of Technology
- Omar Ghattas, The University of Texas at Austin

## License

[GPL3](./LICENSE)
