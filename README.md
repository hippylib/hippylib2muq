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

`hIPPYlib-MUQ` is the interface program between `hIPPYlib` and `MUQ`, which should be, of course, installed first.

We highly recommend to use our prebuilt Docker image, which is the easiest way to
run `hIPPYlib-MUQ`.
With [Docker](https://www.docker.com/) installed on your system, type:
```
docker run -ti --rm -v $(pwd):/home/fenics/hippylib2muq/tutorial \
           -p 127.0.0.1:8888:8888 ktkimyu/hippylib2muq 'jupyter-notebook --ip=0.0.0.0'
```
The notebook will be available at the following address in your web-browser.

`hIPPYlib-MUQ` and its dependencies can be installed via `pip` and [Conda](https://docs.conda.io/en/latest/), which is described in [INSTALL](./INSTALL.md).

## Documentation

A complete API documentation of `hIPPYlib-MUQ` is available at.

## Authors

- Ki-Tae Kim, University of California, Merced
- Umberto Villa, Washington University in St. Louis
- Matthew Parno, The United States Army Corps of Engineers 
- Youssef Marzouk, Massachusetts Institute of Technology
- Omar Ghattas, The University of Texas at Austin
- Noemi Petra, University of California, Merced


## License

[GPL3](./LICENSE)
