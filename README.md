```
 _     ___ ____  ______   ___ _ _           __  __ _   _  ___  
| |__ |_ _|  _ \|  _ \ \ / / (_) |__       |  \/  | | | |/ _ \ 
| '_ \ | || |_) | |_) \ V /| | | '_ \ _____| |\/| | | | | | | |
| | | || ||  __/|  __/ | | | | | |_) |_____| |  | | |_| | |_| |
|_| |_|___|_|   |_|    |_| |_|_|_.__/      |_|  |_|\___/ \__\_\
                                                               
```

> Scalable Markov chain Monte Carlo Sampling Methods for Large-scale Bayesian Inverse Problems Governed by PDEs

## Overview

`hIPPYlib-MUQ` is a Python interface between two open source softwares, `hIPPYlib` 
and `MUQ`, which have complementary capabilities. [hIPPYlib](https://hippylib.github.io) is an extensible 
software package aimed at solving deterministic and linearized Bayesian inverse 
problems governed by PDEs.
[MUQ](http://muq.mit.edu/) is a collection of tools for solving uncertainty quantification problems. 
`hIPPYlib-MUQ` integrates these two libraries into a unique software framework, 
allowing users to implement the state-of-the-art Bayesian inversion algorithms 
in a seamless way. 

To get started, we recommend to follow the interactive tutorial in `tutorial`
folder, which provides step-by-step implementations by solving an example
problem.
A static version of the tutorial is also available [here](https://hippylib.github.io/hippylib2muq/tutorial.html).


## Installation

`hIPPYlib-MUQ` is the interface program between `hIPPYlib` and `MUQ`, which
should be, of course, installed first.

We highly recommend to use our prebuilt Docker image, which is the easiest way
to run `hIPPYlib-MUQ`. With [Docker](https://www.docker.com/) installed on your
system, type: 

``` 
docker run -ti --rm -p 8888:8888 ktkimyu/hippylib2muq 'jupyter-notebook --ip=0.0.0.0' 
``` 
The notebook will be available at the following address in your web-browser.
From there you can run your own interactive notebooks or the tutorial notebook in
`tutorial` folder.

See [INSTALL](./INSTALL.md) for further details.

## Documentation

A complete API documentation of `hIPPYlib-MUQ` is available
[here](https://hippylib.github.io/hippylib2muq/).

## Authors

- Ki-Tae Kim, University of California, Merced
- Umberto Villa, Washington University in St. Louis
- Matthew Parno, The United States Army Corps of Engineers 
- Noemi Petra, University of California, Merced
- Youssef Marzouk, Massachusetts Institute of Technology
- Omar Ghattas, The University of Texas at Austin

## License

[GPL3](./LICENSE)
