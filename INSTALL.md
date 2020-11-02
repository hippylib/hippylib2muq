# Installation

`hIPPYlib-MUQ` builds on [hIPPYlib](https://github.com/hippylib/hippylib)
version 3.0.0 with [FEniCS](https://fenicsproject.org/) version 2019.1 and
[MUQ](https://bitbucket.org/mituq/muq2/src/master/) version 0.2.0.
Installations of these packages are summarized here, but please see the
detailed installation guides given in each github/bitbucket page.

Additional dependencies are 

- jupyter, matplotlib (for tutorial notebooks)
- seaborn, statsmodels (for postprocessing)

Note that we highly recommend to use our prebuilt Docker image, which is the
easiest way to run `hIPPYlib-MUQ`. The docker image with the installation of
all the dependencies is available
[here](https://hub.docker.com/r/ktkimyu/hippylib2muq).

With [Docker](https://www.docker.com/) installed on your system, type: 
```
docker run -ti --rm ktkimyu/hippylib2muq
```
Then, `hIPPYlib-MUQ` is available within the generated Docker container.

If you want to run `hIPPYlib-MUQ` using interactive notebooks, please type

``` 
docker run -ti --rm -p 127.0.0.1:8888:8888 ktkimyu/hippylib2muq 'jupyter-notebook --ip=0.0.0.0' 
``` 
The notebook will be available at the following address in your web-browser.
If you want to mount your local directory on docker container, add it with `-v`
options, e.g., to mount your current directory on /home/fenics/shared/ in
docker container, type
``` 
docker run -ti --rm -v $(pwd):/home/fenics/shared \ 
           -p 127.0.0.1:8888:8888 ktkimyu/hippylib2muq 'jupyter-notebook --ip=0.0.0.0' 
```

## Build the hIPPYlib-MUQ documentation using Sphinx

You can build the documentation on your local machine by using `sphinx`
(tested on version 2.3.0).
Additional required packages are
- `m2r`
- `sphinx_rtd_theme` (current HTML theme)

If you want to use other HTML themes, install the corresponding package and
modify the following line in `conf.py` in `doc/source` folder accordingly:
```
html_theme = 'name_of_the_theme'
```

All the packages above can be installed via `pip` or `conda`.

Once the required packages are installed, run `make html` from `doc` folder to
build the documentation, then the document is available at
`doc/build/html/`.

## Installation of dependencies

Note that `hIPPYlib` depends on [FEniCS](https://fenicsproject.org/) version 2019.1.

#### FEniCS on Conda (Linux and macOS only)

To create a Conda environment for `FEniCS` 2019.01, run the following command in 
your terminal:

```
conda create -n fenics-2019.1 -c conda-forge fenics==2019.1.0
```

#### Installation of hIPPYlib (latest version) using pip

With the supported version of `FEniCS` and its dependencies installed on your
machine, `hippylib` can be installed using `pip`:
```
pip3 install hippylib --user
```


#### Installation of MUQ from source codes

This requires cmake, the GNU Compiler Collection or Clang, and pybind11.
On macOS, you can have these by installing Xcode Command Line Tools. 

To compile and install `MUQ`, type

```
git clone https://bitbucket.org/mituq/muq2
cd muq2/build
cmake -DCMAKE_INSTALL_PREFIX=/your/muq2/install/directory -DMUQ_USE_PYTHON=ON ..
make
make install
```

Then Python static libraries are generated in `/your/muq2/install/directory/lib` folder.

You may append the path to this library folder, for example,

```
export PYTHONPATH=/your/muq2/install/directory/lib:$PYTHONPATH
```
