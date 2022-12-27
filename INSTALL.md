# Installation

`hIPPYlib-MUQ` builds on [hIPPYlib](https://github.com/hippylib/hippylib)
version 3.1.0 with [FEniCS](https://fenicsproject.org/) version 2019.1 and
[MUQ](https://bitbucket.org/mituq/muq2/src/master/) version 0.3.0.
Installations of these packages are summarized here, but please see the
detailed installation guides given in each github/bitbucket page.

Additional dependencies are

- jupyter, matplotlib (for tutorial notebooks)
- seaborn, statsmodels (for postprocessing)

## Docker

We highly recommend to use our prebuilt Docker image, which is the
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
docker run -ti --rm -p 8888:8888 ktkimyu/hippylib2muq 'jupyter-notebook --ip=0.0.0.0'
```
The notebook will be available at the following address in your web-browser.
If you want to mount your local directory on docker container, add it with `-v`
options, e.g., to mount your current directory on /home/fenics/shared/ in
docker container, type
```
docker run -ti --rm -v $(pwd):/home/fenics/shared \
           -p 8888:8888 ktkimyu/hippylib2muq 'jupyter-notebook --ip=0.0.0.0'
```

## Conda

Conda is also a very convenient way to set up an enviroment to use `hIPPYlib-MUQ`.
The script below builds a conda enviroment with `FEniCS 2019` and `MUQ`.
`hIPPYlib 3.0.0` is also downloaded and installed via `pip`.


```
conda create -q -n hippylib2muq -c conda-forge fenics==2019.1.0 muq seaborn statsmodels
conda activate hippylib2muq
git clone --depth 1 --branch 3.0.0 https://github.com/hippylib/hippylib.git
python hippylib/setup.py install
```
### Installation of MUQ from source codes (Expert user/MUQ developers)

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
export PYTHONPATH=/your/muq2/install/directory/python:$PYTHONPATH
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
