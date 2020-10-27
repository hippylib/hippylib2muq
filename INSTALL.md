`hIPPYlib-MUQ` builds on [`hIPPYlib`](https://github.com/hippylib/hippylib)
version 3.0.0 with [FEniCS](https://fenicsproject.org/) version 2019.1 and
[`MUQ`](https://bitbucket.org/mituq/muq2/src/master/) version 0.2.0.  

Installations of these
packages are summarized here, but please see the detailed installation guides
given in each github/bitbucket page.


## Installation of dependencies

Note that `hIPPYlib` depends on [`FEniCS`](https://fenicsproject.org/) version 2019.1.

#### FEniCS on Conda (Linux and macOS only)

To create a Conda environment for `FEniCS` 2019.01, run the following command in 
your terminal:

```sh
conda create -n fenics-2019.1 -c conda-forge fenics==2019.1.0
```

#### Installation of hIPPYlib (latest version) using pip

With the supported version of `FEniCS` and its dependencies installed on your
machine, `hippylib` can be installed using `pip`:
```sh
pip3 install hippylib
```


#### Installation of MUQ from source codes

This requires cmake, the GNU Compiler Collection or Clang, and pybind11.
On macOS, you can have these by installing Xcode Command Line Tools. 

To compile and install `MUQ`, type

```sh
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
