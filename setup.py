#  hIPPYlib-MUQ interface for large-scale Bayesian inverse problems
#  Copyright (c) 2019-2020, The University of Texas at Austin, 
#  University of California--Merced, Washington University in St. Louis,
#  The United States Army Corps of Engineers, Massachusetts Institute of Technology

#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup, find_packages
from os import path
from io import open


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name="hippylib2muq",
      version="1.0.0",
      author="Ki-Tae Kim, Umberto Villa, Matthew Parno, Youssef Marzouk, Omar Ghattas , Noemi Petra",
      author_email="kkim107@ucmerced.edu",
      description="a hippylib-muq interface for large-scale Bayesian inverse problems",
      long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
          "Operating System :: OS Independent",
          "Topic :: Scientific/Engineering",
          "Intended Audience :: Science/Research",
      ],
      packages=find_packages(),
      install_requires=[
          "hippylib"
      ]
      )


