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

"""
This module provides type conversions for the proper use of dependent packages.
"""
import dolfin as df


def dfVector2npArray(vec):
    """
    Convert a ``dolfin:vector`` to a ``numpy:ndarray``.

    :param ``dolfin:vector`` vec: a ``dolfin:vector``
    :returns: converted ``numpy:ndarray``
    """
    vec.apply("")
    y = vec.get_local()
    return y


def const_dfVector(mat, dim):
    """
    Construct and initialize a dolfin vector so that it is compatible with matrix 
    :math:`A` for the multiplication :math:`Ax = b`.

    :param ``dolfin:matrix`` mat: a ``dolfin:matrix``
    :param int dim: 0 for b and 1 for x
    :returns: an initialized ``dolfin:vector`` which is compatiable with ``mat``

    """
    y = df.Vector(mat.mpi_comm())
    mat.init_vector(y, dim)
    return y

def npArray2dfVector(arr, vec):
    """
    Assign values of ``numpy:ndarray`` to ``dolfin:vector``.

    :param ``numpy:ndarray`` arr: a ``numpy:ndarray``
    :param ``dolfin:vector`` vec: a ``dolfin:vector`` assigned by ``arr``

    """
    vec.set_local(arr)
    vec.apply("")
