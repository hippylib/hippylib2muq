"""
Some conversions
"""

import dolfin as df


def dfVector2npArray(vec):
    """Convert dolfin vector to numpy array

    Values on local process are converted.

    :param vec: a dolfin vector
    :returns: converted numpy array

    """
    vec.apply("")
    y = vec.get_local()
    return y


def const_dfVector(mat, dim):
    """Construct and initialize a dolfin vector so that
    it is compatible with matrix :math:`A`
    for multiplication :math:`Ax = b`.

    :param mat: dolfin matrix, A
    :param dim: 0 for b and 1 for x
    :returns: a dolfin vector

    """
    y = df.Vector(mat.mpi_comm())
    mat.init_vector(y, dim)
    return y

def npArray2dfVector(arr, vec):
    """Assign values of numpy array to dolfin vector

    Values on local process are assigned.

    :param arr: a numpy array
    :param vec: a dolfin vector

    """
    vec.set_local(arr)
    vec.apply("")
