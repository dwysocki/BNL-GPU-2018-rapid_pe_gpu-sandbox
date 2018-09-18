import cupy
import numpy

def add_generic(x, y, *ufunc_args, **ufunc_kwargs):
    """
    Adds two arrays, which can either be cupy or numpy arrays.  Uses the right
    module to do the adding regardless of its type.  Will raise a TypeError,
    however, if the two arrays have incompatible types (e.g., one cupy one
    numpy).
    """
    xpy = cupy.get_array_module(x, y)

    return xpy.add(x, y, *ufunc_args, **ufunc_kwargs)


shape = (100, 2)

A_cp = cupy.random.uniform(size=shape)
A_np = cupy.asnumpy(A_cp)

assert cupy.get_array_module(add_generic(A_cp, A_cp)) is cupy
assert cupy.get_array_module(add_generic(A_np, A_np)) is numpy

try:
    add_generic(A_np, A_cp)
    assert False
except TypeError:
    pass
