from __future__ import division, print_function

import cupy
import numpy

add_kernel = cupy.ReductionKernel(
    "T x",
    "T s",
    "x",
    "a + b",
    "s = a",
    "0",
    "add",
)


def trapz(y, dx=1, axis=-1):
    y = cupy.asarray(y)

    ndim = y.ndim

    n_skip = ndim+axis if axis < 0 else axis
    skip_ax = [slice(None)] * n_skip

    reduced_shape = y.shape[:n_skip] + y.shape[n_skip+1:]

    y_mid = y[skip_ax + [slice(1, -1)]]

    s = add_kernel(y_mid, axis=axis)

    s += 0.5 * y[skip_ax + [0]].reshape(reduced_shape)
    s += 0.5 * y[skip_ax + [-1]].reshape(reduced_shape)
    s *= dx

    s = s.reshape(reduced_shape)

#    if s.size == 1:
#        s = s[0]

    return s


def f(x):
    return x**3 + 2*x + 1


def F(a, b):
    return 0.25*(b**4 - a**4) + b**2 - a**2 + b - a

a, b = 0.0, 10.0
dx = 1e-1
x = cupy.arange(a, b+dx, dx)
x = cupy.vstack((x, x, x))
y = f(x)

F_cupy = trapz(y, dx=dx, axis=axis)
F_numpy = numpy.trapz(cupy.asnumpy(y), dx=dx, axis=axis)
F_analytic = F(a, b)

print(F_cupy.shape, F_numpy.shape)
print(F_cupy.dtype, F_numpy.dtype)
print(type(F_cupy), type(F_numpy))
cupy.testing.assert_array_almost_equal(F_cupy, F_numpy)
