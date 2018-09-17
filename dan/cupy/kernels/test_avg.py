import cupy
import numpy

add_kernel = cupy.ReductionKernel(
    "T x",
    "T m",
    "x",
    "a + b",
    "m = a",
    "0",
    "avg",
)

def avg(x):
    return add_kernel(x) / x.size

x = cupy.arange(10, dtype=numpy.float64)

print(avg(x), numpy.mean(x))
