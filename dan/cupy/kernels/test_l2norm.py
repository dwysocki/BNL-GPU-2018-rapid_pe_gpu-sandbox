import cupy
import numpy

l2norm_kernel = cupy.ReductionKernel(
    'T x',  # input params
    'T y',  # output params
    'x * x',  # map
    'a + b',  # reduce
    'y = sqrt(a)',  # post-reduction map
    '0',  # identity value
    'l2norm'  # kernel name
)

x = cupy.arange(10, dtype=numpy.float32).reshape(2, 5)

print(l2norm_kernel(x, axis=1))
