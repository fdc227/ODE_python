import numpy as np
import numpy.ctypeslib as npct

# input type for the cos_doubles function
# must be a double array, with single dimension that is contiguous
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')

# load the library, using numpy mechanisms
libcd = npct.load_library("T_others_f.so",".")

# setup the return types and argument types
libcd.T_others_f.restype = None
libcd.T_others_f.argtypes = [array_1d_double, array_1d_double]


def T_others_ctype(in_array, out_array):
    return libcd.T_others_f(in_array, out_array)

in_array = np.zeros(212)
out_array = np.empty_like(in_array)

T_others_ctype(in_array, out_array)

print(out_array)