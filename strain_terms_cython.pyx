import cython
import numpy as np
cimport numpy as cnp

cdef extern void strain_terms_f(double* state_var, double* strain_terms)

def strain_terms_cython(cnp.ndarray[cnp.double_t, ndim=1, mode="c"] state_var not None):
    cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] strain_terms = np.empty(state_var.size, dtype=np.double)
    strain_terms_f(<double*> state_var.data, <double*> strain_terms.data)
    return strain_terms
