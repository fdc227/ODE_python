import numpy as np
cimport numpy as cnp

cdef extern void T_others_f(double* state_var, double* T_others)
    
def T_others_f_cyx(cnp.ndarray[cnp.double_t, ndim=1] state_var):
    cdef cnp.ndarray[cnp.double_t, ndim=1] T_others = np.empty(state_var.size, dtype=np.double)
    T_others_f(<double *> state_var.data, <double *> T_others.data)
    return T_others
