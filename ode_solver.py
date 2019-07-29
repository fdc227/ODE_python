import numpy as np
import numpy.ctypeslib as npc
import numpy as np
from numpy.linalg import inv
from scipy.integrate import odeint
from scipy.integrate import ode
from scipy.integrate import RK45
from scipy.integrate import solve_ivp
import time
import pickle

array_1d_double = npc.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')

tothers = npc.load_library("T_others_f.so",".")
tdtdt = npc.load_library("T_dt_dt_f.so", '.')
strains = npc.load_library("strain_terms_f.so", '.')

# setup the return types and argument types
tothers.T_others_f.restype = None
tothers.T_others_f.argtypes = [array_1d_double, array_1d_double]

tdtdt.T_dt_dt_f.restype = None
tdtdt.T_dt_dt_f.argtypes = [array_1d_double, array_1d_double]

strains.strain_terms_f.restype = None
strains.strain_terms_f.argtypes = [array_1d_double,array_1d_double]


def T_others_f_ctypes(in_array):
    out_array = np.zeros(106)
    tothers.T_others_f(in_array, out_array)
    return out_array

def T_dt_dt_f_ctypes(in_array):
    out_array = np.zeros(11236)
    tdtdt.T_dt_dt_f(in_array, out_array)
    output = out_array.reshape((106,106))
    return output

def strain_terms_f_ctypes(in_array):
    out_array = np.zeros(106)
    strains.strain_terms_f(in_array, out_array)
    return out_array

ic = np.array([0, 0, 0, 0, 0, 0, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.7999999999999998, 0.5999999999999999, 0.3999999999999999, 0.19999999999999996, 2.0, 2.2, 2.4000000000000004, 2.6, 2.8000000000000003, 3.0, 3.2, 3.4000000000000004, 3.6, 3.8000000000000003, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

def ode_gen(t, y):

    t1 = time.time()

    T_dt_dt_np = T_dt_dt_f_ctypes(y)

    t2 = time.time()

    T_others_np = T_others_f_ctypes(y)

    t3 = time.time()

    strain_term_np = strain_terms_f_ctypes(y)

    t4 = time.time()

    RHS = (-1) * T_others_np + strain_term_np
 
    t5 = time.time()

    T_dt_dt_inv = inv(T_dt_dt_np)

    t6 = time.time()

    y_dt_dt = np.dot(T_dt_dt_inv, RHS)

    t7 = time.time()

    y_dt = y[106 : 212]

    y_out = np.insert(y_dt_dt, 0, y_dt)

    t8 = time.time()

    print(f'now integrating {t}th time_step, {t2-t1}, {t3-t2}, {t4-t3},{t5-t4},{t6-t5},{t7-t6},{t8-t7}')

    return y_out

time_steps = np.linspace(0, 50, 5000)
# sol = odeint(ode_gen, ic, time_steps, mxstep=10000000)
sol = solve_ivp(ode_gen, [0, 100], ic)
# r = ode(ode_gen)
# r.set_integrator('dopri5', nsteps=50, method='bdf')
# r.set_initial_value(ic, 0)
# tn = 10
# dt = 0.1
# t_list = []
# sol = []
# while r.successful and r.t < tn:
#     t_list.append(r.t + dt)
#     sol.append(r.integrate(r.t+dt))

# sol_file = open('Runge_kutta_sol.pkl', 'wb')
# pickle.dump(sol, sol_file)