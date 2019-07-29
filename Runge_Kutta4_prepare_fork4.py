from sympy import *
from sympy.parsing.sympy_parser import parse_expr
from pathos.multiprocessing import ProcessingPool as Pool
# from sympylist_to_txt import sympylist_to_txt
import numpy as np
from numpy.linalg import inv
from scipy.integrate import odeint
from scipy.integrate import ode
import pickle
import time

# start = time.time()
t = symbols('t')
x, w, L, theta_0 = symbols('x, w, L, theta_0')
M, m, x, y, z, g, h, E, I, G, J, x_f, c, s, K, A = symbols('M, m, x, y, z, g, h, E, I, G, J, x_f, c, s, K, A')
rho, V, a_w, gamma, M_thetadot, e = symbols('rho, V, a_w, gamma, M_thetadot, e')
beta, P, Q, R = symbols('beta, P, Q, R')
W_x, W_y, W_z = symbols('W_x, W_y, W_z')
P_s, gamma_alpha = symbols('P_s, gamma_alpha')
MOI = symbols('MOI')

theta = symbols('theta')
phi = symbols('phi')
psi = symbols('psi')
X = symbols('X')
Y = symbols('Y')
Z = symbols('Z')
short_var_list = [theta, phi, psi, X, Y, Z]

theta_dt = symbols('theta_dt')
phi_dt = symbols('phi_dt')
psi_dt = symbols('psi_dt')
X_dt = symbols('X_dt')
Y_dt = symbols('Y_dt')
Z_dt = symbols('Z_dt')
short_var_list_dt = [theta_dt, phi_dt, psi_dt, X_dt, Y_dt, Z_dt]

theta_dt_dt = symbols('theta_dt_dt')
phi_dt_dt = symbols('phi_dt_dt')
psi_dt_dt = symbols('psi_dt_dt')
X_dt_dt = symbols('X_dt_dt')
Y_dt_dt = symbols('Y_dt_dt')
Z_dt_dt = symbols('Z_dt_dt')
short_var_list_dt_dt = [theta_dt_dt, phi_dt_dt, psi_dt_dt, X_dt_dt, Y_dt_dt, Z_dt_dt]

var_q_bending = []
for i in range(10,0,-1):
    globals()[f'p{i}_b'] = symbols(f'p{i}_b')
    var_q_bending.append(globals()[f'p{i}_b'])
for i in range(1,11):
    globals()[f'q{i}_b'] = symbols(f'q{i}_b')
    var_q_bending.append(globals()[f'q{i}_b'])

var_q_bending_dot = []
for i in range(10,0,-1):
    globals()[f'p{i}_b_dot'] = symbols(f'p{i}_b_dot')
    var_q_bending_dot.append(globals()[f'p{i}_b_dot'])
for i in range(1,11):
    globals()[f'q{i}_b_dot'] = symbols(f'q{i}_b_dot')
    var_q_bending_dot.append(globals()[f'q{i}_b_dot'])

var_q_torsion = []
for i in range(10,0,-1):
    globals()[f'p{i}_t'] = symbols(f'p{i}_t')
    var_q_torsion.append(globals()[f'p{i}_t'])
for i in range(1,11):
    globals()[f'q{i}_t'] = symbols(f'q{i}_t')
    var_q_torsion.append(globals()[f'q{i}_t'])

var_q_inplane = []
for i in range(10,0,-1):
    globals()[f'p{i}_i'] = symbols(f'p{i}_i')
    var_q_inplane.append(globals()[f'p{i}_i'])
for i in range(1,11):
    globals()[f'q{i}_i'] = symbols(f'q{i}_i')
    var_q_inplane.append(globals()[f'q{i}_i'])

var_q_inplane_dot = []
for i in range(10,0,-1):
    globals()[f'p{i}_i_dot'] = symbols(f'p{i}_i_dot')
    var_q_inplane_dot.append(globals()[f'p{i}_i_dot'])
for i in range(1,11):
    globals()[f'q{i}_i_dot'] = symbols(f'q{i}_i_dot')
    var_q_inplane_dot.append(globals()[f'q{i}_i_dot'])


var_q_list = [*var_q_bending, *var_q_bending_dot, *var_q_torsion, *var_q_inplane, *var_q_inplane_dot]


var_q_bending_dt = []
for i in range(10, 0, -1):
    globals()[f'p{i}_b_dt'] = symbols(f'p{i}_b_dt')
    var_q_bending_dt.append(globals()[f'p{i}_b_dt'])
for i in range(1, 11):
    globals()[f'q{i}_b_dt'] = symbols(f'q{i}_b_dt')
    var_q_bending_dt.append(globals()[f'q{i}_b_dt'])

var_q_bending_dot_dt = []
for i in range(10, 0, -1):
    globals()[f'p{i}_b_dot_dt'] = symbols(f'p{i}_b_dot_dt')
    var_q_bending_dot_dt.append(globals()[f'p{i}_b_dot_dt'])
for i in range(1, 11):
    globals()[f'q{i}_b_dot_dt'] = symbols(f'q{i}_b_dot_dt')
    var_q_bending_dot_dt.append(globals()[f'q{i}_b_dot_dt'])

var_q_torsion_dt = []
for i in range(10, 0, -1):
    globals()[f'p{i}_t_dt'] = symbols(f'p{i}_t_dt')
    var_q_torsion_dt.append(globals()[f'p{i}_t_dt'])
for i in range(1, 11):
    globals()[f'q{i}_t_dt'] = symbols(f'q{i}_t_dt')
    var_q_torsion_dt.append(globals()[f'q{i}_t_dt'])

var_q_inplane_dt = []
for i in range(10, 0, -1):
    globals()[f'p{i}_i_dt'] = symbols(f'p{i}_i_dt')
    var_q_inplane_dt.append(globals()[f'p{i}_i_dt'])
for i in range(1, 11):
    globals()[f'q{i}_i_dt'] = symbols(f'q{i}_i_dt')
    var_q_inplane_dt.append(globals()[f'q{i}_i_dt'])

var_q_inplane_dot_dt = []
for i in range(10, 0, -1):
    globals()[f'p{i}_i_dot_dt'] = symbols(f'p{i}_i_dot_dt')
    var_q_inplane_dot_dt.append(globals()[f'p{i}_i_dot_dt'])
for i in range(1, 11):
    globals()[f'q{i}_i_dot_dt'] = symbols(f'q{i}_i_dot_dt')
    var_q_inplane_dot_dt.append(globals()[f'q{i}_i_dot_dt'])


var_q_list_dt = [*var_q_bending_dt, *var_q_bending_dot_dt, *var_q_torsion_dt, *var_q_inplane_dt, *var_q_inplane_dot_dt]


var_q_bending_dt_dt = []
for i in range(10, 0, -1):
    globals()[f'p{i}_b_dt_dt'] = symbols(f'p{i}_b_dt_dt')
    var_q_bending_dt_dt.append(globals()[f'p{i}_b_dt_dt'])
for i in range(1, 11):
    globals()[f'q{i}_b_dt_dt'] = symbols(f'q{i}_b_dt_dt')
    var_q_bending_dt_dt.append(globals()[f'q{i}_b_dt_dt'])

var_q_bending_dot_dt_dt = []
for i in range(10, 0, -1):
    globals()[f'p{i}_b_dot_dt_dt'] = symbols(f'p{i}_b_dot_dt_dt')
    var_q_bending_dot_dt_dt.append(globals()[f'p{i}_b_dot_dt_dt'])
for i in range(1, 11):
    globals()[f'q{i}_b_dot_dt_dt'] = symbols(f'q{i}_b_dot_dt_dt')
    var_q_bending_dot_dt_dt.append(globals()[f'q{i}_b_dot_dt_dt'])

var_q_torsion_dt_dt = []
for i in range(10, 0, -1):
    globals()[f'p{i}_t_dt_dt'] = symbols(f'p{i}_t_dt_dt')
    var_q_torsion_dt_dt.append(globals()[f'p{i}_t_dt_dt'])
for i in range(1, 11):
    globals()[f'q{i}_t_dt_dt'] = symbols(f'q{i}_t_dt_dt')
    var_q_torsion_dt_dt.append(globals()[f'q{i}_t_dt_dt'])

var_q_inplane_dt_dt = []
for i in range(10, 0, -1):
    globals()[f'p{i}_i_dt_dt'] = symbols(f'p{i}_i_dt_dt')
    var_q_inplane_dt_dt.append(globals()[f'p{i}_i_dt_dt'])
for i in range(1, 11):
    globals()[f'q{i}_i_dt_dt'] = symbols(f'q{i}_i_dt_dt')
    var_q_inplane_dt_dt.append(globals()[f'q{i}_i_dt_dt'])

var_q_inplane_dot_dt_dt = []
for i in range(10, 0, -1):
    globals()[f'p{i}_i_dot_dt_dt'] = symbols(f'p{i}_i_dot_dt_dt')
    var_q_inplane_dot_dt_dt.append(globals()[f'p{i}_i_dot_dt_dt'])
for i in range(1, 11):
    globals()[f'q{i}_i_dot_dt_dt'] = symbols(f'q{i}_i_dot_dt_dt')
    var_q_inplane_dot_dt_dt.append(globals()[f'q{i}_i_dot_dt_dt'])


var_q_list_dt_dt = [*var_q_bending_dt_dt, *var_q_bending_dot_dt_dt, *var_q_torsion_dt_dt, *var_q_inplane_dt_dt, *var_q_inplane_dot_dt_dt]


q_list = [*short_var_list, *var_q_list]
q_list_dt = [*short_var_list_dt, *var_q_list_dt]
q_list_dt_dt = [*short_var_list_dt_dt, *var_q_list_dt_dt]

var_dict = {c:1, m:1, M:10, MOI: 100, L:1, x_f: 0.5, E:100000, A:3.14, G:100000, J:10000}

T_dt_dt_raw = open('T_dt_dt.pkl', 'rb')
T_dt_dt_simplified = pickle.load(T_dt_dt_raw)

T_others_raw = open('T_others.pkl', 'rb')
T_others_simplified = pickle.load(T_others_raw)

strain_terms_raw = open('strain_terms.pkl', 'rb')
strain_terms_simplified = pickle.load(strain_terms_raw)

T_dt_dt_f = lambdify([*q_list, *q_list_dt], T_dt_dt_simplified, 'numpy')
# for i in T_dt_dt_simplified:
#     inter = []
#     for j in i:
#         var = j.free_symbols
#         inter.append(lambdify(var, j, 'numpy'))
#     T_dt_dt_f.append(inter)
print('first function generated')

T_others_f = lambdify([*q_list, *q_list_dt], T_others_simplified, 'numpy')
# for i in T_others_simplified:
#     inter = []
#     for j in i:
#         var = j.free_symbols
#         inter.append(lambdify(var, j, 'numpy'))
#     T_others_f.append(inter)
print('second function generated')

strain_terms_f = lambdify([*var_q_list, *var_q_list_dt], strain_terms_simplified, 'numpy')
# for i in strain_terms_simplified:
#     inter = []
#     for j in i:
#         var = j.free_symbols
#         inter.append(lambdify(var, j, 'numpy'))
#     strain_terms_f.append(inter)
print('third functions generated')

# T_dt_dt_inv = T_dt_dt_simplified ** (-1)
# print('inverse generated')
# f1 = (-1) * T_dt_dt_inv * T_others
# print('f1 generated')
# f2 = T_dt_dt_inv * strain_terms
# print('f2 generated')

# f = f1 + f2
# print('f generated')

# f_file = open('f.pkl', 'wb')
# pickle.dump(f, f_file)

init_cond_short = {}
for i in short_var_list:
    init_cond_short[i] = 0

init_cond_short_dt = {}
for i in short_var_list_dt:
    init_cond_short_dt[i] = 0

init_cond_bend = {}
for i in range(10):
    init_cond_bend[var_q_bending[i]] = 2 - 0.2 * i
for i in range(10, 20):
    init_cond_bend[var_q_bending[i]] = 0.2 * i

init_cond_bend_dot = {}
for i in range(10):
    init_cond_bend_dot[var_q_bending_dot[i]] = -0.2
for i in range(10, 20):
    init_cond_bend_dot[var_q_bending_dot[i]] = 0.2

init_cond_bend_dt = {}
for i in range(20):
    init_cond_bend_dt[var_q_bending_dt[i]] = 0

init_cond_bend_dot_dt = {}
for i in range(20):
    init_cond_bend_dot_dt[var_q_bending_dot_dt[i]] = 0

init_cond_torsion = {}
for i in range(20):
    init_cond_torsion[var_q_torsion[i]] = 0

init_cond_torsion_dt = {}
for i in range(20):
    init_cond_torsion[var_q_torsion_dt[i]] = 0

init_cond_inplane = {}
for i in range(20):
    init_cond_inplane[var_q_inplane[i]] = 0

init_cond_inplane_dot = {}
for i in range(20):
    init_cond_inplane_dot[var_q_inplane_dot[i]] = 0

init_cond_inplane_dt = {}
for i in range(20):
    init_cond_inplane_dt[var_q_inplane_dt[i]] = 0

init_cond_inplane_dot_dt = {}
for i in range(20):
    init_cond_inplane_dot_dt[var_q_inplane_dot_dt[i]] = 0

# init_condition_ = {**init_cond_short, **init_cond_short_dt, **init_cond_bend, **init_cond_bend_dot, **init_cond_bend_dt, **init_cond_bend_dot_dt, **init_cond_torsion, **init_cond_torsion_dt,
#                     **init_cond_inplane, **init_cond_inplane_dot, **init_cond_inplane_dt, **init_cond_inplane_dot_dt}

init_condition = {**init_cond_short, **init_cond_bend, **init_cond_bend_dot, **init_cond_torsion, **init_cond_inplane, **init_cond_inplane_dot, **init_cond_short_dt, **init_cond_bend_dt,
                  **init_cond_bend_dot_dt,**init_cond_torsion_dt, **init_cond_inplane_dt, **init_cond_inplane_dot_dt}

# init_condition_list_ = [*list(init_cond_short), *list(init_cond_short_dt), *list(init_cond_bend), *list(init_cond_bend_dot), *list(init_cond_bend_dt), *list(init_cond_bend_dot_dt), *list(init_cond_torsion), *list(init_cond_torsion_dt),
#                     *list(init_cond_inplane), *list(init_cond_inplane_dot), *list(init_cond_inplane_dt), *list(init_cond_inplane_dot_dt)]

init_condition_list = [*list(init_cond_short), *list(init_cond_bend), *list(init_cond_bend_dot), *list(init_cond_torsion), *list(init_cond_inplane), *list(init_cond_inplane_dot), *list(init_cond_short_dt), *list(init_cond_bend_dt),
                  *list(init_cond_bend_dot_dt),*list(init_cond_torsion_dt), *list(init_cond_inplane_dt), *list(init_cond_inplane_dot_dt)]

# print(init_condition_list)

T_dt_dt_index = []
for i in T_dt_dt_simplified:
    inter = []
    for j in i:
        index_list = []
        var = j.free_symbols
        for v in var:
            index_list.append(init_condition_list.index(v))
        inter.append(index_list)
    T_dt_dt_index.append(inter)
# print(T_dt_dt_index[0][0])

T_others_index = []
for i in T_others_simplified:
    inter = []
    for j in i:
        index_list = []
        var = j.free_symbols
        for v in var:
            index_list.append(init_condition_list.index(v))
        inter.append(index_list)
    T_others_index.append(inter)

strain_terms_index = []
for i in strain_terms_simplified:
    inter = []
    for j in i:
        index_list = []
        var = j.free_symbols
        for v in var:
            index_list.append(init_condition_list.index(v))
        inter.append(index_list)
    strain_terms_index.append(inter)

initial_cond = init_condition_list.copy()

initial_condition = []
for i in initial_cond:
    initial_condition.append(init_condition[i])
# print(initial_condition)

# print(strain_terms_f)
# print(strain_terms_index)
# print(T_dt_dt_index)
# end = time.time()
# print(end - start)

time_steps = np.linspace(0, 50, 5000)
strain_index_0 = [i for i in range(6, 106)]
strain_index_1 = [i for i in range(112, 212)]
strain_index = [*strain_index_0, *strain_index_1]

def ode_gen(t, y):

    t1 = time.time()

    # T_dt_dt_value = []
    # for i in range(len(T_dt_dt_f)):
    #     inter = []
    #     for j in range(len(T_dt_dt_f[0])):
    #         if len(T_dt_dt_index[i][j]) == 0:
    #             value = T_dt_dt_f[i][j]()
    #         else:
    #             value = T_dt_dt_f[i][j](*y[np.array(T_dt_dt_index[i][j])])
    #         inter.append(value)
    #     T_dt_dt_value.append(inter)
    T_dt_dt_np = T_dt_dt_f(*y)

    t2 = time.time()
    # print(T_dt_dt_np)
    # print(f'shape of T_dt_dt_np is {T_dt_dt_np.shape}')

    # T_others_value = []
    # for i in range(len(T_others_f)):
    #     inter = []
    #     for j in range(len(T_others_f[i])):
    #         if len(T_others_index[i][j]) == 0:
    #             value = T_others_f[i][j]()
    #         else: 
    #             value = T_others_f[i][j](*y[np.array(T_others_index[i][j])])
    #         inter.append(value)
    #     T_others_value.append(inter)
    T_others_np = T_others_f(*y)
    # print(T_others_np)
    # print(f'shape of T_others_np is {T_others_np.shape}')
    t3 = time.time()

    # strain_terms_value = []
    # for i in range(len(strain_terms_f)):
    #     inter = []
    #     for j in range(len(strain_terms_f[i])):
    #         if len(strain_terms_index[i][j]) == 0:
    #             value = strain_terms_f[i][j]()
    #         else:
    #             value = strain_terms_f[i][j](*y[np.array(strain_terms_index[i][j])])
    #         inter.append(value)
    #     strain_terms_value.append(inter)

    strain_term_np = strain_terms_f(*y[strain_index])
    # print(strain_term_np)
    # print(f'shape of strain_term_np is {strain_term_np.shape}')
    t4 = time.time()

    RHS = (-1) * T_others_np + strain_term_np
    # print(RHS)
    # print(f'shape of RHS is {RHS.shape}')
    t5 = time.time()

    T_dt_dt_inv = inv(T_dt_dt_np)
    # print(T_dt_dt_pinv)
    # print(f'shape of T_dt_dt_pinv is {T_dt_dt_pinv.shape}')
    t6 = time.time()
    y_dt_dt = np.dot(T_dt_dt_inv, RHS)
    t7 = time.time()
    y_dt = y[len(init_condition_list) // 2 : len(init_condition_list)]
    # print(f'shape of y_dt_dt is {y_dt_dt.shape}')
    # print(y_dt_dt)
    # print(f'shape of y_dt is {y_dt.shape}')

    y_out = np.insert(y_dt_dt, 0, y_dt)
    t8 = time.time()
    # print(f'shape of y_out is {y_out.shape}')
    print(f'now integrating {t}th time_step, {t2-t1}, {t3-t2}, {t4-t3},{t5-t4},{t6-t5},{t7-t6},{t8-t7}')

    return y_out

# r = ode(ode_gen)
# r.set_integrator('vode', nsteps=50, method='bdf')
# r.set_initial_value(initial_condition, 0)
# tn = 10
# dt = 0.1
# t_list = []
# sol_list = []
# while r.successful and r.t < tn:
#     t_list.append(r.t + dt)
#     sol_list.append(r.integrate(r.t+dt))

sol = odeint(ode_gen, initial_condition, time_steps)

sol_file = open('Runge_kutta_sol.pkl', 'wb')
pickle.dump(sol, sol_file)
