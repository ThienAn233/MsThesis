import numpy as np
import casadi as ca
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlabel("x")
ax.set_ylabel("z")


g  = 9.8 # gravity
x0 = [0,10,20,35,45,50] # landing points
z0 = [0,3,2,7,12,2] # landing points
N  = len(x0)          # number of bounces
ax.set_xlim(x0[0] - 0.1, x0[-1] + 0.1)
ax.scatter(x0,z0,c='r')


def OCP(w=[1,5,10]):
    t_list, x_dot, z_dot = [], [], [] # period of each fly time, horizontal velocity, vertical velocity.
    G = [] # equality constraints list

    for i in range(N-1):
        t_list.append(ca.SX.sym(f't_{i}'))
        x_dot.append(ca.SX.sym(f'dx_{i}'))
        z_dot.append(ca.SX.sym(f'dz_{i}'))

    T = ca.SX.sym('T')  # Total fly time of the robot
    X = ca.vertcat(*t_list,*x_dot,*z_dot,T)


    for i in range(N-1):
        z = z_dot[i] * t_list[i] - 0.5 * g * t_list[i] ** 2 + z0[i] - z0[i+1]
        x = x_dot[i] * t_list[i] + x0[i] - x0[i+1]
        G.append(z)
        G.append(x)
    t_dif = ca.sum(ca.vertcat(*t_list))-T
    G.append(t_dif)
    lbg = [0]*(2*(N-1)+1)
    ubg = [0]*(2*(N-1)+1)

    f=w[0]*T+w[1]*ca.sum(ca.vertcat(*x_dot))+w[2]*ca.sum(ca.vertcat(*z_dot))

    nlp = {'x': X, 'f': f, 'g': ca.vertcat(*G)}
    solver = ca.nlpsol("S", "ipopt", nlp,{'ipopt': {'max_iter': 10000,'print_level':0,'sb': 'yes'},'print_time': 0})

    lbx = [0]*(N-1) + [0]*(N-1) + [0]*(N-1) + [0]
    ubx = [ca.inf]*(N-1) + [20]*(N-1) + [20]*(N-1) + [ca.inf]

    X0 = [1]*(3*(N-1))+[1]
    sol = solver(x0=X0,lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    X_opt = sol['x'].full().flatten()
    t_opt = X_opt[0:N-1]
    x_opt = X_opt[N-1:2*(N-1)]
    z_opt = X_opt[2*(N-1):3*(N-1)]
    T_opt = X_opt[-1]
    solver_stats = solver.stats()
    print(f"{solver_stats['return_status']}",end=": ")
    print(f'Jump in: {np.round(T_opt,2)} sec')

    relative_time = np.linspace(0, 1, 20)
    ax.set_prop_cycle(plt.rcParams["axes.prop_cycle"])
    for i in range(N-1):
        t = t_opt[i]*relative_time
        l = x0[i] + t * x_opt[i]
        h = z0[i] + z_opt[i] * t - 0.5 * g * t * t
        ax.plot(l, h, ".-")
    return t_opt, x_opt, z_opt, T_opt


OCP([1,5,10])
OCP([1,10,5])
plt.show()