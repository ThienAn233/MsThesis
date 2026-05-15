import numpy as np
import casadi as ca
import matplotlib
# matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_robot(x,y):
    return patches.Rectangle((x-.7/2,y-.15/2),width=.7,height=.15,facecolor='red',edgecolor='black',alpha=0.5)

# gra = [9.81,3.71,1.62,6.76,2.67,0.31] # gravity on Earth, Mars, Moon
x0 = [0,1.,2.5,3.75,5.25,6.5] # landing points
z0 = [0.1,0.6,0.4,1.2,0.8,1.] # landing points
# fig, axes = plt.subplots(1,1,figsize=(20*2, 3*9))
# fig.supylabel('z (m)')
# fig.supxlabel('x (m)')
# # for ax in axes:
# #     for a in ax:
# a=axes
# a.set_xlim(x0[0] - 0.5, x0[-1] + 0.5)
# a.scatter(x0,[z+0.25 for z in z0],c='r')
# for i in range(len(x0)):
#     a.add_patch(plot_robot(x0[i],z0[i]+0.25))
# # a.set_xlabel("x")
# # a.set_ylabel("z")
# a.bar(x0, z0, width=.9, label='x0',color='gray',alpha=1)


def OCP(w=[1,0.2,0.2],x0=x0,z0=z0,g=9.81):
    N  = len(x0)          # number of bounces
    t_list, x_dot, z_dot = [], [], [] # period of each fly time, horizontal velocity, vertical velocity.
    G = [] # equality constraints list

    for i in range(N-1):
        t_list.append(ca.SX.sym(f't_{i}'))
        x_dot.append(ca.SX.sym(f'dx_{i}'))
        z_dot.append(ca.SX.sym(f'dz_{i}'))

    T = ca.SX.sym('T')  # Total fly time of the robot
    X = ca.vertcat(*t_list,*x_dot,*z_dot,T)

    zf = []
    for i in range(N-1):
        z = z_dot[i] * t_list[i] - 0.5 * g * t_list[i] ** 2 + z0[i] - z0[i+1]   # Constranits on verticle velocity
        x = x_dot[i] * t_list[i] + x0[i] - x0[i+1]                              # Constranits on horizontal velocity
        zf += [z_dot[i] - g * t_list[i]]                                        # zf < 0 (terminal velocity)
        G.append(z)
        G.append(x)

    for i in range(N-2):
        dif = (x_dot[i]-x_dot[i+1])**2-(0.5*x_dot[i])**2
        G.append(dif)
    
    t_dif = ca.sum(ca.vertcat(*t_list))-T
    G.append(t_dif)
    lbg = [0]*(2*(N-1)) + [-ca.inf]*(N-2)+ [0]*1
    ubg = [0]*(2*(N-1)+(N-2)+1)

    f=T+w[0]*ca.sum(ca.vertcat(*x_dot)**2)+w[1]*ca.sum(ca.vertcat(*z_dot)**2)

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
    print(f'Jump in: {np.round(T_opt,2)} sec; max vx: {np.round(np.max(x_opt),2)}; max vz: {np.round(np.max(z_opt),2)}; min vx: {np.round(np.min(x_opt),2)}; min vz: {np.round(np.min(z_opt),2)}; T_mean: {np.round(np.mean(t_opt),2)} sec  ')
    # ax.set_title(f"Gravity: ${g} m/s^2$")
    # relative_time = np.linspace(0, 1, 20)
    # # ax.set_prop_cycle(plt.rcParams["axes.prop_cycle"])
    # for i in range(N-1):
    #     t = t_opt[i]*relative_time
    #     l = x0[i] + t * x_opt[i]
    #     h = z0[i] + z_opt[i] * t - 0.5 * g * t * t + 0.25
    #     ax.plot(l, h, m+"-"+c)
    #     # ax.axis('equal')
    return t_opt, x_opt, z_opt, T_opt

# w2=[10,10,2]
# print(OCP(ax = axes,g=gra[0],m='-',c='g'))
# print(OCP(ax = axes,w=w2,g=gra[0],m='.',c='g'))
# print(OCP(ax = axes,g=gra[1],m='-',c='b'))
# print(OCP(ax = axes,w=w2,g=gra[1],m='.',c='b'))
# print(OCP(ax = axes,g=gra[2],m='-',c='m'))
# print(OCP(ax = axes,w=w2,g=gra[2],m='.',c='m'))
# print(OCP(ax = axes,g=gra[3],m='-',c='r'))
# print(OCP(ax = axes,w=w2,g=gra[3],m='.',c='r'))
# print(OCP(ax = axes,g=gra[4],m='-',c='c'))
# print(OCP(ax = axes,w=w2,g=gra[4],m='.',c='c'))
# print(OCP(ax = axes,g=gra[5],m='-',c='y'))
# print(OCP(ax = axes,w=w2,g=gra[5],m='.',c='y'))
# axes.set_aspect('equal')
# plt.show()