import numpy as np
import casadi as ca
from utilscasadi import *
from fly2D import OCP as OCP_out
from opti import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches

gra = [9.81,3.71,1.62,6.76,2.67,0.31] # gravity on Earth, Mars, Moon
x0 = [0,1.,2.5,3.75,5.25,6.5] # landing points
z0 = [0.1,0.6,0.4,1.2,0.8,1.] # landing points

robot_params = {
        'L': .50,
        'W': .25,
        'H': 0.0,
        'm': 7.0,
        'l1': .05,
        'l2': 0.0,
        'l3': 0.2,
        'l4': 0.2,
    }
quad = QuadDynamicsCasadi(robot_params)


t_opt, x_opt, z_opt, T_opt = OCP_out(x0=x0,z0=z0,g=gra[0])


foot_points = DM([.25, 0.6/4, z0[0],
                        .25,-0.6/4, z0[0],
                        -.25, 0.6/4, z0[0],
                        -.25,-0.6/4, z0[0]])
initpos = [x0[0],0,z0[0]+.3,0,0,0]
inivel  = [0,0,0,0,0,0]
termvel = [x_opt[0],0,z_opt[0],0,0,0]
landheight = z0[0]
body_traj, joint_traj, ctrl_traj, state_traj= OCP(initpos, foot_points, inivel, termvel, landheight, quad)


body_traj_rollout = roll_foward_dynamics(state_traj[-1,:],joint_traj[-1,:],np.zeros(12),quad, dt=t_opt[0]/50, N=50)


x_landing  = body_traj_rollout[-1][0]
vx_landing = body_traj_rollout[-1][6]
vz_landing = body_traj_rollout[-1][8]
foot_points = DM([x_landing+.25, 0.6/4, z0[1],
                        x_landing+.25,-0.6/4, z0[1],
                        x_landing-.25, 0.6/4, z0[1],
                        x_landing-.25,-0.6/4, z0[1]])
initpos_landing = [x_landing,0,z0[1]+.3,0,0,0]
inivel_landing  = [vx_landing,0,vz_landing,0,0,0]
termvel_landing = [x_opt[1],0,z_opt[1],0,0,0]
landheight_landing = z0[1]
body_traj_landing, joint_traj_landing, ctrl_traj_landing, state_traj_landing = OCP(initpos_landing, foot_points, inivel_landing, termvel_landing, landheight_landing, quad)


body_traj_rollout_landing = roll_foward_dynamics(state_traj_landing[-1,:],joint_traj_landing[-1,:],np.zeros(12),quad, dt=t_opt[1]/50, N=50)


x_landing_stop  = body_traj_rollout_landing[-1][0]
vx_landing_stop = body_traj_rollout_landing[-1][6]
vz_landing_stop = body_traj_rollout_landing[-1][8]
foot_points_stop = DM([x_landing_stop+.25, 0.6/4, z0[2],
                        x_landing_stop+.25,-0.6/4, z0[2],
                        x_landing_stop-.25, 0.6/4, z0[2],
                        x_landing_stop-.25,-0.6/4, z0[2]])
initpos_landing_stop = [x_landing_stop,0,z0[2]+.35,0,0,0]
inivel_landing_stop  = [vx_landing_stop,0,vz_landing_stop,0,0,0]
termvel_landing_stop = [0,0,0,0,0,0]
landheight_landing_stop = z0[2]
body_traj_landing_stop, joint_traj_landing_stop, ctrl_traj_landing_stop, state_traj_landing_stop = OCP(initpos_landing_stop, foot_points_stop, inivel_landing_stop, termvel_landing_stop, landheight_landing_stop, quad)
plot_res(state_traj_landing_stop,ctrl_traj_landing_stop)

fig,ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10,8))
ax = quad.setupView(ax=ax)
ax.set_xlim(2, 3)
ax.set_ylim(-.5, .5)
ax.set_zlim(.3, .9)
ax.set_box_aspect((1,1,.6))
x = np.linspace(-0.35,0.35,2)
y = np.linspace(-0.35,0.35,2)
X,Y = np.meshgrid(x,y)
Z = 0*X+0*Y+0.1
ax.plot_surface(X,Y,Z, color='gray', alpha=.2)
# quad.visualize_robot(joint_traj[0,:], body_traj[0,:])
# for i in range(50):
#     # if i % 5 == 4:
#         quad.visualize_robot(joint_traj[i,:], body_traj[i,:])
# for i in range(50):
#     if i % 5 == 4:
#         quad.visualize_robot(joint_traj[-1,:], body_traj_rollout[i])
# quad.visualize_robot(joint_traj_landing[0,:], body_traj_landing[0,:])       
# for i in range(50):
#     # if i % 10 == 9:
#         quad.visualize_robot(joint_traj_landing[i,:], body_traj_landing[i,:])
# for i in range(50):
#     if i % 5 == 4:
#         quad.visualize_robot(joint_traj_landing[-1,:], body_traj_rollout_landing[i])
# quad.visualize_robot(joint_traj_landing_stop[0,:], body_traj_landing_stop[0,:])
for i in range(50):
    # if i % 5 == 4:
        quad.visualize_robot(joint_traj_landing_stop[i,:], body_traj_landing_stop[i,:])
x = np.linspace(.65,1.35,2)
y = np.linspace(-.35,.35,2)
X,Y = np.meshgrid(x,y)
Z = 0*X+0*Y +0.6
ax.plot_surface(X,Y,Z, color='gray', alpha=.2)
x = np.linspace(2.15,2.85,2)
y = np.linspace(-.35,.35,2)
X,Y = np.meshgrid(x,y)
Z = 0*X+0*Y +0.4
ax.plot_surface(X,Y,Z, color='gray', alpha=.2)
ax.grid(False)
plt.show()