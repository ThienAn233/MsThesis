from casadi import *
from httpx import get
from utilscasadi import *
import numpy as np

Tg = 6.0          # Total ground time (s)
Tf = 6.0          # Total fly time(s)
Tl = 6.0          # Total landing time(s)
Ng = 10           # Number of time steps at ground
Nf = 10           # Number of time steps at fly
Nl = 10           # Number of time steps at landing
N = Ng + Nl + Nf  # Total number of time steps
Nc = Ng + Nl      # Total number of time steps with contact
dtg = Tg / Ng
dtf = Tf / Nf
dtl = Tl / Nl   
mu = 0.5          # friction coefficient
gravity = 1.62    # gravity


robot_params = {
        'L': 2.0,
        'W': 1.0,
        'H': 0.0,
        'm': 5.0,
        'l1': .25,
        'l2': 0.0,
        'l3': 0.8,
        'l4': 0.8,
    }

# Create symbolic variables for each time step
x_list, y_list, z_list, vx_list, vy_list, vz_list, phi_list, theta_list, psi_list, wphi_list, wtheta_list, wpsi_list, jfl1_list, jfl2_list, jfl3_list, jfr1_list, jfr2_list, jfr3_list, jbl1_list, jbl2_list, jbl3_list, jbr1_list, jbr2_list, jbr3_list = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
u_flx_list, u_fly_list, u_flz_list, u_frx_list, u_fry_list, u_frz_list, u_blx_list, u_bly_list, u_blz_list, u_brx_list, u_bry_list, u_brz_list = [], [], [], [], [], [], [], [], [], [], [], []


state_dict = {
    'x': x_list,
    'y': y_list,
    'z': z_list,
    'vx': vx_list,
    'vy': vy_list,
    'vz': vz_list,
    'phi': phi_list,
    'theta': theta_list,
    'psi': psi_list,
    'wphi': wphi_list,
    'wtheta': wtheta_list,
    'wpsi': wpsi_list,
    'jfl': [jfl1_list, jfl2_list, jfl3_list],
    'jfr': [jfr1_list, jfr2_list, jfr3_list],
    'jbl': [jbl1_list, jbl2_list, jbl3_list],
    'jbr': [jbr1_list, jbr2_list, jbr3_list],
} 

control_dict = {
    'u_fl': [u_flx_list, u_fly_list, u_flz_list],
    'u_fr': [u_frx_list, u_fry_list, u_frz_list],
    'u_bl': [u_blx_list, u_bly_list, u_blz_list],
    'u_br': [u_brx_list, u_bry_list, u_brz_list],
}

for i in range(N):
    x_list.append(SX.sym(f'x_{i}'))
    y_list.append(SX.sym(f'y_{i}'))
    z_list.append(SX.sym(f'z_{i}'))
    vx_list.append(SX.sym(f'vx_{i}'))
    vy_list.append(SX.sym(f'vy_{i}'))
    vz_list.append(SX.sym(f'vz_{i}'))
    phi_list.append(SX.sym(f'phi_{i}'))
    theta_list.append(SX.sym(f'theta_{i}'))
    psi_list.append(SX.sym(f'psi_{i}'))
    wphi_list.append(SX.sym(f'wphi_{i}'))
    wtheta_list.append(SX.sym(f'wtheta_{i}'))
    wpsi_list.append(SX.sym(f'wpsi_{i}'))
    jfl1_list.append(SX.sym(f'jfl1_{i}'))
    jfl2_list.append(SX.sym(f'jfl2_{i}'))
    jfl3_list.append(SX.sym(f'jfl3_{i}'))
    jfr1_list.append(SX.sym(f'jfr1_{i}'))
    jfr2_list.append(SX.sym(f'jfr2_{i}'))
    jfr3_list.append(SX.sym(f'jfr3_{i}'))
    jbl1_list.append(SX.sym(f'jbl1_{i}'))
    jbl2_list.append(SX.sym(f'jbl2_{i}'))
    jbl3_list.append(SX.sym(f'jbl3_{i}'))
    jbr1_list.append(SX.sym(f'jbr1_{i}'))
    jbr2_list.append(SX.sym(f'jbr2_{i}'))
    jbr3_list.append(SX.sym(f'jbr3_{i}'))
    
for i in range(Nc):
    u_flx_list.append(SX.sym(f'u_flx_{i}'))
    u_fly_list.append(SX.sym(f'u_fly_{i}'))
    u_flz_list.append(SX.sym(f'u_flz_{i}'))
    u_frx_list.append(SX.sym(f'u_frx_{i}'))
    u_fry_list.append(SX.sym(f'u_fry_{i}'))
    u_frz_list.append(SX.sym(f'u_frz_{i}'))
    u_blx_list.append(SX.sym(f'u_blx_{i}'))
    u_bly_list.append(SX.sym(f'u_bly_{i}'))
    u_blz_list.append(SX.sym(f'u_blz_{i}'))
    u_brx_list.append(SX.sym(f'u_brx_{i}'))
    u_bry_list.append(SX.sym(f'u_bry_{i}'))
    u_brz_list.append(SX.sym(f'u_brz_{i}'))

# Create optimization variable vector by stacking all variables
S = vertcat(*x_list, *y_list, *z_list, *phi_list, *theta_list, *psi_list, *vx_list, *vy_list, *vz_list, *wphi_list, *wtheta_list, *wpsi_list, *jfl1_list, *jfl2_list, *jfl3_list, *jfr1_list, *jfr2_list, *jfr3_list, *jbl1_list, *jbl2_list, *jbl3_list, *jbr1_list, *jbr2_list, *jbr3_list)
U = vertcat(*u_flx_list, *u_fly_list, *u_flz_list, *u_frx_list, *u_fry_list, *u_frz_list, *u_blx_list, *u_bly_list, *u_blz_list, *u_brx_list, *u_bry_list, *u_brz_list)
X = vertcat(S, U)
# Helpers to extract state at a given timestep
def get_b_state(i):
    return vertcat(x_list[i], y_list[i], z_list[i], phi_list[i], theta_list[i], psi_list[i], vx_list[i], vy_list[i], vz_list[i], wphi_list[i], wtheta_list[i], wpsi_list[i])

def get_j_state(i):
    return vertcat(jfl1_list[i], jfl2_list[i], jfl3_list[i],
                   jfr1_list[i], jfr2_list[i], jfr3_list[i],
                   jbl1_list[i], jbl2_list[i], jbl3_list[i],
                   jbr1_list[i], jbr2_list[i], jbr3_list[i])

def get_state(i):
    return vertcat(get_b_state(i), get_j_state(i))

def get_control(i):
    return vertcat(u_flx_list[i], u_fly_list[i], u_flz_list[i],
                   u_frx_list[i], u_fry_list[i], u_frz_list[i],
                   u_blx_list[i], u_bly_list[i], u_blz_list[i],
                   u_brx_list[i], u_bry_list[i], u_brz_list[i])

quad = QuadDynamicsCasadi(robot_params)
def dyanmics(sb, sj, u):
    b_state = sb
    j_state = sj
    rate = quad.get_Euler_rate_matrix(b_state[3], b_state[4], b_state[5])
    M = quad.inertia_matrix(b_state[0:6])
    C = quad.coriolis_term(b_state[0:6],rate@b_state[6:12])
    G = quad.gravity_vector(gravity)
    
    J_cb = quad.J_contact_base(b_state,quad.get_world_frame_contact_point(b_state,j_state))
    ds = solve(M, (J_cb.T @ u - C@rate@b_state[6:12] - G))
    ds = vertcat(rate@b_state[6:12], ds)
    return ds

def get_tau(sb,sj,u):
    J_cj = quad.J_contact_joint(sb,sj)
    return -J_cj.T @ u
    

################### Constraints list ###################
g = []

# Body dynamics constraints
for i in range(N-1):
    # print(f"Adding dynamics constraint for step {i}")
    sb   = get_b_state(i)
    sbp1 = get_b_state(i+1)
    sj   = get_j_state(i)
    sjp1 = get_j_state(i+1)
    if i < Ng:
        u    = get_control(i)
        up1  = get_control(i+1)
    elif (i < Ng + Nl)&(i >= Ng):
        u    = DM.zeros(12)
        up1  = DM.zeros(12)
    else:
        u    = get_control(i-Ng)
        up1  = get_control(i+1-Ng)
    fi   = dyanmics(sb, sj, u) 
    fip1 = dyanmics(sbp1, sjp1, up1)
    if i < Ng:
        g.append(sbp1 - sb - (dtg/2)*(fi + fip1)) 
    elif (i < Ng + Nl)&(i >= Ng):
        g.append(sbp1 - sb - (dtf/2)*(fi + fip1)) 
    else:
        g.append(sbp1 - sb - (dtl/2)*(fi + fip1)) 


# Initial and final conditions
foot_points = DM([1, 0.5, 0,
                     1,-0.5, 0,
                    -1, 0.5, 0,
                    -1,-0.5, 0])
initial_body_position  = vertcat(0,0,.8*sqrt(2),0,0,0,1,0,0,0,0,0)
initial_leg_positions  = quad.inverse_kinematics(initial_body_position, foot_points)
initial_leg_forces     = vertcat(0,0,robot_params['m']*gravity/4, 0,0,robot_params['m']*gravity/4, 0,0,robot_params['m']*gravity/4, 0,0,robot_params['m']*gravity/4)
terminal_body_position = vertcat(10,0,0,0,0,0,1,0,0,0,0,0)
terminal_leg_positions = vertcat(0,pi/3,pi/6,  0,pi/3,pi/6,  0,pi/3,pi/6,  0,pi/3,pi/6)
terminal_leg_forces    = initial_leg_forces

g.append(get_b_state(0)[0:3] - initial_body_position[0:3])    
g.append(get_control(0) - initial_leg_forces)   
# g.append(get_b_state(0)[6:9] - initial_body_position[6:9])                   
g.append(get_b_state(N - 1)[:3] - terminal_body_position[:3])    
g.append(get_control(Ng - 1) - terminal_leg_forces)
# g.append(get_b_state(N - 1)[6:9] - terminal_body_position[6:9]) 

# When leg flying: contact force must be zero
for i in range(Ng,Ng+Nf):
    u = get_control(i)
    for j in range(12):
        g.append(u[j])

# When leg touch ground: foot position constraints + friction cone constraints
for i in range(Ng):
    qb = get_b_state(i)[0:6]
    qj = get_j_state(i)
    u  = get_control(i)
    lp = quad.forward_kinematics(qb,qj)
    for i,idx in enumerate([4,9,14,19]):
        for j in range(3):
            l = lp[idx][j] - foot_points[i*3 + j]
            # foot points constraint
            g.append(l)
for j in range(Ng):
    u = get_control(j)
    qb = get_b_state(j)
    for i in range(4):
        ux = u[i*3 + 0]
        uy = u[i*3 + 1]
        uz = u[i*3 + 2]
        # friction cone constraint
        g.append( sqrt(ux**2 + uy**2 + 1e-6) - mu*uz)
        g.append(1-qb[2]) 
for i in range(Nf,Nf+Nl):
    qb = get_b_state(i)[0:6]
    qj = get_j_state(i)
    u  = get_control(i-Nf)
    lp = quad.forward_kinematics(qb,qj)
    for i,idx in enumerate([4,9,14,19]):
        for j in range(3):
            l = lp[idx][j] - foot_points[i*3 + j]
            # foot points constraint
            g.append(l)
for j in range(Nf,Nf+Nl):
    u = get_control(j-Nf)
    qb = get_b_state(j)
    for i in range(4):
        ux = u[i*3 + 0]
        uy = u[i*3 + 1]
        uz = u[i*3 + 2]
        # friction cone constraint
        g.append( sqrt(ux**2 + uy**2 + 1e-6) - mu*uz)
        g.append(1-qb[2])


################### End of constraints list ###################



################### Objective: minimize total control effort ###################
f = 0
for i in range(N - 1):
    u_i   = get_control(i)
    u_ip1 = get_control(i + 1)
    sb    = get_b_state(i)
    sbp1  = get_b_state(i + 1)
    rot_i = sb[3:6]
    rot_ip1 = sbp1[3:6]
    sj  = get_j_state(i) - terminal_leg_positions
    sj_ip1 = get_j_state(i + 1) - terminal_leg_positions
    tau = get_tau(sb,get_j_state(i), u_i)
    tau_ip1 = get_tau(sbp1,get_j_state(i + 1), u_ip1)
    f +=  0.5 * dtg * (0*(u_i.T @ u_i + u_ip1.T @ u_ip1) + 0*(tau.T @ tau + tau_ip1.T @ tau_ip1) + 10000*(rot_i.T @ rot_i + rot_ip1.T @ rot_ip1) + (sj.T @ sj + sj_ip1.T @ sj_ip1))
################### End of objective ###################



################### Set up NLP ###################
G = vertcat(*g)
nlp = {'x': X, 'f': f, 'g': G}
solver = nlpsol('S', 'ipopt', nlp)
################### End of NLP setup ###################



################### Setup bounds for variables ###################
# print("Shape of optimization variable X:", X.shape)
lbx = [-inf]*12*N + [-pi/6]*N + [-pi/2]*N + [ 0]*N + [-pi/6]*N + [-pi/2]*N + [ 0]*N + [-pi/6]*N + [-pi/2]*N + [ 0]*N + [-pi/6]*N + [-pi/2]*N + [ 0]*N + [-100]*2*Nc + [0]*Nc + [-100]*2*Nc + [0]*Nc + [-100]*2*Nc + [0]*Nc +[-100]*2*Nc + [0]*Nc
ubx = [ inf]*12*N + [ pi/6]*N + [ pi/2]*N + [pi]*N + [ pi/6]*N + [ pi/2]*N + [pi]*N + [ pi/6]*N + [ pi/2]*N + [pi]*N + [ pi/6]*N + [ pi/2]*N + [pi]*N + [ 100]*12*Nc
 
lbg = [0.]*(G.shape[0]-16*Nc) + [-inf]*16*Nc
ubg = [0.]*(G.shape[0]-16*Nc) + [ 0  ]*16*Nc
#################### End of bounds setup ###################



################# Initial guess for optimization variables ###################
x0 = []
# X position
for i in range(Ng-1):
    x0 += [0]
for i in range(Ng-1, N):
    x0 += [(i-Ng+1)*terminal_body_position[0]/Ng]
# Y position
for i in range(N):
    x0 += [0]
# Z position
for i in range(Ng-1):
    x0 += [0]
for i in range(Ng-1, N):
    x0 += [(i-Ng+1)*terminal_body_position[2]/Ng]
# Roll, pitch, yaw
for i in range(N):
    x0 += [0,0,0]
# X velocity
for i in range(N):
    x0 += [0]
# Y velocity
for i in range(N):
    x0 += [0]
# Z velocity
for i in range(N):
    x0 += [0]
# angular velocities
for i in range(N):
    x0 += [0,0,0]
# Leg joint angles
for i in range(4):
    for j in range(3):
        for k in range(N):
            x0 += [initial_leg_positions[i*3 + j]]
# Control inputs (forces at legs)

for i in range(12*N):
    x0 += [0]
################# End of initial guess ###################
# print(x0)
# print(lbx)
# print(ubx)
# print(lbg)
# print(ubg)
###################
sol = solver(
    x0 = DM(x0),
    lbx = DM(lbx),
    ubx = DM(ubx),
    lbg = DM(lbg),
    ubg = DM(ubg)
)

Xopt     = sol['x'].full().flatten()
xopt     = Xopt[0:N]
yopt     = Xopt[N:2*N]
zopt     = Xopt[2*N:3*N]
phiopt   = Xopt[3*N:4*N]
thetaopt = Xopt[4*N:5*N]
psiopt   = Xopt[5*N:6*N]
vxopt    = Xopt[6*N:7*N]
vyopt    = Xopt[7*N:8*N]
vzopt    = Xopt[8*N:9*N]
jfl1_opt = Xopt[12*N:13*N]
jfl2_opt = Xopt[13*N:14*N]
jfl3_opt = Xopt[14*N:15*N]
jfr1_opt = Xopt[15*N:16*N]
jfr2_opt = Xopt[16*N:17*N]
jfr3_opt = Xopt[17*N:18*N]
jbl1_opt = Xopt[18*N:19*N]
jbl2_opt = Xopt[19*N:20*N]
jbl3_opt = Xopt[20*N:21*N]
jbr1_opt = Xopt[21*N:22*N]
jbr2_opt = Xopt[22*N:23*N]
jbr3_opt = Xopt[23*N:24*N]
uflx_opt = Xopt[24*N:25*N]
ufly_opt = Xopt[25*N:26*N]
uflz_opt = Xopt[26*N:27*N]
ufrx_opt = Xopt[27*N:28*N]
ufry_opt = Xopt[28*N:29*N]
ufrz_opt = Xopt[29*N:30*N]
ublx_opt = Xopt[30*N:31*N]
ubly_opt = Xopt[31*N:32*N]
ublz_opt = Xopt[32*N:33*N]
ubrx_opt = Xopt[33*N:34*N]
ubry_opt = Xopt[34*N:35*N]
ubrz_opt = Xopt[35*N:36*N]

# get body trajectory
body_traj = np.stack((xopt, yopt, zopt, phiopt, thetaopt, psiopt), axis=1)
joint_traj = np.stack((jfl1_opt, jfl2_opt, jfl3_opt,
                       jfr1_opt, jfr2_opt, jfr3_opt,
                       jbl1_opt, jbl2_opt, jbl3_opt,
                       jbr1_opt, jbr2_opt, jbr3_opt), axis=1)
#plot_results
ax = quad.setupView()
ax.set_xlim(-1, 12)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 12)
ax.set_box_aspect((12,2,12))
# plot a flat surface
x,y = np.meshgrid(np.linspace(-5,5,10), np.linspace(-5,5,10))
z = 0*x
plt.plot(x, y, z, alpha=0.5)
quad.visualize_robot(joint_traj[0,:], body_traj[0,:])
for i in range(N):
    # if i % 5 == 4:
        quad.visualize_robot(joint_traj[i,:], body_traj[i,:])
# plot control 
fig, ax = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True)
ax[0,0].plot(uflx_opt,'r'), ax[0,0].set_title('FLX')
ax[1,0].plot(ufly_opt,'g'), ax[1,0].set_title('FLY')
ax[2,0].plot(uflz_opt,'b'), ax[2,0].set_title('FLZ')
ax[0,1].plot(ufrx_opt,'r'), ax[0,1].set_title('FRX')
ax[1,1].plot(ufry_opt,'g'), ax[1,1].set_title('FRY')
ax[2,1].plot(ufrz_opt,'b'), ax[2,1].set_title('FRZ')
ax[0,2].plot(ublx_opt,'r'), ax[0,2].set_title('BLX')
ax[1,2].plot(ubly_opt,'g'), ax[1,2].set_title('BLY')
ax[2,2].plot(ublz_opt,'b'), ax[2,2].set_title('BLZ')
ax[0,3].plot(ubrx_opt,'r'), ax[0,3].set_title('BRX')
ax[1,3].plot(ubry_opt,'g'), ax[1,3].set_title('BRY')
ax[2,3].plot(ubrz_opt,'b'), ax[2,3].set_title('BRZ')
# plot position and linear velocity
fig, ax = plt.subplots(3, 2, figsize=(15, 10), sharex=True, sharey=True)
ax[0,0].plot(xopt,'r'), ax[0,0].set_title('X position')
ax[1,0].plot(yopt,'g'), ax[1,0].set_title('Y position')
ax[2,0].plot(zopt,'b'), ax[2,0].set_title('Z position')
ax[0,1].plot(vxopt,'r'), ax[0,1].set_title('X velocity')
ax[1,1].plot(vyopt,'g'), ax[1,1].set_title('Y velocity')
ax[2,1].plot(vzopt,'b'), ax[2,1].set_title('Z velocity')

plt.legend()
plt.show()