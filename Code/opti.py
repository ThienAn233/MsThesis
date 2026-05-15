from casadi import *
from utilscasadi import *
import numpy as np

def roll_foward_dynamics(s_bopt,s_jopt, u_opt,quad,gravity=9.81,dt=0.1,N=20):
    out = []
    s_now = s_bopt
    for i in range(N):
        out += [s_now+dt*dyanmics(quad,s_now,s_jopt, u_opt,gravity)]
        s_now = out[-1]
    return out   

def dyanmics(quad, sb, sj, u,gravity=9.81 ):
        b_state = sb
        j_state = sj
        M = quad.inertia_matrix(b_state[0:6])
        G = quad.gravity_vector(gravity)
        
        J_cb = quad.J_contact_base(b_state,quad.get_world_frame_contact_point(b_state,j_state))
        ds = solve(M, (J_cb.T @ u + G))
        ds = vertcat(b_state[6:12], ds)
        return ds

def OCP(initpos,initfoot,initvel,termvel,landheight,quad,N=50,mu=0.5,gravity=9.81):
    T  = ca.SX.sym('T') # Total ground time (s)
    dt = T / N 
    # Helper functions 
    def get_b_state(i):
        return vertcat(x_list[i], y_list[i], z_list[i], phi_list[i], theta_list[i], psi_list[i], vx_list[i], vy_list[i], vz_list[i], wphi_list[i], wtheta_list[i], wpsi_list[i])
    def get_j_state(i):
        return vertcat(jfl1_list[i], jfl2_list[i], jfl3_list[i],
                    jfr1_list[i], jfr2_list[i], jfr3_list[i],
                    jbl1_list[i], jbl2_list[i], jbl3_list[i],
                    jbr1_list[i], jbr2_list[i], jbr3_list[i])
    def get_control(i):
        return vertcat(u_flx_list[i], u_fly_list[i], u_flz_list[i],
                    u_frx_list[i], u_fry_list[i], u_frz_list[i],
                    u_blx_list[i], u_bly_list[i], u_blz_list[i],
                    u_brx_list[i], u_bry_list[i], u_brz_list[i])
    def get_tau(sb,sj,u):
        J_cj = quad.J_contact_joint(sb,sj)
        return -J_cj.T @ u
     
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
    X = vertcat(S, U, T)
    
    
    ################### Constraints list ###################
    g = []

    # Body dynamics constraints
    for i in range(N-1):
        # print(f"Adding dynamics constraint for step {i}")
        sb   = get_b_state(i)
        sbp1 = get_b_state(i+1)
        sj   = get_j_state(i)
        sjp1 = get_j_state(i+1)
        u    = get_control(i)
        up1  = get_control(i+1)
        fi   = dyanmics(quad,sb, sj, u, gravity) 
        fip1 = dyanmics(quad,sbp1, sjp1, up1, gravity)
        g.append(sbp1 - sb - (dt/2)*(fi + fip1)) 
        
    # Initial and final conditions
    foot_points = initfoot
    initial_body_position  = vertcat(*initpos,*initvel)
    terminal_body_position = vertcat(0,0,0,0,0,0,*termvel)
    g.append(get_b_state(0)[0:12] - initial_body_position[0:12])    
    g.append(get_b_state(N - 1)[6:12] - terminal_body_position[6:12]) 
    # When leg touch ground: foot position constraints + friction cone constraints
    for i in range(N):
        qb = get_b_state(i)[0:6]
        qj = get_j_state(i)
        u  = get_control(i)
        lp = quad.forward_kinematics(qb,qj)
        for i,idx in enumerate([4,9,14,19]):
            for j in range(3):
                l = lp[idx][j] - foot_points[i*3 + j]
                # foot points constraint
                g.append(l)
    for j in range(N):
        u = get_control(j)
        qb = get_b_state(j)
        for i in range(4):
            ux = u[i*3 + 0]
            uy = u[i*3 + 1]
            uz = u[i*3 + 2]
            # friction cone constraint
            g.append( sqrt(ux**2 + uy**2 + 1e-6) - mu*uz)
            g.append(landheight+.7/4-qb[2])
    ################### End of constraints list ###################
    
    
    ################### Objective: minimize total control effort ###################
    f = 0#-100*vx_list[-1]**2 - vz_list[-1]**2 # take-off velocity objective
    for i in range(N-1):
        u_i   = get_control(i)
        u_ip1 = get_control(i + 1)
        sb    = get_b_state(i)
        sbp1  = get_b_state(i + 1)


        tau = get_tau(sb,get_j_state(i), u_i)
        tau_ip1 = get_tau(sbp1,get_j_state(i + 1), u_ip1)
        f +=  0.5 * dt * (0.1*(u_i.T @ u_i + u_ip1.T @ u_ip1) + 0*(tau.T @ tau + tau_ip1.T @ tau_ip1))
    ################### End of objective ###################
    
    
    ################### Set up NLP ###################
    G = vertcat(*g)
    nlp = {'x': X, 'f': f, 'g': G}
    options = {'ipopt': {'max_iter': 50000,}}# 'print_level': 0, 'sb': 'yes'}, 'print_time': 0}
    solver = nlpsol('S', 'ipopt', nlp, options)
    ################### End of NLP setup ###################



    ################### Setup bounds for variables ###################
    # print("Shape of optimization variable X:", X.shape)
    lbx = [-inf]*12*N + [-pi/6]*N + [-pi/2]*N + [ 0]*N + [-pi/6]*N + [-pi/2]*N + [ 0]*N + [-pi/6]*N + [-pi/2]*N + [ 0]*N + [-pi/6]*N + [-pi/2]*N + [ 0]*N + [-500]*12*N + [0] 
    ubx = [ inf]*12*N + [ pi/6]*N + [ pi/2]*N + [pi]*N + [ pi/6]*N + [ pi/2]*N + [pi]*N + [ pi/6]*N + [ pi/2]*N + [pi]*N + [ pi/6]*N + [ pi/2]*N + [pi]*N + [ 500]*12*N + [9] 
    
    lbg = [-0.01]*(G.shape[0]-8*N) + [-inf]*8*N
    ubg = [0.01]*(G.shape[0]-8*N) + [ 0.01  ]*8*N
    #################### End of bounds setup ###################
    
    
    
    ################# Initial guess for optimization variables ###################
    def interpolation(start,end,num,i):
        return start + (end - start) * i / (num - 1)
    x0 = []
    # XYZ position
    for i in range(3*N):
        x0 += [0]
    # Roll, pitch, yaw
    for i in range(3*N):
        x0 += [0]
    # XYZ velocity
    for i in range(N):
        x0 += [interpolation(initial_body_position[6], terminal_body_position[6], N, i)]
    for i in range(N):
        x0 += [interpolation(initial_body_position[7], terminal_body_position[7], N, i)]
    for i in range(N):
        x0 += [interpolation(initial_body_position[8], terminal_body_position[8], N, i)]
    # angular velocities
    for i in range(N):
        x0 += [0,0,0]
    # Leg joint angles
    for i in range(4):
        for j in range(3):
            for k in range(N):
                x0 += [0]
    # Control inputs (forces at legs)
    for i in range(12*N):
        x0 += [0]
    x0 += [1] # initial guess for total time T
    ################# End of initial guess ###################
    
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
    wphiopt  = Xopt[9*N:10*N]
    wthetaopt= Xopt[10*N:11*N]
    wpsiopt  = Xopt[11*N:12*N]
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
    print("Optimal total time T:", Xopt[-1])
    # get body trajectory
    body_traj = np.stack((xopt, yopt, zopt, phiopt, thetaopt, psiopt), axis=1)
    joint_traj = np.stack((jfl1_opt, jfl2_opt, jfl3_opt,
                        jfr1_opt, jfr2_opt, jfr3_opt,
                        jbl1_opt, jbl2_opt, jbl3_opt,
                        jbr1_opt, jbr2_opt, jbr3_opt), axis=1)
    ctrl_traj = np.stack((uflx_opt, ufly_opt, uflz_opt,
                        ufrx_opt, ufry_opt, ufrz_opt,
                        ublx_opt, ubly_opt, ublz_opt,
                        ubrx_opt, ubry_opt, ubrz_opt), axis=1)
    state_traj = np.stack((xopt, yopt, zopt, phiopt, thetaopt, psiopt, vxopt, vyopt, vzopt, wphiopt, wthetaopt, wpsiopt), axis=1)
    return body_traj, joint_traj, ctrl_traj, state_traj


def plot_res(state_traj,ctrl_traj):
    fig, ax = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True)
    ax[0,0].plot(ctrl_traj[:,0],'r'), ax[0,0].set_title('FLX')
    ax[1,0].plot(ctrl_traj[:,1],'g'), ax[1,0].set_title('FLY')
    ax[2,0].plot(ctrl_traj[:,2],'b'), ax[2,0].set_title('FLZ')
    ax[0,1].plot(ctrl_traj[:,3],'r'), ax[0,1].set_title('FRX')
    ax[1,1].plot(ctrl_traj[:,4],'g'), ax[1,1].set_title('FRY')
    ax[2,1].plot(ctrl_traj[:,5],'b'), ax[2,1].set_title('FRZ')
    ax[0,2].plot(ctrl_traj[:,6],'r'), ax[0,2].set_title('BLX')
    ax[1,2].plot(ctrl_traj[:,7],'g'), ax[1,2].set_title('BLY')
    ax[2,2].plot(ctrl_traj[:,8],'b'), ax[2,2].set_title('BLZ')
    ax[0,3].plot(ctrl_traj[:,9],'r'), ax[0,3].set_title('BRX')
    ax[1,3].plot(ctrl_traj[:,10],'g'), ax[1,3].set_title('BRY')
    ax[2,3].plot(ctrl_traj[:,11],'b'), ax[2,3].set_title('BRZ')
    # plot position and linear velocity
    fig, ax = plt.subplots(3, 2, figsize=(15, 10), sharex=True, sharey=True)
    ax[0,0].plot(state_traj[:,0],'r'), ax[0,0].set_title('X position')
    ax[1,0].plot(state_traj[:,1],'g'), ax[1,0].set_title('Y position')
    ax[2,0].plot(state_traj[:,2],'b'), ax[2,0].set_title('Z position')
    ax[0,1].plot(state_traj[:,6],'r'), ax[0,1].set_title('X velocity')
    ax[1,1].plot(state_traj[:,7],'g'), ax[1,1].set_title('Y velocity')
    ax[2,1].plot(state_traj[:,8],'b'), ax[2,1].set_title('Z velocity')

    fig, ax = plt.subplots(3, 2, figsize=(15, 10), sharex=True, sharey=True)
    ax[0,0].plot(state_traj[:,3],'r'), ax[0,0].set_title('Roll (x)')
    ax[1,0].plot(state_traj[:,4],'g'), ax[1,0].set_title('Pitch (y)')
    ax[2,0].plot(state_traj[:,5],'b'), ax[2,0].set_title('Yaw (z)')
    ax[0,1].plot(state_traj[:,9],'r'), ax[0,1].set_title('Roll rate (x)')
    ax[1,1].plot(state_traj[:,10],'g'), ax[1,1].set_title('Pitch rate (y)')
    ax[2,1].plot(state_traj[:,11],'b'), ax[2,1].set_title('Yaw rate (z)')
    plt.legend()


if __name__ == "__main__":
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
    foot_points = DM([.25, 0.6/4, 0.1,
                            .25,-0.6/4, 0.1,
                            -.25, 0.6/4, 0.1,
                            -.25,-0.6/4, 0.1])
    initpos = [0,0,.3,0,0,0]
    inivel  = [0,0,0,0,0,0]
    termvel = [1.52,0,3.99,0,0,0]
    landheight = 0.1



    body_traj, joint_traj, ctrl_traj, state_traj= OCP(initpos, foot_points, inivel, termvel, landheight, quad)
    body_traj_rollout = roll_foward_dynamics(state_traj[-1,:],joint_traj[-1,:],np.zeros(12),quad, dt=0.065890226, N=10)

    x_landing  = body_traj_rollout[-1][0]
    vx_landing = body_traj_rollout[-1][6]
    vz_landing = body_traj_rollout[-1][8]
    foot_points = DM([x_landing+.25, 0.6/4, 0.6,
                            x_landing+.25,-0.6/4, 0.6,
                            x_landing-.25, 0.6/4, 0.6,
                            x_landing-.25,-0.6/4, 0.6])
    initpos_landing = [x_landing,0,1,0,0,0]
    print(initpos_landing)
    inivel_landing  = [vx_landing,0,vz_landing,0,0,0]
    print(inivel_landing)
    termvel_landing = [1.87,0,3.69,0,0,0]
    print(termvel_landing)
    landheight_landing = 0.6
    body_traj_landing, joint_traj_landing, ctrl_traj_landing, state_traj_landing = OCP(initpos_landing, foot_points, inivel_landing, termvel_landing, landheight_landing, quad)
    body_traj_rollout_landing = roll_foward_dynamics(state_traj_landing[-1,:],joint_traj_landing[-1,:],np.zeros(12),quad, dt=0.080285942, N=10)

    fig,ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10,8))
    ax = quad.setupView(ax=ax)
    ax.set_xlim(-.5, .5)
    ax.set_ylim(-.5, .5)
    ax.set_zlim(-.1, 1.5)
    ax.set_box_aspect((1,1,1.6))
    x = np.linspace(-0.35,0.35,2)
    y = np.linspace(-0.35,0.35,2)
    X,Y = np.meshgrid(x,y)
    Z = 0*X+0*Y+0.1
    ax.plot_surface(X,Y,Z, color='gray', alpha=.2)
    quad.visualize_robot(joint_traj[0,:], body_traj[0,:])
    for i in range(50):
        if i % 10 == 9:
            quad.visualize_robot(joint_traj[i,:], body_traj[i,:])
    # for i in range(11):
    #     # if i % 2 == 1:
    #         quad.visualize_robot(joint_traj[-1,:], body_traj_rollout[i])
    # quad.visualize_robot(joint_traj_landing[0,:], body_traj_landing[0,:])       
    # for i in range(50):
    #     if i % 10 == 9:
    #         quad.visualize_robot(joint_traj_landing[i,:], body_traj_landing[i,:])
    # for i in range(11):
    #     # if i % 2 == 1:
    #         quad.visualize_robot(joint_traj_landing[-1,:], body_traj_rollout_landing[i])
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