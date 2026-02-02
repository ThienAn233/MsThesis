import numpy as np
import matplotlib.pyplot as plt
from math import *

### Helper functions ###
def Rotx(theta): # Rotation about x-axis
    return np.array([[1,0,0,0],
                     [0,cos(theta),-sin(theta),0],
                     [0,sin(theta),cos(theta),0],
                     [0,0,0,1]])
    
def Roty(theta): # Rotation about y-axis
    return np.array([[cos(theta),0,sin(theta),0],
                     [0,1,0,0],
                     [-sin(theta),0,cos(theta),0],
                     [0,0,0,1]])
    
def Rotz(theta): # Rotation about z-axis
    return np.array([[cos(theta),-sin(theta),0,0],
                     [sin(theta),cos(theta),0,0],
                     [0,0,1,0],
                     [0,0,0,1]])
    
def R_zyx(phi, theta, psi):
    """Rotation matrix R = Rz(psi) * Ry(theta) * Rx(phi)."""
    return (Rotz(psi) @ Roty(theta) @ Rotx(phi))[:3,:3]

def skew(v):
    return np.array([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0   ]
    ])
### End of helper functions ###





### Kinematics and Dynamics functions ###
class QuadDynamics:
    ''' Class for quadrupeds robot dynamics computations '''
    def __init__(self,robot_params):
        '''
        Initialize with robot parameters dictionary where keys include:
        'L' : body length
        'W' : body width
        'H' : body height
        'm' : total mass
        'Ib' : body inertia tensor
        'l1': leg segment 1 length
        'l2': leg segment 2 length
        'l3': leg segment 3 length
        'l4': leg segment 4 length
        '''
        self.robot_params = robot_params
        self.L = robot_params['L']
        self.W = robot_params['W']
        self.H = robot_params['H']
        self.m = robot_params['m']
        self.Ib = robot_params.get('Ib', np.diag([(1/12.)*self.m*(self.H**2+self.W**2),(1/12.)*self.m*(self.L**2+self.H**2),(1/12.)*self.m*(self.W**2+self.L**2)]))  # default inertia tensor
        self.l1 = robot_params['l1']
        self.l2 = robot_params['l2']    
        self.l3 = robot_params['l3']
        self.l4 = robot_params['l4']
        self.sHp = np.sin(pi/2)
        self.cHp = np.cos(pi/2)
        self.Lo = np.array([0,0,0,1])
        self.Iy = np.array([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
            
    def drawLegPoints(self,p):
        plt.plot([x[0] for x in p],[x[1] for x in p],[x[2] for x in p], 'k-', lw=3)
        plt.plot([p[0][0]],[p[0][1]],[p[0][2]],'bo',lw=2)
        plt.plot([p[4][0]],[p[4][1]],[p[4][2]],'ro',lw=2)  
    
    def setupView(self,limit,fig=None):
        if fig is None:
            ax = plt.axes(projection="3d")
        else:
            ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        return ax
    
    def world_2_body(self,qb):
        '''       
        Input: qb = [xm, ym, zm, omega, phi, psi] (body position and euler angles)
        Output: transformation matrices of body centers
        '''
        xm, ym, zm = qb[0], qb[1], qb[2]
        omega, phi, psi = qb[3], qb[4], qb[5]
        Rx = Rotx(omega)
        Ry = Roty(phi)
        Rz = Rotz(psi)
        Rxyz=Rx@Ry@Rz

        T   = np.array([[0,0,0,xm],[0,0,0,ym],[0,0,0,zm],[0,0,0,0]]) #translation matrix
        Tm  = T+Rxyz
        return Tm
    
    def body_2_leg(self):
        Tfl = Roty(pi/2)+np.array([[0,0,0,self.L/2],[0,0,0,self.W/2],[0,0,0,self.H],[0,0,0,0]])
        Tfr = Roty(pi/2)+np.array([[0,0,0,self.L/2],[0,0,0,-self.W/2],[0,0,0,self.H],[0,0,0,0]])
        Tbl = Roty(pi/2)+np.array([[0,0,0,-self.L/2],[0,0,0,self.W/2],[0,0,0,self.H],[0,0,0,0]])
        Tbr = Roty(pi/2)+np.array([[0,0,0,-self.L/2],[0,0,0,-self.W/2],[0,0,0,self.H],[0,0,0,0]])
        return np.array([Tfl,Tfr,Tbl,Tbr])
    
    def world_2_leg(self,qb):
        '''       
        Input: qb = [xm, ym, zm, omega, phi, psi] (body position and euler angles)
        Output: transformation matrices of 4 body corners (Tlf,Trf,Tlb,Trb)
        '''
        Tm  = self.world_2_body(qb)
        (Tfl, Tfr, Tbl, Tbr) = self.body_2_leg()
        Tfl = Tm @ Tfl
        Tfr = Tm @ Tfr      
        Tbl = Tm @ Tbl
        Tbr = Tm @ Tbr
        
        return np.array([Tfl,Tfr,Tbl,Tbr])
    
    def calcLegPoints(self,angles):
        '''
        input: leg joint angles (theta1,theta2,theta3) in rad, in order of fl, fr, bl, br
        output: list of transformation matrices of leg points (T0,T1,T2,T3,T4) from leg base
        '''
        (theta1,theta2,theta3)=angles
        theta23=theta3+theta2

        T0=self.Lo
        T1=T0+np.array([-self.l1*sin(theta1),self.l1*cos(theta1),0,0])
        T2=T1+np.array([self.l2*cos(theta1),self.l2*sin(theta1),0,0])
        T3=T2+np.array([self.l3*cos(theta1)*cos(theta2),self.l3*sin(theta1)*cos(theta2),-self.l3*sin(theta2),0])
        T4=T3+np.array([-self.l4*cos(theta1)*cos(theta23),-self.l4*sin(theta1)*cos(theta23),self.l4*sin(theta23),0])
        return np.array([T0,T1,T2,T3,T4])
    
    def get_base_frame_contact_point(self,angles):
        '''
        Input: leg joint angles (theta1,theta2,theta3) in rad, in order of fl, fr, bl, br
        Output: contact points positions in base frame
        '''
        contact_points = []
        (Tfl,Tfr,Tbl,Tbr) = self.body_2_leg()
        contact_points += [Tfl@self.calcLegPoints(angles[0])[-1]]
        contact_points += [Tfr@self.Iy@self.calcLegPoints(angles[1])[-1]]
        contact_points += [Tbl@self.calcLegPoints(angles[2])[-1]]
        contact_points += [Tbr@self.Iy@self.calcLegPoints(angles[3])[-1]]
        return np.array(contact_points)

    def get_contact_joints_axis(self,angles):
        '''
        Input: leg joint angles (theta1,theta2,theta3) in rad, in order of fl, fr, bl, br
        Output: contact positions in base frame, shape: (4,3)
                joint positions in base frame, shape: (12,3)
                joint axes in base frame, shape: (12,3)
        '''
        contact_pts = self.get_base_frame_contact_point(angles)[:,:3]
        Tjoint = []
        for i in range(4):
            Tjoint += [self.calcLegPoints(angles[i])[j] for j in [0,1,3]]
        Tf     = self.body_2_leg()
        T_idx  = [0,0,0,1,1,1,2,2,2,3,3,3]
        joint_positions = [Tf[T_idx[i]]@x for i,x in enumerate(Tjoint)]
        joint_axes = []
        for i in range(4):
            joint_axes += [[0,0,1]]
            joint_axes += [(Tf[i]@Rotz(angles[i][0])@np.array([0,1,0,0]))[:3]]
            joint_axes += [(Tf[i]@Rotz(angles[i][0])@np.array([0,1,0,0]))[:3]]
        return  contact_pts, np.array(joint_positions)[:,:3], np.array(joint_axes)[:,:3]

    def leg_ik(self,point):
        '''
        input: foot position in leg frame (x,y,z)
        output: leg joint angles (theta1,theta2,theta3) in rad
        ---------------------------------------------------------------------------
        theta1: hip yaw angle
        theta2: hip pitch angle
        theta3: knee pitch angle
        ---------------------------------------------------------------------------
        '''
        (x,y,z)=(point[0],point[1],point[2])
        F=sqrt(x**2+y**2-self.l1**2)
        G=F-self.l2  
        H=sqrt(G**2+z**2)
        theta1=-atan2(x,y)+atan2(F,self.l1)

        D=(self.l3**2+self.l4**2-H**2)/(2*self.l3*self.l4)
        theta3=acos(D) 

        theta2=-atan2(z,G)+atan2(self.l4*sin(theta3),self.l3-self.l4*cos(theta3))
        return np.array([theta1,theta2,theta3])
    
    def inertia_matrix(self,qb):
        """
        Build full 6x6 inertia matrix M(q) for:
            q_b = [x,y,z,phi,theta,psi]  (floating base)
        m  = total body mass
        Ib = 3x3 body inertia tensor in body frame
        """
        # Extract Euler angles
        phi, theta, psi = qb[3], qb[4], qb[5]

        # Rotation from body to world
        R = R_zyx(phi, theta, psi)

        # Base inertia block 6x6
        Mbb = np.zeros((6, 6))
        Mbb[0:3, 0:3] = self.m * np.eye(3)                 # linear inertia
        Mbb[3:6, 3:6] = R @ self.Ib @ R.T                  # rotational inertia
        return Mbb
    
    def gravity_vector(self,g=9.81):
        """
        Gravity vector g(q) for 6-DOF quadruped.
        ONLY affects base because legs are massless.
        """
        gvec = np.zeros(6)
        gvec[2] = -self.m * g     # only z-axis gravity force
        return gvec

    def coriolis_term(self,dq):
        """
        Compute C(q,dq)*dq for floating base.
        Massless legs => zero contribution.
        """
        # Extract velocities
        v = dq[0:3]      # linear velocity of base
        omega = dq[3:6]  # angular velocity of base

        # Coriolis term for a rigid body
        C_term = np.zeros(6)

        # Linear part: m * (omega x v)
        C_term[0:3] = self.m * np.cross(omega, v)

        # Angular part: omega x (I*omega)
        C_term[3:6] = np.cross(omega, self.Ib @ omega)

        return C_term
    
    def J_contact_base_one_point(self,r_cb):
        """
        Input: r_cb : 3-vector from base COM to contact point
        Returns J_cb : (3x6)
        [ I   -S(r) ]
        """
        J_trans = np.eye(3)
        J_rot   = -skew(r_cb)
        return np.hstack([J_trans, J_rot])
    
    def J_contact_base(self,r_cb):
        """
        r_cb : list of N 3d-vectors from base COM to contact points (N contact points)
        Returns J_cb : (3N x 6)
        """
        J = np.zeros((3*len(r_cb), 6))
        for i, r in enumerate(r_cb):
            J[i*3:(i+1)*3, :] = self.J_contact_base_one_point(r)
        return J
    
    def J_contact_joint_one_leg(self, p_c, joint_positions, joint_axes):
        """
        p_c : (3,) contact position in base frame
        joint_positions : list of joint origins p_k (each 3,) in base frame
        joint_axes : list of joint axes z_k (each 3,) in base frame
        Returns J_leg : (3x3)
        """
        J = np.zeros((3, len(joint_axes)))

        for k, (p_k, z_k) in enumerate(zip(joint_positions, joint_axes)):
            J[:, k] = np.cross(z_k, p_c - p_k)

        return J


    def J_contact_joint(self, q_j):
        """
        q_j : (N,3) joint vector
        fk_leg_funcs : list of N FK functions (one per leg)

        Each fk_leg(q_leg) must return:
        {
            "p_c": (3,),
            "joint_positions": [p1, p2, p3],
            "joint_axes": [z1, z2, z3]
        }

        Returns:
        J_cj : (12x12)
        """
        N = len(q_j)
        J = np.zeros((3*N, 3*N))
        contact_pts, joint_positions_all, joint_axes_all = self.get_contact_joints_axis(q_j)
        
        for leg_id in range(N):
            
            J_leg = self.J_contact_joint_one_leg(
                contact_pts[leg_id],
                joint_positions_all[leg_id*3:(leg_id+1)*3],
                joint_axes_all[leg_id*3:(leg_id+1)*3]
            )

            # Insert block
            J[3*leg_id:3*(leg_id+1),
            3*leg_id:3*(leg_id+1)] = J_leg

        return J
    def forward_kinematics(self,qb,j_angles):
        '''
        Input:
        qb = [xm, ym, zm, omega, phi, psi] (body position and euler angles)
        j_angles = [[theta1,theta2,theta3] for each leg in order fl, fr, bl, br]
        Output: leg_points = list of foot positions in world frame, from leg base to hip abductor joint, hip flexion joint, knee joint and feet in order fl, fr, bl, br
        '''
        leg_points = []
        try:
            (Tlf,Trf,Tlb,Trb)= self.world_2_leg(qb)
        except ValueError:
            print("FK Error: check body angles/position")
            return
        leg_points += [Tlf@x for x in self.calcLegPoints(j_angles[0])]
        leg_points += [Trf@self.Iy@x for x in self.calcLegPoints(j_angles[1])]
        leg_points += [Tlb@x for x in self.calcLegPoints(j_angles[2])]
        leg_points += [Trb@self.Iy@x for x in self.calcLegPoints(j_angles[3])]
        return np.array(leg_points)
    
    def inverse_kinematics(self,qb,leg_points):
        '''
        Input:
        qb = [xm, ym, zm, omega, phi, psi] (body position and euler angles)
        leg_points = list of foot positions in world frame in order fl, fr, bl, br
        Output: j_angles = [[theta1,theta2,theta3] for each leg in order fl, fr, bl, br]
        '''
        j_angles = []
        try:
            Tlf, Trf, Tlb, Trb = self.world_2_leg(qb)
        except ValueError:
            print("IK Error: check body angles/position")
            return
        try:
            j_angles += [self.leg_ik(np.linalg.inv(Tlf)@leg_points[0])]
            j_angles += [self.leg_ik(self.Iy@np.linalg.inv(Trf)@leg_points[1])]
            j_angles += [self.leg_ik(np.linalg.inv(Tlb)@leg_points[2])]
            j_angles += [self.leg_ik(self.Iy@np.linalg.inv(Trb)@leg_points[3])]
        except ValueError:
            print("IK Error: check leg lengths/positions")
            return
        return np.array(j_angles)
    
    def visualize_robot(self, Lp, qb):
        """_summary_

        Args:
            Lp (numpy array): Leg positions in world space
            qb (numpy array): Body state (xm, ym, zm, omega, phi, psi)

        Returns:
            None: draws the robot in 3D space with matplotlib
        """
        
        self.setupView(2).view_init(elev=12., azim=28)
        try:
            angles = self.inverse_kinematics(qb, Lp)
        except:
            print("IK Error: cannot visualize robot, check leg lengths/positions (IK failure)")
            return
        try:
            leg_points = self.forward_kinematics(qb, angles)
        except:
            print("FK Error: cannot visualize robot, check body angles/position (FK failure)")
            return
        CPs=[leg_points[x] for x in [0,5,15,10,0]]
        plt.plot([x[0] for x in CPs],[x[1] for x in CPs],[x[2] for x in CPs], 'bo-', lw=2)
        self.drawLegPoints(leg_points[0:5])
        self.drawLegPoints(leg_points[5:10])
        self.drawLegPoints(leg_points[10:15])
        self.drawLegPoints(leg_points[15:20])
        
        
        # for i in range(4):
        #     plt.plot([leg_points[i][0], leg_points[i+4][0]], [leg_points[i][1], leg_points[i+4][1]], [leg_points[i][2], leg_points[i+4][2]], 'k-', lw=3)
        plt.show()

    
if __name__ == "__main__":

    # -----------------------------
    # Robot parameters
    # -----------------------------
    robot_params = {
        'L' : 2.0,
        'W' : 1.0,
        'H' : 0.0,
        'm' : 5.0,
        'l1': 0.25,
        'l2': 0.0,
        'l3': 0.8,
        'l4': 0.8,
    }

    quad = QuadDynamics(robot_params)

    # -----------------------------
    # Base state
    # -----------------------------
    qb = np.array([0.0, 0.0, 0.0,   # x y z
                   0.0, 0.0, 0.0])  # roll pitch yaw

    dq = np.zeros(6)

    # -----------------------------
    # Joint configuration
    # -----------------------------
    qj = np.radians(np.array([[0, 90, 30],
                           [0, 90, 30],
                           [0, 90, 30],
                           [0, 90, 30]]))  # 4 legs × 3 joints

    # -----------------------------
    # World / body transforms
    # -----------------------------
    T_body = quad.world_2_body(qb)
    T_legs = quad.world_2_leg(qb)

    print("T_body shape:", T_body.shape)        # (4,4)
    print("T_legs shape:", T_legs.shape)        # (4,4,4)

    # -----------------------------
    # Forward kinematics
    # -----------------------------
    leg_points = quad.forward_kinematics(qb, qj)
    print("leg_points shape:", leg_points.shape)
    # Expected: (4 legs × 5 points, 4)

    # -----------------------------
    # Inverse kinematics
    # -----------------------------
    foot_points = [
        np.array([0.6,  0.4, -0.8, 1.0]),
        np.array([0.6, -0.4, -0.8, 1.0]),
        np.array([-0.6, 0.4, -0.8, 1.0]),
        np.array([-0.6,-0.4, -0.8, 1.0]),
    ]

    qj_ik = quad.inverse_kinematics(qb, foot_points)
    print("IK joint angles shape:", qj_ik.shape)  # (4,3)

    # -----------------------------
    # Contact points in base frame
    # -----------------------------
    contact_pts = quad.get_base_frame_contact_point(qj)
    print("Contact points shape:", contact_pts.shape)  # (4,4)

    r_cb = contact_pts[:, :3]  # strip homogeneous coord

    # -----------------------------
    # Base Jacobian
    # -----------------------------
    J_cb = quad.J_contact_base(r_cb)
    print("J_contact_base shape:", J_cb.shape)  # (12,6)

    # -----------------------------
    # Dummy FK function for joint Jacobian
    # -----------------------------
    data = quad.get_contact_joints_axis(qj)

    # -----------------------------
    # Joint Jacobian
    # -----------------------------
    J_cj = quad.J_contact_joint(qj)
    print("J_contact_joint shape:", J_cj.shape)  # (12,12)

    # -----------------------------
    # Inertia, gravity, Coriolis
    # -----------------------------
    M = quad.inertia_matrix(qb)
    g = quad.gravity_vector()
    C = quad.coriolis_term(dq)


    print("Inertia matrix M shape:", M.shape)    # (6,6)
    print("Gravity vector g shape:", g.shape)    # (6,)
    print("Coriolis term C shape:", C.shape)     # (6,)
    print("Joint positions shape:", data[1].shape)  # (12,3)
    print("Joint axes shape:", data[2].shape)       # (12,3)
    print("Contact points shape:", data[0].shape)  # (4,3)
    contact = [leg_points[i] for i in [4,9,14,19]]
    print(leg_points[4])
    quad.visualize_robot(contact, qb)