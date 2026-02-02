import casadi as ca
import numpy as np
from math import pi
import matplotlib.pyplot as plt

# ==========================================================
# Helper functions (CasADi)
# ==========================================================
def Rotx(theta):
    c = ca.cos(theta)
    s = ca.sin(theta)
    return ca.vertcat(
        ca.horzcat(1, 0, 0, 0),
        ca.horzcat(0, c,-s, 0),
        ca.horzcat(0, s, c, 0),
        ca.horzcat(0, 0, 0, 1)
    )

def Roty(theta):
    c = ca.cos(theta)
    s = ca.sin(theta)
    return ca.vertcat(
        ca.horzcat(c, 0, s, 0),
        ca.horzcat(0, 1, 0, 0),
        ca.horzcat(-s,0, c, 0),
        ca.horzcat(0, 0, 0, 1)
    )

def Rotz(theta):
    c = ca.cos(theta)
    s = ca.sin(theta)
    return ca.vertcat(
        ca.horzcat(c,-s, 0, 0),
        ca.horzcat(s, c, 0, 0),
        ca.horzcat(0, 0, 1, 0),
        ca.horzcat(0, 0, 0, 1)
    )

def R_zyx(phi, theta, psi):
    return (Rotz(psi) @ Roty(theta) @ Rotx(phi))[0:3,0:3]

def skew(v):
    return ca.vertcat(
        ca.horzcat(0, -v[2],  v[1]),
        ca.horzcat(v[2], 0,  -v[0]),
        ca.horzcat(-v[1], v[0], 0)
    )
# ==========================================================
# Quad Dynamics (CasADi)
# ==========================================================
class QuadDynamicsCasadi:

    def __init__(self, params):
        self.L  = params['L']
        self.W  = params['W']
        self.H  = params['H']
        self.m  = params['m']
        self.l1 = params['l1']
        self.l2 = params['l2']
        self.l3 = params['l3']
        self.l4 = params['l4']

        self.Ib= params.get(
            'Ib',
            ca.DM(np.diag([
            (1/12)*params['m']*(params['H']**2+params['W']**2),
            (1/12)*params['m']*(params['L']**2+params['H']**2),
            (1/12)*params['m']*(params['W']**2+params['L']**2),]))
        )


        self.Lo = ca.DM([0,0,0,1])
        self.Iy = ca.DM([
            [1,0,0,0],
            [0,-1,0,0],
            [0,0,1,0],
            [0,0,0,1]
        ])

    # ------------------------------------------------------
    def world_2_body(self, qb):
        x,y,z,phi,theta,psi = qb[0], qb[1], qb[2], qb[3], qb[4], qb[5]
        T = ca.vertcat(
            ca.horzcat(0,0,0,x),
            ca.horzcat(0,0,0,y),
            ca.horzcat(0,0,0,z),
            ca.horzcat(0,0,0,0)
        )
        return T + Rotx(phi) @ Roty(theta) @ Rotz(psi)

    # ------------------------------------------------------
    def body_2_leg(self):
        Ry = Roty(pi/2)

        def T(dx,dy):
            return Ry + ca.vertcat(
                ca.horzcat(0,0,0,dx),
                ca.horzcat(0,0,0,dy),
                ca.horzcat(0,0,0,self.H),
                ca.horzcat(0,0,0,0)
            )
        return [
            T( self.L/2,  self.W/2),
            T( self.L/2, -self.W/2),
            T(-self.L/2,  self.W/2),
            T(-self.L/2, -self.W/2)
        ]

    # ------------------------------------------------------
    def world_2_leg(self, qb):
        Tm = self.world_2_body(qb)
        return [Tm @ T for T in self.body_2_leg()]

    # ------------------------------------------------------
    def calcLegPoints(self, q):
        t1,t2,t3 = q[0], q[1], q[2]
        t23 = t2 + t3

        T0 = self.Lo
        T1 = T0 + ca.vertcat(-self.l1*ca.sin(t1), self.l1*ca.cos(t1), 0, 0)
        T2 = T1 + ca.vertcat(self.l2*ca.cos(t1), self.l2*ca.sin(t1), 0, 0)
        T3 = T2 + ca.vertcat(
            self.l3*ca.cos(t1)*ca.cos(t2),
            self.l3*ca.sin(t1)*ca.cos(t2),
            -self.l3*ca.sin(t2),
            0
        )
        T4 = T3 + ca.vertcat(
            -self.l4*ca.cos(t1)*ca.cos(t23),
            -self.l4*ca.sin(t1)*ca.cos(t23),
            self.l4*ca.sin(t23),
            0
        )
        return [T0,T1,T2,T3,T4]

    # ------------------------------------------------------
    def get_world_frame_contact_point(self,qb,qj):
        Tlegs = self.world_2_leg(qb)
        cps = []
        cps.append((Tlegs[0] @ self.calcLegPoints(qj[0:3])[-1])[:3])
        cps.append((Tlegs[1] @ self.Iy @ self.calcLegPoints(qj[3:6])[-1])[:3])
        cps.append((Tlegs[2] @ self.calcLegPoints(qj[6:9])[-1])[:3])
        cps.append((Tlegs[3] @ self.Iy @ self.calcLegPoints(qj[9:12])[-1])[:3])
        return cps  # (4×3)
    
    # ------------------------------------------------------
    def get_contact_joints_axis(self,qb,qj):
        contact_pts = self.get_world_frame_contact_point(qb,qj)
        joint_positions = []
        joint_axes = []
        fl = self.calcLegPoints(qj[0:3])
        fr = self.calcLegPoints(qj[3:6])
        bl = self.calcLegPoints(qj[6:9])
        br = self.calcLegPoints(qj[9:12])
        jfl = [fl[0], fl[1], fl[3]]
        jfr = [fr[0], fr[1], fr[3]]
        jbl = [bl[0], bl[1], bl[3]]
        jbr = [br[0], br[1], br[3]]
        Tf = self.world_2_leg(qb)
        joint_positions += [(Tf[0]@x)[:3] for x in jfl]
        joint_positions += [(Tf[1]@self.Iy@x)[:3] for x in jfr]
        joint_positions += [(Tf[2]@x)[:3] for x in jbl]
        joint_positions += [(Tf[3]@self.Iy@x)[:3] for x in jbr]
        joint_axes += [ca.DM([0,0,1])]
        joint_axes += [(Tf[0]@Rotz(qj[0][0])@ca.DM([0,1,0,0]))[:3]]
        joint_axes += [(Tf[0]@Rotz(qj[0][0])@ca.DM([0,1,0,0]))[:3]]
        joint_axes += [ca.DM([0,0,1])]
        joint_axes += [(Tf[1]@self.Iy@Rotz(qj[1][0])@ca.DM([0,1,0,0]))[:3]]
        joint_axes += [(Tf[1]@self.Iy@Rotz(qj[1][0])@ca.DM([0,1,0,0]))[:3]]
        joint_axes += [ca.DM([0,0,1])]
        joint_axes += [(Tf[2]@Rotz(qj[2][0])@ca.DM([0,1,0,0]))[:3]]
        joint_axes += [(Tf[2]@Rotz(qj[2][0])@ca.DM([0,1,0,0]))[:3]]
        joint_axes += [ca.DM([0,0,1])]
        joint_axes += [(Tf[3]@self.Iy@Rotz(qj[3][0])@ca.DM([0,1,0,0]))[:3]]
        joint_axes += [(Tf[3]@self.Iy@Rotz(qj[3][0])@ca.DM([0,1,0,0]))[:3]]
        return contact_pts, joint_positions, joint_axes

    # ------------------------------------------------------
    def leg_ik(self, point):
        x, y, z = point[0], point[1], point[2]
        F = ca.sqrt(x**2 + y**2 - self.l1**2)
        G = F - self.l2
        H = ca.sqrt(G**2 + z**2)
        theta1 = -ca.atan2(x, y) + ca.atan2(F, self.l1)
        D = (self.l3**2 + self.l4**2 - H**2) / (2 * self.l3 * self.l4)
        theta3 = ca.acos(D)
        theta2 = -ca.atan2(z, G) + ca.atan2(self.l4 * ca.sin(theta3), self.l3 - self.l4 * ca.cos(theta3))
        out = ca.SX.zeros(3)
        out[0] = theta1
        out[1] = theta2
        out[2] = theta3
        return out
        
    # ------------------------------------------------------
    def inertia_matrix(self, qb):
        R = R_zyx(qb[3], qb[4], qb[5])
        M = ca.SX.zeros(6,6)
        M[0:3,0:3] = self.m * ca.SX.eye(3)
        M[3:6,3:6] = R @ self.Ib @ R.T
        return M

    # ------------------------------------------------------
    def gravity_vector(self, g=9.81):
        return ca.vertcat(0,0,-self.m*g,0,0,0)

    # ------------------------------------------------------
    def coriolis_term(self,qb, dq):
        R = R_zyx(qb[3], qb[4], qb[5])
        w = dq[3:6]
        Sw = skew(w)
        return ca.blockcat([[self.m * Sw,         ca.DM.zeros(3,3)],
                            [ca.DM.zeros(3,3),    Sw @ R @ self.Ib @ R.T]])

    # ------------------------------------------------------
    def J_contact_base(self, qb, world_foot_points):
        COM_world = qb[0:3]
        J = ca.SX.zeros(12, 6)
        for i in range(4):
            r_world = world_foot_points[i] - COM_world
            J[3*i:3*i+3, 0:3] = ca.SX.eye(3)
            J[3*i:3*i+3, 3:6] = -skew(r_world)
        return J

    # ------------------------------------------------------
    def J_contact_joint_one_leg(self, r_cj, joint_positions, joint_axes):
        N = len(joint_axes)
        J = ca.SX.zeros(3,N)
        for k in range(N):
            p_k = joint_positions[k]
            z_k = joint_axes[k]
            J[:,k] = ca.cross(z_k, r_cj - p_k)
        return J
    
    # ------------------------------------------------------
    def J_contact_joint(self,qb, q_j):
        N = 4
        J = ca.SX.zeros(3*N, 3*N)
        contact_pts, joint_positions_all, joint_axes_all = self.get_contact_joints_axis(qb,q_j)
        
        for leg_id in range(N):
            J_leg = self.J_contact_joint_one_leg(
                contact_pts[leg_id],
                joint_positions_all[leg_id*3:(leg_id+1)*3],
                joint_axes_all[leg_id*3:(leg_id+1)*3]
            )
            J[3*leg_id:3*leg_id+3, 3*leg_id:3*leg_id+3] = J_leg
        return J
    # ------------------------------------------------------
    def get_Euler_rate_matrix(self,phi,theta,psi):
        rate = E = ca.vertcat(
            ca.horzcat(1, 0, -ca.sin(theta)),
            ca.horzcat(0, ca.cos(phi), ca.cos(theta)*ca.sin(phi)),
            ca.horzcat(0, -ca.sin(phi), ca.cos(theta)*ca.cos(phi))
        )
        out = ca.SX.eye(6)
        out[3:6,3:6] = rate
        return out

    # ------------------------------------------------------
    
    def forward_kinematics(self,qb,j_angles):
        leg_points = []
        try:
            Tfl, Tfr, Tbl, Tbr = self.world_2_leg(qb)
        except ValueError:
            print("FK Error: check body angles/position")
            return
        leg_points += [(Tfl @ point)[:3] for point in self.calcLegPoints(j_angles[0:3])]
        leg_points += [(Tfr @ self.Iy @ point)[:3] for point in self.calcLegPoints(j_angles[3:6])]
        leg_points += [(Tbl @ point)[:3] for point in self.calcLegPoints(j_angles[6:9])]
        leg_points += [(Tbr @ self.Iy @ point)[:3] for point in self.calcLegPoints(j_angles[9:12])]
        return leg_points

    # ------------------------------------------------------
    def inverse_kinematics(self,qb,leg_points):
        j_angles = []
        try:
            Tfl, Tfr, Tbl, Tbr = self.world_2_leg(qb)
        except ValueError:
            print("IK Error: check body angles/position")
            return
        j_angles += [self.leg_ik(ca.inv(Tfl)@ca.vertcat(leg_points[0],leg_points[1],leg_points[2],ca.DM(1)))]
        j_angles += [self.leg_ik(self.Iy@ca.inv(Tfr)@ca.vertcat(leg_points[3],leg_points[4],leg_points[5],ca.DM(1)))]
        j_angles += [self.leg_ik(ca.inv(Tbl)@ca.vertcat(leg_points[6],leg_points[7],leg_points[8],ca.DM(1)))]
        j_angles += [self.leg_ik(self.Iy@ca.inv(Tbr)@ca.vertcat(leg_points[9],leg_points[10],leg_points[11],ca.DM(1)))]
        return ca.vertcat(*j_angles)
    # ------------------------------------------------------
    def setupView(self,ax=None,limit=5.0):
        if ax is None:
            ax = plt.axes(projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        return ax
    # ------------------------------------------------------
    def drawLegPoints(self,p):
        plt.plot([x[0] for x in p],[x[1] for x in p],[x[2] for x in p], 'k-', lw=3)
        plt.plot([p[0][0]],[p[0][1]],[p[0][2]],'bo',lw=2)
        plt.plot([p[4][0]],[p[4][1]],[p[4][2]],'ro',lw=2)  
    # ------------------------------------------------------
    def visualize_robot(self, angles, qb):
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
    # ------------------------------------------------------
    
if __name__ == "__main__":
    
    #robot parameters
    params = {
        'L': 2.0,
        'W': 1.0,
        'H': 0.0,
        'm': 5.0,
        'l1': .25,
        'l2': 0.0,
        'l3': 0.8,
        'l4': 0.8,
    }
    quad = QuadDynamicsCasadi(params)
    qb = ca.DM([0,0,0,0,0,0])
    dq = ca.DM.zeros(6)

    qj = ca.DM([0.0, -pi/4, -pi/2,
                0.0, -pi/4, -pi/2,
                0.0, -pi/4, -pi/2,
                0.0, -pi/4, -pi/2])

    
    T_body = quad.world_2_body(qb)
    T_legs = quad.body_2_leg()
    T      = quad.world_2_leg(qb)
    print("Body Transformation dim:", T_body.shape)
    print("Leg Transformations dim:", [T.shape for T in T_legs])
    print("World to Leg Transformations dim:", [T.shape for T in T])
    
    leg_points = quad.forward_kinematics(qb, qj)
    print("Leg Points dim:", [p.shape for p in leg_points])
    
    foot_points = ca.DM([1, 0.5, 0,
                     1,-0.5, 0,
                    -1, 0.5, 0,
                    -1,-0.5, 0])
    
    j_angles = quad.inverse_kinematics(qb, foot_points)
    print("Joint Angles dim:", [j_angles.shape ])
    
    contact_pts = quad.get_world_frame_contact_point(qb,qj)
    print("Contact Points dim:",[cp.shape for cp in contact_pts])
    
    J_cb = quad.J_contact_base(qb, contact_pts)
    print("J_contact_base dim:", J_cb.shape)
    
    data = quad.get_contact_joints_axis(qb, qj)
    print("Contact pts dim:", [dat.shape for dat in data[0]])
    print("Joint positions dim:", [dat.shape for dat in data[1]])
    print("Joint axes dim:", [dat.shape for dat in data[2]])
    
    J_cj = quad.J_contact_joint(qb,qj)
    print("J_contact_joint dim:", J_cj.shape)
    
    M = quad.inertia_matrix(qb)
    print("Inertia matrix dim:", M.shape)
    print("inverse Inertia matrix dim:", ca.inv(M).shape)
    G = quad.gravity_vector()
    print("Gravity vector dim:", G.shape)
    C = quad.coriolis_term(qb,dq)
    print("Coriolis term dim:", C.shape)
    quad.setupView(limit=2.0)
    quad.visualize_robot(qj,qb)
    plt.show()