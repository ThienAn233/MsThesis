import numpy as np
from math import *

def Rotx(theta):
    return np.array([[1,0,0,0],
                     [0,cos(theta),-sin(theta),0],
                     [0,sin(theta),cos(theta),0],
                     [0,0,0,1]])
def Roty(theta):
    return np.array([[cos(theta),0,sin(theta),0],
                     [0,1,0,0],
                     [-sin(theta),0,cos(theta),0],
                     [0,0,0,1]])
def Rotz(theta):
    return np.array([[cos(theta),-sin(theta),0,0],
                     [sin(theta),cos(theta),0,0],
                     [0,0,1,0],
                     [0,0,0,1]])

class IK:
    def __init__(self, l1=25, l2=0, l3=80, l4=80, L=120, W=90, H = 10):
        """ 
        Initialize the Inverse Kinematics parameters 
        where l1, l2, l3, l4 are the lengths of the leg segments,
        L is the length of the body, and W is the width of the body. 
        Default values are provided.
        """
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        self.L = L
        self.W = W
        self.H = H
        
        self.sHp=np.sin(pi/2)
        self.cHp=np.cos(pi/2)
        self.Lo =np.array([0,0,0,1])
        self.Ix = np.array([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
    
    def bodyIK(self,omega,phi,psi,xm,ym,zm):
        '''
        input: body euler angles (rad) and body center position (xm,ym,zm)
        output: transformation matrices of 4 body corners (Tlf,Trf,Tlb,Trb)
        ---------------------------------------------------------------------------
        Tlf: transformation matrix of left front corner
        Trf: transformation matrix of right front corner
        Tlb: transformation matrix of left back corner
        Trb: transformation matrix of right back corner
        ---------------------------------------------------------------------------
        Here we use X for heading, Z for vertical, Y for lateral
        '''
        Rx = Rotx(omega)
        Ry = Roty(phi)
        Rz = Rotz(psi)
        Rxyz=Rx@Ry@Rz

        T   = np.array([[0,0,0,xm],[0,0,0,ym],[0,0,0,zm],[0,0,0,0]]) #translation matrix
        Tm  = T+Rxyz
        Tfl = Tm@(Roty(pi/2)+np.array([[0,0,0,self.L/2],[0,0,0,self.W/2],[0,0,0,self.H],[0,0,0,0]]))
        Tfr = Tm@(Roty(pi/2)+np.array([[0,0,0,self.L/2],[0,0,0,-self.W/2],[0,0,0,self.H],[0,0,0,0]]))
        Tbl = Tm@(Roty(pi/2)+np.array([[0,0,0,-self.L/2],[0,0,0,self.W/2],[0,0,0,self.H],[0,0,0,0]]))
        Tbr = Tm@(Roty(pi/2)+np.array([[0,0,0,-self.L/2],[0,0,0,-self.W/2],[0,0,0,self.H],[0,0,0,0]]))

        return np.array([Tfl,Tfr,Tbl,Tbr])

    def legIK(self,point):
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
        print(f"IK angles: {degrees(theta1):.2f}, {degrees(theta2):.2f}, {degrees(theta3):.2f}")
        return(theta1,theta2,theta3)
    
    def drawRobot(self,Lp,angles,center):
        """_summary_

        Args:
            Lp (numpy array): Leg positions in world space
            angles (numpy array): Body angles (omega, phi, psi)
            center (numpy array): Body center position (xm, ym, zm)

        Returns:
            _numpy array: 12 joint angles for 4 legs
        """
        (omega,phi,psi)=angles
        (xm,ym,zm)=center
        try:
            (Tlf,Trf,Tlb,Trb)= self.bodyIK(omega,phi,psi,xm,ym,zm)
        except ValueError:
            print("IK Error: check body angles/position")
            return
        try:
            out = []
            out+=[*self.legIK(np.linalg.inv(Tlf)@Lp[0])]
            out+=[*self.legIK(self.Ix@np.linalg.inv(Trf)@Lp[1])]
            out+=[*self.legIK(np.linalg.inv(Tlb)@Lp[2])]
            out+=[*self.legIK(self.Ix@np.linalg.inv(Trb)@Lp[3])]
        except ValueError:
            print("IK Error: check leg lengths/positions")
            return 
        return np.array(out)
    
    def calcLegPoints(self,angles):
        '''
        input: leg joint angles (theta1,theta2,theta3) in rad
        output: transformation matrices of leg points (T0,T1,T2,T3,T4) from leg base to foot
        '''
        (theta1,theta2,theta3)=angles
        theta23=theta3+theta2

        T0=self.Lo
        T1=T0+np.array([-self.l1*sin(theta1),self.l1*cos(theta1),0,0])
        T2=T1+np.array([self.l2*cos(theta1),self.l2*sin(theta1),0,0])
        T3=T2+np.array([self.l3*cos(theta1)*cos(theta2),self.l3*sin(theta1)*cos(theta2),-self.l3*sin(theta2),0])
        T4=T3+np.array([-self.l4*cos(theta1)*cos(theta23),-self.l4*sin(theta1)*cos(theta23),self.l4*sin(theta23),0])
        return np.array([T0,T1,T2,T3,T4])

if __name__ == "__main__":
    ik = IK()
    angles = ik.drawRobot(np.array([[100,100,-100,1],[100,-100,-100,1],[-100,100,-100,1],[-100,-100,-100,1]]),(0,0,0),(0,0,0))
    print(angles)