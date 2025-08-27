import numpy as np
from math import *

class IK:
    def __init__(self, l1=25, l2=0, l3=80, l4=80, L=120, W=90):
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
        
        self.sHp=np.sin(pi/2)
        self.cHp=np.cos(pi/2)
        self.Lo =np.array([0,0,0,1])
        self.Ix = np.array([[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    
    def bodyIK(self, omega, phi, psi, xm, ym, zm):
        """
        Calculate the four Transformation-Matrices for our Legs
        Rx=X-Axis Rotation Matrix
        Ry=Y-Axis Rotation Matrix
        Rz=Z-Axis Rotation Matrix
        Rxyz=All Axis Rotation Matrix
        T=Translation Matrix
        Tm=Transformation Matrix
        Trb,Trf,Tlb,Tlf=final Matrix for RightBack,RightFront,LeftBack and LeftFront
        """
        Rx = np.array([[1,0,0,0],
                   [0,np.cos(omega),-np.sin(omega),0],
                   [0,np.sin(omega),np.cos(omega),0],[0,0,0,1]])
        Ry = np.array([[np.cos(phi),0,np.sin(phi),0],
                    [0,1,0,0],
                    [-np.sin(phi),0,np.cos(phi),0],[0,0,0,1]])
        Rz = np.array([[np.cos(psi),-np.sin(psi),0,0],
                    [np.sin(psi),np.cos(psi),0,0],[0,0,1,0],[0,0,0,1]])
        Rxyz=Rx@Ry@Rz

        T = np.array([[0,0,0,xm],[0,0,0,ym],[0,0,0,zm],[0,0,0,0]])
        Tm = T+Rxyz

        return([Tm @ np.array([[self.cHp,0,self.sHp,self.L/2],[0,1,0,0],[-self.sHp,0,self.cHp,self.W/2],[0,0,0,1]]),
            Tm @ np.array([[self.cHp,0,self.sHp,self.L/2],[0,1,0,0],[-self.sHp,0,self.cHp,-self.W/2],[0,0,0,1]]),
            Tm @ np.array([[self.cHp,0,self.sHp,-self.L/2],[0,1,0,0],[-self.sHp,0,self.cHp,self.W/2],[0,0,0,1]]),
            Tm @ np.array([[self.cHp,0,self.sHp,-self.L/2],[0,1,0,0],[-self.sHp,0,self.cHp,-self.W/2],[0,0,0,1]])
            ])
        
    def legIK(self,point):
        """
        x/y/z=Position of the Foot in Leg-Space
        F=Length of shoulder-point to target-point on x/y only
        G=length we need to reach to the point on x/y
        H=3-Dimensional length we need to reach
        """
        (x,y,z)=(point[0],point[1],point[2])
        F=sqrt(x**2+y**2-self.l1**2)
        G=F-self.l2  
        H=sqrt(G**2+z**2)
        theta1=-atan2(y,x)-atan2(F,-self.l1)

        D=(H**2-self.l3**2-self.l4**2)/(2*self.l3*self.l4)
        theta3=acos(D) 

        theta2=atan2(z,G)-atan2(self.l4*sin(theta3),self.l3+self.l4*cos(theta3))

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


if __name__ == "__main__":
    ik = IK()
    print(ik.drawRobot(np.array([[100,-100,100,1],[100,-100,-100,1],[-100,-100,100,1],[-100,-100,-100,1]]),(0.4,0,0),(0,0,0)))