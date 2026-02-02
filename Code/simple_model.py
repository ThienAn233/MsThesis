import casadi as ca
import numpy as np
from math import pi
import matplotlib.pyplot as plt

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
    
    def body2world(self,qb):
        x,y,z,phi,theta,psi = qb[0], qb[1], qb[2], qb[3], qb[4], qb[5]
        T = ca.eye(4)
        T[0:3,3] = ca.vertcat(x,y,z)
        return T @ R_zyx(phi, theta, psi)
    
    def euler2angular(self,qb,dqb):
        phi, theta = qb[3], qb[4]
        p, q, r = dqb[3], dqb[4], dqb[5]
        E = ca.vertcat(
            ca.horzcat(1, 0, -ca.sin(theta)),
            ca.horzcat(0, ca.cos(phi), ca.cos(theta)*ca.sin(phi)),
            ca.horzcat(0, -ca.sin(phi), ca.cos(theta)*ca.cos(phi))
        )
        return E @ ca.vertcat(p,q,r)
    
    def inertia(self,qb):
        # Inertia in world frame
        phi, theta, psi = qb[3], qb[4], qb[5]
        R = R_zyx(phi, theta, psi)
    
    def gravity(self,g=9.81):
        # Gravity in world frame
        return ca.vertcat(0, 0, -self.m*g, 0, 0, 0)
    
    def coriolis(self,qb,euler_rates):
        # Coriolis term in world frame
        R = self.body2world(qb)
        
        return