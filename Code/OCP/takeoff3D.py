import numpy as np
import casadi as ca

# “To jump while keeping a constant horizontal speed, 
# set the robot’s horizontal velocity at takeoff equal to the desired speed; 
# during flight (neglecting aerodynamic drag) horizontal velocity remains constant. 
# Vertical motion is governed by ballistic dynamics under gravity and is controlled by the vertical takeoff velocity. 
# If drag is significant, continuous horizontal forces are required to cancel drag so horizontal acceleration is zero.”

# Optimization vars: [COMpos,COMvel,COMagl,COMang] and reaction force on the ground


g  = 9.8 # gravity
Lp0 = np.array([[100,100,0,1],[100,-100,0,1],[-100,100,0,1],[-100,-100,0,1]]) # (x,y,z) order LF, RF, LB, RB
angles0 = [0,0,0]
center0 = [0,0,100]