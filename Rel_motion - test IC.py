

"""
################################################################################################################

Author: Eliott HUBIN
Project: Cubsats’s study in formation for Earth observation
File: Rel_motion - test IC.py
Description: Initial tests performed to understand the dynamics of the different parameters of the CW equations
Date: 30/08/2023

################################################################################################################
"""




import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Functions import *


mu = 398600.5 #km3/s2
r_tgt = 6378.137 + 400 #km
w = np.sqrt(mu/r_tgt**3) #rad/s



# Initial CW equations tested (without reformulation)
def interceptor_3Dmotion(t,x0,x0_dot,y0,y0_dot,z0,z0_dot):
    
    x = x0_dot/w*np.sin(w*t) - (3*x0 + 2*y0_dot/w)*np.cos(w*t) + 2*(2*x0 + y0_dot/w)
    y = 2*(3*x0 + 2*y0_dot/w)*np.sin(w*t) + 2*x0_dot/w*np.cos(w*t) - 3*(2*x0 + y0_dot/w)*w*t + (y0 - 2*x0_dot/w)
    z = z0*np.cos(w*t) + z0_dot/w*np.sin(w*t)

    return x,y,z

def eccentricity(x,y):

    a = np.max(y)/2
    b = np.min(x)/2
    c = np.sqrt(a**2 - b**2)

    e = c/a
    return e,a


test1 = 0
test2 = 0
test3 = 0
test4 = 0
test5 = 0
test6 = 0
test7 = 1
test8 = 0



#------------------1. Deputy Satellite Motion for x0 and y0 Displacements, x0_dot and y0_dot Variations----------------------
if (test1):

    t = np.linspace(0, 10000, 100000)
    x0_dot_list = [0.05, 0.01, 0.0]

    for x0_dot in x0_dot_list:

        x0, y0, y0_dot = 25 , 100 , -0.04415 
        x,y,z = interceptor_3Dmotion(t,x0,x0_dot,y0,y0_dot,0,0)
        plt.plot(y, x, label='x0_dot = {:.2f} m/s'.format(x0_dot))

        x0, y0, y0_dot = -25 , -100 , 0.04415
        x,y,z = interceptor_3Dmotion(t,x0,-x0_dot,y0,y0_dot,0,0)
        plt.plot(y, x, label='x0_dot = -{:.2f} m/s'.format(x0_dot))

    plt.scatter(100, 25, marker='o', s=50, color='black', label='Initial point')
    plt.scatter(-100, -25, marker='o', s=50, color='black')

    plt.xlim([-250, 250])
    plt.ylim([-150, 150])
    plt.xlabel('y [m]')
    plt.ylabel('x [m]')
    plt.title('Deputy Satellite motion for x0 and y0 Displacements, x0_dot and y0_dot Variations')






#--------------2. Motion of the Deputy Satellite for Various x0 Displacements-------------
if (test2):

    # Compute satellite position over time
    t = np.linspace(0, 10000, 100000)
    x0_list = [100, 200, 300, 400] #[m]

    z0,z0_dot,phi0,theta0 = 0,0,0,0

    x0_dot, y0, y0_dot = 0.0 , 0.0 , -2*w*x0_list[0]     #Set y0_dot=-3*w*x0/2 to obtain a straight line (C=0) and to -2wx0 to have ellipse
    x,y,z = relative_3Dmotion(t,x0_list[0] ,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
    plt.plot(y, x, label=r'$x_0$ = 100 m')

    x0_dot, y0, y0_dot = 0.0 , 0.0 , -2*w*x0_list[1]     
    x,y,z = relative_3Dmotion(t,x0_list[1] ,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
    plt.plot(y, x, label=r'$x_0$ = 200 m')

    x0_dot, y0, y0_dot = 0.0 , 0.0 , -2*w*x0_list[2]    
    x,y,z = relative_3Dmotion(t,x0_list[2] ,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
    plt.plot(y, x, label=r'$x_0$ = 300 m')

    x0_dot, y0, y0_dot = 0.0 , 0.0 , -2*w*x0_list[3]   
    x,y,z = relative_3Dmotion(t,x0_list[3] ,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
    plt.plot(y, x, label=r'$x_0$ = 400 m')


    plt.scatter(0,0,color='k',label='Chief satellite')
    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.axvline(x=0, color='black', linewidth=0.5)

    plt.xlabel('y [m]', fontsize=24)
    plt.ylabel('x [m]', fontsize=24)
    plt.title('Motion of the deputy satellite for Various x0 Displacements')

    plt.legend(loc='upper right',fontsize=17)
    plt.tick_params(axis='both', labelsize=17)





#------------------3. Motion of the Deputy Satellite for x0_dot Variations (no other dispalcement)-----------------------

if (test3):
    t = np.linspace(0, 10000, 100000)
    x0_dot_list = [0.5, 1, 5, 10, 50]

    x0, y0, y0_dot = 0.0 , 0.0 , 0.0  #set y0=2*x0_dot/w affect the center of the ellipse
    x,y,z = interceptor_3Dmotion(t,x0,x0_dot_list[0],y0,y0_dot,0,0)
    plt.plot(y, x, label= r'$\dot{x}_0$ = 0.5 m/s')
    x,y,z = interceptor_3Dmotion(t,x0,x0_dot_list[1],y0,y0_dot,0,0)
    plt.plot(y, x, label= r'$\dot{x}_0$ = 1 m/s')
    x,y,z = interceptor_3Dmotion(t,x0,x0_dot_list[2],y0,y0_dot,0,0)
    plt.plot(y, x, label= r'$\dot{x}_0$ = 5 m/s')
    x,y,z = interceptor_3Dmotion(t,x0,x0_dot_list[3],y0,y0_dot,0,0)
    plt.plot(y, x, label= r'$\dot{x}_0$ = 10 m/s')
    x,y,z = interceptor_3Dmotion(t,x0,x0_dot_list[4],y0,y0_dot,0,0)
    plt.plot(y, x, label= r'$\dot{x}_0$ = 50 m/s')

    x,y,z = interceptor_3Dmotion(t,x0,-x0_dot_list[0],y0,y0_dot,0,0)
    plt.plot(y, x, label= r'$\dot{x}_0$ = -0.5 m/s')
    x,y,z = interceptor_3Dmotion(t,x0,-x0_dot_list[1],y0,y0_dot,0,0)
    plt.plot(y, x, label= r'$\dot{x}_0$ = -1 m/s')
    x,y,z = interceptor_3Dmotion(t,x0,-x0_dot_list[2],y0,y0_dot,0,0)
    plt.plot(y, x, label= r'$\dot{x}_0$ = -5 m/s')
    x,y,z = interceptor_3Dmotion(t,x0,-x0_dot_list[3],y0,y0_dot,0,0)
    plt.plot(y, x, label= r'$\dot{x}_0$ = -10 m/s')
    x,y,z = interceptor_3Dmotion(t,x0,-x0_dot_list[4],y0,y0_dot,0,0)
    plt.plot(y, x, label= r'$\dot{x}_0$ = -50 m/s')


    for x0_dot in x0_dot_list:

        x0, y0, y0_dot = 0.0 , 0.0 , 0.0  
        x,y,z = interceptor_3Dmotion(t,x0,x0_dot,y0,y0_dot,0,0)
        plt.plot(y, x, label= r'$\dot{x}_0$ = {:.2f} m/s')

        x0, y0, y0_dot = 0.0 , 0.0 , 0.0 
        x,y,z = interceptor_3Dmotion(t,x0,-x0_dot,y0,y0_dot,0,0)


    plt.scatter(0, 0, color='k', label='Chief satellite')

    plt.legend()
    plt.xlabel('y [m]')
    plt.ylabel('x [m]')
    plt.title('Motion of the deputy satellite for x0_dot Variations (no other dispalcement)')
    plt.xlim([-3000, 3000])
    plt.ylim([-2000, 2000])




#------------------3bis. Motion of the Deputy Satellite for x0_dot Variations (x0 dispalcement and no drift)-----------------------
if (test4):
    t = np.linspace(0, 10000, 100000)
    x0_dot_list = [0.5, 1, 5, 10, 50]

    for x0_dot in x0_dot_list:

        x0 = 200.0
        y0 = 2*x0_dot/w   #set y0=-2*x0_dot/w affect the center of the ellipse
        y0_dot = -2*w*x0
        x,y,z = relative_3Dmotion(t,x0,x0_dot,y0,y0_dot,0,0,0,0,w)
        plt.plot(y, x, label='x0_dot = {:.2f} m/s'.format(x0_dot))


    # Return the eccentricity of the orbit
    e,a = eccentricity(x,y)
    print('eccentricity =',e)
    print('semimajor axis =',a)

    plt.scatter(0, 0, color='r', label='Target satellite')

    plt.xlabel('y (m)')
    plt.ylabel('x (m)')
    plt.title('Motion of the Deputy Satellite (bis) for x0_dot Variations (no other dispalcement)')





#------------------5. Motion of the Deputy Satellite for y0_dot Variations (no drift)-----------------------
if (test5):
    t = np.linspace(0, 10000, 100000)
    #y0_dot_list = [-0.117744] #for x0=50 m/s
    #y0_dot_list = [-11.7744]
    y0_dot_list = [0.113, 0.25, 0.5, 1, 5, 10, 50]

    # Loop over different values of x0_dot
    for y0_dot in y0_dot_list:
        x0 = 5000
        x0_dot = 500 
        y0 = 2*x0_dot/w   #with this y0 value, the initial center of the ellipse will be about the target
        x,y,z = relative_3Dmotion(t,x0,x0_dot,y0,y0_dot,0,0,0,0,w)
        plt.plot(y, x, label='y0_dot = {:.3f} m/s'.format(y0_dot))

    # Return the eccentricity of the orbit
    e,a = eccentricity(x,y)
    print('eccentricity =',e)
    print('semimajor axis =',a)

    plt.scatter(0, 0, color='r', label='Target satellite')

    plt.xlabel('y (m)')
    plt.ylabel('x (m)')
    plt.title('Motion of the Interceptor for y0_dot Variations (no other dispalcement)')

    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.axvline(x=0, color='black', linewidth=0.5)




#----------------------6. Motion of the Deputy Satellite for x0 displacements and x0_dot variations-------------------
if (test6):
    t = np.linspace(0, 10000, 100000)
    x0_dot_list = [0.5, 1, 5.0, 10, 50] #[m/s]

    for x0_dot in x0_dot_list:

        x0, y0, y0_dot = 50 , 0 , 0.0
        x,y,z = interceptor_3Dmotion(t,x0,-x0_dot,y0,y0_dot,0,0)
        plt.plot(y, x, label='x0_dot = -{:.1f} m/s'.format(x0_dot))

    plt.scatter(0, 50, marker='o', s=50, color='black', label='Initial point')

    plt.xlim([-3000, 3000])
    plt.ylim([-2000, 2000])
    plt.xlabel('y [m]')
    plt.ylabel('x [m]')
    plt.title('Motion of the Deputy Satellite for x0 Displacements and x0_dot Variations')

    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.axvline(x=0, color='black', linewidth=0.5)





#----------------------7. Motion of the Deputy Satellite for x0 displacements and y0_dot variation (no other displacements)-------------------
if (test7):
    
    z0,z0_dot = 0,0
    t = np.linspace(0, 20000, 10000)
    y0_dot_list1 = [0.2, 2.2, 12, 24] #[m/s]
    y0_dot_list2 = [-2*w*500, -1.3, -2.5, -12, -24] #Notice the special case of an “orbit” (with target satellite being the center) at –0.117 m/s (=-2wx0).

    x0, y0, x0_dot = 500.0 , 0.0 , 0.0
    x,y,z = interceptor_3Dmotion(t,x0,x0_dot,y0,y0_dot_list1[0],z0,z0_dot) 
    plt.plot(y, x, label= r'$\dot{y}_0$ = 0.2 m/s')
    x,y,z  = interceptor_3Dmotion(t,x0,x0_dot,y0,y0_dot_list1[1],z0,z0_dot)
    plt.plot(y, x, label= r'$\dot{y}_0$ = 2.2 m/s')
    x,y,z  = interceptor_3Dmotion(t,x0,x0_dot,y0,y0_dot_list1[2],z0,z0_dot)
    plt.plot(y, x, label= r'$\dot{y}_0$ = 12 m/s')
    x,y,z  = interceptor_3Dmotion(t,x0,x0_dot,y0,y0_dot_list1[3],z0,z0_dot)
    plt.plot(y, x, label= r'$\dot{y}_0$ = 24 m/s')

    x,y,z  = interceptor_3Dmotion(t,x0,x0_dot,y0,y0_dot_list2[0],z0,z0_dot)
    plt.plot(y, x, label= r'$\dot{y}_0$ = $-2\omega x_o$ m/s')
    x,y,z  = interceptor_3Dmotion(t,x0,x0_dot,y0,y0_dot_list2[1],z0,z0_dot)
    plt.plot(y, x, label= r'$\dot{y}_0$ = -1.3 m/s')
    x,y,z  = interceptor_3Dmotion(t,x0,x0_dot,y0,y0_dot_list2[2],z0,z0_dot)
    plt.plot(y, x, label= r'$\dot{y}_0$ = -2.5 m/s')
    x,y,z  = interceptor_3Dmotion(t,x0,x0_dot,y0,y0_dot_list2[3],z0,z0_dot)
    plt.plot(y, x, label= r'$\dot{y}_0$ = -12 m/s')
    x,y,z  = interceptor_3Dmotion(t,x0,x0_dot,y0,y0_dot_list2[4],z0,z0_dot)
    plt.plot(y, x, label= r'$\dot{y}_0$ = -24 m/s')


    plt.scatter(0, 0, marker='o', s=50, color='black', label='Chief satellite')

    plt.xlim([-4000, 4000])
    plt.ylim([-3000, 3000])
    plt.xlabel('y [m]', fontsize=24) 
    plt.ylabel('x [m]', fontsize=24)
    plt.title('Motion of the Deputy Satellite for x0 Displacements and y0_dot Variations')

    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.axvline(x=0, color='black', linewidth=0.5)

    plt.legend(fontsize=17) #17
    plt.tick_params(axis='both', labelsize=17)




#--------------------8. Relative Motion of the Deputy Satellite with a z Variation--------------------

if (test8):
    t = np.linspace(0, 20000, 10000)
    x0_list = [50] #[m/s]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for x0 in x0_list:

        if x0==50:
            x0_dot = 0.0
            y0, y0_dot, z0, z0_dot = 2*x0_dot/w , -2*w*x0 , 50 , 0.
            x2,y2,z2 = relative_3Dmotion(t,x0,x0_dot,y0,y0_dot,z0,z0_dot,0,np.radians(90),w)
            x3,y3,z3 = relative_3Dmotion(t,x0,x0_dot,y0,y0_dot,z0,z0_dot,0,np.radians(0),w)
            ax.plot(x2, y2, z2,label= r'$\theta_0$ = 90°')
            ax.plot(x3, y3, z3,label= r'$\theta_0$ = 0°')

    ax.scatter(0, 0, 0, marker='o', s=50, color='black', label='Chief satellite')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')


plt.legend()
plt.show()



