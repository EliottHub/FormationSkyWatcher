

"""
##################################################################

Author: Eliott HUBIN
Project: Cubsats’s study in formation for Earth observation
File: TypeOfFormation.py
Description: Lists all the physically accepted formation tested.
Date: 30/08/2023

##################################################################
"""




import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Functions import *

mu = 398600.5 #km3/s2
r_tgt = 6378.137 + 400 #km
w = np.sqrt(mu/r_tgt**3) #rad/s
T = 2*np.pi/w

t = np.linspace(0, T, 3600)


# Small on-board computer for easy management of flight formation type activation

Spiral_3D = 1
Spiral_2D = 0
Sym_Formation_2D = 0
Sym_Formation_3D = 0
Single_Orbit_Formation = 0
Phase_Shift_2D = 0
Y_Formation_3D = 0
Y_Formation_2D = 0




"""
##################################################################
#################### Tested Formation Flights ####################
##################################################################
"""


#------------------------ 3D Spiral (concentric orbits) or 2D Spiral to control the slope -------------------------
if (Spiral_3D):

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')

    t = np.linspace(0, T, 3600)

    phi0_list = [0,90,180,270] 
    x0_list =  [5,11,17,22,24,30,35] 

    dephasage, index = 0,1

    count = 0

    num_sat = len(phi0_list)*len(x0_list)+1  
    print('Number of satellite :',num_sat)

    colors = ['r', 'm', 'g', 'b', 'r', 'y']

    # Slope 
    x_slope = []
    z_slope = []

    coef = 1.3

    for x0 in x0_list:
        for phi0, color in zip(phi0_list, colors):

            x0_dot = 0.0
            y0 = 2*x0_dot/w
            y0_dot = -2*w*x0
            z0 = coef*x0  
            z0_dot = 0.0
            theta0 = -(90-dephasage+phi0)

            x,y,z = relative_3Dmotion(t[2700],x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0-dephasage),np.radians(theta0),w)
            x_slope.append(x)
            z_slope.append(z)
            if count == 0:
                #ax.scatter(x, y, z, marker='.', s=50, color='b',label='Deputy satellites')
                plt.scatter(x, z, marker='.', color='b',label='Deputy satellites')
                count+=1
            else:
                #ax.scatter(x, y, z, marker='.', s=50, color='b')
                plt.plot(x, z, marker='.', color='b')
        dephasage+=220/len(x0_list) 


    # Plot concentric relative orbits
    T = 2*np.pi/w
    t2 = np.linspace(0, T, 3600)

    x0 = 35
    x0_dot = 0.0
    y0 = 2*x0_dot/w
    y0_dot = -2*w*x0
    z0 = coef*x0   
    z0_dot = 0.0
    phi0 = 0
    theta0 = -(90+dephasage+phi0)

    x,y,z = relative_3Dmotion(t2,x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0+dephasage),np.radians(theta0),w)

    
    #ax.plot(x, y, z, color ='blue', alpha=0.2, label='Outer relative orbit')

    # Calculate highest and lowest points along the z-axis
    highest_point = np.max(z)
    lowest_point = np.min(z)

    # Plot line connecting highest and lowest points
    z_length = np.sqrt((x.max() - x.min())**2 + (0 - 0)**2 + (highest_point - lowest_point)**2)

    #ax.plot([x.min(), x.max()], [0, 0], [highest_point, lowest_point], color='red', linestyle='dashed', label='Minor axis')

    highest_point_y = np.max(y)
    lowest_point_y = np.min(y)

    # Plot line connecting highest and lowest points
    y_length = np.sqrt((0 - 0)**2 + (lowest_point_y - highest_point_y)**2 + (0 - 0)**2)

    #ax.plot([0,0], [lowest_point_y, highest_point_y], [0, 0], color='orange', linestyle='dashed', label='Major axis')

    print("Length of z-axis line:", z_length)
    print("Length of y-axis line:", y_length)

    # Compute slope of the tilt
    slope = np.polyfit(z_slope, x_slope, 1)[0]
    x = np.array([min(z_slope), max(z_slope)])  # Plage des valeurs z
    y = slope * x  # Calcul des valeurs y en utilisant la pente
    #plt.plot(y, x, 'r-', label='Pente')  # Tracé de la pente

    # Compute angle of the slope
    angle_rad = np.arctan(slope)
    angle_deg = np.degrees(angle_rad)

    print("inclinaison =",90-(-round(angle_deg, 1)))
    

    plt.xlabel('x [m]', fontsize=24)
    plt.ylabel('z [m]', fontsize=24)
    plt.xlim(-100,100)
    plt.ylim(-100,100)

    # ax.set_xlabel('x [m]', fontsize=18)
    # ax.set_ylabel('y [m]', fontsize=18)
    # ax.set_zlabel('z [m]', fontsize=18)
    # ax.set_xlim(-100,100)
    # ax.set_ylim(-110,110)
    # ax.set_zlim(-100,100)

    plt.axhline(y=0, color='k', linestyle='--', label='Absolute orbital plane')
    plt.legend(fontsize=18)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tick_params(axis='both', labelsize=20)

    #plt.savefig('test1.png')









#------------------------ Symetrical Orbits Formation Flight 2D - left and right -------------------------

if (Sym_Formation_2D):

    x0_dot = 0.
    x0 = 17
    y0_dot = -2*w*x0
    z0 = 0
    z0_dot = 0.0
    theta0 = 0
    C = np.sqrt( (3*x0+2*y0_dot/w)**2 + (x0_dot/w)**2 )
    y0_list = [(2*C + 10) + 2*x0_dot/w , -(2*C + 10) + 2*x0_dot/w]
    phi0_list = np.linspace(0,360,15)
    #count = 0

    num_sat = len(y0_list)*len(phi0_list)+1  #min:20 / max:32
    print('Number of satellite :',num_sat)
    pos_sat = np.zeros((num_sat,3))

    colors = ['b', 'orange']
    index=1

    labels = ['Upstream orbit', 'Downstream orbit']

    for y0, color, label in zip(y0_list, colors, labels):
        for phi0 in phi0_list:
            x, y, z = relative_3Dmotion(0, x0, x0_dot, y0, y0_dot, z0, z0_dot, np.radians(phi0), np.radians(theta0), w)
            pos_sat[index,:] = np.array([x, y, z])
            plt.scatter(pos_sat[index,1], pos_sat[index,0], color=color)

            if index == 1:  # Add label for the first point in upstream orbit
                plt.scatter(pos_sat[index,1], pos_sat[index,0], color=color, label='Upstream deputy satellites')
            elif index == len(phi0_list) + 1:  # Add label for the first point in downstream orbit
                plt.scatter(pos_sat[index,1], pos_sat[index,0], color=color, label='Downstream deputy satellites')
            else:
                plt.scatter(pos_sat[index,1], pos_sat[index,0], color=color)

            index += 1

        # Print orbits
        num_points = 3600
        T = 2*np.pi/w #period
        t = np.linspace(0, T, num_points)
        x, y, z = relative_3Dmotion(t, x0, x0_dot, y0, y0_dot, z0, z0_dot, np.radians(phi0), np.radians(theta0), w)
        plt.plot(y, x, alpha = 0.3, label=label)

    plt.scatter(pos_sat[0,0], pos_sat[0,1], color='k', label='Chief satellite')

    plt.xlabel("y [m]", fontsize=24)
    plt.ylabel("x [m]", fontsize=24)
    plt.legend(fontsize=17,loc='upper center')
    plt.tick_params(axis='both', labelsize=17)

    plt.grid()




#------------------------ Symetrical Orbits Formation Flight 3D - left and right -------------------------
if (Sym_Formation_3D):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    num_points = 3600
    T = 2*np.pi/w #period
    t = np.linspace(0, T, num_points)

    x0_dot = 0.
    y0_list = [(799.9998476066492 + 100) + 2*x0_dot/w , -(799.9998476066492 + 100) + 2*x0_dot/w]
    phi0_list = [0]
    count = 0

    colors = ['b', 'r']

    for y0, color in zip(y0_list, colors):
        count = 0
        for phi0 in phi0_list:

            x0 = 400
            y0_dot = -2*w*x0
            z0 = 0
            z0_dot = 0.0
            theta0 = 0

            x, y, z = relative_3Dmotion(t[0], x0, x0_dot, y0, y0_dot, x0, z0_dot, np.radians(phi0), theta0, w)
            x2, y2, z2 = relative_3Dmotion(t[1800], x0, x0_dot, y0, y0_dot, x0, z0_dot, np.radians(phi0), theta0, w)

            if count == 0 and color == 'r':
                ax.scatter(y, x, z, marker='.', s=50, color=color, label=r'Satellites positions on left orbit, $t_0$')
            elif count == 0 and color == 'b':
                ax.scatter(y, x, z, marker='.', s=50, color=color, label=r'Satellites positions on right orbit, $t_0$')
            else:
                ax.scatter(y, x, z, marker='.', s=50, color=color) 


            if color == 'b':
                if count == 0:
                    ax.scatter(y2, x2, z2, marker='.', s=50, color='steelblue', label=r'Satellites positions on right orbit, $t_{T/2}$')
                else:
                    ax.scatter(y2, x2, z2, marker='.', s=50, color='steelblue') 
            else:
                if count == 0:
                    ax.scatter(y2, x2, z2, marker='.', s=50, color='#FF6666', label=r'Satellites positions on left orbit, $t_{T/2}$')
                else:
                    ax.scatter(y2, x2, z2, marker='.', s=50, color='#FF6666') 

            
            count += 1
            if count <= 15:
                phi0_list.append(count * 360/15)
            else:
                break


    t2=np.linspace(0, T, num_points)
    dephasage = 0

    # Plot relative orbits
    y0 = (799.9998476066492 + 100)   #100m a droite du target (a = 938.78m)
    for phi0 in phi0_list:
        x, y, z = relative_3Dmotion(t2, x0, x0_dot, y0, y0_dot, x0, z0_dot, np.radians(phi0), theta0, w)
        ax.plot(y, x, z, alpha = 0.2)
    y_max = np.max(y)
    y_min = np.min(y)
    a = (y_max-y_min)/2
    print("a =",a)

    y0 = -(799.9998476066492 + 100) + 2*x0_dot/w  #100m a gauche du target (a = 938.78m)
    for phi0 in phi0_list:
        x, y, z = relative_3Dmotion(t2, x0, x0_dot, y0, y0_dot, x0, z0_dot, np.radians(phi0), theta0, w)
        ax.plot(y, x, z, alpha = 0.2)
    #ax.plot(y, x, z, label='Second relative orbit', alpha = 0.5)

    # Plot the target satellite
    ax.scatter(0, 0, 0, marker='.', s=50, color='k', label='Chief satellite')

    # Set plot parameters
    plt.legend(loc='upper right')
    #plt.xlabel('y [m]')
    #plt.ylabel('x [m]')
    ax.set_xlabel('y [m]')
    ax.set_ylabel('x [m]')
    ax.set_zlabel('z [m]')
    #plt.xlim(-1500,1500)
    #plt.ylim(-1000,1000)





#------------------------ Spiral 2D Formation flight - concentric orbits -------------------------

if (Spiral_2D):
    T = 2*np.pi/w
    t = np.linspace(0, T, 3600)

    # Plot the spiral 
    phi0_list = [0,90,180,270] #5[0,72,144,216,288] 
    x0_list = np.linspace(25,300,7)
    print(x0_list)

    dephasage, index = 0,1

    Time = 0

    num_sat = len(phi0_list)*len(x0_list)+1  #min:20 / max:32
    print('Number of satellite :',num_sat)
    pos_sat = np.zeros((num_sat,3))


    colors = ['orange', 'b', 'g', 'r', 'm', 'y']


    for x0 in x0_list:
        count = 0
        for phi0, color in zip(phi0_list, colors):
        #for phi0 in phi0_list:

            x0_dot = 0.0
            y0 = 2*x0_dot/w
            y0_dot = -2*w*x0
            z0 = 0.0
            z0_dot = 0.0
            theta0 = -(90+phi0+dephasage)

            x,y,z = relative_3Dmotion(Time,x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0-dephasage),np.radians(theta0),w) 
            pos_sat[index,:] = np.array([x,y,z])
            if color == 'orange' and x0 == 25:
                plt.scatter(pos_sat[index,1], pos_sat[index,0], color=color, label='Satellite positions, arm 1')
            elif color == 'b' and x0 == 25:
                plt.scatter(pos_sat[index,1], pos_sat[index,0], color=color, label='Satellite positions, arm 2')
            elif color == 'g' and x0 == 25:
                plt.scatter(pos_sat[index,1], pos_sat[index,0], color=color, label='Satellite positions, arm 3')
            elif color == 'r' and x0 == 25:
                plt.scatter(pos_sat[index,1], pos_sat[index,0], color=color, label='Satellite positions, arm 4')
            elif color == 'm' and x0 == 25:
                plt.scatter(pos_sat[index,1], pos_sat[index,0], color=color, label='Satellite positions, arm 5')
            else:
                plt.scatter(pos_sat[index,1], pos_sat[index,0], color=color)
            

            # Pour orbite unique
            #count += 1
            #if count <= num_sat-2:
            #    phi0_list.append(count * 360/(num_sat-1))
            #else:
            #    break

            index+=1
        dephasage+=192/len(x0_list) #deg 


    # Plot concentric relative orbits
    for x0 in x0_list:
        x0_dot = 0.0
        y0 = 2*x0_dot/w
        y0_dot = -2*w*x0
        z0, z0_dot, phi0, theta0 = 0,0,0,0

        x,y,z = relative_3Dmotion(t,x0,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
        if x0 == 25:
            plt.plot(y, x,alpha=0.2, color ='blue', label='Deputy satellite relative orbits')
        else:
            plt.plot(y, x,alpha=0.2, color ='blue')

    plt.scatter(pos_sat[0,0], pos_sat[0,1], color='k', label='Chief satellite')

    #plt.axhline(y=0, color='black', linewidth=0.5)
    #plt.axvline(x=0, color='black', linewidth=0.5)
    plt.legend()
    plt.xlabel("y [m]", fontsize=24)
    plt.ylabel("x [m]", fontsize=24)
    plt.legend(fontsize=15,loc='upper right')
    plt.tick_params(axis='both', labelsize=17)





#----------------------------- Single Orbite Formation Flight 2D ----------------------------- 

if (Single_Orbit_Formation):
    dephasage, index = 0,1
    oui=0

    num_sat = 28+1  #min:20 / max:32
    print('Number of satellite :',num_sat)
    pos_sat = np.zeros((num_sat,3))

    plt.scatter(pos_sat[0,0], pos_sat[0,1], color='k', label='Chief satellite')

    for i in range(num_sat-1):

        phi0 = 0.0
        x0 = 22
        x0_dot = 0.0
        y0 = 2*x0_dot/w
        y0_dot = -2*w*x0
        z0 = 0.0
        z0_dot = 0.0
        theta0 = 0

        x,y,z = relative_3Dmotion(0,x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0-dephasage),np.radians(theta0),w) 
        pos_sat[index,:] = np.array([x,y,z])
        if oui == 0:
            plt.scatter(pos_sat[index,1], pos_sat[index,0],color='b', label='Deputy satellites')
            oui+=1
        else:
            plt.scatter(pos_sat[index,1], pos_sat[index,0],color='b')
        index+=1
        dephasage+=360/(num_sat-1) #deg 

    x,y,z = relative_3Dmotion(t,x0,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
    plt.plot(y, x,alpha=0.2, color ='blue', label='Deputy satellites orbit')

    plt.xlabel("y [m]", fontsize=24)
    plt.ylabel("x [m]", fontsize=24)
    plt.grid()
    plt.legend(loc='upper right', fontsize=17)
    plt.tick_params(axis='both', labelsize=17)





#------------------------ Concept of Phase shift 2D -------------------------

if (Phase_Shift_2D):

    t = 0.0
    t2 = np.linspace(0, 20000, 10000)

    phi0_list = [0.0, 90.0, 180.0, 270.0]
    x0_list = [100]

    x0 = 100
    x0_dot = 0.0
    y0 = 2 * x0_dot/w
    y0_dot = -2*w*x0
    z0, z0_dot, theta0 = 0,0,0
    x, y, z = relative_3Dmotion(t2, x0, x0_dot, y0, y0_dot, z0, z0_dot, np.radians(0), theta0, w)
    plt.plot(y, x, color='k', alpha=0.3, label=r'$x_0$ = 100 m')

    x, y, z = relative_3Dmotion(t, x0, x0_dot, y0, y0_dot, z0, z0_dot, np.radians(phi0_list[0]), theta0, w)
    plt.scatter(y, x, color='r', label=r'$\phi_o$ = 0°')
    x, y, z = relative_3Dmotion(t, x0, x0_dot, y0, y0_dot, z0, z0_dot, np.radians(phi0_list[1]), theta0, w)
    plt.scatter(y, x, color='g', label=r'$\phi_o$ = 90°')
    x, y, z = relative_3Dmotion(t, x0, x0_dot, y0, y0_dot, z0, z0_dot, np.radians(phi0_list[2]), theta0, w)
    plt.scatter(y, x, color='b', label=r'$\phi_o$ = 180°')
    x, y, z = relative_3Dmotion(t, x0, x0_dot, y0, y0_dot, z0, z0_dot, np.radians(phi0_list[3]), theta0, w)
    plt.scatter(y, x, color='orange', label=r'$\phi_o$ = 270°')



    x0 = 200
    x0_dot = 0.0
    y0 = 2 * x0_dot/w
    y0_dot = -2*w*x0
    z0, z0_dot, theta0 = 0,0,0
    x, y, z = relative_3Dmotion(t2, x0, x0_dot, y0, y0_dot, z0, z0_dot, np.radians(0), theta0, w)
    plt.plot(y, x, color='b', alpha=0.3, label=r'$x_0$ = 200 m')

    dephasage=45
    x, y, z = relative_3Dmotion(t, x0, x0_dot, y0, y0_dot, z0, z0_dot, np.radians(phi0_list[0]+dephasage), theta0, w)
    plt.scatter(y, x, color='c', label=r'$\phi_o = 0° + \Delta \phi_o$')
    x, y, z = relative_3Dmotion(t, x0, x0_dot, y0, y0_dot, z0, z0_dot, np.radians(phi0_list[1]+dephasage), theta0, w)
    plt.scatter(y, x, color='m', label=r'$\phi_o = 90° + \Delta \phi_o$')
    x, y, z = relative_3Dmotion(t, x0, x0_dot, y0, y0_dot, z0, z0_dot, np.radians(phi0_list[2]+dephasage), theta0, w)
    plt.scatter(y, x, color='y', label=r'$\phi_o = 180° + \Delta \phi_o$')
    x, y, z = relative_3Dmotion(t, x0, x0_dot, y0, y0_dot, z0, z0_dot, np.radians(phi0_list[3]+dephasage), theta0, w)
    plt.scatter(y, x, color='indigo', label=r'$\phi_o = 270° + \Delta \phi_o$')


    colors = ['r', 'g', 'b', 'orange']  
    dephasage = 0

    # Plot the target satellite
    plt.scatter(0, 0, color='k', label='Chief satellite')

    plt.xlabel('y [m]', fontsize=24)
    plt.ylabel('x [m]', fontsize=24)

    plt.legend(loc='upper right', fontsize=17, bbox_to_anchor=(1.05, 1))
    plt.tick_params(axis='both', labelsize=17)
    plt.axis('equal')









#--------------------------- X and Y-shaped Formation Flight 2D (y-z plane) ---------------------------

if (Y_Formation_2D):
    t = np.linspace(0,T,3600)
    ti = 0 
    tf = t[3599]

    Y_shaped = 1
    X_shaped = 0

    y0_list = np.linspace(1,12,12)

    if Y_shaped == 1:
        y0_list2 = np.linspace(-1,-10,10)

    #y0_list = list(map(int, y0_list)) #converti float en int
    #y0_list = np.array(y0_list) *5
    print(y0_list)
    x0_dot,x0,y0_dot,z0_dot = 0.0,0.0,0.0,0.0
    z0 = 0.0
    phi0, theta0 = 0,0

    index=1
    count=0

    # Pour la légende des points rouges, oranges et trajectoires
    x,y,z = relative_3Dmotion(ti,x0,x0_dot,1,y0_dot,2,z0_dot,phi0,theta0,w)
    plt.scatter(z, y, color='b', label=r'Deputy satellites positions, $t_0$')
    x,y,z = relative_3Dmotion(tf,x0,x0_dot,1,y0_dot,2,z0_dot,phi0,theta0,w)
    plt.scatter(z, y, color='orange', label=r'Deputy satellites positions, $t_{T/2}$')
    x2,y2,z2 = relative_3Dmotion(t,x0,x0_dot,2,y0_dot,0,z0_dot,phi0,theta0,w)
    plt.plot(z2, y2, color='b', alpha=0.2, label='Motion range')

    for y0 in y0_list:
        if y0>=1 and y0 % 2 == 0:
            z0+=2.

            xi,yi,zi = relative_3Dmotion(ti,x0,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
            plt.scatter(zi, yi, color='b')

            xf,yf,zf = relative_3Dmotion(tf,x0,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
            plt.scatter(-zf,yf, color='orange')

            x2,y2,z2 = relative_3Dmotion(t,x0,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
            plt.plot(z2,y2, color='b', alpha=0.2)

        
        elif y0>=1 and y0 % 2 != 0:
            z0+=2.

            xi,yi,zi = relative_3Dmotion(ti,x0,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
            plt.scatter(-zi, yi, color='b')

            xf,yf,zf = relative_3Dmotion(tf,x0,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
            plt.scatter(zf,yf, color='orange')

            x2,y2,z2 = relative_3Dmotion(t,x0,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
            plt.plot(-z2,y2, color='b', alpha=0.2)

        else: 
            x,y,z = relative_3Dmotion(ti,x0,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
            plt.scatter(z, y, color='r')
        
        index+=1

    if Y_shaped == 1:
        for y0 in y0_list2:
            x0_dot,x0,y0_dot,z0_dot = 0.0,0.0,0.0,0.0
            z0 = 0.0
            phi0, theta0 = 0,0
            x,y,z = relative_3Dmotion(0,x0,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
            plt.scatter(z, y, color='b')



    # Pour former le X
    if X_shaped == 1:
        y0_list2 = np.linspace(-1,-12,12)
        z0 = 0

        for y0 in y0_list2:
            if y0<=-1 and y0 % 2 == 0:
                z0+=2.

                xi,yi,zi = relative_3Dmotion(ti,x0,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
                plt.scatter(zi, yi, color='b')


                xf,yf,zf = relative_3Dmotion(tf,x0,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
                plt.scatter(-zf,yf, color='orange')

                x2,y2,z2 = relative_3Dmotion(t,x0,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
                plt.plot(z2,y2, color='b', alpha=0.2)

            
            elif y0<=-1 and y0 % 2 != 0:
                z0+=2.

                xi,yi,zi = relative_3Dmotion(ti,x0,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
                plt.scatter(-zi, yi, color='b')

                xf,yf,zf = relative_3Dmotion(tf,x0,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
                plt.scatter(zf,yf, color='orange')

                x2,y2,z2 = relative_3Dmotion(t,x0,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
                plt.plot(-z2,y2, color='b', alpha=0.2)

            else: 
                x,y,z = relative_3Dmotion(ti,x0,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
                plt.scatter(z, y, color='b')
            
            index+=1

        num_sat = len(y0_list)+len(y0_list2)+1  #min:20 / max:32
        print('Number of satellite :',num_sat)

    # Plot the target satellite
    plt.scatter(0, 0, color='k', label='Chief satellite')

    # Set plot parameters
    plt.xlabel("z [m]", fontsize=24)
    plt.ylabel("y [m]", fontsize=24)
    plt.grid()
    plt.legend(loc='lower right', fontsize=17)
    plt.tick_params(axis='both', labelsize=17)
    #plt.xlim(-200,250)
    #plt.ylim(-2,30)








#------------------------ Y-shaped Formation Flight 3D -------------------------

if (Y_Formation_3D):

    t = np.linspace(0, T, 3600)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # First satellite
    x0_dot = 0.0
    x0 = 0.0
    y0 = 10.0
    y0_dot = 0.0
    z0 = 0.0
    z0_dot = 0.0
    phi0,theta0 = 0,0

    x,y,z = relative_3Dmotion(t,x0,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
    ax.scatter(y, x, z, marker='.', s=50, color='red', label='First satellite')



    # Second satellite
    x0_dot = 0.0
    x0 = 0.0
    y0 = 20.0
    y0_dot = 0.0
    z0 = 0.0
    z0_dot = 0.0

    x,y,z = relative_3Dmotion(t,x0,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
    ax.scatter(y, x, z, marker='.', s=50, color='red', label='Second satellite')


    # Third satellite
    x0_dot = 0.0
    x0 = 0.0
    y0 = 30.0
    y0_dot = 0.0
    z0 = 100
    z0_dot = 1.0

    x,y,z = relative_3Dmotion(t,x0,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
    ax.plot(y, x, z,label='Third satellite')


    # Fourth satellite
    x0_dot = 0.0
    x0 = 0.0
    y0 = 40.0
    y0_dot = 0.0
    z0 = 100
    z0_dot = 3.0

    x,y,z = relative_3Dmotion(t,x0,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w)
    ax.plot(y, x, z,label='Fourth satellite')


    # add points for starting positions
    ax.scatter(0, 0, 0, marker='.', s=50, color='black', label='Target satellite')

    # set labels and title
    ax.set_xlabel('y [m]')
    ax.set_ylabel('x [m]')
    ax.set_zlabel('z [m]')
    ax.set_title('Motion of the Interceptor in 3D')

    plt.legend()


#plt.legend(loc='upper right')
#plt.savefig("Single Orbit Config.svg")
plt.show()