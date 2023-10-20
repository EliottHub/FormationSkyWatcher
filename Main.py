

"""
##################################################################

Author: Eliott HUBIN
Project: Cubsats’s study in formation for Earth observation.
File: Main.py
Description: Main function to orchestrate the plotting functions.
Date: 30/08/2023

##################################################################
"""





import numpy as np
import matplotlib.pyplot as plt
from Functions import *
import matplotlib.colors as colors




# Control activation of the plot
Absolute_motion = 1
Relative_motion = 0

Optimisation = 0
Pattern = 0
Pattern3D = 0
GroundSpot = 0



# Computation resolution
acc = 0.1
print('Accuracy =',acc)

# Time at wich the following test are performed in the range [0;3600] (x100 for better precision)
Time = 2900*100
print("Time =", int(Time/100))

if Absolute_motion == 1:
    pos_sat,specular_point = pos_sat_with_specular_point(Time)
    specular_cart_coord = specular_point
if Relative_motion == 1:
    pos_sat = pos_sat(Time)
    specular_cart_coord = np.array([0,0,0])


# Computation of the position of the Cubesats formation 
center_array_sats = np.array([pos_sat[0][0],pos_sat[0][1],pos_sat[0][2]]) 
print("center_array_sats =",center_array_sats)






#-------------- Plot Radiation Pattern - 2D ------------
if (Pattern): 
    index_theta = int(90/acc) # for antenna arrays in the x-y plane
    F,D,integral = Radiation_Pattern(pos_sat, center_array_sats, specular_cart_coord, acc)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    p = np.arange(0,360,acc) 
    phi = np.radians(p[:, np.newaxis])
    ax.plot(phi,F[:,index_theta])
    ax.set_xlabel(r'$\varphi$')


#-------------- Plot Radiation Pattern - 3D ------------
if (Pattern3D):
    
    number_sat = 29

    F,D,integral = Radiation_Pattern(pos_sat, center_array_sats, specular_cart_coord, acc)
    p = np.arange(0,360,acc) 
    phi = np.radians(p[:, np.newaxis])  

    thets = np.arange(0, 180, acc) # elevation
    theta = np.radians(thets)
    #theta = np.concatenate([theta, np.flip(theta)])

    x = F * (np.sin(phi) * np.cos(theta))
    y = F * (np.sin(phi) * np.sin(theta))
    z = F * (np.cos(phi))

    # Plot the 3D radiation pattern
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x, y, z, facecolors=plt.cm.jet(F))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()




#------------------ Plot Ground Spot -------------------
if (GroundSpot) and (Absolute_motion):

    width = 10000
    G,F = ground_spot(specular_cart_coord, center_array_sats, pos_sat, width)
    F = 10*np.log10(F) 

    x = np.linspace(-(width+1)/2, (width+1)/2, int(width/10) + 1)
    y = np.linspace(-(width+1)/2, (width+1)/2, int(width/10) + 1)
    X, Y = np.meshgrid(x, y)

    # Define the desired color intervals
    color_intervals = [-39, -36, -33, -30, -27, -24, -21, -18, -15, -12, -9, -6, -3, 0]

    # Create a custom colormap with the desired color intervals
    color_sequence = ['#440154', '#48186a', '#472d7b', '#424086', '#3b528b', '#33638d',
                      '#2c728e', '#26828e', '#21918c', '#1fa088', '#28ae80', '#3fbc73',
                      '#5ec962', '#84d44b', '#addc30', '#d8e219', '#fde725','#ff0000']

    colormap = colors.LinearSegmentedColormap.from_list('custom_colormap', color_sequence, N=20 )

    plt.figure()
    plt.imshow(F, cmap=colormap, vmin=-39, vmax=0, extent=[x.min(), x.max(), y.min(), y.max()])

    colorbar = plt.colorbar(ticks=color_intervals)
    colorbar.set_label("Intensity [dB]")

    plt.axis("square")
    plt.grid()
    
    plt.xlabel("[m]")
    plt.ylabel("[m]")
    #plt.savefig("GNSS t=3330 z0=6x0.svg")




#----------------- Plot Partial Radiation Pattern ----------------
if (Relative_motion):
    plt.figure()

    # Accuracy for the partial radiation pattern
    acc = 0.01 # [deg] 

    # Computation of the glistening zone
    width = 10 #[km] width 
    alpha = glistening_angle(width)

    # Radiation pattern with angular notation
    min = 170
    max = 190
    p = np.arange(min,max+acc,acc)
    F = radiation_pattern_partial(pos_sat,p,acc)

    # Change the x-axis into kilometers
    pp = 400*np.tan(np.deg2rad(p-180))
    plt.plot(pp,10*np.log10(F[:,0]),'b')

    x_ticks = [-15, -10, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 10, 15]
    plt.xticks(x_ticks)

    plt.axvline(x=-5,color="black",linestyle='--',label="Glistening zone")
    plt.axvline(x=5,color="black",linestyle='--')
    plt.ylim(-15,0)
    plt.xlim(-15,15)
    plt.grid()
    plt.xlabel("Distance from the specular point [km]", fontsize=14)
    plt.ylabel("Intensity [dB]", fontsize=14)
    plt.legend(fontsize=12)

    #plt.savefig("Radiation Pattern.svg")




#-------------- Compute Resolution ----------------
if (Relative_motion):
    try:
        res = compute_resolution_partial(F,acc,min,max)
        print("The resolution is equal to {:.2f} km".format(res))
    except:
        print("Resolution Computation failed")




plt.show()


