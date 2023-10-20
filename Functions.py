

"""
##################################################################

Author: Eliott HUBIN
Project: Cubsats’s study in formation for Earth observation.
File: Functions.py
Description: Lists all the functions used throughout the project.
Date: 30/08/2023

##################################################################
"""





import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from numba import jit
from PIL import Image


# Compute angular rate of the chief satellite
def angular_rate(alt_sat):

    """
    Calculates the angular rate (angular velocity) of a satellite in orbit.

    Parameters:
    - alt_sat (float): Altitude of the satellite above the Earth's surface (in meters).

    Returns:
    - float: Angular rate in radians per second.
    """
        
    mu = 3.986e14 #m3/s2
    #mu = 398600.5 #km3/s2
    r_tgt = 6371000 + alt_sat #km
    w = np.sqrt(mu/r_tgt**3) #rad/s
    return w



def relative_3Dmotion(t,x0,x0_dot,y0,y0_dot,z0,z0_dot,phi0,theta0,w):

    """
    Calculates the relative position of a satellite in orbit based on CW equations.

    Parameters:
    - t (float): Time parameter.
    - x0, x0_dot, y0, y0_dot, z0, z0_dot (float): Initial position and velocity components.
    - phi0, theta0 (float): Initial angles.
    - w (float): Angular frequency.

    Returns:
    - Tuple (x, y, z): Relative coordinates of the satellite in motion.
    """

    y_c0 = y0 - (2*x0_dot / w)
    y_c_dot = -6*w*x0 - 3*y0_dot
    x_c = -2*y_c_dot / (3*w)
    z_c = z0

    C = np.sqrt( (3*x0+2*y0_dot/w)**2 + (x0_dot/w)**2 )
    D = np.sqrt((z0_dot/w)**2 + z0**2)
    
    x = x_c  + C*np.sin(w*t + phi0)
    y = y_c0 + y_c_dot*t + 2*C*np.cos(w*t + phi0) 
    z = D*np.cos(w*t - theta0)

    return x,y,z 



def glistening_angle(x):
    
    """
    Calculates the opening (glistening) angle of the glistening zone (in degrees) based on the its width.

    Parameters:
    - x (float): Width parameter in kilometers.

    Returns:
    - float: Glistening zone's opening angle in degrees.
    """
    
    alpha = 0.5*np.arctan(x/400)
    return np.rad2deg(alpha)




def compute_resolution_partial(F,acc,phi_min,phi_max): 

    """
    Computes the resolution of the main beam.

    Parameters:
    - F (numpy array): Radiation pattern.
    - acc (float): Accuracy parameter of the calculation.
    - phi_min (float): Minimum azimuthal angle.
    - phi_max (float): Maximum azimuthal angle.

    Returns:
    - float: Computed resolution in kilometers.
    """
        
    dx = 0.0001
    F = 10*np.log10(F)
    x = np.linspace(phi_min,phi_max,int((F.shape[0]-1)/dx)+1)
    xp = np.arange(phi_min,phi_max+acc,acc)
    y = np.interp(x,xp,F[: ,0])
    idx = int(((phi_max - phi_min)/2)/(acc))
    yf = np.argwhere((y[:] < (F[idx ,0] -3)) & (y[:] > (F[idx ,0] -3.001) ) )
    yf1 = np.extract(yf>idx/dx,yf)
    angle = x[yf1[0]]-180
    #x = 2*400000*np.tan(np.deg2rad(angle))
    x = 2*400000*np.sin(np.deg2rad(angle))
    return x/1000



def geo_to_cart_coord(longitude, latitude, altitude):

    """
    Transforms geocentric coordinates into Cartesian coordinates.

    Parameters:
    - longitude (float): Longitude in degrees.
    - latitude (float): Latitude in degrees.
    - altitude (float): Altitude in meters.

    Returns:
    - numpy array: Cartesian coordinates [x, y, z].
    """

    longitude = np.radians(longitude)
    latitude = np.radians(latitude)

    # Transform into Cartesian coordinates
    x = altitude * np.cos(latitude) * np.cos(longitude)
    y = altitude * np.cos(latitude) * np.sin(longitude)
    z = altitude * np.sin(latitude)

    specular_cart_coord = np.array([x,y,z])

    return specular_cart_coord



@jit(nopython=True)
def Radiation_Pattern(pos_sat, sat_center_coord, specular_coord, acc):

    """
    Computes the radiation pattern and directivity of a satellite formation.

    Parameters:
    - pos_sat (numpy array): Array of satellite positions.
    - sat_center_coord (numpy array): Coordinates of the central (chief) satellite.
    - specular_coord (numpy array): Coordinates of the specular point.
    - acc (float): Accuracy parameter.

    Returns:
    - F (numpy array): Radiation pattern.
    - D (numpy array): Directivity.
    - integral (float): Integral value used in directivity calculation.
    """

    phi = int(360/acc)
    theta = int(180/acc)
    lambdaa = 0.2 #m
    num_sat = pos_sat.shape[0]
    k = 2*np.pi/lambdaa
    u = np.zeros((phi,theta,3)) 
    F = np.zeros((phi,theta),dtype=np.complex128) 

    #u0 = (specular_coord-sat_center_coord)/np.linalg.norm(specular_coord-sat_center_coord)

    for i in range(phi): #loop over the phi
        for l in range(theta): #loop over the theta theta
            u[i,l,:] = [np.sin(np.radians(i*acc))*np.cos(np.radians(l*acc)),np.sin(np.radians(i*acc))*np.sin(np.radians(l*acc)),np.cos(np.radians(i*acc))] #unit (rho=1) director vector u in cartesian coord.
        
            for j in range(num_sat): #loop over the number of satellite 
                rho = pos_sat[j,:]-sat_center_coord
                u[i,l,:] /= np.linalg.norm(u[i,l,:])
                r = 1 #np.linalg.norm(specular_coord-sat_center_coord)
                F[i,l] = F[i,l] + (1/r)*np.exp(1j*k*np.dot(u[i,l,:],rho))#*np.exp(-1j*k*np.dot(u0,rho))  #(1/r)*np.exp(1j*k*np.dot((rounded_u-rounded_u0),rho)) #(additional phase shift)  

    F = np.abs(F)
    F = F/np.amax(F) #normalize values in array F

    # Directivity 
    t = np.deg2rad(np.arange(0,180,acc))
    p = np.deg2rad(np.arange(0,360,acc))
    sin_theta = np.sin(t)
    integral = 0
    for i in range(theta):
        for j in range(phi):
            integral = integral + (np.pi/theta)*(2*np.pi/phi) * F[j,i]**2 * sin_theta[i]
    D = (4*np.pi/integral)*(F**2)

    return F,D,integral


        

def radiation_pattern_partial(pos_sat,p,acc):

    """
    Computes the radiation pattern for a partial region (used for higher "acc").

    Parameters:
    - pos_sat (numpy array): Array of satellite positions.
    - p (numpy array): Array of azimuth angles in degrees.
    - acc (float): Accuracy parameter.

    Returns:
    - F (numpy array): Radiation pattern for the partial region.
    """

    lambd = 0.2 #m
    k = 2*np.pi/lambd
    m = int(p.shape[0])
    u = np.zeros((m,1,3))
    F = np.zeros((m,1),dtype="complex")
    for i in range(m): #boucle sur les phi
        phi = p[i]
        u[i,0,:] = [0,np.sin(phi*np.pi/180),np.cos(phi*np.pi/180)] #unit director vector u for theta=90°
        #u[i,0,:] = [np.sin(np.radians(phi)) , 0 , np.cos(np.radians(phi))] #unit director vector u for theta=0°
        for j in range(pos_sat.shape[0]):
            F[i,0] = F[i,0] + np.exp(1j*k*np.dot(u[i,0,:],pos_sat[j,:])) 
    F = np.abs(F)
    F = F/np.amax(F)
    
    return F



@jit(nopython=True)
def ground_spot(specular_coord, sat_center_coord, pos_sat, width):

    """
    Computes the ground spot illumination pattern.

    Parameters:
    - specular_coord (numpy array): Coordinates of the specular point.
    - sat_center_coord (numpy array): Coordinates of the central (chief) satellite.
    - pos_sat (numpy array): Array of satellite positions.
    - width (float): Width of the ground spot in meters.

    Returns:
    - G (float): Ground spot normalization factor.
    - F (numpy array): Ground spot illumination pattern.
    """
   
    F = np.zeros((int(width/10)+1,int(width/10)+1),dtype="complex") 
    lambdaa = 0.2
    k = 2*np.pi/lambdaa
    
    u = np.zeros((int(width/10)+1,int(width/10)+1,3))
    u0 = (specular_coord - sat_center_coord)/np.linalg.norm(specular_coord - sat_center_coord) 
    ground_coord = np.zeros((int(width/10)+1,int(width/10)+1,3))

    num_sat = pos_sat.shape[0]
    x_grid = int(width/10) + 1
    y_grid = int(width/10) + 1
    
    for i in range(x_grid):
        for j in range(y_grid):
            ground_coord[i,j,0] = specular_coord[0] + 10*(i - int(width/20))
            ground_coord[i,j,1] = specular_coord[1] + 10*(j - int(width/20))
            ground_coord[i,j,2] = specular_coord[2] + 0
            
            u[i,j,0] = ground_coord[i,j,0] - sat_center_coord[0]
            u[i,j,1] = ground_coord[i,j,1] - sat_center_coord[1]
            u[i,j,2] = ground_coord[i,j,2] - sat_center_coord[2]

            u[i,j,:] /= np.linalg.norm(u[i,j,:])
            
            for n in range(num_sat):
                rho = pos_sat[n,:] - sat_center_coord
                r = 1 #np.linalg.norm(specular_coord-sat_center_coord)
                F[i,j] += (1/r)*np.exp(1j*k*np.dot((u[i,j,:]-u0),rho))
                
    F = np.abs(F)
    F = F/np.amax(F)

    G = 1
    
    return G,F




def circular_orbit(a, w, inclination, raan, argp, t):

    """
    Computes the position and velocity vectors of a satellite in a circular orbit.

    Parameters:
    - a (float): Semi-major axis of the orbit.
    - w (float): Angular velocity of the orbit.
    - inclination (float): Inclination angle of the orbit in degrees.
    - raan (float): Right Ascension of Ascending Node (RAAN) in degrees.
    - argp (float): Argument of Perigee in degrees.
    - t (numpy array): Time array.

    Returns:
    - orbit (numpy array): Position vector of the satellite in the Earth-centered inertial (ECI) frame.
    - velocity (numpy array): Velocity vector of the satellite in the ECI frame.
    """

    inclination = np.deg2rad(inclination)
    raan = np.deg2rad(raan)  # right ascension of the ascending node
    argp = np.deg2rad(argp)  # argument of perigee

    x = a * np.cos(w*t)
    y = a * np.sin(w*t)
    z = np.zeros_like(t)

    # Rotation about the Z axis: aligns the orbit plane with the equatorial plane
    R3_W = np.array([[np.cos(raan), np.sin(raan), 0], [-np.sin(raan), np.cos(raan), 0], [0, 0, 1]])
    # Rotation around the X-axis: tilts the orbit plane by the given tilt angle
    R1_i = np.array([[1, 0, 0], [0, np.cos(inclination), np.sin(inclination)], [0, -np.sin(inclination), np.cos(inclination)]])
    # Z-axis rotation: rotates the orbit around the Z-axis according to the argument of the periapsis and the ascending node
    R3_w = np.array([[np.cos(argp), np.sin(argp), 0], [-np.sin(argp), np.cos(argp), 0], [0, 0, 1]])

    R = R3_W.dot(R1_i).dot(R3_w)
    orbit = np.vstack((x, y, z))
    velocity = np.vstack((-a*w*np.sin(w*t), a*w*np.cos(w*t), np.zeros_like(t)))

    return R.dot(orbit), R.dot(velocity)



def sat_visibility(center, radius, pos_Cubesat, pos_GNSS):

    """
    Calculates the intersection points between a line segment (Cubesat to GNSS satellite) and a sphere (Earth).
    (Used to compute if the Cubesat and the GNSS satellite are in light-of-sight.)

    Parameters:
    - center (numpy array): Coordinates of the center of the Earth.
    - radius (float): Radius of the Earth.
    - pos_Cubesat (numpy array): Position vector of the CubeSat.
    - pos_GNSS (numpy array): Position vector of the GNSS satellite.

    Returns:
    - intersections (list of numpy arrays): List of intersection points between the line segment and the Earth's surface.
    """

    
    # Calculate the direction vector of the line segment
    vec_Cubesat_to_GNSS = pos_GNSS - pos_Cubesat
    
    # Calculate the vector from the segment's first point to the sphere center
    oc = pos_Cubesat - center
    
    # Calculate the coefficients of the quadratic equation
    a = np.dot(vec_Cubesat_to_GNSS, vec_Cubesat_to_GNSS)
    b = 2 * np.dot(vec_Cubesat_to_GNSS, oc)
    c = np.dot(oc, oc) - radius**2
    
    discriminant = b**2 - 4*a*c
    
    # No intersection
    if discriminant < 0:
        return []
    
    # Calculate the intersection points
    t1 = (-b + np.sqrt(discriminant)) / (2*a)
    t2 = (-b - np.sqrt(discriminant)) / (2*a)
    
    # Check if the intersection points are within the segment bounds
    intersections = []
    if 0 <= t1 <= 1:
        intersection1 = pos_Cubesat + t1 * vec_Cubesat_to_GNSS
        intersections.append(intersection1)
    if 0 <= t2 <= 1:
        intersection2 = pos_Cubesat + t2 * vec_Cubesat_to_GNSS
        intersections.append(intersection2)
    
    return intersections




def minimize_distance_on_sphere(pos_Cubesat, pos_GNSS, center, radius):

    """
    Minimizes the sum of distances from a point on the sphere to both the CubeSat and GNSS satellite.
    Used to compute the location of the specular point on the surface of a spherical Earth.

    Parameters:
    - pos_Cubesat (numpy array): Position vector of the CubeSat.
    - pos_GNSS (numpy array): Position vector of the GNSS satellite.
    - center (numpy array): Coordinates of the center of the Earth.
    - radius (float): Radius of the Earth.

    Returns:
    - result (OptimizeResult object): Result of the optimization, containing information about the optimized point.
    """

    
    scale_factor = np.linalg.norm(center) + radius
    pos_Cubesat_normalized = pos_Cubesat / scale_factor
    pos_GNSS_normalized = pos_GNSS / scale_factor
    center_normalized = center / scale_factor
    radius_normalized = radius / scale_factor
    
    # Objective function for optimization
    def objective_function(point):
        d1 = np.linalg.norm(point - pos_Cubesat_normalized)
        d2 = np.linalg.norm(point - pos_GNSS_normalized)
        return d1 + d2
    
    # Constraint: the distance between the point and the sphere must be equal to the radius
    constraint = {'type': 'eq', 'fun': lambda point: np.linalg.norm(point - center_normalized) - radius_normalized}

    # Initial point for optimization
    initial_point = np.array([0.0, 0.0, 1.0])
    
    result = minimize(objective_function, initial_point, constraints=constraint)
    
    return result



def compare_angles(d1, d2, normal):

    """
    Determines if the incident and reflection angles are the same, providing evidence that the specular point computation is accurate.

    Parameters:
    - d1 (numpy array): Incident direction vector.
    - d2 (numpy array): Reflection direction vector.
    - normal (numpy array): Surface normal vector.

    Returns:
    - result (bool): True if incident and reflection angles are very close; False otherwise.
    """

    d1_normalized = d1 / np.linalg.norm(d1)
    d2_normalized = d2 / np.linalg.norm(d2)
    
    # Calculate the angles of incidence and reflection (for 3 decimals)
    angle_incidence = round(np.arccos(np.dot(d2_normalized, normal)),3)
    angle_reflection = round(np.arccos(np.dot(d1_normalized, normal)),3)
    
    # Compare the angles
    return np.isclose(angle_incidence, angle_reflection)




def pos_sat_with_specular_point(Time):

    """
    Compute the ABSOLUTE positions of the deputy satellites in a formation (for different configurations) and 
    the specular point on the Earth's surface for a given time.

    Parameters:
    - Time (int): The time instant for which to compute the positions.

    Returns:
    - pos_sat (numpy.ndarray): Array containing the positions of deputy satellites.
    - specular_point (numpy.ndarray): Coordinates of the specular point on the Earth's surface.
    """


    # Small on-board computer for easy management of flight formation type activation

    X_shape = 0
    Spiral  = 1



    """
    ####################################################
    ########## Absolute X_shape configuration ##########
    ####################################################
    """

    if X_shape == 1:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Define the ciruclar chief orbit of the Cubesat
        Re = 6371000
        alt_Cubesat = 400000 #km
        a_Cubesat = Re + alt_Cubesat  # semi-major axis in km
        inclination = 90 # determines how tilted the orbit is relative to the Earth's equator (deg)
        raan = 0
        argp = 0
        w_Cubesat = angular_rate(alt_Cubesat) #rad/s
        num_points = 360000
        T = 2*np.pi/w_Cubesat #period
        t_Cubesat = np.linspace(0, T, num_points)
        circular_orbit_Cubesat = circular_orbit(a_Cubesat, w_Cubesat, inclination, raan, argp, t_Cubesat)

        # Define the ciruclar orbit of the GNSS
        alt_GNSS = 23222000 #m (Galileo)
        a_GNSS = Re + alt_GNSS
        inclination_GNSS = 56 
        raan_GNSS = 0
        argp_GNSS = 0
        w_GNSS = angular_rate(alt_GNSS)
        T = 2*np.pi/w_GNSS
        t_GNSS = np.linspace(0, T, num_points)
        circular_orbit_GNSS = circular_orbit(a_GNSS, w_GNSS, inclination_GNSS, raan_GNSS, argp_GNSS, t_GNSS)

        # Absolute position of the deputy satellites
        index=1

        y0_list = np.linspace(5,75,20)
        y0_list2 = np.linspace(-10,-100,10)

        num_sat = len(y0_list)+len(y0_list2)+1  #min:20 / max:32
        print('Number of satellite :',num_sat)
        pos_sat = np.zeros((num_sat,3))

        pos_sat[0][0] = circular_orbit_Cubesat[0][0][Time]
        pos_sat[0][1] = circular_orbit_Cubesat[0][1][Time]
        pos_sat[0][2] = circular_orbit_Cubesat[0][2][Time]

        x0_dot,x0,y0_dot,z0_dot = 0.0,0.0,0.0,0.0
        z0 = 0.0
        phi0 = 0.0
        theta0 = 0
        count= 0

        z0_both = 3.5

        for y0 in y0_list:
            if y0>=1:
                z0+=z0_both
                x_rel, y_rel, z_rel = relative_3Dmotion(t_Cubesat[Time],x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0),np.radians(theta0),w_Cubesat)
            
            else: 
                x_rel, y_rel, z_rel = relative_3Dmotion(t_Cubesat[Time],x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0),np.radians(theta0),w_Cubesat)
            
            # Position of the chief satellite
            pos1 = np.array([circular_orbit_Cubesat[0][0][Time], circular_orbit_Cubesat[0][1][Time], circular_orbit_Cubesat[0][2][Time]])
            vit1 = np.array([circular_orbit_Cubesat[1][0][Time], circular_orbit_Cubesat[1][1][Time], circular_orbit_Cubesat[1][2][Time]])

            # Vector pointing from the centre of the Earth to the first satellite
            earth_to_sat1 = pos1/np.linalg.norm(pos1)
            normal_dir = np.cross(earth_to_sat1, vit1)
            tangential_dir = np.cross(pos1, normal_dir)

            # Vector normalization
            tangential_dir /= np.linalg.norm(tangential_dir)
            normal_dir /= np.linalg.norm(normal_dir)

            # Rotation matrix
            R = np.array([earth_to_sat1, tangential_dir, normal_dir]).T

            pos_rel = np.vstack((x_rel, y_rel, z_rel))
            x_rel = R.dot(pos_rel)[0]
            y_rel = R.dot(pos_rel)[1]
            z_rel = R.dot(pos_rel)[2]
            
            x = x_rel + circular_orbit_Cubesat[0][0][Time]
            y = y_rel + circular_orbit_Cubesat[0][1][Time]
            z = z_rel + circular_orbit_Cubesat[0][2][Time]

            if y0 >= -1:
                if count % 2 != 0:
                    pos_sat[index,:] = np.array([x[0], -y[0], z[0]])
                    ax.scatter(pos_sat[index][0], pos_sat[index][1], pos_sat[index][2], marker='.')
                    index+=1
                else:
                    pos_sat[index,:] = np.array([x[0], y[0], z[0]])
                    ax.scatter(pos_sat[index][0], pos_sat[index][1], pos_sat[index][2], marker='.')
                    index+=1
                count += 1
            else:
                pos_sat[index,:] = np.array([x[0], y[0], z[0]])
                ax.scatter(pos_sat[index][0], pos_sat[index][1], pos_sat[index][2], marker='.')
                index+=1

        z0 = 0
        count=0

        for y0 in y0_list2:
            z0+=0 
            x_rel,y_rel,z_rel = relative_3Dmotion(t_Cubesat[Time],x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0),np.radians(theta0),w_Cubesat) 

            pos1 = np.array([circular_orbit_Cubesat[0][0][Time], circular_orbit_Cubesat[0][1][Time], circular_orbit_Cubesat[0][2][Time]])
            vit1 = np.array([circular_orbit_Cubesat[1][0][Time], circular_orbit_Cubesat[1][1][Time], circular_orbit_Cubesat[1][2][Time]])
            earth_to_sat1 = pos1/np.linalg.norm(pos1)
            normal_dir = np.cross(earth_to_sat1, vit1)
            tangential_dir = np.cross(pos1, normal_dir)
            tangential_dir /= np.linalg.norm(tangential_dir)
            normal_dir /= np.linalg.norm(normal_dir)
            R = np.array([earth_to_sat1, tangential_dir, normal_dir]).T

            pos_rel = np.vstack((x_rel, y_rel, z_rel))
            x_rel = R.dot(pos_rel)[0]
            y_rel = R.dot(pos_rel)[1]
            z_rel = R.dot(pos_rel)[2]
            
            x = x_rel + circular_orbit_Cubesat[0][0][Time]
            y = y_rel + circular_orbit_Cubesat[0][1][Time]
            z = z_rel + circular_orbit_Cubesat[0][2][Time]

            if y0 <= -1:
                if count % 2 != 0:
                    pos_sat[index,:] = np.array([x[0], -y[0], z[0]])
                    ax.scatter(pos_sat[index][0], pos_sat[index][1], pos_sat[index][2], marker='.')
                    index+=1
                else:
                    pos_sat[index,:] = np.array([x[0], y[0], z[0]])
                    ax.scatter(pos_sat[index][0], pos_sat[index][1], pos_sat[index][2], marker='.')
                    index+=1
                count += 1
            else:
                pos_sat[index,:] = np.array([x[0], y[0], z[0]])
                ax.scatter(pos_sat[index][0], pos_sat[index][1], pos_sat[index][2], marker='.')
                index+=1

        #---------------------
        center = np.array([0,0,0])

        # Plot the orbits
        pos_Cubesat = np.array([circular_orbit_Cubesat[0][0][Time],circular_orbit_Cubesat[0][1][Time],circular_orbit_Cubesat[0][2][Time]])
        ax.scatter(pos_Cubesat[0], pos_Cubesat[1], pos_Cubesat[2], color='k',label='Chief Satellite')
        #ax.plot(circular_orbit_Cubesat[0][0], circular_orbit_Cubesat[0][1],circular_orbit_Cubesat[0][2], color='green',label='Absolute Chief Orbit')

        pos_GNSS = np.array([circular_orbit_GNSS[0][0][Time],circular_orbit_GNSS[0][1][Time],circular_orbit_GNSS[0][2][Time]])
        print("pos_GNSS =",pos_GNSS)
        #ax.scatter(pos_GNSS[0], pos_GNSS[1], pos_GNSS[2], color='m',label='GNSS')
        #ax.plot(circular_orbit_GNSS[0][0], circular_orbit_GNSS[0][1],circular_orbit_GNSS[0][2], color='orange',label='GNSS Orbit')


        print("---------------------------------------------------------------")
        # ---------------------Compute visibility-------------------------
        visibility = 0
        intersections = sat_visibility(center, Re, pos_Cubesat, pos_GNSS)
        if intersections:
            print("Intersection points:")
        else:
            print("No intersections, Cubesat and GNSS are in line-of-sight")
            visibility = 1

        #ax.plot([pos_Cubesat[0], pos_GNSS[0]], [pos_Cubesat[1], pos_GNSS[1]], [pos_Cubesat[2], pos_GNSS[2]], linestyle='dashed', color='orange')

        # ---------------------Compute Minimisation---------------------
        if visibility == 1:
            result = minimize_distance_on_sphere(pos_Cubesat, pos_GNSS, center, Re)
            scale_factor = np.linalg.norm(center) + Re
            specular_point = result.x * scale_factor

            #ax.scatter(specular_point[0], specular_point[1], specular_point[2], color='r', label='Specular Point')
            print("Specular Point :", specular_point)
        print("---------------------------------------------------------------")

        """
        # Create a mesh grid for the sphere
        u, v = np.mgrid[0:2 * np.pi:200j, 0:np.pi:100j]
        x = center[0] + Re * np.cos(u) * np.sin(v)
        y = center[1] + Re * np.sin(u) * np.sin(v)
        z = center[2] + Re * np.cos(v)
        ax.plot_surface(x, y, z, color='blue',alpha=0.2)
        """

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')

        # Limit for t=2700 view of the whole system zoomed
        # ax.set_xlim(-80-6.3626e6,10-6.3626e6)
        # ax.set_ylim(-45,45)
        # ax.set_zlim(840+2.315e6,930+2.315e6)

        # Limit for t = 2700
        # ax.set_xlim(10,160)
        # ax.set_ylim(-80,80)
        # ax.set_zlim(-60+6.771e6,60+6.771e6)

        #ax.set_xlim(-15000000,15000000)
        #ax.set_ylim(-15000000,15000000)
        #ax.set_zlim(-13500000,13500000)

        plt.legend()





    """
    ###################################################
    ########## Absolute spiral configuration ##########
    ###################################################
    """

    if Spiral == 1:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Define the ciruclar chief orbit of the Cubesat
        Re = 6371000
        alt_Cubesat = 400000 #km
        a_Cubesat = Re + alt_Cubesat  # semi-major axis in km
        inclination = 90 # determines how tilted the orbit is relative to the Earth's equator (deg)
        raan = 0
        argp = 0
        w_Cubesat = angular_rate(alt_Cubesat) #rad/s
        num_points = 360000
        T = 2*np.pi/w_Cubesat #period
        t_Cubesat = np.linspace(0, T, num_points)
        circular_orbit_Cubesat = circular_orbit(a_Cubesat, w_Cubesat, inclination, raan, argp, t_Cubesat)

        # Define the ciruclar orbit of the GNSS
        alt_GNSS = 23222000 #m (Galileo)
        a_GNSS = Re + alt_GNSS
        inclination_GNSS = 56 #90
        raan_GNSS = 0 #90
        argp_GNSS = 0
        w_GNSS = angular_rate(alt_GNSS)
        T = 2*np.pi/w_GNSS
        t_GNSS = np.linspace(0, T, num_points)
        circular_orbit_GNSS = circular_orbit(a_GNSS, w_GNSS, inclination_GNSS, raan_GNSS, argp_GNSS, t_GNSS)

        # Absolute position of the deputy satellites
        x0_list = [5,11,17,22,24,30,35] # Top 4: #phase shift=220
        # Top 2: x0_list = [5,9,13,16,19,22,24,26,28,30,31,32,33,34,35] #dephasage=150  
        # Top 5: x0_list = [3, 11, 13, 21, 22, 35]  #dephase = 210
        # Top 3: x0_list = [5,10,14,18,19,23,29,30,33,35] #dephasage=230
        
        # Increase the size of the satellite positions 
        for i in range(len(x0_list)):
            x0_list[i] *= 2000

        phi0_list = [0,90,180,270] #[0,180] #[0,72,144,216,288]
        dephasage = 0
        index,oui = 1,0

        num_sat = len(phi0_list)*len(x0_list)+1  #min:20 / max:32
        print('Number of satellite :',num_sat)
        pos_sat = np.zeros((num_sat,3))

        pos_sat[0][0] = circular_orbit_Cubesat[0][0][Time]
        pos_sat[0][1] = circular_orbit_Cubesat[0][1][Time]
        pos_sat[0][2] = circular_orbit_Cubesat[0][2][Time]

        colors = ['r', 'm', 'g', 'b', 'r', 'y']

        for x0 in x0_list:
            for phi0, color in zip(phi0_list, colors):

                x0_dot = 0.0
                y0 = 2*x0_dot/w_Cubesat
                y0_dot = -2*w_Cubesat*x0
                z0 = x0*2 # variable
                z0_dot = 0.0
                theta0 = -(90+phi0+dephasage)

                x_rel, y_rel, z_rel = relative_3Dmotion(t_Cubesat[Time],x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0+dephasage),np.radians(theta0),w_Cubesat)

                # Position of the chief satellite
                pos1 = np.array([circular_orbit_Cubesat[0][0][Time], circular_orbit_Cubesat[0][1][Time], circular_orbit_Cubesat[0][2][Time]])
                vit1 = np.array([circular_orbit_Cubesat[1][0][Time], circular_orbit_Cubesat[1][1][Time], circular_orbit_Cubesat[1][2][Time]])

                # Vector pointing from the centre of the Earth to the first satellite
                earth_to_sat1 = pos1/np.linalg.norm(pos1)
                normal_dir = np.cross(earth_to_sat1, vit1)
                tangential_dir = np.cross(pos1, normal_dir)

                # Vector normalization
                tangential_dir /= np.linalg.norm(tangential_dir)
                normal_dir /= np.linalg.norm(normal_dir)

                # Rotation matrix
                R = np.array([earth_to_sat1, tangential_dir, normal_dir]).T

                pos_rel = np.vstack((x_rel, y_rel, z_rel))
                x_rel = R.dot(pos_rel)[0]
                y_rel = R.dot(pos_rel)[1]
                z_rel = R.dot(pos_rel)[2]

                x = x_rel + circular_orbit_Cubesat[0][0][Time]
                y = y_rel + circular_orbit_Cubesat[0][1][Time]
                z = z_rel + circular_orbit_Cubesat[0][2][Time]
            
                pos_sat[index,:] = np.array([x[0], y[0], z[0]])
                if oui == 0:
                    ax.scatter(pos_sat[index][0], pos_sat[index][1], pos_sat[index][2], color='b', marker='.',label='Deputy Satellites')
                    oui+=1
                else:
                    ax.scatter(pos_sat[index][0], pos_sat[index][1], pos_sat[index][2], color='b', marker='.')
                index+=1

            dephasage += 220 / len(x0_list)
        

        #---------------------
        center = np.array([0,0,0])

        # Plot the orbits
        pos_Cubesat = np.array([circular_orbit_Cubesat[0][0][Time],circular_orbit_Cubesat[0][1][Time],circular_orbit_Cubesat[0][2][Time]])
        ax.scatter(pos_Cubesat[0], pos_Cubesat[1], pos_Cubesat[2], color='k',label='Chief Satellite')
        ax.plot(circular_orbit_Cubesat[0][0], circular_orbit_Cubesat[0][1],circular_orbit_Cubesat[0][2], color='green',label='Absolute Chief Orbit')

        Time_GNSS = 2950*100     #Time #Oral : 2400-2950-3250
        pos_GNSS = np.array([circular_orbit_GNSS[0][0][Time_GNSS],circular_orbit_GNSS[0][1][Time_GNSS],circular_orbit_GNSS[0][2][Time_GNSS]])
        #ax.scatter(pos_GNSS[0], pos_GNSS[1], pos_GNSS[2], color='m', label='GNSS$_{t_1}$')
        #ax.plot(circular_orbit_GNSS[0][0], circular_orbit_GNSS[0][1],circular_orbit_GNSS[0][2], color='orange',label='GNSS Orbit')


        print("---------------------------------------------------------------")
        # ---------------------Compute visibility-------------------------
        visibility = 0
        intersections = sat_visibility(center, Re, pos_Cubesat, pos_GNSS)
        if intersections:
            specular_point = np.array([0,0,0])
            print("Intersection points:")
            for intersection in intersections:
                print(intersection)
                ax.scatter(intersection[0], intersection[1], intersection[2], color='green', label='Intersection Point')
        else:
            print("No intersections, Cubesat and GNSS are in line-of-sight")
            visibility = 1


        # ---------------------Compute Minimisation---------------------
        if visibility == 1:
            result = minimize_distance_on_sphere(pos_Cubesat, pos_GNSS, center, Re)
            scale_factor = np.linalg.norm(center) + Re
            specular_point = result.x * scale_factor

            ax.scatter(specular_point[0], specular_point[1], specular_point[2], color='r', label='Specular Point')
            ax.scatter(pos_GNSS[0], pos_GNSS[1], pos_GNSS[2], color='m', label='GNSS$_{t_1}$')
            print("Specular Point :", specular_point)
            #ax.plot([pos_Cubesat[0], specular_point[0]], [pos_Cubesat[1], specular_point[1]], [pos_Cubesat[2], specular_point[2]], linestyle='dashed', color='orange',label='Cubesats -> Specular Point')
            ax.plot([pos_GNSS[0], specular_point[0]], [pos_GNSS[1], specular_point[1]], [pos_GNSS[2], specular_point[2]], linestyle='dashed', color='orangered',label='GNSS -> Specular Point')
            

            # Compute the vector representing the line
            line_vector = specular_point - pos_Cubesat
            inclination_angle = np.arccos(np.dot(line_vector, [0, 0, 1]) / np.linalg.norm(line_vector))
            inclination_angle_deg = np.degrees(inclination_angle)
            #print("Inclination Angle:", inclination_angle_deg)


        #------------------------------------------------------------------------------------------
        if visibility == 1:
            # 1. Plot a plane at the points of the spiral using the least squares method
            A = np.column_stack((pos_sat[:, 0], pos_sat[:, 1], np.ones_like(pos_sat[:, 0])))
            b = pos_sat[:, 2]
            coefficients1, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            x_plane, y_plane = np.meshgrid(np.linspace(pos_sat[:, 0].min(), pos_sat[:, 0].max(), 100),
                                        np.linspace(pos_sat[:, 1].min(), pos_sat[:, 1].max(), 100))
            z_plane = coefficients1[0] * x_plane + coefficients1[1] * y_plane + coefficients1[2]
            #ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.5, color='r')


            # 2. Draw a plane at the points of the two lines GNSS-specular point and Cubesat-specular point using the least squares method.
            line_points = np.array([pos_Cubesat, specular_point, pos_GNSS])
            A = np.column_stack((line_points[:, 0], line_points[:, 1], np.ones_like(line_points[:, 0])))
            b = line_points[:, 2]
            coefficients2, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            x_plane, y_plane = np.meshgrid(np.linspace(line_points[:, 0].min(), line_points[:, 0].max(), 100),
                                        np.linspace(line_points[:, 1].min(), line_points[:, 1].max(), 100))
            z_plane = coefficients2[0] * x_plane + coefficients2[1] * y_plane + coefficients2[2]
            #ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.5, color='orange')


            # 3. Draw the line of intersection between the 2 planes
            normal_plane1 = np.array([coefficients1[0], coefficients1[1], -1])
            normal_plane2 = np.array([coefficients2[0], coefficients2[1], -1])
            direction_vector = np.cross(normal_plane1, normal_plane2)
            intersection_point = np.array([pos_Cubesat[0], pos_Cubesat[1], pos_Cubesat[2]]) 
            # Parametric equation of the line of intersection
            t = np.linspace(-0, 0.0005, 100)  
            x_line = intersection_point[0] + direction_vector[0] * t
            y_line = intersection_point[1] + direction_vector[1] * t
            z_line = intersection_point[2] + direction_vector[2] * t
            #ax.plot3D(x_line, y_line, z_line, 'b')


            # 4. Calculate the vector representing the line connecting pos_Cubesat to the specular point
            line_vector = pos_Cubesat-specular_point 
            cos_angle = np.dot(direction_vector, line_vector) / (np.linalg.norm(direction_vector) * np.linalg.norm(line_vector))
            angle_rad = np.arccos(cos_angle)
            angle_deg1 = np.degrees(angle_rad)

            print("Angle between Cubesats line and line connecting the SP:", 180-angle_deg1)


            # 5. Calculate the angle between the inclination of the Cubesat plane and the normal to the centre of the Earth
            vector1 = specular_point - pos_Cubesat
            vector2 = pos_Cubesat - center
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)
            dot_product = np.dot(vector1, vector2)
            angle_rad = np.arccos(dot_product / (norm1 * norm2))
            angle_deg2 = np.degrees(angle_rad)

            print("Angle between the line connecting the SP and the normal to the centre of the Earth :", 180-angle_deg2)

            print("Angle between Cubesat line and the normal to the centre of the Earth :", 180-angle_deg1 + 180-angle_deg2)


            # 6. Calculate angle between horizon line and GNSS elevation
            #ax.plot([pos_Cubesat[0], pos_GNSS[0]], [pos_Cubesat[1], pos_GNSS[1]], [pos_Cubesat[2], pos_GNSS[2]], linestyle='dashed', color='y',label='GNSS -> Cubesats')
            #ax.plot3D([pos_Cubesat[0],0], [pos_Cubesat[1], -2e7], [pos_Cubesat[2], 6.771e6], 'k', linestyle='dashed', label='Horizon')

            line1 = np.array([pos_Cubesat[0], pos_Cubesat[1], pos_Cubesat[2]]) - np.array([0, -2e7, 6.771e6])
            line2 = np.array([pos_Cubesat[0], pos_Cubesat[1], pos_Cubesat[2]]) - np.array([pos_GNSS[0], pos_GNSS[1], pos_GNSS[2]])
            angle = np.arccos(np.dot(line1, line2) / (np.linalg.norm(line1) * np.linalg.norm(line2)))
            angle_degrees = np.degrees(angle)

            print("Angle between horizon and GNSS-Cubesats line: ", angle_degrees)

        # Create a mesh grid for the sphere
        texture_path = "/Users/eliotthubin/Downloads/earth_texture.jpg"
        texture_image = Image.open(texture_path)
        texture_image_rgba = texture_image.convert("RGBA")

        u, v = np.mgrid[0:2 * np.pi:200j, 0:np.pi:100j]
        x = center[0] + Re * np.cos(u) * np.sin(v)
        y = center[1] + Re * np.sin(u) * np.sin(v)
        z = center[2] + Re * np.cos(v)
        #ax.plot_surface(x, y, z, color='blue',alpha=0.2) #a enlever quand on veut plot la terre
        #ax.scatter(0,0,0,s=100,color='blue',alpha=0.2,label='Earth')

        #Normalize texture coordinates to range [0, 1]
        norm_u = (u - u.min()) / (u.max() - u.min())
        norm_v = (v - v.min()) / (v.max() - v.min())
        texture_image_resized = texture_image_rgba.resize((u.shape[0], v.shape[0]))
        texture_data = np.array(texture_image_resized)/255.0  # Normalize RGBA values to [0, 1] range
        alpha = 0.1  # Set the desired transparency value (0.0 - transparent, 1.0 - opaque)
        texture_data[..., 3] = alpha
        ax.plot_surface(x, y, z, facecolors=texture_data, rstride=1, cstride=1, shade=False) # Apply the texture to the surface of the sphere


        #------------------------------------------------------------------------------------------

        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')

        # Different view of the system

        # Limit for t=2700 view of the whole system zoomed
        # ax.set_xlim(-0.5e6,0.5e6)
        # ax.set_ylim(-0.5e6,0.5e6)
        # ax.set_zlim(6e6,7e6)

        # ax.set_box_aspect([1,1,1])

        # Limit for t=2900 view of the whole system zoomed
        # ax.set_xlim(1.8e6,2.8e6)
        # ax.set_ylim(-1e6,1e6)
        # ax.set_zlim(5.5e6,6.5e6)

        # Limit for t=2200 view of the whole system zoomed
        # ax.set_xlim(-5.5e6,-4.5e6)
        # ax.set_ylim(-0.5e6,0.5e6)
        # ax.set_zlim(4e6,4.5e6)

        # Limit for t=800 view of the whole system zoomed
        # ax.set_xlim(0.5e6,1.5e6)
        # ax.set_ylim(-1e6,1e6)
        # ax.set_zlim(-6.9e6,-5.9e6)


        # Limit for t = 2700
        # ax.set_xlim(10,160)
        # ax.set_ylim(-80,80)
        # ax.set_zlim(-60+6.771e6,60+6.771e6)

        # Limit for t = 900
        # ax.set_xlim(-100,40)
        # ax.set_ylim(-80,80)
        # ax.set_zlim(-40-6.771e6,40-6.771e6)

        # Limit for t = 1300
        # ax.set_xlim(-420-4.352e6,-280-4.352e6)
        # ax.set_ylim(-80,80)
        # ax.set_zlim(-920-5.186e6,-800-5.186e6)

        # General view
        # ax.set_xlim(-2e7,2e7)
        # ax.set_ylim(-2.0e7,2.0e7)
        # ax.set_zlim(-1.6e7,1.6e7)

        # General view zoomed on the Earth
        ax.set_xlim(-0.9e7,0.9e7)
        ax.set_ylim(-0.9e7,0.9e7)
        ax.set_zlim(-0.7e7,0.7e7)

        ax.view_init(elev=0, azim=90)
        
        #plt.savefig("Inclinaison 12x05.svg", bbox_inches='tight')

        plt.legend()


    return pos_sat,specular_point






def pos_sat(Time):

    """
    Compute the RELATIVE positions of deputy satellites for a given time in various formation configurations.

    Parameters:
    - Time (int): The time instant for which to compute the positions.

    Returns:
    - pos_sat (numpy.ndarray): Array containing the positions of deputy satellites.
    """


    # Small on-board computer for easy management of flight formation type activation

    Spiral_shape = 1        #Spiral configuration (different type of arm (see below))
    Single_orbit = 0        #Single orbit formation
    xy_shape = 0            #X or Y-shaped configuration
    Left_right_orbit = 0    #Symetrical orbits 


    mu = 398600.5 #km3/s2
    r_tgt = 6371 + 400 #km
    w = np.sqrt(mu/r_tgt**3) #rad/s
    print("w=",w)
    num_points = 3600*100
    T = 2*np.pi/w #period
    t_Cubesat = np.linspace(0, T, num_points)

    
    # -------------------------- 2D Spiral Formation -------------------------
    if Spiral_shape == 1: 

        arms_2 = 0
        arms_3 = 0
        arms_4 = 1
        arms_5 = 0

        if (arms_2):
            phi0_list = [0,180]
            x0_list = [2,5,9,13,17,20,23,25,27,29,30,31,32,33,34] 
            phase_shift = 150
            colors = ['blue', 'r']
        if (arms_3):
            phi0_list = [0,120,240]
            x0_list = [4,8,14,19,19,21,29,30,33,35] #[5,10,14,18,19,23,29,30,33,35]   
            phase_shift = 230
            colors = ['blue', 'r', 'g']
        if (arms_4):
            phi0_list = [0,90,180,270]
            x0_list = [5,11,17,22,24,30,35]     
            phase_shift = 220
            colors = ['blue', 'r', 'g', 'orange']
        if (arms_5):
            phi0_list = [0,72,144,216,288]
            x0_list = [3, 11, 13, 21, 22, 35] 
            phase_shift = 210
            colors = ['blue', 'r', 'g', 'orange']

        dephasage, index = 0,1

        num_sat = len(phi0_list)*len(x0_list)+1  #min:20 / max:32
        print('Number of satellite :',num_sat)
        pos_sat = np.zeros((num_sat,3))

        plt.scatter(pos_sat[0,1], pos_sat[0,0], color='k', label='Chief satellite')

        lines = [[] for _ in range(len(colors))]

        for x0 in x0_list:
            for phi0, color in zip(phi0_list, colors):

                x0_dot = 0.0
                y0 = 2*x0_dot / w
                y0_dot = -2*w*x0
                z0 = 0
                z0_dot = 0.0
                theta0 = 0.0 

                x, y, z = relative_3Dmotion(t_Cubesat[Time], x0, x0_dot, y0, y0_dot, z0, z0_dot, np.radians(phi0 + dephasage), np.radians(theta0), w) 
                pos_sat[index,:] = np.array([x, y, z])
                plt.scatter(pos_sat[index,1], pos_sat[index,0], color=color)
                lines[colors.index(color)].append((y, x))

                index += 1
            dephasage += phase_shift / len(x0_list) # deg 

        for line_coords, color in zip(lines, colors):
            xs, ys = zip(*line_coords)
            plt.plot(xs, ys, color=color, alpha=0.3)

        plt.xlabel("y [m]", fontsize=16)
        plt.ylabel("x [m]", fontsize=16)

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.grid()
        plt.legend(fontsize=14)

        plt.gcf().set_size_inches(8, 8)

        #plt.axis("equal")
        plt.savefig("Spiral_4_Config.svg", format="svg")
    


    # ------------------ Y/X formations (and straight line) -------------------
    if xy_shape == 1:
        
        Y_shape = 1
        X_shape = 0

        if Y_shape == 1:
            y0_list = np.linspace(5,50,20) 
            y0_list2 = np.linspace(-5,-50,10)
        elif X_shape == 1:
            y0_list = np.linspace(1,48,15)
            y0_list2 = np.linspace(-1,-48,15) 


        x0_dot,x0,y0_dot,z0_dot = 0.0,0.0,0.0,0.0
        z0 = 0.0
        phi0 = 0
        theta0 = 0

        index=1
        count=0

        num_sat = len(y0_list)+len(y0_list2)+1  #min:20 / max:32
        print('Number of satellite :',num_sat)
        pos_sat = np.zeros((num_sat,3))

        z0_both = 200000

        for y0 in y0_list:
            z0+=z0_both
            if count % 2 != 0:
                x,y,z = relative_3Dmotion(t_Cubesat[Time],x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0),np.radians(theta0),w) 
                pos_sat[index,:] = np.array([z, y, x])
                plt.scatter(pos_sat[index][0], pos_sat[index][1],color='b')
                index+=1
            else:
                x,y,z = relative_3Dmotion(t_Cubesat[Time],x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0),np.radians(theta0),w) 
                pos_sat[index,:] = np.array([-z, y, x])
                plt.scatter(pos_sat[index][0], pos_sat[index][1],color='b')
                index+=1
            count += 1
            

        z0 = 0

        for y0 in y0_list2:

            if Y_shape == 1:
                z0+=0
            elif X_shape == 1:
                z0+=z0_both

            if count % 2 != 0:
                x,y,z = relative_3Dmotion(t_Cubesat[Time],x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0),np.radians(theta0),w) 
                pos_sat[index,:] = np.array([z, y, x])
                plt.scatter(pos_sat[index][0], pos_sat[index][1],color='b')
                index+=1
            else:
                x,y,z = relative_3Dmotion(t_Cubesat[Time],x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0),np.radians(theta0),w) 
                pos_sat[index,:] = np.array([-z, y, x])
                plt.scatter(pos_sat[index][0], pos_sat[index][1],color='b')
                index+=1
            count += 1
            

        # Plot the target satellite
        plt.scatter(pos_sat[0,0], pos_sat[0,1], color='k', label='Chief satellite')

        # Set plot parameters
        plt.legend(loc='upper center',fontsize=12)
        plt.xlabel('z [m]', fontsize=14)
        plt.ylabel('y [m]', fontsize=14)
        # plt.xlim(-100,100)
        # plt.ylim(-100,100)
        plt.grid()
        plt.savefig("Y_shaped Config.svg")
    

    

    #--------------------------- Single Orbite Formation --------------------------------- 
    if Single_orbit == 1:
        dephasage, index = 0,1
        count = 1

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
            theta0 = -(90+phi0+dephasage)

            x,y,z = relative_3Dmotion(Time,x0,x0_dot,y0,y0_dot,z0,z0_dot,np.radians(phi0-dephasage),np.radians(theta0),w) 
            pos_sat[index,:] = np.array([x,y,z])
            if count == 1:
                plt.scatter(pos_sat[index,1], pos_sat[index,0],color='b', label ="Deputy satellites")
                count+=1
            else:
                plt.scatter(pos_sat[index,1], pos_sat[index,0],color='b')
            index+=1
            dephasage+=360/(num_sat-1) #deg 

        # Print orbits
        num_points = 3600
        T = 2*np.pi/w #period
        t = np.linspace(0, T, num_points)
        x, y, z = relative_3Dmotion(t, x0, x0_dot, y0, y0_dot, z0, z0_dot, np.radians(phi0), np.radians(theta0), w)
        plt.plot(y, x, alpha = 0.3, label="Deputy satellites orbit")

        plt.xlabel("y [m]", fontsize=14)
        plt.ylabel("x [m]", fontsize=14)
        plt.grid()
        #plt.xlim(-27,27)
        #plt.ylim(-52,52)
        plt.legend(fontsize=12,loc='upper center')
        plt.savefig("Single Orbit Config.svg")
    


    #---------------------- Symetrical Orbits Formation - left and right -------------------- 
    if Left_right_orbit == 1:
    
        x0_dot = 0.
        x0 = 17
        y0_dot = -2*w*x0
        z0 = 0
        z0_dot = 0.0
        theta0 = 0
        C = np.sqrt( (3*x0+2*y0_dot/w)**2 + (x0_dot/w)**2 )
        y0_list = [(2*C + 10) + 2*x0_dot/w , -(2*C + 10) + 2*x0_dot/w]
        phi0_list = np.linspace(0,360,15)

        num_sat = len(y0_list)*len(phi0_list)+1  #min:20 / max:32
        print('Number of satellite :',num_sat)
        pos_sat = np.zeros((num_sat,3))

        plt.scatter(pos_sat[0,0], pos_sat[0,1], color='k', label='Chief satellite')

        colors = ['b', 'r']
        index=1

        labels = ['Upstream orbit', 'Downstream orbit']

        for y0, color, label in zip(y0_list, colors, labels):
            for phi0 in phi0_list:
                x, y, z = relative_3Dmotion(Time, x0, x0_dot, y0, y0_dot, z0, z0_dot, np.radians(phi0), np.radians(theta0), w)
                pos_sat[index,:] = np.array([x, y, z])
                plt.scatter(pos_sat[index,1], pos_sat[index,0], color=color)
                index += 1


            num_points = 3600
            T = 2*np.pi/w #period
            t = np.linspace(0, T, num_points)
            x, y, z = relative_3Dmotion(t, x0, x0_dot, y0, y0_dot, z0, z0_dot, np.radians(phi0), np.radians(theta0), w)
            plt.plot(y, x, alpha = 0.3, label=label)

        plt.xlabel("y [m]", fontsize=14)
        plt.ylabel("x [m]", fontsize=14)
        plt.legend(fontsize=12,loc='upper center')
        plt.grid()
        #plt.savefig("Sym orbit Config.svg")


    return pos_sat



