# Placeholder functions for orbital dynamics and related calculations
import numpy as np
from scipy.signal import argrelmin, argrelmax
import numpy.ma as ma

# Function to reintegrate pericenter and apocenter properties based on input data
def peri_apo_props_reintegrate(time, Nstep, tmax, pos_vel, min_loc, max_loc, satdist, t_new):
    """
    Reintegrates the properties of pericenter and apocenter over time.

    Parameters:
    - time: array of time steps in orbit
    - Nstep: number of steps in orbit 
    - tmax: maximum time of orbit
    - pos_vel: array containing positions and velocities
    - min_loc: pericenter indicies
    - max_loc: apocenter indices
    - satdist: array of satellite distances
    - t_new: new time array

    Returns:
    - all_apo: distances at apocenter
    - all_peri: distances at percenter
    - coordinates_apo: coordinates at apocenter
    - coordinates_peri: coordinates at pericenter
    - time_between_apo: time intervals between successive apocenter events
    - vels_peri: velocities at pericenter
    - vels_apo: velocities at apocenter
    """
    t = t_new
    all_apo = satdist[max_loc]
    all_peri = satdist[min_loc]
    coordinates_peri = pos_vel[min_loc,:3]
    coordinates_apo = pos_vel[max_loc,:3]
    vels_peri = pos_vel[min_loc,3:]
    vels_apo = pos_vel[max_loc,3:]
    time_between_apo = np.diff(max_loc) * np.abs(t[1] - t[0])
    
    return all_apo, all_peri, coordinates_apo, coordinates_peri, time_between_apo, vels_peri, vels_apo

# Function to estimate orbital eccentricity based on pericenter and apocenter distances
def eccentricity_est(rp, ra):
    """
    Estimates the eccentricity of an orbit.

    Parameters:
    - rp: pericenter distance
    - ra: apocenter distance

    Returns:
    - ecc: orbital eccentricity
    """
    numerator = ra - rp
    denominator = ra + rp
    ecc = np.divide(numerator, denominator)
    
    return ecc

# Function to convert cylindrical coordinates and velocities to Cartesian coordinates and velocities
def rvcyltocart(coords_cyl, vels_cyl):
    """
    Converts cylindrical coordinates and velocities to Cartesian coordinates and velocities.

    Parameters:
    - coords_cyl: cylindrical coordinates [s, phi, z]
    - vels_cyl: cylindrical velocities [v_s, v_phi, v_z]

    Returns:
    - coords_cart: Cartesian coordinates [x, y, z]
    - vels_cart: Cartesian velocities [v_x, v_y, v_z]
    """
    s = coords_cyl[0]
    phi = coords_cyl[1]
    z = coords_cyl[2]
    
    v_s = vels_cyl[0]
    v_phi = vels_cyl[1]
    v_z = vels_cyl[2]
    
    s_dot = v_s
    phi_dot = v_phi / s
    z_dot = v_z
    
    x = s * np.cos(phi)
    y = s * np.sin(phi)
    
    x_dot = s_dot * np.cos(phi) - s * phi_dot * np.sin(phi)
    y_dot = s_dot * np.sin(phi) + s * phi_dot * np.cos(phi)
    
    coords_cart = [x, y, z]
    vels_cart = [x_dot, y_dot, z_dot]
    
    return coords_cart, vels_cart

# Function to convert Cartesian coordinates and velocities to cylindrical coordinates and velocities
def rvcarttocyl(coords_cart, vels_cart):
    """
    Converts Cartesian coordinates and velocities to cylindrical coordinates and velocities.

    Parameters:
    - coords_cart: Cartesian coordinates [x, y, z]
    - vels_cart: Cartesian velocities [v_x, v_y, v_z]

    Returns:
    - coords_cyl: cylindrical coordinates [rho, theta, z]
    - vels_cyl: cylindrical velocities [v_rho, v_theta, v_z]
    """
    x = coords_cart[0]
    y = coords_cart[1]
    z = coords_cart[2]
    
    v_x = vels_cart[0]
    v_y = vels_cart[1]
    v_z = vels_cart[2]
    
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    v_rho = (x * v_x + y * v_y) / rho
    v_theta = (x * v_y - y * v_x) / rho

    coords_cyl = [rho, theta, z]
    vels_cyl = [v_rho, v_theta, v_z]
    
    return coords_cyl, vels_cyl

# Function to calculate energy and angular momentum of an object
def EnergyAngMom(coords_cart, vels_cart, host_potential, m_sat):
    """
    Calculates the total energy and angular momentum of a satellite in a gravitational field.

    Parameters:
    - coords_cart: Cartesian coordinates [x, y, z]
    - vels_cart: Cartesian velocities [v_x, v_y, v_z]
    - host_potential: potential energy due to the host (e.g., galaxy or planet)
    - m_sat: mass of the satellite

    Returns:
    - E: total energy (kinetic + potential)
    - L: angular momentum magnitude
    """
    L = np.linalg.norm(np.cross(coords_cart, vels_cart))
    v_sat = np.linalg.norm(vels_cart)
    E = host_potential + 0.5 * v_sat**2

    return E, L

# Function to calculate the tidal radius of a satellite
def calc_rtide(m, m_rp, rp):
    """
    Estimates the tidal radius for a satellite orbiting a larger mass.

    Parameters:
    - m: mass of the satellite
    - m_rp: mass of the host at pericenter
    - rp: pericenter distance

    Returns:
    - r_tide: tidal radius
    """
    s = np.divide(m, (3 * m_rp))**(1/3)
    r_tide = np.multiply(s, rp)
    
    return r_tide

# Function to estimate the orbital period of an object
def orbital_period_estimate(r, M):
    """
    Estimates the orbital period of an object in a circular orbit.

    Parameters:
    - r: radius of the orbit in kiloparsecs (kpc)
    - M: mass of the central object (e.g., galaxy or star) in solar masses

    Returns:
    - T: orbital period in gigayears (Gyr)
    """
    G = 1.327 * 10**11  # gravitational constant in km^3 MSun^-1 s^-2
    r = r * 3.086 * 10**16  # convert kpc to km
    T = np.sqrt((4 * np.pi**2 * r**3) / (G * M))  # orbital period in seconds
    T = T / (3.16 * 10**16)  # convert seconds to gigayears (Gyr)
    
    return T

# Function to find pericenter and apocenter based on position and velocity data
def find_apo_peri_from_pos_vel(pos_vel_arr):
    """
    Identifies the apocenter and pericenter points from position and velocity data.

    Parameters:
    - pos_vel_arr: array containing positions and velocities [x, y, z, vx, vy, vz]
    - forward: direction of integration (optional)

    Returns:
    - satdist: array of satellite distances from the host
    - min_loc: indices of pericenter points (local minima)
    - max_loc: indices of apocenter points (local maxima)
    """
    z = pos_vel_arr[:, 2]
    x = pos_vel_arr[:, 0] * np.cos(pos_vel_arr[:, 1])
    y = pos_vel_arr[:, 0] * np.sin(pos_vel_arr[:, 1])
    
    satdist = np.linalg.norm(np.array([x, y, z]).T, axis=1)
    
    min_loc = argrelmin(satdist, order=5)
    max_loc = argrelmax(satdist, order=5)
        
    return satdist, min_loc, max_loc

# Function to predict values using Huber regression and outlier handling
def get_huber_predictions(x_in, y_in):
    """
    Applies Huber regression to predict values while handling outliers.

    Parameters:
    - x_in: input feature array
    - y_in: target variable array

    Returns:
    - predictions: predicted values after Huber regression
    """
    predictions = []
    if len(y_in) > 1: 
        try:
            y_start = y_in
            x_start = x_in
            z_score = zscore(y_start)
            y = y_start[np.where((np.abs(z_score) < 1) & (np.abs(y_start) > 0.01))]
            x = x_start[np.where((np.abs(z_score) < 1) & (np.abs(y_start) > 0.01))]
            
            # Standardize the input data
            x_scaler, y_scaler = StandardScaler(), StandardScaler()
            x_train = x_scaler.fit_transform(x[..., None])
            y_train = y_scaler.fit_transform(y[..., None])
            
            # Fit the Huber regression model
            model = HuberRegressor(epsilon=1)
            model.fit(x_train, y_train.ravel())
            
            # Make predictions
            test_x = x_in
            predictions = y_scaler.inverse_transform(model.predict(x_scaler.transform(test_x[..., None])))
        except: 
            print("Huber exception!")
            predictions = y_in
    else: 
        predictions = y_in
    return predictions
