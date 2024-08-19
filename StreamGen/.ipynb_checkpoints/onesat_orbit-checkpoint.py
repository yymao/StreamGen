# Orbits simulation script for integrating satellite orbits

import sys
sys.path.insert(0, '/tigress/dropulic/SatGen/')  # Add custom library path
import aux
import orbit as orb
from profiles import Dekel, MN, EnergyAngMomGivenRpRa, Phi  # Importing various galaxy profiles and utilities
import multiprocessing as mp
from tqdm import tqdm  # Progress bar for loops
import scipy
import math
import time as tt  # Importing time module for performance measurement
import numpy as np
from galhalo import Reff  # Function to calculate effective radius
import helpers

def integrate_orbit(galaxy, coord_at, host_coords, mass, redshift_id, time_btwn_apo, ecc_est, period_est_correction=0, forward=True, random_draw=None, inLLcorner=False):
    """
    Integrate the orbit of a satellite galaxy, estimating its orbit over time.

    Parameters:
    - galaxy: Galaxy object containing the host profile information.
    - coord_at: Coordinates of the satellite at different redshifts.
    - host_coords: Coordinates of the host galaxy.
    - mass: Array of satellite masses over redshifts.
    - redshift_id: Index for the current redshift.
    - time_btwn_apo: Array of time intervals between successive apocenters.
    - ecc_est: Estimated eccentricity of the orbit.
    - period_est_correction: Correction factor for the period estimate (default is 0).
    - forward: Boolean flag for forward or backward integration (default is True).
    - random_draw: Random velocities for initial conditions, if provided (default is None).
    - inLLcorner: Boolean flag for low-likelihood corner cases (default is False).

    Returns:
    - time: Array of time steps.
    - Nstep: Number of integration steps.
    - tmax: Maximum integration time.
    - pos_vel: Array of positions and velocities.
    - min_locs: Indices of pericenter locations.
    - max_locs: Indices of apocenter locations.
    - satdist: Array of satellite distances from the host.
    - t_new: Adjusted time array for integration.
    """
    print('in integrate_orbit')
    print('coord_at', coord_at)
    print('coord_at[redshift_id]', coord_at[redshift_id])
    print(random_draw)
    print('period_est_correction', period_est_correction)
    
    # Initial conditions for the orbit
    xv_o = coord_at[redshift_id]
    z = xv_o[2]
    x = xv_o[0] * np.cos(xv_o[1])
    y = xv_o[0] * np.sin(xv_o[1])
    satdist_init = np.linalg.norm(np.array([x, y, z]))  # Initial distance of the satellite from the host
    mass = mass[redshift_id]
    
    # Set the integration timestep based on eccentricity and low-likelihood corner cases
    timestep = 0.01
    if ecc_est > 0.3:
        timestep = 0.001
    if inLLcorner:
        timestep /= 10
        if ecc_est < 0.3:
            timestep /= 10
    
    # Estimate the orbital period based on the satellite distance and mass
    if len(time_btwn_apo) > 0:
        period_est = np.abs(time_btwn_apo[0])
        if period_est_correction != 0:
            period_est = np.abs(time_btwn_apo[period_est_correction])
    else: 
        period_est = orbital_period_estimate(satdist_init, mass)

    # Cap the period estimate to avoid overly long integration times
    if period_est > 20:
        period_est = 20
    
    # Initializing the input conditions for the orbit integration
    if random_draw is not None:
        xv_in = np.concatenate((xv_o[:3], random_draw), axis=None)
    else:
        xv_in = xv_o

    # Create orbit object and host profile
    o = orb.orbit(xv_in)
    total_profile = galaxy.get_host_profile(redshift_id)

    # Set integration time parameters based on the period estimate
    if period_est < 1.0:
        print('Period less than 1 Gyr! Integrating longer to decrease noise')
        tmax = math.ceil(period_est * 10)  # Extend integration time for short periods
        Nstep = math.ceil(tmax / timestep)
    elif 1 < period_est < 5:
        tmax = math.ceil(period_est * 5)
        Nstep = math.ceil(tmax / timestep)
    else:
        tmax = math.ceil(period_est)
        Nstep = math.ceil(tmax / timestep)

    # Set the time array for integration, depending on forward or backward integration
    if forward:
        tmin = 0
        tmax = tmax
    else:
        tmin = 0
        tmax = -tmax

    t = np.linspace(tmin, tmax, Nstep + 1)

    # Integrate the orbit over the specified time
    start = tt.time()
    o.integrate(t, total_profile, mass)
    end = tt.time()
    timer = end - start

    # Extract the results from the orbit integration
    time = o.t
    pos_vel = o.xvArray

    # Find apocenter and pericenter locations from the position and velocity data
    t_new = []
    satdist, min_loc, max_loc = helpers.find_apo_peri_from_pos_vel(pos_vel, forward)
    distdiff = np.abs(satdist[0] - satdist_init)
    
    # Check if the integration needs to be extended based on the number of pericenters
    n = 2
    condition = False  # Boolean flag to check if we have enough pericenters
    if len(min_loc[0]) < 5:
        condition = True
    if timer > 120:
        condition = False
        t_new = t

    # Extend the integration if needed
    while condition:
        if period_est < 1.0:
            tmax = math.ceil(period_est * 10) * n
            Nstep = math.ceil(tmax / timestep)
        elif 1 < period_est < 5:
            tmax = math.ceil(period_est * 5) * n
            Nstep = math.ceil(tmax / timestep)
        else:
            if period_est > 20:
                period_est = 20
            tmax = math.ceil(period_est) * n
            Nstep = math.ceil(tmax / timestep)

        # Reinitialize orbit and integrate over the extended time
        o = orb.orbit(xv_in)
        if n > 5:
            condition = False
        if forward:
            tmin = 0
            tmax = tmax
        else:
            tmin = 0
            tmax = -tmax

        t_new = np.linspace(tmin, tmax, Nstep + 1)
        o.integrate(t_new, total_profile, mass)
        time = o.t
        pos_vel = o.xvArray

        # Recalculate apocenter and pericenter locations
        satdist, min_loc, max_loc = helpers.find_apo_peri_from_pos_vel(pos_vel, forward)

        # Check if enough pericenters have been found
        if len(min_loc[0]) < 5 and condition != False:
            condition = True
        else:
            condition = False
        n += 1
    
    # Adjust the min/max locations to ensure consistent lengths
    min_locs = []
    max_locs = []
    if len(min_loc[0]) > len(max_loc[0]):
        min_locs = min_loc[0][:len(max_loc[0])]
        max_locs = max_loc[0]
    elif len(min_loc[0]) < len(max_loc[0]):
        max_locs = max_loc[0][:len(min_loc[0])]
        min_locs = min_loc[0]
    else:
        max_locs = max_loc[0]
        min_locs = min_loc[0]

    # Ensure that min/max locations are within bounds of the data
    min_locs = min_locs[np.where(min_locs > 4)]
    max_locs = max_locs[np.where(max_locs < (len(satdist) - 4))]

    # Further adjustment of min/max locations to ensure consistent lengths
    if len(min_locs) > len(max_locs):
        min_locs = min_locs[:len(max_locs)]
    elif len(min_locs) < len(max_locs):
        max_locs = max_locs[:len(min_locs)]

    # Return the results
    if len(t_new) == 0:
        t_new = t

    return time, Nstep, tmax, pos_vel, min_locs, max_locs, satdist, t_new
