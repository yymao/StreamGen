# Orbits simulation and Monte Carlo integration for satellite galaxies
import sys
sys.path.insert(0, '/tigress/dropulic/SatGen/') #!change this to your path to SatGen to load in necessary 
from profiles import Dekel, MN, EnergyAngMomGivenRpRa, Phi
import multiprocessing as mp
from tqdm import tqdm
import scipy
import onesat_orbit
import numpy as np
import helpers

# Function to perform Monte Carlo integration of orbits for a satellite galaxy
def integrate_mc_orbits(galaxy, coordinate, sigma, coord_at, host_coords, mass, redshift_id, time_btwn_apo, ecc_est, num_mc, period_est_correction=0):
    """
    Integrates satellite "particles" after sampling from satellite's velocity distribution.

    Parameters:
    - galaxy (object): The galaxy object, providing necessary properties.
    - coordinate (2D array): Array of satellite positions and velocities at each redshift step.
    - sigma (float): Velocity dispersion of the satellite (in kpc/Gyr).
    - coord_at (array): Current coordinates of the satellite.
    - host_coords (2D array): Array of host galaxy properties at different time steps.
    - mass (float): Mass of the satellite.
    - redshift_id (int): Index indicating the current redshift in the time evolution.
    - time_btwn_apo (float): Time between successive apocenter events (in Gyr).
    - ecc_est (float): Estimated orbital eccentricity of the satellite.
    - num_mc (int): Number of Monte Carlo iterations.
    - period_est_correction (float, optional): Correction factor for the estimated orbital period.

    Returns:
    - ca_list (list): List of apocenter coordinates from the sampled orbits.
    - cp_list (list): List of pericenter coordinates from the sampled orbits.
    - vels_peri_list (list): List of satellite velocities at pericenter from the sampled orbits.
    - vels_apo_list (list): List of satellite velocities at apocenter from the sampled orbits.
    - satdist_list (list): List of satellite distances from the host galaxy from the sampled orbits.
    - t_list (list): List of time steps from the sampled orbits.
    - tba_list (list): List of times between successive apocenter events from the sampled orbits.
    - total_profile (object): The host galaxy's potential profile.
    - pos_vel_list (list): List of satellite positions and velocities at each time step from the sampled orbits.
    - num_mc (int): Number of sampled orbits.
    
    Notes:
    - The function uses parallel processing to integrate sampled satellite particles.
    """
    num_draw = num_mc

    # Initialize lists to store results
    ca_list = []
    cp_list = []
    vels_peri_list = []
    vels_apo_list = []
    satdist_list = []
    t_list = []
    tba_list = []
    total_profile = []
    pos_vel_list = []

    # Set up multiprocessing pool for parallel Monte Carlo simulations
    pool = mp.Pool(mp.cpu_count() - 1)
    print("cpu count", mp.cpu_count())
    
    # Perform Monte Carlo simulations using parallel processing
    results = [pool.apply_async(MC_orbits_SG, args=(galaxy, coordinate, sigma, coord_at, host_coords, mass, redshift_id, time_btwn_apo, ecc_est, period_est_correction)) for _ in tqdm(range(0, num_draw), position=0, leave=True)]
    
    pool.close()
    res = [f.get() for f in tqdm(results)]

    # Collect results from Monte Carlo simulations
    for mc_i in range(0, num_draw):
        ca_list.append(res[mc_i][0])
        cp_list.append(res[mc_i][1])
        vels_peri_list.append(res[mc_i][2])
        vels_apo_list.append(res[mc_i][3])
        satdist_list.append(res[mc_i][4])
        t_list.append(res[mc_i][5])
        tba_list.append(res[mc_i][6])
        total_profile = res[mc_i][7]
        pos_vel_list.append(res[mc_i][8])

    return ca_list, cp_list, vels_peri_list, vels_apo_list, satdist_list, t_list, tba_list, total_profile, pos_vel_list, num_mc

# Function to simulate individual Monte Carlo orbits for a satellite galaxy
def MC_orbits_SG(galaxy, coordinate, sigma, coord_at, host_coords, mass, redshift_id, time_btwn_apo, ecc_est, period_est_correction):
    """
    Simulates a single Monte Carlo orbit for a satellite galaxy by drawing random velocities.
    
    Parameters:
    - galaxy (object): The galaxy object, providing necessary properties.
    - coordinate (2D array): Array of satellite positions and velocities at each redshift step.
    - sigma (float): Velocity dispersion of the satellite (in kpc/Gyr).
    - coord_at (array): Current coordinates of the satellite.
    - host_coords (2D array): Array of host galaxy properties at different time steps.
    - mass (float): Mass of the satellite.
    - redshift_id (int): Index indicating the current redshift in the time evolution.
    - time_btwn_apo (float): Time between successive apocenter events (in Gyr).
    - ecc_est (float): Estimated orbital eccentricity of the satellite.
    - period_est_correction (float, optional): Correction factor for the estimated orbital period.

    Returns:
    - ca (array): List of apocenter coordinates from the sampled orbit.
    - cp (array): List of pericenter coordinates from the sampled orbit.
    - vels_peri (array): List of satellite velocities at pericenter from the sampled orbit.
    - vels_apo (array): List of satellite velocities at apocenter from the sampled orbit.
    - satdist (array): List of satellite distances from the host galaxy from the sampled orbit.
    - t (array): List of time steps from the sampled orbit.
    - tba (array): List of times between successive apocenter events from the sampled orbit.
    - total_profile (object): The host galaxy's potential profile.
    - pos_vel (2D array): List of satellite positions and velocities at each time step from the sampled orbit.

    - The function performs a random velocity draw based on the velocity dispersion (sigma) and integrates the satellite's orbit.
    - Uses cylindrical to Cartesian coordinate conversion to apply the velocity draw in 3D space.
    - Calls `onesat_orbit.integrate_orbit` to integrate the orbit of the satellite and calculates apocenter and pericenter properties.
    """
    # Seed the random number generator for different Monte Carlo runs
    np.random.seed()
    
    # Extract satellite coordinates and velocities at the current redshift
    coordinate = coordinate[redshift_id]
    
    coord_init = coordinate[:3]
    vels_init = coordinate[3:]

    # Convert initial cylindrical coordinates to Cartesian coordinates
    coords_cart, vels_cart = helpers.rvcyltocart(coord_init, vels_init)
    
    # Generate a random velocity draw from a multivariate normal distribution
    cov = np.array([[sigma**2, 0, 0], [0, sigma**2, 0], [0, 0, sigma**2]])
    random_vel_draw = np.random.multivariate_normal(vels_cart, cov, 1).flatten()
    
    # Convert the drawn velocities back to cylindrical coordinates
    coords_cyl, coord_rand_vel_cyl = helpers.rvcarttocyl(coords_cart, random_vel_draw)

    # Integrate the satellite's orbit using the random velocity draw
    timetot, Nstep, tmax, pos_vel, min_loc, max_loc, satdist, t = onesat_orbit.integrate_orbit(galaxy, coord_at, host_coords, mass, redshift_id, time_btwn_apo, ecc_est, period_est_correction=period_est_correction,  random_draw=coord_rand_vel_cyl)
    
    # Reintegrate pericenter and apocenter properties
    all_apo, all_peri, ca, cp, tba, vels_peri, vels_apo = helpers.peri_apo_props_reintegrate(timetot, Nstep, tmax, pos_vel, min_loc, max_loc, satdist, t)
    
    # Get the total host profile at the current redshift
    total_profile = galaxy.get_host_profile(redshift_id)

    return ca, cp, vels_peri, vels_apo, satdist, t, tba, total_profile, pos_vel
