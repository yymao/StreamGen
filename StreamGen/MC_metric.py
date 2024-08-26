# Import necessary libraries and custom modules
import numpy as np
import helpers
import sys, os
satgen_path = os.path.abspath(os.path.join(__file__, "./../../../SatGen/"))
sys.path.insert(0, satgen_path)
from profiles import Dekel, MN, EnergyAngMomGivenRpRa, Phi  # Import galaxy profile functions
import pandas as pd
import scipy
import onesat_orbit
from galhalo import Reff  # Function to calculate effective radius

def calc_E_L_Tr_dPsi_mc(ca_list, cp_list, vels_peri_list, vels_apo_list, satdist_list, t_list, tba_list, total_profile, m_sat):
    """
    Calculate energy (E), angular momentum (L), and deltaPsi (angle between successive apocenters) for each stream, 
    selecting the closest to z=0.

    Parameters:
    - ca_list: List of apocenter coordinates.
    - cp_list: List of pericenter coordinates.
    - vels_peri_list: List of velocities at pericenter.
    - vels_apo_list: List of velocities at apocenter.
    - satdist_list: List of satellite distances from the host.
    - t_list: List of time steps.
    - tba_list: Time between apocenters list.
    - total_profile: Total host galaxy profile.
    - m_sat: Mass of the satellite galaxy.

    Returns:
    - E_arr: Array of calculated energies.
    - L_arr: Array of calculated angular momenta.
    - deltaPsi_arr: Array of angles between successive apocenters.
    - tba_list: Updated time between apocenters list.
    """
    E_arr = []
    L_arr = []
    deltaPsi_arr = []

    # Loop over all apocenter and pericenter coordinates
    for ca, cp, va, vp in zip(ca_list, cp_list, vels_apo_list, vels_peri_list):
        # Convert cylindrical coordinates to Cartesian coordinates
        z_a = ca[2]
        x_a = ca[0] * np.cos(ca[1])
        y_a = ca[0] * np.sin(ca[1])
        
        z_p = cp[2]
        x_p = cp[0] * np.cos(cp[1])
        y_p = cp[0] * np.sin(cp[1])
        
        # Calculate satellite distances
        satdist_a = np.linalg.norm(np.array([x_a, y_a, z_a]))
        satdist_p = np.linalg.norm(np.array([x_p, y_p, z_p]))
        
        # Convert to Cartesian coordinates and calculate energy and angular momentum
        coords_cart, vels_cart = helpers.rvcyltocart(cp, vp)
        E, L = helpers.EnergyAngMom(coords_cart, vels_cart, Phi(total_profile, satdist_p), m_sat)
        E_arr.append(E)
        L_arr.append(L)

    # Calculate deltaPsi (angle between successive apocenters)
    range_low = 0
    range_high = len(E_arr) - 2
    skip = 1

    for i in range(range_low, range_high, skip):
        increment = i + 1

        z_0 = ca_list[i][2]
        x_0 = ca_list[i][0] * np.cos(ca_list[i][1])
        y_0 = ca_list[i][0] * np.sin(ca_list[i][1])

        z_1 = ca_list[increment][2]
        x_1 = ca_list[increment][0] * np.cos(ca_list[increment][1])
        y_1 = ca_list[increment][0] * np.sin(ca_list[increment][1])

        coord_apo_0 = np.array([x_0, y_0, z_0])
        coord_apo_1 = np.array([x_1, y_1, z_1])

        # Calculate angle between successive apocenters
        deltaPsi = np.arccos(np.dot(coord_apo_0, coord_apo_1) / (np.linalg.norm(coord_apo_0) * np.linalg.norm(coord_apo_1)))
        deltaPsi_arr.append(deltaPsi)
    
    return E_arr, L_arr, deltaPsi_arr, tba_list

def calc_rosette_angle(max_loc, min_loc, pos_vel):
    """
    Calculate the rosette angle for satellite orbits.

    Parameters:
    - max_loc: Array of indices corresponding to maximum positions along the orbit.
    - min_loc: Array of indices corresponding to minimum positions along the orbit.
    - pos_vel: Array containing position and velocity data in cylindrical coordinates.

    Returns:
    - half_period_locs: List of positions at half the orbital period.
    - rosette_angle_arr: Array of calculated rosette angles.
    """
    minmax_loc_diff = np.abs(np.subtract(max_loc, min_loc))
    quarter_period_timestep = np.round(minmax_loc_diff / 2)
    half_period_locs = []
    rosette_angle_arr = []

    for max_loc_i, mm_loc_diff_i in zip(max_loc, quarter_period_timestep):
        # Calculate half-period locations
        hf_pd_loc_i = np.array([int(max_loc_i + mm_loc_diff_i), int(max_loc_i - mm_loc_diff_i)])
        half_period_locs.append(hf_pd_loc_i)
        
        # Ensure indices are within bounds
        if (hf_pd_loc_i[0] > 0) and (hf_pd_loc_i[1] > 0) and (hf_pd_loc_i[1] < len(pos_vel[:, 0])) and (hf_pd_loc_i[0] < len(pos_vel[:, 0])):
            # Convert cylindrical coordinates to Cartesian and compute rosette angle
            coord_rosette_0, vels_rosette_0 = helpers.rvcyltocart(pos_vel[hf_pd_loc_i[0], :3], pos_vel[hf_pd_loc_i[0], 3:])
            coord_rosette_1, vels_rosette_1 = helpers.rvcyltocart(pos_vel[hf_pd_loc_i[1], :3], pos_vel[hf_pd_loc_i[1], 3:])
            rosette_angle = np.arccos(np.dot(coord_rosette_0, coord_rosette_1) / (np.linalg.norm(coord_rosette_0) * np.linalg.norm(coord_rosette_1)))
            rosette_angle_arr.append(rosette_angle)
        else:
            rosette_angle_arr.append(np.inf)
    
    return half_period_locs, rosette_angle_arr

def detrivatives_metric(ca_mc, cp_mc, vels_peri_mc, vels_apo_mc, satdist_mc, t_mc, tba_mc, host_total_profile, mod_mass_at, E_sat, L_sat, num_mc, redshift):
    """
    Calculate derivatives for Monte Carlo simulations, fitting L vs deltaPsi and E vs tba (time between apocenters).

    Parameters:
    - ca_mc: Monte Carlo apocenter coordinates.
    - cp_mc: Monte Carlo pericenter coordinates.
    - vels_peri_mc: Monte Carlo velocities at pericenter.
    - vels_apo_mc: Monte Carlo velocities at apocenter.
    - satdist_mc: Monte Carlo satellite distances from the host.
    - t_mc: Monte Carlo time steps.
    - tba_mc: Monte Carlo time between apocenters.
    - host_total_profile: Total host galaxy profile.
    - mod_mass_at: Modified satellite mass at current redshift.
    - E_sat: Satellite energy.
    - L_sat: Satellite angular momentum.
    - num_mc: Number of Monte Carlo simulations.
    - redshift: Redshift value.

    Returns:
    - mean_slope_L_Psi: Mean slope of L vs deltaPsi.
    - mean_slope_E_tba: Mean slope of E vs tba.
    """
    E_z0 = []
    L_z0 = []
    tba_z0 = []
    deltaPsi_z0 = []

    # Loop over Monte Carlo simulations
    for mc_i in range(num_mc):
        E_arr_mc, L_arr_mc, deltaPsi_arr_mc, tba_list_mc = calc_E_L_Tr_dPsi_mc(ca_mc[mc_i], cp_mc[mc_i], vels_peri_mc[mc_i], vels_apo_mc[mc_i], satdist_mc[mc_i], t_mc[mc_i], tba_mc[mc_i], host_total_profile, mod_mass_at[0])
        
        try:
            # Append the closest to z=0 values
            E_z0.append(E_arr_mc[0])
            L_z0.append(L_arr_mc[0])
            tba_z0.append(tba_list_mc[0])
            deltaPsi_z0.append(np.pi * 2 - deltaPsi_arr_mc[0])
        except:
            print("Exception occurred, check sampled orbits!")

    # Create DataFrame for Monte Carlo results
    df_MC = pd.DataFrame({'E': E_z0, 'L': L_z0, 'tba': tba_z0, 'deltaPsi': deltaPsi_z0})

    # Bin the data for energy and angular momentum
    hist_E, bin_edges_E = np.histogram(df_MC.E, bins='auto')
    hist_L, bin_edges_L = np.histogram(df_MC.L, bins='auto')
    bin_means_E, bin_edges_E, binnumber_E = scipy.stats.binned_statistic(df_MC.E, df_MC.E, statistic='count', bins=bin_edges_E)
    bin_means_L, bin_edges_L, binnumber_L = scipy.stats.binned_statistic(df_MC.L, df_MC.L, statistic='count', bins=bin_edges_L)

    E = E_sat
    L = L_sat
    bin_no_LdeltaPsi = []
    bin_no_Etba = []

    # Find the bin number for the satellite's energy and angular momentum
    for bin_edgeE in range(0, len(bin_edges_E) - 1):
        bin_no_LdeltaPsi = bin_edgeE
        if (bin_edges_E[bin_edgeE] <= E) and (E < bin_edges_E[bin_edgeE + 1]):
            break
        if E < bin_edges_E[0]:
            break
                
    for bin_edgeL in range(0, len(bin_edges_L) - 1):
        bin_no_Etba = bin_edgeL
        if (bin_edges_L[bin_edgeL] <= L) and (L < bin_edges_L[bin_edgeL + 1]):
            break
        if L < bin_edges_L[0]:
            break
                
    # Fit L vs deltaPsi and E vs tba for each bin
    df_MC['binnumber_E'] = binnumber_E
    df_MC['binnumber_L'] = binnumber_L
    fit_L_Psi_arr = []
    fit_E_tba_arr = []

    for bin_num_i in range(np.min(binnumber_E), np.max(binnumber_E) + 1):
        df_MC_hold = df_MC.loc[df_MC.binnumber_E == bin_num_i].reset_index(drop=True)
        if len(df_MC_hold) > 1:
            fit_L = np.polyfit(df_MC_hold.L, df_MC_hold.deltaPsi, 1)
            fit_L_Psi_arr.append(fit_L[0])
        else:
            fit_L_Psi_arr.append(np.nan)

    for bin_num_i in range(np.min(binnumber_L), np.max(binnumber_L) + 1):
        df_MC_hold = df_MC.loc[df_MC.binnumber_L == bin_num_i].reset_index(drop=True)
        if len(df_MC_hold) > 1:
            fit_E = np.polyfit(df_MC_hold.E, df_MC_hold.tba, 1)
            fit_E_tba_arr.append(fit_E[0])
        else:
            fit_E_tba_arr.append(np.nan)

    # Calculate mean slopes and handle NaNs
    mean_slope_L_Psi = fit_L_Psi_arr[bin_no_LdeltaPsi]
    mean_slope_E_tba = fit_E_tba_arr[bin_no_Etba]

    i = 0
    while np.isnan(mean_slope_L_Psi):
        if bin_no_LdeltaPsi > len(fit_L_Psi_arr) / 2:
            mean_slope_L_Psi = fit_L_Psi_arr[bin_no_LdeltaPsi - i]
            if bin_no_LdeltaPsi - i == 0:
                break
        else:
            mean_slope_L_Psi = fit_L_Psi_arr[bin_no_LdeltaPsi + i]
            if bin_no_LdeltaPsi + i == len(fit_L_Psi_arr) - 1:
                break
        i += 1

    i = 0
    while np.isnan(mean_slope_E_tba):
        if bin_no_Etba > len(fit_E_tba_arr) / 2:
            mean_slope_E_tba = fit_E_tba_arr[bin_no_Etba - i]
            if bin_no_Etba - i == 0:
                break
        else:
            mean_slope_E_tba = fit_L_Psi_arr[bin_no_Etba + i]
            if bin_no_Etba + i == len(fit_E_tba_arr) - 1:
                break
        i += 1

    return mean_slope_L_Psi, mean_slope_E_tba

def stream_or_shell_integrate(galaxy, m_sat, sat_host_dist, peri_arr, apo_arr, coord_peri, coord_apo, time_between_apo, alltime_hostcoords, redshift_id, t, max_loc, N_orb_from_back, vel_peri, min_loc, rosette_angle_arr, sigma_vel, vel_z0, dPsidL, dtbadE, forward=True):
    """
    Integrates satellite orbits and determines if the satellite is a stream or shell.

    Parameters:
    - galaxy: Galaxy object containing the host profile information.
    - m_sat: Satellite masses.
    - sat_host_dist: Distance between satellite and host galaxy.
    - peri_arr, apo_arr: Arrays of pericenter and apocenter distances.
    - coord_peri, coord_apo: Arrays of pericenter and apocenter coordinates.
    - time_between_apo: Time intervals between successive apocenters.
    - alltime_hostcoords: Array of host galaxy coordinates over time.
    - redshift_id: Index for redshift in the simulation.
    - t: Time array.
    - max_loc, min_loc: Indices of maximum and minimum positions in the orbit.
    - N_orb_from_back: Number of orbits from backward time integration.
    - vel_peri: Velocity data at pericenters.
    - rosette_angle_arr: Array of precomputed rosette angles.
    - sigma_vel: Velocity dispersion.
    - vel_z0: Velocity at redshift zero.
    - dPsidL, dtbadE: Derivatives for Psi_L and Psi_E.
    - forward: Boolean flag for forward or backward integration (default is True).

    Returns:
    - Various arrays of computed physical quantities (Psi_E, Psi_L, deltaPsi, etc.).
    - Integer indicating the classification of the orbit (0: intact satellite, 1: stream, 2: shell).
    """
    what_is_it = 0

    # Initialization
    E_sat, L_sat, l_s = [], [], []
    deltaPsi_arr, mod_deltaPsi_arr = [], []
    mod_t_apo_arr, d_deltaPsi_dL_Eorb_arr = [], []
    Psi_L_arr, Psi_E_arr = [], []
    N_orb, mod_E_arr, mod_L_arr, predictions = [], [], [], []
    M_Rp_arr = []

    N_orb = N_orb_from_back

    if len(peri_arr) >= 3:
        ii = 0

        # Loop over pericenter and apocenter arrays
        for rp, ra, sathd in zip(peri_arr, apo_arr, sat_host_dist):
            # Define host and satellite profiles
            host_profile = Dekel(alltime_hostcoords[0, redshift_id], alltime_hostcoords[1, redshift_id],
                                 alltime_hostcoords[2, redshift_id], alltime_hostcoords[3, redshift_id], z=0.)
            r_half_host = Reff(alltime_hostcoords[4, redshift_id], host_profile.ch)
            disk_scale_length_host = 0.766421 / (1. + 1. / galaxy.flattening) * r_half_host
            disk_scale_height_host = disk_scale_length_host / galaxy.flattening
            host_disk_profile = MN(alltime_hostcoords[0, redshift_id] * galaxy.fd, disk_scale_length_host, disk_scale_height_host)
            host_stellar_disk_profile = MN(alltime_hostcoords[5, redshift_id] * galaxy.fd, disk_scale_length_host, disk_scale_height_host)
            total_profile = [host_profile, host_disk_profile]
            s = (m_sat[redshift_id] / (3 * (host_profile.M(sathd) + host_disk_profile.M(sathd)))) ** (1 / 3)
            M_Rp_arr.append(host_profile.M(sathd) + host_disk_profile.M(sathd))
            
            # Convert pericenter coordinates to Cartesian and calculate energy and angular momentum
            coords_cart, vels_cart = helpers.rvcyltocart(coord_peri[ii], vel_peri[ii])
            E_new, L_new = helpers.EnergyAngMom(coords_cart, vels_cart, Phi(total_profile, sathd), m_sat[redshift_id])
            E_sat.append(E_new)
            L_sat.append(L_new)
            ii += 1

        mod_E_arr = np.array(E_sat)
        mod_L_arr = np.array(L_sat)

        range_low = 0
        range_high = len(peri_arr) - 2
        skip = 1

        # Loop to calculate deltaPsi (angle between successive apocenters)
        for i in range(range_low, range_high, skip):
            increment = i + 1

            z_0 = coord_apo[i][2]
            x_0 = coord_apo[i][0] * np.cos(coord_apo[i][1])
            y_0 = coord_apo[i][0] * np.sin(coord_apo[i][1])

            z_1 = coord_apo[increment][2]
            x_1 = coord_apo[increment][0] * np.cos(coord_apo[increment][1])
            y_1 = coord_apo[increment][0] * np.sin(coord_apo[increment][1])

            coord_apo_0 = np.array([x_0, y_0, z_0])
            coord_apo_1 = np.array([x_1, y_1, z_1])

            deltaPsi = np.arccos(np.dot(coord_apo_0, coord_apo_1) / (np.linalg.norm(coord_apo_0) * np.linalg.norm(coord_apo_1)))
            deltaPsi = np.pi * 2 - deltaPsi
            deltaPsi_arr.append(deltaPsi)

        mod_deltaPsi_arr = np.array(deltaPsi_arr)
        mod_t_apo_arr = time_between_apo
        maxlocflip = max_loc
        modtflip = mod_t_apo_arr
        rosette_angle_arr_flip = rosette_angle_arr

        range_low = 0
        range_high = len(peri_arr) - 3
        skip = 1

        # Loop to compute Psi_L and Psi_E for classification
        for i in range(range_low, range_high, skip):
            increment = i + 1

            host_profile = Dekel(alltime_hostcoords[0, redshift_id], alltime_hostcoords[1, redshift_id],
                                 alltime_hostcoords[2, redshift_id], alltime_hostcoords[3, redshift_id], z=0.)
            r_half_host = Reff(alltime_hostcoords[4, redshift_id], host_profile.ch)
            disk_scale_length_host = 0.766421 / (1. + 1. / galaxy.flattening) * r_half_host
            disk_scale_height_host = disk_scale_length_host / galaxy.flattening
            host_disk_profile = MN(alltime_hostcoords[0, redshift_id] * galaxy.fd, disk_scale_length_host, disk_scale_height_host)
            host_stellar_disk_profile = MN(alltime_hostcoords[5, redshift_id] * galaxy.fd, disk_scale_length_host, disk_scale_height_host)
            total_profile = [host_profile, host_disk_profile]
            s = (m_sat[redshift_id] / (3 * (host_profile.M(sat_host_dist[i]) + host_disk_profile.M(sat_host_dist[i])))) ** (1 / 3)
            M_host = host_profile.M(sat_host_dist[i]) + host_disk_profile.M(sat_host_dist[i])
            scale_radius_host = host_profile.rs
            r_tide = sat_host_dist[i] * s
            R1 = sat_host_dist[i] * (1. + .001)
            R2 = sat_host_dist[i] * (1. - .001)
            Phi1 = Phi(total_profile, R1)
            Phi2 = Phi(total_profile, R2)
            dPhidR_Rp = (Phi1 - Phi2) / (R1 - R2)
            e_s = 2 * r_tide * dPhidR_Rp
            l_s.append((np.sqrt(3) + 2) * s * mod_L_arr[i])

            L1 = mod_L_arr[i]
            L2 = mod_L_arr[increment]
            deltaPsi = mod_deltaPsi_arr[i]
            Psi_L = l_s[i] * dPsidL * N_orb
            Psi_L_arr.append(Psi_L)

            E1 = mod_E_arr[i]
            E2 = mod_E_arr[increment]
            Tr = mod_t_apo_arr[i]
            deltaTr = mod_t_apo_arr[i] - mod_t_apo_arr[increment]
            deltaE = E1 - E2

            Psi_E = e_s * (deltaPsi / Tr) * dtbadE * N_orb
            if np.minimum(np.abs(Psi_E), rosette_angle_arr_flip[i]) == np.abs(Psi_E):
                Psi_E_arr.append(Psi_E)
            else:
                Psi_E = rosette_angle_arr_flip[i]
                Psi_E_arr.append(rosette_angle_arr_flip[i])
                
        predictions = np.divide(Psi_L_arr, Psi_E_arr)

        # Determine if the orbit is a stream or shell based on predictions
        if len(Psi_L_arr) > 1 and forward:
            if np.abs(predictions[0]) < 1:
                what_is_it = 1
            elif np.abs(predictions[0]) > 1:
                what_is_it = 2
        elif len(Psi_L_arr) > 1 and not forward:
            if np.abs(predictions[0]) < 1:
                what_is_it = 1
            elif np.abs(predictions[0]) > 1:
                what_is_it = 2
        elif len(Psi_L_arr) == 1:
            if np.abs(predictions) < 1:
                what_is_it = 1
            elif np.abs(predictions) > 1:
                what_is_it = 2

    # Return the results
    return_Psi_E = np.array(Psi_E_arr)
    return_Psi_L = np.array(Psi_L_arr)
    return_deltaPsi = np.array(deltaPsi_arr)
    return_mod_deltaPsi = np.array(mod_deltaPsi_arr)
    return_L_sat = np.array(L_sat)
    return_E_sat = np.array(E_sat)
    return_Lmod_sat = np.array(mod_L_arr)
    return_Emod_sat = np.array(mod_E_arr)
    return_predictions = predictions

    return int(what_is_it), return_Psi_E, return_Psi_L, return_deltaPsi, return_mod_deltaPsi, return_L_sat, return_E_sat, N_orb, return_Lmod_sat, return_Emod_sat, return_predictions, M_Rp_arr
