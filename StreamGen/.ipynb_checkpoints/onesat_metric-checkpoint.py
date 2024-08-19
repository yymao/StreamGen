# One Satellite Metric

def calc_rosette_angle_SatGen(max_loc, min_loc, pos_vel):
    """
    Calculate the rosette angle for satellite orbits.

    Parameters:
    - max_loc: Array of indices corresponding to apocenters along the orbit.
    - min_loc: Array of indices corresponding to pericenters along the orbit.
    - pos_vel: Array containing position and velocity data in cylindrical coordinates.

    Returns:
    - half_period_locs: List of positions at half the orbital period.
    - rosette_angle_arr: Array of calculated rosette angles.
    """
    # Calculate the difference between max and min positions
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
            coord_rosette_0, vels_rosette_0 = rvcyltocart(pos_vel[hf_pd_loc_i[0], :3], pos_vel[hf_pd_loc_i[0], 3:])
            coord_rosette_1, vels_rosette_1 = rvcyltocart(pos_vel[hf_pd_loc_i[1], :3], pos_vel[hf_pd_loc_i[1], 3:])
            rosette_angle = np.arccos(np.dot(coord_rosette_0, coord_rosette_1) / (np.linalg.norm(coord_rosette_0) * np.linalg.norm(coord_rosette_1)))
            rosette_angle_arr.append(rosette_angle)
        else:
            rosette_angle_arr.append(np.inf)
    
    return half_period_locs, rosette_angle_arr

def stream_or_shell_integrate_OLD(m_sat, sat_host_dist, peri_arr, apo_arr, coord_peri, coord_apo, time_between_apo, alltime_hostcoords, redshift_id, t, max_loc, ax, N_orb_from_back, vel_apo, min_loc, rosette_angle_arr, vel_peri, forward=True):
    """
    Integrates satellite orbits and determines if the satellite is a stream or shell.
    
    Parameters:
    - m_sat: Satellite masses.
    - sat_host_dist: Distance between satellite and host galaxy.
    - peri_arr, apo_arr: Arrays of pericenter and apocenter distances.
    - coord_peri, coord_apo: Arrays of pericenter and apocenter coordinates.
    - time_between_apo: Time intervals between successive apocenters.
    - alltime_hostcoords: Array of host galaxy coordinates over time.
    - redshift_id: Index for redshift in the simulation.
    - t: Time array.
    - max_loc, min_loc: Indices of apocenters and pericenters in the orbit.
    - ax: Axes object for plotting.
    - N_orb_from_back: Number of orbits from backward time integration.
    - vel_apo, vel_peri: Velocity data at apocenters and pericenters.
    - rosette_angle_arr: Array of precomputed rosette angles.
    - forward: Boolean flag indicating forward or backward time integration.
    
    Returns:
    - Various arrays of computed physical quantities (Psi_E, Psi_L, deltaPsi, etc.)
    - Integer indicating the classification of the orbit (0: intact satellite, 1: stream, 2: shell).
    """
    print('in stream_or_shell_integrate')
    
    # Initialization
    what_is_it = 0
    time_between_apo = np.abs(time_between_apo)
    E_sat, L_sat, l_s = [], [], []
    deltaPsi_arr, mod_deltaPsi_arr = [], []
    mod_t_apo_arr, d_deltaPsi_dL_Eorb_arr = [], []
    Psi_L_arr, Psi_E_arr = [], []
    N_orb, mod_E_arr, mod_L_arr, predictions = [], [], [], []
    M_Rp_arr = []
    ii = 0

    if len(peri_arr) >= 3:
        # Loop over pericenter and apocenter arrays
        for rp, ra, sathd in zip(peri_arr, apo_arr, sat_host_dist):
            # Define host and satellite profiles
            host_profile = Dekel(alltime_hostcoords[0, redshift_id], alltime_hostcoords[1, redshift_id],
                                 alltime_hostcoords[2, redshift_id], alltime_hostcoords[3, redshift_id], z=0.)
            r_half_host = Reff(alltime_hostcoords[4, redshift_id], host_profile.ch)
            disk_scale_length_host = 0.766421 / (1. + 1. / flattening) * r_half_host
            disk_scale_height_host = disk_scale_length_host / flattening
            host_disk_profile = MN(alltime_hostcoords[0, redshift_id] * fd, disk_scale_length_host, disk_scale_height_host)
            host_stellar_disk_profile = MN(alltime_hostcoords[5, redshift_id] * fd, disk_scale_length_host, disk_scale_height_host)
            total_profile = [host_profile, host_disk_profile]
            s = (m_sat[redshift_id] / (3 * (host_profile.M(sathd) + host_disk_profile.M(sathd)))) ** (1 / 3)
            M_Rp_arr.append(host_profile.M(sathd) + host_disk_profile.M(sathd))
            
            # Convert pericenter coordinates to Cartesian and calculate energy and angular momentum
            coords_cart, vels_cart = rvcyltocart(coord_peri[ii], vel_peri[ii])
            E_new, L_new = EnergyAngMom(coords_cart, vels_cart, Phi(total_profile, sathd), m_sat[redshift_id])
            ii += 1
            E, L = EnergyAngMomGivenRpRa(total_profile, rp, ra)
            E_sat.append(E_new)
            L_sat.append(L_new)

        mod_E_arr = np.array(E_sat)
        mod_L_arr = np.array(L_sat)

        # Loop to calculate the deltaPsi angles between successive apocenters
        range_low = 0
        range_high = len(peri_arr) - 2
        skip = 1
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
            deltaPsi_arr.append(deltaPsi)

        mod_deltaPsi_arr = np.array(deltaPsi_arr)
        mod_t_apo_arr = t[max_loc[0:len(np.array(time_between_apo))]]
        mod_t_apo_arr = np.abs(mod_t_apo_arr)
        maxlocflip = max_loc
        modtflip = mod_t_apo_arr
        rosette_angle_arr_flip = rosette_angle_arr

        # Loop to compute Psi_L and Psi_E for classification
        range_low = 0
        range_high = len(peri_arr) - 3
        skip = 1
        for i in range(range_low, range_high, skip):
            increment = i + 1
            host_profile = Dekel(alltime_hostcoords[0, redshift_id], alltime_hostcoords[1, redshift_id],
                                 alltime_hostcoords[2, redshift_id], alltime_hostcoords[3, redshift_id], z=0.)
            disk_scale_length_host = 0.766421 / (1. + 1. / flattening) * r_half_host
            disk_scale_height_host = disk_scale_length_host / flattening
            host_disk_profile = MN(alltime_hostcoords[0, redshift_id] * fd, disk_scale_length_host, disk_scale_height_host)
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
            deltaPsi_2 = mod_deltaPsi_arr[increment]
            d_deltaPsi_dL_Eorb = np.abs(deltaPsi - deltaPsi_2) / (L1 - L2)
            d_deltaPsi_dL_Eorb_arr.append(d_deltaPsi_dL_Eorb)
            Psi_L = l_s[i] * d_deltaPsi_dL_Eorb * N_orb
            Psi_L_arr.append(Psi_L)

            E1 = mod_E_arr[i]
            E2 = mod_E_arr[increment]
            Tr = mod_t_apo_arr[i]
            deltaTr = mod_t_apo_arr[i] - mod_t_apo_arr[increment]
            deltaE = E1 - E2
            dTr_dE_Lorb = np.abs(deltaTr) / deltaE
            Psi_E = e_s * (deltaPsi / Tr) * dTr_dE_Lorb * N_orb
            if np.minimum(np.abs(Psi_E), rosette_angle_arr_flip[i]) == np.abs(Psi_E):
                Psi_E_arr.append(Psi_E)
            else:
                Psi_E_arr.append(rosette_angle_arr_flip[i])

        predictions = np.divide(Psi_L_arr, Psi_E_arr)

        # Classify based on predictions
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

    # Return results
    return int(what_is_it), np.array(Psi_E_arr), np.array(Psi_L_arr), np.array(deltaPsi_arr), np.array(mod_deltaPsi_arr), np.array(L_sat), np.array(E_sat), N_orb, np.array(mod_L_arr), np.array(mod_E_arr), predictions, M_Rp_arr


def stream_or_shell(m_sat, host_coords, sat_host_dist, peri_arr, apo_arr, coord_peri, coord_apo, time_between_apo, sm_loss, alltime_hostcoords, idx_massres_sat, order, vel_apo, t, max_loc, rosette_angle_arr, vel_peri):
    """
    Classifies satellite orbits as intact satellites, streams, or shells.

    Parameters:
    - m_sat: Satellite masses.
    - host_coords: Coordinates of the host galaxy.
    - sat_host_dist: Distance between the satellite and host galaxy.
    - peri_arr, apo_arr: Arrays of pericenter and apocenter distances.
    - coord_peri, coord_apo: Arrays of pericenter and apocenter coordinates.
    - time_between_apo: Time intervals between successive apocenters.
    - sm_loss: Stellar mass loss during the orbit.
    - alltime_hostcoords: Array of host galaxy coordinates over time.
    - idx_massres_sat: Index for satellite mass resolution.
    - order: Order of integration (forward/backward).
    - vel_apo, vel_peri: Velocity data at apocenters and pericenters.
    - t: Time array.
    - max_loc: Indices of maximum positions in the orbit.
    - rosette_angle_arr: Array of precomputed rosette angles.

    Returns:
    - Classification of the orbit (0: intact satellite, 1: stream, 2: shell).
    - Various arrays of computed physical quantities (Psi_E, Psi_L, deltaPsi, etc.).
    """
    what_is_it = 0
    
    # Initialization
    E_sat, L_sat, l_s = [], [], []
    deltaPsi_arr, mod_deltaPsi_arr = [], []
    mod_t_apo_arr, d_deltaPsi_dL_Eorb_arr = [], []
    Psi_L_arr, Psi_E_arr = [], []
    N_orb, mod_E_arr, mod_L_arr, predictions = [], [], [], []

    if len(peri_arr) >= 3:
        j = 0
        # Loop over pericenter and apocenter arrays
        for rp, ra, msat, sathd in zip(peri_arr, apo_arr, m_sat, sat_host_dist):
            # Define host and satellite profiles
            host_profile = Dekel(alltime_hostcoords[0, j], alltime_hostcoords[1, j], alltime_hostcoords[2, j],
                                 alltime_hostcoords[3, j], z=0.)
            r_half_host = Reff(alltime_hostcoords[4, j], host_profile.ch)
            disk_scale_length_host = 0.766421 / (1. + 1. / flattening) * r_half_host
            disk_scale_height_host = disk_scale_length_host / flattening
            host_disk_profile = MN(alltime_hostcoords[0, j] * fd, disk_scale_length_host, disk_scale_height_host)
            host_stellar_disk_profile = MN(alltime_hostcoords[5, j] * fd, disk_scale_length_host, disk_scale_height_host)
            total_profile = [host_profile, host_disk_profile]
            s = (msat / (3 * (host_profile.M(sathd) + host_disk_profile.M(sathd)))) ** (1 / 3)
            coords_cart, vels_cart = rvcyltocart(coord_peri[j], vel_peri[j])
            E_new, L_new = EnergyAngMom(coords_cart, vels_cart, Phi(total_profile, sathd), msat)
            E, L = EnergyAngMomGivenRpRa(total_profile, rp, ra)
            E_sat.append(E_new)
            L_sat.append(L_new)
            l_s.append((np.sqrt(3) + 2) * s * L_new)
            j += 1

        mod_E_arr = np.array(E_sat)
        mod_L_arr = np.array(L_sat)

        range_low = 0
        range_high = len(peri_arr) - 1
        skip = 1
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
            deltaPsi_arr.append(deltaPsi)

        mod_deltaPsi_arr = np.array(deltaPsi_arr)
        mod_t_apo_arr = time_between_apo
        maxlocflip = max_loc
        modtflip = mod_t_apo_arr
        rosette_angle_arr_flip = rosette_angle_arr

        range_low = 0
        range_high = len(peri_arr) - 2
        skip = 1
        for i in range(range_low, range_high, skip):
            increment_0 = i
            increment = i + 1

            host_profile = Dekel(alltime_hostcoords[0, i], alltime_hostcoords[1, i], alltime_hostcoords[2, i],
                                 alltime_hostcoords[3, i], z=0.)
            r_half_host = Reff(alltime_hostcoords[4, i], host_profile.ch)
            disk_scale_length_host = 0.766421 / (1. + 1. / flattening) * r_half_host
            disk_scale_height_host = disk_scale_length_host / flattening
            host_disk_profile = MN(alltime_hostcoords[0, i] * fd, disk_scale_length_host, disk_scale_height_host)
            host_stellar_disk_profile = MN(alltime_hostcoords[5, i] * fd, disk_scale_length_host, disk_scale_height_host)
            total_profile = [host_profile, host_disk_profile]
            N_orb = len(peri_arr[:increment])
            s = (m_sat[i] / (3 * (host_profile.M(sat_host_dist[i]) + host_disk_profile.M(sat_host_dist[i])))) ** (1 / 3)
            r_tide = sat_host_dist[i] * s
            M_host = host_profile.M(sat_host_dist[i]) + host_disk_profile.M(sat_host_dist[i])
            scale_radius_host = host_profile.rs
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
            deltaPsi_2 = mod_deltaPsi_arr[increment]
            d_deltaPsi_dL_Eorb = np.abs(deltaPsi - deltaPsi_2) / (L1 - L2)
            d_deltaPsi_dL_Eorb_arr.append(d_deltaPsi_dL_Eorb)
            Psi_L = l_s[i] * d_deltaPsi_dL_Eorb * N_orb
            Psi_L_arr.append(Psi_L)

            E1 = mod_E_arr[i]
            E2 = mod_E_arr[increment]
            Tr = mod_t_apo_arr[i]
            deltaTr = mod_t_apo_arr[i] - mod_t_apo_arr[increment]
            deltaE = E1 - E2
            dTr_dE_Lorb = np.abs(deltaTr) / deltaE
            Psi_E = e_s * (deltaPsi / Tr) * dTr_dE_Lorb * N_orb
            if np.minimum(np.abs(Psi_E), rosette_angle_arr_flip[i]) == np.abs(Psi_E):
                Psi_E_arr.append(Psi_E)
            else:
                Psi_E_arr.append(rosette_angle_arr_flip[i])

        predictions = np.divide(Psi_L_arr, Psi_E_arr)

        if len(Psi_L_arr) > 1:
            if np.abs(predictions[0]) < 1:
                what_is_it = 1
            elif np.abs(predictions[0]) > 1:
                what_is_it = 2
        elif len(Psi_L_arr) == 1:
            if np.abs(predictions) < 1:
                what_is_it = 1
            elif np.abs(predictions) > 1:
                what_is_it = 2

    return int(what_is_it), np.array(Psi_E_arr), np.array(Psi_L_arr), np.array(mod_deltaPsi_arr), np.array(mod_L_arr), np.array(mod_E_arr), np.array(predictions)
