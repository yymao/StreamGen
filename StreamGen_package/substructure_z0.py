# Substructure class for analyzing satellite orbits and their evolution

import numpy as np
import matplotlib.pyplot as plt
import os
import helpers
import MC_metric
import onesat_metric
import onesat_orbit
import monte_carlo_orbits
import astropy
import sys
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.cosmology import units as cu
from profiles import Dekel, MN, EnergyAngMomGivenRpRa, Phi  # Import galaxy profile models

# Define cosmology model
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
u.add_enabled_units(cu)

# Add custom library path for SatGen
sys.path.insert(0, '/tigress/dropulic/SatGen/')

class Substructure:
    def __init__(self, df, tag, galaxy):
        """
        Initialize the Substructure class with satellite data, a tag, and a galaxy object.

        Parameters:
        - df: DataFrame containing satellite data.
        - tag: String tag for identifying the substructure analysis.
        - galaxy: Galaxy object containing host galaxy information.
        """
        self.df = df
        self.tag = tag
        self.galaxy = galaxy
        self.initialize_variables()

    def initialize_variables(self):
        """
        Initialize variables for storing results of the analysis.
        """
        self.id_sat_stream = []          # IDs for satellite streams
        self.lookback = []               # Lookback time array
        self.peri_all_reintegrate = []   # Reintegrated pericenter positions
        self.apo_all_reintegrate = []    # Reintegrated apocenter positions
        self.PL_over_PE_arr = []         # Ratio of PL to PE
        self.PE_list = []                # List of PE values
        self.PL_list = []                # List of PL values
        self.host_profiles = []          # Host galaxy profiles
        self.Norb_list = []              # Number of orbits at z=0
        self.sats_per_gal = [len(self.df)]  # Number of satellites per galaxy

    def handle_directory(self, file_path):
        """
        Create a directory if it doesn't exist.

        Parameters:
        - file_path: Path to the directory to be created.
        """
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            print("Directory created:", file_path)
        else:
            print("Directory already exists:", file_path)

    def load_or_integrate_mc(self, file_path, tag, sat_tag, integrate_fn, *args):
        """
        Load Monte Carlo orbit data from files if they exist, otherwise integrate orbits and save the results.

        Parameters:
        - file_path: Path to save/load the files.
        - tag: Identifier for the current analysis.
        - sat_tag: Satellite tag used for saving/loading files.
        - integrate_fn: Function for orbit integration.
        - *args: Arguments passed to the integration function.

        Returns:
        - data: Loaded or computed data.
        """
        # Define paths for different data files
        paths = [f"{file_path}{name}_{sat_tag}_{tag}.npy" for name in [
            'ca_mc', 'cp_mc', 'vels_peri_mc', 'vels_apo_mc', 'satdist_mc', 't_mc', 'tba_mc', 'host_total_profile', 'pos_vel'
        ]]
        
        # Load data if files exist, otherwise compute and save the data
        if all(os.path.exists(path) for path in paths):
            data = [np.load(path, allow_pickle=True) for path in paths]
            print("Loaded files")
        else:
            data = integrate_fn(*args)
            for path, d in zip(paths, data):
                np.save(path, np.array(d, dtype=object), allow_pickle=True)
            print("Saved new data")
        
        return data

    def preprocess_orbits(self, sat_i, idx_zacc, pericenter_locs1, all_peris1, apocenter_locs1, all_apos1, time_between_apocenters, allz_m1):
        """
        Preprocess the orbital data to filter pericenters and apocenters, and calculate the number of orbits at z=0.

        Parameters:
        - sat_i: Index of the current satellite.
        - idx_zacc: Index of the satellite's accretion redshift.
        - pericenter_locs1: List of pericenter locations.
        - all_peris1: List of pericenter coordinates.
        - apocenter_locs1: List of apocenter locations.
        - all_apos1: List of apocenter coordinates.
        - time_between_apocenters: List of times between apocenters.
        - allz_m1: Array of satellite masses over time.

        Returns:
        - Filtered and processed lists of pericenters and apocenters, and the number of orbits at z=0.
        """
        pericenter_locs_hold, apocenter_locs_hold = [], []
        all_peris_hold, all_apos_hold = [], []
        idx_mass_res_hold = 0
        entered = False
        peri_num = 0
        
        # Find the index where the mass drops below a threshold
        try:
            idx_mass_res = np.where((allz_m1[:][0:idx_zacc] <= 10**6))[0][-1]
            idx_mass_res_hold = 0
        except:
            idx_mass_res = 0
            idx_mass_res_hold = 0
            
        # Filter pericenters based on accretion redshift
        for elem_i, elem_j in zip(pericenter_locs1, all_peris1):
            if elem_i < idx_zacc:
                pericenter_locs_hold.append(elem_i)
                all_peris_hold.append(elem_j)
                if (elem_i > idx_mass_res) and not entered and idx_mass_res != 0:
                    idx_mass_res_hold = peri_num
                    entered = True
                peri_num += 1

        # Filter apocenters based on accretion redshift
        for elem_i, elem_j in zip(apocenter_locs1, all_apos1):
            if elem_i < idx_zacc:
                apocenter_locs_hold.append(elem_i)
                all_apos_hold.append(elem_j)

        # Ensure equal lengths of pericenter and apocenter lists
        if len(pericenter_locs_hold) > len(apocenter_locs_hold):
            pericenter_locs_hold = pericenter_locs_hold[:len(apocenter_locs_hold)]
        elif len(pericenter_locs_hold) < len(apocenter_locs_hold):
            apocenter_locs_hold = apocenter_locs_hold[:len(pericenter_locs_hold)]

        # Calculate the number of orbits at z=0
        Norb_z0 = len(pericenter_locs_hold)
        if len(pericenter_locs_hold) < 1:
            Norb_z0 = 1

        return pericenter_locs_hold, apocenter_locs_hold, all_peris_hold, all_apos_hold, Norb_z0

    def perform_calculations(self, sat_i):
        """
        Perform calculations and orbit integrations for a single satellite.

        Parameters:
        - sat_i: Index of the current satellite.
        """
        data = self.df.iloc[sat_i]  # Get satellite data from DataFrame

        # Extract necessary data fields
        coord_peris1 = data['coordinates_peri']
        vel_peris1 = data['velocities_peri']
        idx_zacc = data['idx_zaccs']
        coord1 = data['coordinates_hold']
        vel_arr1 = data['velocities_hold']
        allz_m1 = data['allz_m_s']
        allz_mstar1 = data['allz_mstar_s']
        sat_distances1 = data['sat_distances']
        pericenter_locs1 = data['pericenter_locs']
        apocenter_locs1 = data['apocenter_locs']
        all_peris1 = data['all_peri']
        all_apos1 = data['all_apo']
        time_between_apocenters = data['time_between_apocenters']
        redshift = data['redshift']
        dekel_host_coords_at = self.galaxy.host_coords_alltime
        vel_disp_z0 = data['velocity_dispersion']

        # Calculate energy and angular momentum for the satellite at the first pericenter
        coords_cart, vels_cart = helpers.rvcyltocart(coord_peris1[0], vel_peris1[0])
        E, L = helpers.EnergyAngMom(coords_cart, vels_cart, Phi(self.galaxy.get_host_profile(0), np.linalg.norm(coords_cart)), allz_m1[0])

        # Filter distances and calculate lookback time
        dists = sat_distances1[:idx_zacc]
        origin_mass = np.max(allz_mstar1[:idx_zacc][allz_mstar1[:idx_zacc] != -99])
        self.lookback.append(14 - cosmo.age(self.galaxy.redshift[idx_zacc] * cu.redshift) / u.Gyr)
        mstar_over_mstartot = np.log10(allz_mstar1[:idx_zacc][allz_mstar1[:idx_zacc] != -99] / origin_mass)[:idx_zacc]
        self.Norb_list.append(len(all_peris1))

        # Preprocess orbits and extract pericenters and apocenters
        pericenter_locs_hold, apocenter_locs_hold, all_peris_hold, all_apos_hold, Norb_z0 = self.preprocess_orbits(sat_i, idx_zacc, pericenter_locs1, all_peris1, apocenter_locs1, all_apos1, time_between_apocenters, allz_m1)

        # Handle Monte Carlo orbit integration and load necessary data
        ca_mc, cp_mc, vels_peri_mc, vels_apo_mc, satdist_mc, t_mc, tba_mc, host_total_profile, pos_vel , mod_mass_at, num_mc, ecc_est = self.handle_mc_files(data, sat_i, pericenter_locs_hold, apocenter_locs_hold, all_peris_hold, all_apos_hold, Norb_z0)

        # Calculate mean slopes for L-deltaPsi and E-tba metrics
        mean_slope_L_Psi, mean_slope_E_tba = MC_metric.detrivatives_metric(ca_mc, cp_mc, vels_peri_mc, vels_apo_mc, satdist_mc, t_mc, tba_mc, host_total_profile, mod_mass_at, E, L, num_mc, 0)
        print(mean_slope_L_Psi, mean_slope_E_tba)

        # Forward integration from z=0
        redshift_start = 0
        timetot, Nstep, tmax, pos_vel, min_loc, max_loc, satdist, t = onesat_orbit.integrate_orbit(self.galaxy, np.hstack([coord1, vel_arr1]), dekel_host_coords_at, mod_mass_at, redshift_start, time_between_apocenters, ecc_est)
        
        # Calculate rosette angles and reintegrate peri-apo properties
        half_period_locs_fwd, rosette_angle_fwd = MC_metric.calc_rosette_angle(max_loc, min_loc, pos_vel)
        rosette_angle_fwd_fit = helpers.get_huber_predictions(t[max_loc[0:len(rosette_angle_fwd)]], np.array(rosette_angle_fwd))
        all_apo, all_peri, ca, cp, tba, vels_peri, vels_apo = helpers.peri_apo_props_reintegrate(timetot, Nstep, tmax, pos_vel, min_loc, max_loc, satdist, t)
        
        # Determine if the satellite is a stream or shell
        id_ss_fwd, PE_fwd, PL_fwd, deltaPsiarr_fwd, mod_deltaPsiarr_fwd, Lsat_fwd, Esat_fwd, n_orb, Lmod_sat, Emod_sat, fit_PL_PE_fwd, M_Rp_arr_fwd = MC_metric.stream_or_shell_integrate(self.galaxy, mod_mass_at, satdist[min_loc], all_peri, all_apo, cp, ca, tba, dekel_host_coords_at, redshift_start, t + np.max(cosmo.age(self.galaxy.redshift[0:idx_zacc] * cu.redshift) / u.Gyr), max_loc, Norb_z0, vels_peri, min_loc, rosette_angle_fwd_fit, vel_disp_z0, vel_arr1, mean_slope_L_Psi, mean_slope_E_tba, forward=True)

        # Store results in class attributes
        if len(all_peri) > 0:
            self.peri_all_reintegrate.append(all_peri[0])
            self.apo_all_reintegrate.append(all_apo[0])
        else:
            self.peri_all_reintegrate.append(np.nan)
            self.apo_all_reintegrate.append(np.nan)
        
        if len(PL_fwd) > 0:
            self.PL_over_PE_arr.append(np.divide(PL_fwd[0], PE_fwd[0]))
            self.PL_list.append(PL_fwd[0])
            self.PE_list.append(PE_fwd[0])
        else:
            self.PL_over_PE_arr.append(np.nan)
            self.PL_list.append(np.nan)
            self.PE_list.append(np.nan)

        self.id_sat_stream.append(id_ss_fwd)

    def handle_mc_files(self, data, sat_i, pericenter_locs_hold, apocenter_locs_hold, all_peris_hold, all_apos_hold, Norb_z0):
        """
        Handle loading or computing Monte Carlo orbits and saving the data to files.

        Parameters:
        - data: Data for the current satellite.
        - sat_i: Index of the current satellite.
        - pericenter_locs_hold: Filtered list of pericenter locations.
        - apocenter_locs_hold: Filtered list of apocenter locations.
        - all_peris_hold: Filtered list of pericenter coordinates.
        - all_apos_hold: Filtered list of apocenter coordinates.
        - Norb_z0: Number of orbits at z=0.

        Returns:
        - Various arrays and values related to Monte Carlo orbit integration and the satellite's properties.
        """
        # Directory for saving files
        file_mc = f'/scratch/gpfs/dropulic/StreamGen/StreamGen_package/{self.tag}/'
        self.handle_directory(file_mc)

        # Extract satellite mass and eccentricity estimate
        zacc_mass = np.max(data.allz_m_s[:data.idx_zaccs], axis=0)
        mod_mass_at = data.allz_m_s
        ecc_est = 0.0
        if (len(all_apos_hold) > 1) and (len(all_peris_hold) > 1):
            ecc_est = (all_apos_hold[0] - all_peris_hold[0]) / (all_apos_hold[0] + all_peris_hold[0])

        # Generate a tag for the satellite
        str_mass = "{:.2e}".format(zacc_mass)
        sat_tag = 'zaccidx_' + str(data.idx_zaccs) + '_zaccmass_' + str_mass

        # Paths for saving and loading Monte Carlo files
        paths = [f"{file_mc}{name}_{sat_tag}_{self.tag}.npy" for name in [
            'ca_mc', 'cp_mc', 'vels_peri_mc', 'vels_apo_mc', 'satdist_mc', 't_mc', 'tba_mc', 'host_total_profile', 'pos_vel'
        ]]
        
        # Load or compute Monte Carlo data
        if all(os.path.exists(path) for path in paths):
            ca_mc, cp_mc, vels_peri_mc, vels_apo_mc, satdist_mc, t_mc, tba_mc, host_total_profile, pos_vel = [np.load(path, allow_pickle=True) for path in paths]
            host_total_profile = self.galaxy.get_host_profile(0)
            num_mc = len(ca_mc)
            print(num_mc)
            print("Loaded MC files")
        else:
            # Integrate Monte Carlo orbits if data is not available
            ca_mc, cp_mc, vels_peri_mc, vels_apo_mc, satdist_mc, t_mc, tba_mc, host_total_profile, pos_vel, num_mc = monte_carlo_orbits.integrate_mc_orbits(self.galaxy, np.hstack([data.coordinates_hold, data.velocities_hold]), data.velocity_dispersion, np.hstack([data.coordinates_hold, data.velocities_hold]), self.galaxy.host_coords_alltime, mod_mass_at, 0, data.time_between_apocenters, ecc_est, 20, period_est_correction=0)
            
            # Save computed data
            for path, data in zip(paths, [ca_mc, cp_mc, vels_peri_mc, vels_apo_mc, satdist_mc, t_mc, tba_mc, host_total_profile, pos_vel]):
                np.save(path, np.array(data, dtype=object), allow_pickle=True)
            print("Saved new MC data")

        return ca_mc, cp_mc, vels_peri_mc, vels_apo_mc, satdist_mc, t_mc, tba_mc, host_total_profile, pos_vel, mod_mass_at, num_mc, ecc_est

    def main_processing_loop(self):
        """
        Main loop for processing all satellites in the DataFrame.
        """
        for sat_i in range(6, 7):  # Limit processing to one satellite for now (change range as needed)
            self.perform_calculations(sat_i)
            print(self.id_sat_stream)
            print(self.PL_over_PE_arr)


