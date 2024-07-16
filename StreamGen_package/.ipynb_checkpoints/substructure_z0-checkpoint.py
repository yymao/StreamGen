#Substructure class
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
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
import astropy.units as u
import astropy.cosmology.units as cu
u.add_enabled_units(cu)
sys.path.insert(0, '/tigress/dropulic/SatGen/')
from profiles import Dekel, MN, EnergyAngMomGivenRpRa, Phi

class Substructure:
    def __init__(self, df, tag, galaxy):
        self.df = df
        self.tag = tag
        self.galaxy = galaxy
        self.initialize_variables()

    def initialize_variables(self):
        self.id_sat_stream = []
        self.lookback = []
        self.peri_all_reintegrate = []
        self.apo_all_reintegrate = []
        self.PL_over_PE_arr = []
        self.PE_list = []
        self.PL_list = []
        self.host_profiles = []
        self.Norb_list = []
        self.sats_per_gal = [len(self.df)]

    def handle_directory(self, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            print("Directory created:", file_path)
        else:
            print("Directory already exists:", file_path)

    def load_or_integrate_mc(self, file_path, tag, sat_tag, integrate_fn, *args):
        paths = [f"{file_path}{name}_{sat_tag}_{tag}.npy" for name in [
            'ca_mc', 'cp_mc', 'vels_peri_mc', 'vels_apo_mc', 'satdist_mc', 't_mc', 'tba_mc', 'host_total_profile', 'pos_vel'
        ]]
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
        pericenter_locs_hold = []
        apocenter_locs_hold = []
        all_peris_hold = []
        all_apos_hold = []
        idx_mass_res_hold = 0
        entered = False
        peri_num = 0
        try:
            idx_mass_res = np.where((allz_m1[:][0:idx_zacc] <= 10**(6)))[0][-1]
            idx_mass_res_hold = 0
        except:
            idx_mass_res = 0
            idx_mass_res_hold = 0
            

        for elem_i, elem_j in zip(pericenter_locs1, all_peris1):
            if elem_i < idx_zacc:
                pericenter_locs_hold.append(elem_i)
                all_peris_hold.append(elem_j)
                if (elem_i > idx_mass_res) and not entered and idx_mass_res != 0:
                    idx_mass_res_hold = peri_num
                    entered = True
                peri_num += 1

        for elem_i, elem_j in zip(apocenter_locs1, all_apos1):
            if elem_i < idx_zacc:
                apocenter_locs_hold.append(elem_i)
                all_apos_hold.append(elem_j)

        if len(pericenter_locs_hold) > len(apocenter_locs_hold):
            pericenter_locs_hold = pericenter_locs_hold[:len(apocenter_locs_hold)]
        elif len(pericenter_locs_hold) < len(apocenter_locs_hold):
            apocenter_locs_hold = apocenter_locs_hold[:len(pericenter_locs_hold)]

        Norb_z0 = len(pericenter_locs_hold)
        if len(pericenter_locs_hold) < 1:
            Norb_z0 = 1

        return pericenter_locs_hold, apocenter_locs_hold, all_peris_hold, all_apos_hold, Norb_z0

    def perform_calculations(self, sat_i):
        data = self.df.iloc[sat_i]

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
        
        coords_cart, vels_cart = helpers.rvcyltocart(coord_peris1[0], vel_peris1[0])
        E, L = helpers.EnergyAngMom(coords_cart, vels_cart, Phi(self.galaxy.get_host_profile(0), np.linalg.norm(coords_cart)), allz_m1[0])

        dists = sat_distances1[:idx_zacc]
        origin_mass = np.max(allz_mstar1[:idx_zacc][allz_mstar1[:idx_zacc] != -99])
        self.lookback.append(14 - cosmo.age(self.galaxy.redshift[idx_zacc] * cu.redshift) / u.Gyr)
        mstar_over_mstartot = np.log10(allz_mstar1[:idx_zacc][allz_mstar1[:idx_zacc] != -99] / origin_mass)[:idx_zacc]
        self.Norb_list.append(len(all_peris1))

        pericenter_locs_hold, apocenter_locs_hold, all_peris_hold, all_apos_hold, Norb_z0 = self.preprocess_orbits(sat_i, idx_zacc, pericenter_locs1, all_peris1, apocenter_locs1, all_apos1, time_between_apocenters, allz_m1)

        ca_mc, cp_mc, vels_peri_mc, vels_apo_mc, satdist_mc, t_mc, tba_mc, host_total_profile, pos_vel , mod_mass_at, num_mc, ecc_est = self.handle_mc_files(data, sat_i, pericenter_locs_hold, apocenter_locs_hold, all_peris_hold, all_apos_hold, Norb_z0)

        mean_slope_L_Psi, mean_slope_E_tba = MC_metric.detrivatives_metric(ca_mc, cp_mc, vels_peri_mc, vels_apo_mc, satdist_mc, t_mc, tba_mc, host_total_profile, mod_mass_at, E, L, num_mc, 0)
        print(mean_slope_L_Psi, mean_slope_E_tba)

        # Forwards from z = 0
        redshift_start = 0
        timetot, Nstep, tmax, pos_vel, min_loc, max_loc, satdist, t = onesat_orbit.integrate_orbit(self.galaxy,np.hstack([coord1,vel_arr1]), dekel_host_coords_at, mod_mass_at, redshift_start, time_between_apocenters, ecc_est)
        half_period_locs_fwd, rosette_angle_fwd = MC_metric.calc_rosette_angle(max_loc, min_loc, pos_vel)
        rosette_angle_fwd_fit = helpers.get_huber_predictons(t[max_loc[0:len(rosette_angle_fwd)]], np.array(rosette_angle_fwd))
        all_apo, all_peri, ca, cp, tba, vels_peri, vels_apo = helpers.peri_apo_props_reintegrate(timetot, Nstep, tmax, pos_vel, min_loc, max_loc, satdist, t)
        id_ss_fwd, PE_fwd, PL_fwd, deltaPsiarr_fwd, mod_deltaPsiarr_fwd, Lsat_fwd, Esat_fwd, n_orb, Lmod_sat, Emod_sat, fit_PL_PE_fwd, M_Rp_arr_fwd = MC_metric.stream_or_shell_integrate(self.galaxy,mod_mass_at, satdist[min_loc], all_peri, all_apo, cp, ca, tba, dekel_host_coords_at, redshift_start, t + np.max(cosmo.age(self.galaxy.redshift[0:idx_zacc] * cu.redshift) / u.Gyr), max_loc,Norb_z0, vels_peri, min_loc, rosette_angle_fwd_fit, vel_disp_z0, vel_arr1, mean_slope_L_Psi, mean_slope_E_tba, forward=True)

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

    def handle_mc_files(self, data, sat_i, pericenter_locs_hold, apocenter_locs_hold,all_peris_hold, all_apos_hold, Norb_z0):
        # Directory for saving files
        file_mc = f'/scratch/gpfs/dropulic/StreamGen/StreamGen_package/{self.tag}/'
        self.handle_directory(file_mc)

        zacc_mass = np.max(data.allz_m_s[:data.idx_zaccs], axis=0)
        mod_mass_at = data.allz_m_s

        ecc_est = 0.0
        if (len(all_apos_hold) > 1) and (len(all_peris_hold) > 1):
            ecc_est = (all_apos_hold[0] - all_peris_hold[0]) / (all_apos_hold[0] + all_peris_hold[0])

        str_mass = "{:.2e}".format(zacc_mass)
        sat_tag = 'zaccidx_' + str(data.idx_zaccs) + '_zaccmass_' + str_mass

        # Paths for saving and loading MC files
        paths = [f"{file_mc}{name}_{sat_tag}_{self.tag}.npy" for name in [
            'ca_mc', 'cp_mc', 'vels_peri_mc', 'vels_apo_mc', 'satdist_mc', 't_mc', 'tba_mc', 'host_total_profile', 'pos_vel'
        ]]
        
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

        if all(os.path.exists(path) for path in paths):
            ca_mc, cp_mc, vels_peri_mc, vels_apo_mc, satdist_mc, t_mc, tba_mc, host_total_profile, pos_vel = [np.load(path, allow_pickle=True) for path in paths]
            host_total_profile = self.galaxy.get_host_profile(0)
            num_mc = len(ca_mc)
            print(num_mc)
            print("Loaded MC files")
        else:
            # Integrate orbits and save results
            
            ca_mc, cp_mc, vels_peri_mc, vels_apo_mc, satdist_mc, t_mc, tba_mc, host_total_profile, pos_vel, num_mc = monte_carlo_orbits.integrate_mc_orbits(self.galaxy,
                np.hstack([data.coordinates_hold,data.velocities_hold]), data.velocity_dispersion, np.hstack([data.coordinates_hold,data.velocities_hold]), self.galaxy.host_coords_alltime,mod_mass_at,  0, data.time_between_apocenters, ecc_est, 20, period_est_correction = 0, forward = True,inLLcorner = False
            )
            for path, data in zip(paths, [ca_mc, cp_mc, vels_peri_mc, vels_apo_mc, satdist_mc, t_mc, tba_mc, host_total_profile, pos_vel]):
                np.save(path, np.array(data, dtype=object), allow_pickle=True)
            print("Saved new MC data")

        return ca_mc, cp_mc, vels_peri_mc, vels_apo_mc, satdist_mc, t_mc, tba_mc, host_total_profile, pos_vel, mod_mass_at, num_mc, ecc_est

    def main_processing_loop(self):
        for sat_i in range(6,7):#len(self.df)
            self.perform_calculations(sat_i)
            print(self.id_sat_stream)
            print(self.PL_over_PE_arr)

