#Substructure class
import numpy as np
import matplotlib.pyplot as plt
import os
import helpers
import astropy
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
import astropy.units as u
import astropy.cosmology.units as cu
u.add_enabled_units(cu)

class Substructure:
    def __init__(self, df, tag, galaxy):
        self.df = df
        self.tag = tag
        self.galaxy = galaxy
        self.initialize_variables()

    def initialize_variables(self):
        self.id_sat_stream = []
        self.id_sat_stream_zp99 = []
        self.id_ss_SatGen = []
        self.lookback = []
        self.peri_all_reintegrate = []
        self.apo_all_reintegrate = []
        self.peri_all_reintegrate_zp99 = []
        self.apo_all_reintegrate_zp99 = []
        self.PL_over_PE_arr = []
        self.PL_over_PE_arr_zp99 = []
        self.PE_list_zp99 = []
        self.PL_list_zp99 = []
        self.PE_list = []
        self.PL_list = []
        self.host_profiles = []
        self.Norb_list = []
        self.real_zp99_TF_list = []
        self.sats_per_gal = [len(self.df)]

    def integrate_orbit_wrapper(self, coord, dekel_host_coords, mod_mass_at, start, apocenters, ecc, forward=True, correction=None, inLLcorner=False):
        timetot, Nstep, tmax, pos_vel, min_loc, max_loc, satdist, t = integrate_orbit(
            coord, dekel_host_coords, mod_mass_at, start, apocenters, ecc, forward, correction, inLLcorner
        )
        half_period_locs, rosette_angle = calc_rosette_angle(max_loc, min_loc, pos_vel)
        rosette_angle_fit = get_huber_predictons(t[max_loc[:len(rosette_angle)]], np.array(rosette_angle))
        all_apo, all_peri, ca, cp, tba, vels_peri, vels_apo = peri_apo_props_reintegrate(
            timetot, Nstep, tmax, pos_vel, min_loc, max_loc, satdist, t
        )
        return timetot, Nstep, tmax, pos_vel, min_loc, max_loc, satdist, t, half_period_locs, rosette_angle, rosette_angle_fit, all_apo, all_peri, ca, cp, tba, vels_peri, vels_apo

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

    def preprocess_orbits(self, sat_i, idx_zacc, pericenter_locs1, all_peris1, apocenter_locs1, all_apos1, time_between_apocenters,allz_m1):
        pericenter_locs_hold = []
        apocenter_locs_hold = []
        idx_mass_res_hold = 0
        idx_zp99_hold = 0
        entered = False
        entered_zp99 = False
        peri_num = 0
        try:
            idx_mass_res = np.where((allz_m1[:][0:idx_zacc] <= 10**(6)))[0][-1]
            idx_mass_res_hold = 0
        except:
            idx_mass_res = 0
            idx_mass_res_hold = 0

        for elem_i, elem_j, elem_k in zip(pericenter_locs1[sat_i], all_peris1[sat_i], time_between_apocenters[sat_i]):
            if elem_i < idx_zacc:
                pericenter_locs_hold.append(elem_i)
                all_peris_hold.append(elem_j)
                time_between_apo_hold.append(elem_k)
                if (elem_i > idx_mass_res) and not entered and idx_mass_res != 0:
                    idx_mass_res_hold = peri_num
                    entered = True
                if (elem_i > idx_zp99) and not entered_zp99 and idx_zp99 != 0:
                    idx_zp99_hold = peri_num
                    entered_zp99 = True
                peri_num += 1

        for elem_i, elem_j in zip(apocenter_locs1[sat_i], all_apos1[sat_i]):
            if elem_i < idx_zacc:
                apocenter_locs_hold.append(elem_i)
                all_apos_hold.append(elem_j)

        if len(pericenter_locs_hold) > len(apocenter_locs_hold):
            pericenter_locs_hold = pericenter_locs_hold[:len(apocenter_locs_hold)]
        elif len(pericenter_locs_hold) < len(apocenter_locs_hold):
            apocenter_locs_hold = apocenter_locs_hold[:len(pericenter_locs_hold)]
            
        Norb_zp99 = 1
        Norb_z0 = len(pericenter_locs_hold)
        if len(pericenter_locs_hold) < 1:
            Norb_z0 = 1

        try:
            idx_zp99 = pericenter_locs_hold[0]
            Norb_zp99 = len(pericenter_locs_hold) - 1
        except:
            idx_zp99 = 0
        real_zp99 = False
        try:
            real_idx_zp99 = np.where((np.divide(allz_m1[:idx_zacc], allz_m1[idx_zacc]) <= 1 / 300))[0][-1]
            if real_idx_zp99 > idx_zp99:
                idx_zp99 = real_idx_zp99
                real_zp99 = True
                try:
                    p_i = 0
                    for perii in pericenter_locs_hold[1:]:
                        p_i += 1
                        if idx_zp99 < perii:
                            idx_zp99 = perii
                            Norb_zp99 = len(pericenter_locs_hold) - p_i
                            break
                except:
                    print('idx_zp99 is not a pericenter')
        except:
            real_zp99 = False

        return pericenter_locs_hold, apocenter_locs_hold, idx_zp99, idx_zp99_hold, entered, entered_zp99, peri_num, Norb_zp99, Norb_z0, real_zp99


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
        E, L = helpers.EnergyAngMom(coords_cart, vels_cart, Phi(self.galaxy.get_host_profile(self, 0), np.linalg.norm(coords_cart)), allz_m1[0])
        #E, L = EnergyAngMomGivenRpRa(prof_list[host_in], all_peris1[0], all_apos1[0])

        dists = sat_distances1[:idx_zacc]
        origin_mass = np.max(allz_mstar1[:idx_zacc][allz_mstar1[:idx_zacc] != -99])
        self.lookback.append(14 - cosmo.age(redshift[idx_zacc] * cu.redshift) / u.Gyr)
        mstar_over_mstartot = np.log10(allz_mstar1[:idx_zacc][allz_mstar1[:idx_zacc] != -99] / origin_mass)[:idx_zacc]
        self.Norb_list.append(len(all_peris1))

        pericenter_locs_hold, apocenter_locs_hold, idx_zp99, idx_zp99_hold, entered, entered_zp99, peri_num, Norb_zp99, Norb_z0, real_zp99 = self.preprocess_orbits(sat_i, idx_zacc, pericenter_locs1, all_peris1, apocenter_locs1, all_apos1, time_between_apocenters,allz_m1)

        self.real_zp99_TF_list.append(real_zp99)

        ca_mc, cp_mc, vels_peri_mc, vels_apo_mc, satdist_mc, t_mc, tba_mc, host_total_profile, pos_vel, ca_mc_zp99, cp_mc_zp99, vels_peri_mc_zp99, vels_apo_mc_zp99, satdist_mc_zp99, t_mc_zp99, tba_mc_zp99, host_total_profile_zp99, pos_vel_zp99 = self.handle_mc_files(sat_i, real_zp99, pericenter_locs_hold, apocenter_locs_hold, idx_mass_res_hold, idx_zp99_hold, entered, entered_zp99, peri_num, Norb_z0, Norb_zp99)
        
        mean_slope_L_Psi, mean_slope_E_tba  = detrivatives_metric(ax13, ca_mc, cp_mc, vels_peri_mc, vels_apo_mc, satdist_mc, t_mc, tba_mc, host_total_profile, mod_mass_at, E, L, num_mc, 0)
        
        mean_slope_L_Psi_zp99, mean_slope_E_tba_zp99  = detrivatives_metric(ax20, ca_mc_zp99, cp_mc_zp99, vels_peri_mc_zp99, vels_apo_mc_zp99, satdist_mc_zp99, t_mc_zp99, tba_mc_zp99, host_total_profile_zp99, mod_mass_at, E_zp99, L_zp99, num_mc, idx_zp99)

        if  idx_zp99 > 0:
            #forwards from idx_zp99
            timetot, Nstep, tmax, pos_vel_zp99, min_loc, max_loc, satdist, t = integrate_orbit(coord1[sat_i], dekel_host_coords_at,mod_mass_at, idx_zp99, time_between_apocenters[sat_i], ecc_est_zp99,period_est_correction = idx_zp99_hold, forward = True, inLLcorner = inLLcorner_sat)
            print('time_between_apocenters[idx_zp99_hold]',time_between_apocenters[idx_zp99_hold])
            half_period_locs_fwd, rosette_angle_fwd = calc_rosette_angle(max_loc, min_loc, pos_vel)
            rosette_angle_fwd_fit = get_huber_predictons(t[max_loc[0:len(rosette_angle_fwd)]], np.array(rosette_angle_fwd))
            all_apo, all_peri, ca, cp, tba, vels_peri, vels_apo = peri_apo_props_reintegrate(timetot, Nstep, tmax, pos_vel_zp99, min_loc, max_loc, satdist, t)
            id_ss_fwd_zp99, PE_fwd_zp99, PL_fwd_zp99, deltaPsiarr_fwd_zp99, mod_deltaPsiarr_fwd_zp99, Lsat_fwd_zp99, Esat_fwd_zp99, n_orb_zp99 ,Lmod_sat_zp99, Emod_sat_zp99, fit_PL_PE_fwd_zp99, M_Rp_arr_fwd_zp99 = stream_or_shell_integrate(mod_mass_at, satdist[min_loc], all_peri, all_apo, cp, ca, tba, dekel_host_coords_at, idx_zp99, t+ np.max(cosmo.age(redshift[idx_zp99:idx_zacc]*cu.redshift)/u.Gyr), max_loc, ax2,Norb_zp99,vels_peri ,min_loc, rosette_angle_fwd_fit, vel_disp_z0[sat_i], vel_arr1[sat_i][0], mean_slope_L_Psi_zp99, mean_slope_E_tba_zp99, forward = True)
            t = t + np.max(cosmo.age(redshift[idx_zp99:idx_zacc]*cu.redshift)/u.Gyr) 
            if len(all_peri > 0):
                peri_all_reintegrate_zp99.append(all_peri[0])
                apo_all_reintegrate_zp99.append(all_apo[0])
            else:
                peri_all_reintegrate_zp99.append(np.nan)
                apo_all_reintegrate_zp99.append(np.nan)
            try:
                if len(PL_fwd > 0):
                    PL_over_PE_arr_zp99.append(np.divide(PL_fwd_zp99[0], PE_fwd_zp99[0]))
                    PL_list_zp99.append(PL_fwd_zp99[0])
                    PE_list_zp99.append(PE_fwd_zp99[0])
                else:
                    PL_over_PE_arr_zp99.append(np.nan)  
                    PL_list_zp99.append(np.nan)
                    PE_list_zp99.append(np.nan)
            except:
                PL_over_PE_arr_zp99.append(np.nan)
                PL_list_zp99.append(np.nan)
                PE_list_zp99.append(np.nan)
                                         
            id_sat_stream_zp99.append(id_ss_fwd_zp99)
        else:
            peri_all_reintegrate_zp99.append(np.nan)
            apo_all_reintegrate_zp99.append(np.nan)
            id_sat_stream_zp99.append(np.nan)
            PL_over_PE_arr_zp99.append(np.nan)
            PL_list_zp99.append(np.nan)
            PE_list_zp99.append(np.nan)

        #forwards from z = 0
        timetot, Nstep, tmax, pos_vel, min_loc, max_loc, satdist, t = integrate_orbit(coord1[sat_i], dekel_host_coords_at,mod_mass_at, redshift_start, time_between_apocenters[sat_i], ecc_est, forward = True, inLLcorner = inLLcorner_sat)
        half_period_locs_fwd, rosette_angle_fwd = calc_rosette_angle(max_loc, min_loc, pos_vel)
        rosette_angle_fwd_fit = get_huber_predictons(t[max_loc[0:len(rosette_angle_fwd)]], np.array(rosette_angle_fwd))
        all_apo, all_peri, ca, cp, tba, vels_peri, vels_apo = peri_apo_props_reintegrate(timetot, Nstep, tmax, pos_vel, min_loc, max_loc, satdist, t)
        id_ss_fwd, PE_fwd, PL_fwd, deltaPsiarr_fwd, mod_deltaPsiarr_fwd, Lsat_fwd, Esat_fwd, n_orb ,Lmod_sat, Emod_sat, fit_PL_PE_fwd, M_Rp_arr_fwd= stream_or_shell_integrate(mod_mass_at, satdist[min_loc], all_peri, all_apo, cp, ca, tba, dekel_host_coords_at, redshift_start, t+ np.max(cosmo.age(redshift[0:idx_zacc]*cu.redshift)/u.Gyr), max_loc, ax2,Norb_z0,vels_peri ,min_loc, rosette_angle_fwd_fit, vel_disp_z0[sat_i], vel_arr1[sat_i][0], mean_slope_L_Psi, mean_slope_E_tba, forward = True)
        t = t + np.max(cosmo.age(redshift[0:idx_zacc]*cu.redshift)/u.Gyr) 
        
        if len(all_peri > 0):
            peri_all_reintegrate.append(all_peri[0])
            apo_all_reintegrate.append(all_apo[0])
        else:
            peri_all_reintegrate.append(np.nan)
            apo_all_reintegrate.append(np.nan)
        if len(PL_fwd > 0):
            PL_over_PE_arr.append(np.divide(PL_fwd[0], PE_fwd[0]))
            PL_list.append(PL_fwd[0])
            PE_list.append(PE_fwd[0])
        else:
            PL_over_PE_arr.append(np.nan)
            PL_list.append(np.nan)
            PE_list.append(np.nan)

        print('PE_fwd',PE_fwd)
        print('PL_fwd',PL_fwd)
        print(id_ss_fwd)

        id_sat_stream.append(id_ss_fwd)
        
        self.id_sat_stream.append(id_ss_fwd)
        self.id_sat_stream_zp99.append(id_sat_stream_zp99)

        return
        
    def handle_directory(self, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            print("Directory created:", file_path)
        else:
            print("Directory already exists:", file_path)

    def handle_mc_files(self, sat_i, real_zp99, pericenter_locs_hold, apocenter_locs_hold, idx_mass_res_hold, idx_zp99_hold, entered, entered_zp99, peri_num, Norb_z0, Norb_zp99):
        # Directory for saving files
        file_mc = f'/scratch/gpfs/dropulic/StreamGen/StreamGen_package/{self.tag}/'
        self.handle_directory(file_mc)
        
        
        zacc_mass = np.max(allz_m1[sat_i,:idx_zacc], axis = 0)
        mod_mass_at = allz_m1[sat_i,:]

        ecc_est = 0.0
        ecc_est_zp99 = 0.0
        if (len(all_apos_hold) > 1) and (len(all_peris_hold) > 1):
            ecc_est = (all_apos_hold[0] - all_peris_hold[0])/(all_apos_hold[0] + all_peris_hold[0])
            ecc_est_zp99 = (all_apos_hold[idx_zp99_hold] - all_peris_hold[idx_zp99_hold])/(all_apos_hold[idx_zp99_hold] + all_peris_hold[idx_zp99_hold])

        str_mass = "{:.2e}".format(zacc_mass)
        sat_tag = 'zaccidx_'+str(idx_zacc)+'_zaccmass_'+str_mass

        # Paths for saving and loading MC files
        paths = [f"{file_mc}{name}_{sat_tag}_{self.tag}.npy" for name in [
            'ca_mc', 'cp_mc', 'vels_peri_mc', 'vels_apo_mc', 'satdist_mc', 't_mc', 'tba_mc', 'host_total_profile', 'pos_vel'
        ]]

        if all(os.path.exists(path) for path in paths):
            ca_mc, cp_mc, vels_peri_mc, vels_apo_mc, satdist_mc, t_mc, tba_mc, host_total_profile, pos_vel = [np.load(path, allow_pickle=True) for path in paths]
            host_total_profile = self.galaxy.get_host_profile(self, 0)
            print("Loaded MC files")
        else:
            # Integrate orbits and save results
            ca_mc, cp_mc, vels_peri_mc, vels_apo_mc, satdist_mc, t_mc, tba_mc, host_total_profile, pos_vel = integrate_mc_orbits(
                coord1, vel_disp_z0, coord1, dekel_host_coords_at, vel_disp_z0, 0, time_between_apocenters[sat_i], Norb_z0, num_mc=300, forward=True, correction=idx_mass_res_hold, inLLcorner=False
            )
            for path, data in zip(paths, [ca_mc, cp_mc, vels_peri_mc, vels_apo_mc, satdist_mc, t_mc, tba_mc, host_total_profile, pos_vel]):
                np.save(path, np.array(data, dtype=object), allow_pickle=True)
            print("Saved new MC data")

        # Repeat for zp99 if real_zp99 is True
        if real_zp99:
            paths_zp99 = [f"{file_mc}{name}_{sat_tag}_zp99_{self.tag}.npy" for name in [
                'ca_mc', 'cp_mc', 'vels_peri_mc', 'vels_apo_mc', 'satdist_mc', 't_mc', 'tba_mc', 'host_total_profile', 'pos_vel'
            ]]

            if all(os.path.exists(path) for path in paths_zp99):
                ca_mc_zp99, cp_mc_zp99, vels_peri_mc_zp99, vels_apo_mc_zp99, satdist_mc_zp99, t_mc_zp99, tba_mc_zp99, host_total_profile_zp99, pos_vel_zp99 = [np.load(path, allow_pickle=True) for path in paths_zp99]
                print("Loaded MC zp99 files")
                host_total_profile_zp99 = self.galaxy.get_host_profile(self, idx_zp99)
            else:
                ca_mc_zp99, cp_mc_zp99, vels_peri_mc_zp99, vels_apo_mc_zp99, satdist_mc_zp99, t_mc_zp99, tba_mc_zp99, host_total_profile_zp99, pos_vel_zp99 = integrate_mc_orbits(
                    coord1, vel_disp_z0, coord1, dekel_host_coords_at, vel_disp_z0, idx_zp99_hold, time_between_apocenters[sat_i], Norb_zp99, num_mc=300, forward=True, correction=idx_mass_res_hold, inLLcorner=False
                )
                for path, data in zip(paths_zp99, [ca_mc_zp99, cp_mc_zp99, vels_peri_mc_zp99, vels_apo_mc_zp99, satdist_mc_zp99, t_mc_zp99, tba_mc_zp99, host_total_profile_zp99, pos_vel_zp99]):
                    np.save(path, np.array(data, dtype=object), allow_pickle=True)
                print("Saved new MC zp99 data")
        return ca_mc, cp_mc, vels_peri_mc, vels_apo_mc, satdist_mc, t_mc, tba_mc, host_total_profile, pos_vel, ca_mc_zp99, cp_mc_zp99, vels_peri_mc_zp99, vels_apo_mc_zp99, satdist_mc_zp99, t_mc_zp99, tba_mc_zp99, host_total_profile_zp99, pos_vel_zp99

    def main_processing_loop(self):
        for sat_i in range(len(self.df)):
            self.perform_calculations(sat_i)