import numpy as np
import helpers
import sys
sys.path.insert(0, '/tigress/dropulic/SatGen/')
from profiles import Dekel, MN, EnergyAngMomGivenRpRa, Phi
import pandas as pd
import scipy
import onesat_orbit
from galhalo import Reff


def calc_E_L_Tr_dPsi_mc(ca_list, cp_list, vels_peri_list, vels_apo_list, satdist_list, t_list,tba_list,total_profile, m_sat):
    #not sure where this should go in the code because I need the profile to calculate the E & L
    # also need the max_loc
    #going to calculate these values for each stream, then select the ones closest to the time of z = 0
    ca_list = ca_list
    cp_list = cp_list
    vels_peri_list = vels_peri_list
    vels_apo_list = vels_apo_list
    E_arr = []
    L_arr = []
    deltaPsi_arr = []
    
    for ca, cp, va, vp in zip(ca_list,cp_list,vels_apo_list,vels_peri_list ):
        z_a = ca[2]
        x_a = ca[0] * np.cos(ca[1])
        y_a = ca[0] * np.sin(ca[1])
        
        z_p = cp[2]
        x_p = cp[0] * np.cos(cp[1])
        y_p = cp[0] * np.sin(cp[1])
        
        satdist_a = np.linalg.norm(np.array([x_a,y_a,z_a]))
        satdist_p = np.linalg.norm(np.array([x_p,y_p,z_p]))
        coords_cart, vels_cart = helpers.rvcyltocart(cp, vp)
        #print('peri', satdist_p)
        E,L = helpers.EnergyAngMom(coords_cart, vels_cart, Phi(total_profile,satdist_p) , m_sat)
        E_arr.append(E)
        L_arr.append(L)
    range_low = 0
    range_high = len(E_arr)-2
    skip = 1

    for i in range(range_low, range_high, skip):
        #if forward == True:
        increment = i + 1
        increment_2 = i + 2

        z_0 = ca_list[i][2]
        x_0 = ca_list[i][0] * np.cos(ca_list[i][1])
        y_0 = ca_list[i][0] * np.sin(ca_list[i][1])

        z_1 = ca_list[increment][2]
        x_1 = ca_list[increment][0] * np.cos(ca_list[increment][1])
        y_1 = ca_list[increment][0] * np.sin(ca_list[increment][1])

        coord_apo_0 = np.array([x_0,y_0,z_0])
        coord_apo_1 = np.array([x_1,y_1,z_1])


        deltaPsi = np.arccos(np.dot(coord_apo_0, coord_apo_1)/(np.linalg.norm(coord_apo_0)*np.linalg.norm(coord_apo_1)))#angle between successive apocenters ##wait not sure this gives me the right angle in cylindrical coordiantes ###
        deltaPsi_arr.append(deltaPsi)
    
    return  E_arr, L_arr, deltaPsi_arr, tba_list

def calc_rosette_angle(max_loc, min_loc, pos_vel):
    minmax_loc_diff = np.abs(np.subtract(max_loc,min_loc))
    quarter_period_timestep = np.round(minmax_loc_diff/2)
    half_period_locs = []
    rosette_angle_arr = []
    for max_loc_i, mm_loc_diff_i in zip(max_loc, quarter_period_timestep):
        hf_pd_loc_i = np.array([int(max_loc_i + mm_loc_diff_i),int(max_loc_i - mm_loc_diff_i )])
        half_period_locs.append(hf_pd_loc_i)
        
        if (hf_pd_loc_i[0] > 0) and (hf_pd_loc_i[1] > 0) and (hf_pd_loc_i[1] < len(pos_vel[:,0]))and (hf_pd_loc_i[0] < len(pos_vel[:,0])):
        
            coord_rosette_0, vels_rosette_0 = helpers.rvcyltocart(pos_vel[hf_pd_loc_i[0],:3],pos_vel[hf_pd_loc_i[0],3:])
            coord_rosette_1, vels_rosette_1 = helpers.rvcyltocart(pos_vel[hf_pd_loc_i[1],:3],pos_vel[hf_pd_loc_i[1],3:])
            rosette_angle = np.arccos(np.dot(coord_rosette_0, coord_rosette_1)/(np.linalg.norm(coord_rosette_0)*np.linalg.norm(coord_rosette_1)))
            rosette_angle_arr.append(rosette_angle)
        else:
            rosette_angle_arr.append(np.inf)
    return half_period_locs,rosette_angle_arr

def detrivatives_metric(ca_mc, cp_mc, vels_peri_mc, vels_apo_mc, satdist_mc, t_mc, tba_mc, host_total_profile, mod_mass_at, E_sat, L_sat, num_mc, redshift):
        E_z0 = []
        L_z0 = []
        tba_z0 = []
        deltaPsi_z0 = []
        for mc_i in range(num_mc):#was should have been len(ca_mc)

            E_arr_mc, L_arr_mc, deltaPsi_arr_mc, tba_list_mc = calc_E_L_Tr_dPsi_mc(ca_mc[mc_i], cp_mc[mc_i], vels_peri_mc[mc_i], vels_apo_mc[mc_i], satdist_mc[mc_i], t_mc[mc_i],tba_mc[mc_i],host_total_profile, mod_mass_at[0])
            #print('E_arr_mc',  E_arr_mc)
            try:
                E_z0.append(E_arr_mc[0])
                L_z0.append(L_arr_mc[0])
                tba_z0.append(tba_list_mc[0])
                deltaPsi_z0.append(np.pi*2 -deltaPsi_arr_mc[0])
            except:
                print("except!")
        
        df_MC = pd.DataFrame()
        df_MC['E'] = E_z0
        df_MC['L'] = L_z0
        df_MC['tba'] = tba_z0
        df_MC['deltaPsi'] = deltaPsi_z0
        hist_E, bin_edges_E = np.histogram(df_MC.E, bins = 'auto')
        hist_L, bin_edges_L = np.histogram(df_MC.L, bins = 'auto')
        bin_means_E, bin_edges_E, binnumber_E = scipy.stats.binned_statistic(df_MC.E, df_MC.E, statistic='count', bins=bin_edges_E)
        bin_means_L, bin_edges_L, binnumber_L = scipy.stats.binned_statistic(df_MC.L, df_MC.L, statistic='count', bins=bin_edges_L)
        E = E_sat
        L = L_sat
        bin_no_LdeltaPsi = []
        bin_no_Etba = []
        for bin_edgeE in range(0,len(bin_edges_E)-1):
            bin_no_LdeltaPsi = bin_edgeE
            if (bin_edges_E[bin_edgeE] <= E) and (E < bin_edges_E[bin_edgeE + 1]):
                print('upper_bin_edge', bin_edges_E[bin_edgeE+1])
                print('bin_no', bin_edgeE)
                break;
            if E < bin_edges_E[0]:
                break;
                
        for bin_edgeL in range(0,len(bin_edges_L)-1):
            bin_no_Etba = bin_edgeL
            if (bin_edges_L[bin_edgeL] <= L) and (L < bin_edges_L[bin_edgeL + 1]):
                print('upper_bin_edge', bin_edges_L[bin_edgeL+1])
                print('bin_no', bin_edgeL)
                break;
            if L < bin_edges_L[0]:
                break;
                
        #try:
        bin_means_E, bin_edges_E, binnumber_E = scipy.stats.binned_statistic(df_MC.E, df_MC.E, statistic='count', bins=bin_edges_E)
        bin_means_L, bin_edges_L, binnumber_L = scipy.stats.binned_statistic(df_MC.L, df_MC.L, statistic='count', bins=bin_edges_L)
        df_MC['binnumber_E'] = binnumber_E
        df_MC['binnumber_L'] = binnumber_L
        fit_L_Psi_arr = []
        fit_E_tba_arr = []
        from matplotlib.pyplot import cm
        color = cm.tab20(np.linspace(0, 1, np.max(binnumber_E)+1))
        for bin_num_i in range(np.min(binnumber_E), np.max(binnumber_E)+1):
            df_MC_hold = df_MC.loc[df_MC.binnumber_E == bin_num_i].reset_index(drop = True)
            if len(df_MC_hold)> 1:
                fit_L = np.polyfit(df_MC_hold.L,df_MC_hold.deltaPsi, 1)
                line_L = np.poly1d(fit_L)
                fit_L_Psi_arr.append(fit_L[0])
            else:
                fit_L_Psi_arr.append(np.nan)

        for bin_num_i in range(np.min(binnumber_L), np.max(binnumber_L)+1):
            df_MC_hold = df_MC.loc[df_MC.binnumber_L == bin_num_i].reset_index(drop = True)
            if len(df_MC_hold)> 1:
                fit_E = np.polyfit(df_MC_hold.E,df_MC_hold.tba, 1)
                line_E = np.poly1d(fit_E)


                fit_E_tba_arr.append(fit_E[0])
            else:
                fit_E_tba_arr.append(np.nan)

        mean_slope_L_Psi = fit_L_Psi_arr[bin_no_LdeltaPsi]
        mean_slope_E_tba = fit_E_tba_arr[bin_no_Etba]
        i=0
        print(np.isnan(mean_slope_L_Psi))
        while np.isnan(mean_slope_L_Psi) == True:
            if bin_no_LdeltaPsi > len(fit_L_Psi_arr)/2:
                mean_slope_L_Psi = fit_L_Psi_arr[bin_no_LdeltaPsi-i]
                if bin_no_LdeltaPsi-i ==0: break
            else:
                mean_slope_L_Psi = fit_L_Psi_arr[bin_no_LdeltaPsi+i]
                if bin_no_LdeltaPsi+i == len(fit_L_Psi_arr)-1: break
            
            i +=1
        i==0      
        while np.isnan(mean_slope_E_tba) == True:
            if bin_no_Etba > len(fit_E_tba_arr)/2:
                mean_slope_E_tba = fit_E_tba_arr[bin_no_Etba-i]
                if bin_no_Etba-i ==0: break
            else:
                mean_slope_E_tba = fit_L_Psi_arr[bin_no_Etba+i]
                if bin_no_Etba+i == len(fit_E_tba_arr)-1: break
            i +=1

            
        return mean_slope_L_Psi, mean_slope_E_tba

def stream_or_shell_integrate(galaxy,m_sat, sat_host_dist, peri_arr, apo_arr, coord_peri, coord_apo, time_between_apo, alltime_hostcoords, redshift_id, t, max_loc, N_orb_from_back , vel_peri, min_loc, rosette_angle_arr,sigma_vel, vel_z0, dPsidL , dtbadE , forward = True):
    print('in stream_or_shell_integrate')
    what_is_it = 0

    ###time_between_apo = np.abs(time_between_apo)
        
        
    E_sat = []
    L_sat = []
    l_s = []
    deltaPsi_arr = []
    mod_deltaPsi_arr = []
    mod_t_apo_arr = []
    d_deltaPsi_dL_Eorb_arr = []
    Psi_L_arr = []
    Psi_E_arr = []
    N_orb = []
    mod_E_arr = []
    mod_L_arr = []
    predictions = []
    
    #if forward == True:
    N_orb =  N_orb_from_back
    #else:
    #    N_orb = 0
    
    
    #print('peri_arr', peri_arr)
    #print('apo_arr', apo_arr)
    #print(sat_host_dist)
    M_Rp_arr = []
    ii = 0
    if (len(peri_arr) >= 3):
        
        for rp, ra, sathd in zip(peri_arr, apo_arr,sat_host_dist):
            host_profile = Dekel(alltime_hostcoords[0,redshift_id],alltime_hostcoords[1,redshift_id],alltime_hostcoords[2,redshift_id],alltime_hostcoords[3,redshift_id],z=0.)
            r_half_host = Reff(alltime_hostcoords[4,redshift_id],host_profile.ch)
            disk_scale_length_host = 0.766421/(1.+1./galaxy.flattening) * r_half_host
            disk_scale_height_host = disk_scale_length_host / galaxy.flattening
            #define the host disk profile
            host_disk_profile = MN(alltime_hostcoords[0,redshift_id]*galaxy.fd,disk_scale_length_host,disk_scale_height_host )
            host_stellar_disk_profile = MN(alltime_hostcoords[5,redshift_id]*galaxy.fd,disk_scale_length_host,disk_scale_height_host )
            total_profile = [host_profile, host_disk_profile]
            s = (m_sat[redshift_id]/(3*(host_profile.M(sathd)+host_disk_profile.M(sathd))))**(1/3) #also an issue that the mass keeps decreasing below the mass resolution limit, for some sats, but does not get updated here
            M_Rp_arr.append((host_profile.M(sathd)+host_disk_profile.M(sathd)))
            coords_cart, vels_cart = helpers.rvcyltocart(coord_peri[ii], vel_peri[ii])
            E_new, L_new = helpers.EnergyAngMom(coords_cart, vels_cart, Phi(total_profile,sathd) , m_sat[redshift_id]) #looks good
            ii = ii+1
            #E,L = EnergyAngMomGivenRpRa(total_profile,rp,ra)

            
            E_sat.append(E_new)
            L_sat.append(L_new)

        #if (np.abs(np.diff(np.array(E_sat))[0]) < 1) or (np.abs(np.diff(np.array(L_sat))[0]) < 1):

        #    fit_E = np.polyfit(t[max_loc[0:len(np.array(E_sat))]], np.array(E_sat), 1)
        #    fit_L = np.polyfit(t[max_loc[0:len(np.array(L_sat))]], np.array(L_sat), 1)
        #    mod_E_arr = fit_E[0]*t[max_loc[0:len(np.array(E_sat))]]+fit_E[1]
        #    mod_L_arr = fit_L[0]*t[max_loc[0:len(np.array(L_sat))]]+fit_L[1]
        #else: 
        mod_E_arr = np.array(E_sat)
        mod_L_arr = np.array(L_sat)

        range_low = 0
        range_high = len(peri_arr)-2
        skip = 1
            
        for i in range(range_low, range_high, skip):
            increment = i + 1
            increment_2 = i + 2

            z_0 = coord_apo[i][2]
            x_0 = coord_apo[i][0] * np.cos(coord_apo[i][1])
            y_0 = coord_apo[i][0] * np.sin(coord_apo[i][1])

            z_1 = coord_apo[increment][2]
            x_1 = coord_apo[increment][0] * np.cos(coord_apo[increment][1])
            y_1 = coord_apo[increment][0] * np.sin(coord_apo[increment][1])

            z_2 = coord_apo[increment_2][2]
            x_2 = coord_apo[increment_2][0] * np.cos(coord_apo[increment_2][1])
            y_2 = coord_apo[increment_2][0] * np.sin(coord_apo[increment_2][1])

            coord_apo_0 = np.array([x_0,y_0,z_0])
            coord_apo_1 = np.array([x_1,y_1,z_1])
            coord_apo_2 = np.array([x_2,y_2,z_2])

            
            deltaPsi = np.arccos(np.dot(coord_apo_0, coord_apo_1)/(np.linalg.norm(coord_apo_0)*np.linalg.norm(coord_apo_1)))#angle between successive apocenters ##wait not sure this gives me the right angle in cylindrical coordiantes ###
            deltaPsi = np.pi *2 - deltaPsi
            deltaPsi_arr.append(deltaPsi) 

        #if len(np.array(deltaPsi_arr)) > 1:
        #    fit_1 = np.polyfit(t[max_loc[:(len(np.array(deltaPsi_arr)))]], np.array(deltaPsi_arr), 1)
        #    x = t[max_loc[0:len(np.array(deltaPsi_arr))]]

        #    mod_deltaPsi_arr = fit_1[0]*x+fit_1[1]

                
        #else: 
        mod_deltaPsi_arr = np.array(deltaPsi_arr)

        
        mod_t_apo_arr = time_between_apo
        #print('mod_deltaPsi_arr SG',mod_deltaPsi_arr)
        #print(mod_t_apo_arr)
        maxlocflip = []
        modtflip = []
        rosette_angle_arr_flip = []

        maxlocflip = max_loc
        modtflip = mod_t_apo_arr
        rosette_angle_arr_flip = rosette_angle_arr

        range_low = 0
        range_high = len(peri_arr)-3
        skip = 1
            
        for i in range(range_low, range_high, skip):
            #if forward == True:
            increment_0 = i
            increment = i + 1
            increment_2 = i + 2

            host_profile = Dekel(alltime_hostcoords[0,redshift_id],alltime_hostcoords[1,redshift_id],alltime_hostcoords[2,redshift_id],alltime_hostcoords[3,redshift_id],z=0.)
            r_half_host = Reff(alltime_hostcoords[4,redshift_id],host_profile.ch)
            disk_scale_length_host = 0.766421/(1.+1./galaxy.flattening) * r_half_host
            disk_scale_height_host = disk_scale_length_host / galaxy.flattening
            #define the host disk profile
            host_disk_profile = MN(alltime_hostcoords[0,redshift_id]*galaxy.fd,disk_scale_length_host,disk_scale_height_host )
            host_stellar_disk_profile = MN(alltime_hostcoords[5,redshift_id]*galaxy.fd,disk_scale_length_host,disk_scale_height_host )
            #define the total profile
            total_profile = [host_profile, host_disk_profile]
            #if forward == True:
            #N_orb = N_orb + len(peri_arr[:increment_2]) #need to account for backward one 
            #else: 
            #     N_orb = len(peri_arr[increment_2:])
            #print('N_orb', N_orb)
            s = (m_sat[redshift_id]/(3*(host_profile.M(sat_host_dist[i])+host_disk_profile.M(sat_host_dist[i]))))**(1/3) # so these change in time...might need to take that into account in round 2 #also an issue that the mass keeps decreasing below the mass resolution limit, for some sats, but does not get updated here
            M_host = host_profile.M(sat_host_dist[i])+host_disk_profile.M(sat_host_dist[i])
            scale_radius_host = host_profile.rs
            r_tide = sat_host_dist[i]*s
            R1 = sat_host_dist[i]*(1.+.001)
            R2 = sat_host_dist[i]*(1.-.001)
            Phi1 = Phi(total_profile,R1)
            Phi2 = Phi(total_profile,R2)
            dPhidR_Rp = (Phi1-Phi2) / (R1-R2)
            e_s = 2*r_tide*dPhidR_Rp
            l_s.append((np.sqrt(3)+2)*s*mod_L_arr[i])
            #print('l_s',(np.sqrt(3)+2)*s*mod_L_arr[i])
            #print('l_s[i]', l_s[i])
            #print('Norb', N_orb )
            #print('e_s', e_s)
            #going to need E and L at different timesteps because orbit is decaying 

            
            L1 = mod_L_arr[i]
            L2 = mod_L_arr[increment]
            #if forward == True:
            deltaPsi = mod_deltaPsi_arr[i]
            deltaPsi_2 = mod_deltaPsi_arr[increment] 
            ###d_deltaPsi_dL_Eorb = np.abs(deltaPsi - deltaPsi_2) / (L1-L2)

            ###d_deltaPsi_dL_Eorb_arr.append(d_deltaPsi_dL_Eorb)
            Psi_L = l_s[i] *dPsidL *N_orb
            Psi_L_arr.append(Psi_L)
            
            E1 = mod_E_arr[i]
            E2 = mod_E_arr[increment]
            Tr = mod_t_apo_arr[i]
            deltaTr = mod_t_apo_arr[i] - mod_t_apo_arr[increment]
            deltaE = E1-E2
            print('deltaPsi', deltaPsi)
            print('Tr', Tr)
            
            ###dTr_dE_Lorb = np.abs(deltaTr)/ deltaE
            Psi_E = e_s  * (deltaPsi/Tr) * dtbadE * N_orb
            if np.minimum(np.abs(Psi_E) ,rosette_angle_arr_flip[i]) == np.abs(Psi_E):
                Psi_E_arr.append(Psi_E)
            else: 
                Psi_E = rosette_angle_arr_flip[i]
                Psi_E_arr.append(rosette_angle_arr_flip[i])
            print('mu', np.divide(Psi_L, Psi_E))
            
        #writing the fits for the angles to exclude outliers in PL/PE
        #predictions = []
        if (len(Psi_L_arr) > 1): 
            try:
                y_start = np.divide(Psi_L_arr,Psi_E_arr)
                x_start = t[max_loc[0:len(np.array(Psi_L_arr))]]
                z_score = zscore(y_start)
                #print('zscore',z_score)
                y_hold = y_start[np.where((np.abs(y_start) > 0.01))] #might need to cut on 1 #(np.abs(z_score) < 1.25) & #np.abs(np.median(y_start))
                x_hold = x_start[np.where((np.abs(y_start) > 0.01))]#(np.abs(z_score) < 1.25) & #np.abs(np.median(y_start))
                z_score_hold = zscore(y_hold)
                #print('zscore_hold',z_score_hold)
                y = y_hold[np.where((np.abs(z_score_hold) < 1.05))]
                x = x_hold[np.where((np.abs(z_score_hold) < 1.05))]
                #standardize
                x_scaler, y_scaler = StandardScaler(), StandardScaler()
                x_train = x_scaler.fit_transform(x[..., None])
                y_train = y_scaler.fit_transform(y[..., None])
                #fit omdel
                model = HuberRegressor(epsilon=1)
                model.fit(x_train, y_train.ravel())
                #predict
                test_x = t[max_loc[0:len(np.array(Psi_L_arr))]]
                predictions = y_scaler.inverse_transform(model.predict(x_scaler.transform(test_x[..., None])))
                print('predictions',predictions)
            except: 
                predictions = np.divide(Psi_L_arr, Psi_E_arr)
        else: 
            predictions = np.divide(Psi_L_arr, Psi_E_arr)
            
        #need to determine rosette angle because this will change whether high-eccentricity orbits are labelled as streams or shells
        
        if (len(Psi_L_arr) > 1) and (forward == True):
            if np.abs(predictions[0]) < 1:
                what_is_it = 1
            elif np.abs(predictions[0]) > 1:
                what_is_it = 2
        elif (len(Psi_L_arr) > 1) and (forward == False):
            if np.abs(predictions[0]) < 1:
                what_is_it = 1
            elif np.abs(predictions[0]) > 1:
                what_is_it = 2
        elif len(Psi_L_arr) == 1:
            if np.abs(predictions) < 1:
                what_is_it = 1
            elif np.abs(predictions) > 1:
                what_is_it = 2
            
            

    return_Psi_E = []
    return_Psi_L = []
    return_deltaPsi = []
    return_mod_deltaPsi = []
    return_L_sat = []
    return_E_sat = []
    return_Lmod_sat = []
    return_Emod_sat = []
    #if forward == True:
    return_Psi_E = np.array(Psi_E_arr)
    return_Psi_L = np.array(Psi_L_arr)
    return_deltaPsi = np.array(deltaPsi_arr)
    return_mod_deltaPsi  = np.array(mod_deltaPsi_arr)
    return_L_sat = np.array(L_sat)
    return_E_sat = np.array(E_sat)
    return_Lmod_sat = np.array(mod_L_arr)
    return_Emod_sat = np.array(mod_E_arr)
    return_predictions = predictions


    return int(what_is_it), return_Psi_E, return_Psi_L, return_deltaPsi, return_mod_deltaPsi, return_L_sat, return_E_sat, N_orb, return_Lmod_sat, return_Emod_sat, return_predictions, M_Rp_arr