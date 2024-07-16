#one sat metric

def calc_rosette_angle_SatGen(max_loc, min_loc, pos_vel):
    minmax_loc_diff = np.abs(np.subtract(max_loc,min_loc))
    quarter_period_timestep = np.round(minmax_loc_diff/2)
    half_period_locs = []
    rosette_angle_arr = []
    for max_loc_i, mm_loc_diff_i in zip(max_loc, quarter_period_timestep):
        hf_pd_loc_i = np.array([int(max_loc_i + mm_loc_diff_i),int(max_loc_i - mm_loc_diff_i )])
        half_period_locs.append(hf_pd_loc_i)
        
        if (hf_pd_loc_i[0] > 0) and (hf_pd_loc_i[1] > 0) and (hf_pd_loc_i[1] < len(pos_vel[:,0]))and (hf_pd_loc_i[0] < len(pos_vel[:,0])):
        
            coord_rosette_0, vels_rosette_0 = rvcyltocart(pos_vel[hf_pd_loc_i[0],:3],pos_vel[hf_pd_loc_i[0],3:])
            coord_rosette_1, vels_rosette_1 = rvcyltocart(pos_vel[hf_pd_loc_i[1],:3],pos_vel[hf_pd_loc_i[1],3:])
            rosette_angle = np.arccos(np.dot(coord_rosette_0, coord_rosette_1)/(np.linalg.norm(coord_rosette_0)*np.linalg.norm(coord_rosette_1)))
            rosette_angle_arr.append(rosette_angle)
        else:
            rosette_angle_arr.append(np.inf)
    return half_period_locs,rosette_angle_arr

def stream_or_shell_integrate_OLD(m_sat, sat_host_dist, peri_arr, apo_arr, coord_peri, coord_apo, time_between_apo, alltime_hostcoords, redshift_id, t, max_loc,ax, N_orb_from_back , vel_apo, min_loc, rosette_angle_arr,vel_peri, forward = True):
    print('in stream_or_shell_integrate')
    what_is_it = 0
    time_between_apo = np.abs(time_between_apo)
        
        
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
            disk_scale_length_host = 0.766421/(1.+1./flattening) * r_half_host
            disk_scale_height_host = disk_scale_length_host / flattening
            #define the host disk profile
            host_disk_profile = MN(alltime_hostcoords[0,redshift_id]*fd,disk_scale_length_host,disk_scale_height_host )
            host_stellar_disk_profile = MN(alltime_hostcoords[5,redshift_id]*fd,disk_scale_length_host,disk_scale_height_host )
            total_profile = [host_profile, host_disk_profile]
            s = (m_sat[redshift_id]/(3*(host_profile.M(sathd)+host_disk_profile.M(sathd))))**(1/3) #also an issue that the mass keeps decreasing below the mass resolution limit, for some sats, but does not get updated here
            M_Rp_arr.append((host_profile.M(sathd)+host_disk_profile.M(sathd)))
            coords_cart, vels_cart = rvcyltocart(coord_peri[ii], vel_peri[ii])
            E_new, L_new = EnergyAngMom(coords_cart, vels_cart, Phi(total_profile,sathd) , m_sat[redshift_id]) #looks good
            print('sathd ssIO', sathd)
            ii = ii+1
            E,L = EnergyAngMomGivenRpRa(total_profile,rp,ra)
            #print('ELnew', E_new, L_new)
            #print('ELold',E,L)
            
            E_sat.append(E_new)
            L_sat.append(L_new)

        mod_E_arr = np.array(E_sat)
        mod_L_arr = np.array(L_sat)


        #if forward == True:
        range_low = 0
        range_high = len(peri_arr)-2
        skip = 1

        for i in range(range_low, range_high, skip):
            #if forward == True:
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
            deltaPsi_arr.append(deltaPsi) 

        mod_deltaPsi_arr = np.array(deltaPsi_arr)
        mod_t_apo_arr = t[max_loc[0:len(np.array(time_between_apo))]]
        
        mod_t_apo_arr = np.abs(mod_t_apo_arr)
       # print(mod_deltaPsi_arr)
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
            disk_scale_length_host = 0.766421/(1.+1./flattening) * r_half_host
            disk_scale_height_host = disk_scale_length_host / flattening
            #define the host disk profile
            host_disk_profile = MN(alltime_hostcoords[0,redshift_id]*fd,disk_scale_length_host,disk_scale_height_host )
            host_stellar_disk_profile = MN(alltime_hostcoords[5,redshift_id]*fd,disk_scale_length_host,disk_scale_height_host )
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
            #going to need E and L at different timesteps because orbit is decaying 
            #deltaPsi = mod_deltaPsi_arr[i-2]
            #deltaPsi_2 = mod_deltaPsi_arr[increment-2] 
            
            L1 = mod_L_arr[i]
            L2 = mod_L_arr[increment]
            #if forward == True:
            deltaPsi = mod_deltaPsi_arr[i]
            deltaPsi_2 = mod_deltaPsi_arr[increment] 
            d_deltaPsi_dL_Eorb = np.abs(deltaPsi - deltaPsi_2) / (L1-L2)
            #else:
            #    deltaPsi = mod_deltaPsi_arr[i-3]
            #    deltaPsi_2 = mod_deltaPsi_arr[increment-3] 
            #    d_deltaPsi_dL_Eorb = np.abs(deltaPsi_2 - deltaPsi) / (L2-L1)
            d_deltaPsi_dL_Eorb_arr.append(d_deltaPsi_dL_Eorb)
            Psi_L = l_s[i] *d_deltaPsi_dL_Eorb *N_orb
           # print('l_s',l_s[i])
           # print('d_deltaPsi_dL_Eorb',d_deltaPsi_dL_Eorb)
            Psi_L_arr.append(Psi_L)
            
            E1 = mod_E_arr[i]
            E2 = mod_E_arr[increment]
            #if forward == True:
            Tr = mod_t_apo_arr[i]
            deltaTr = mod_t_apo_arr[i] - mod_t_apo_arr[increment]
            deltaE = E1-E2
            #else: 
            #    Tr = mod_t_apo_arr[i-3]
            #    deltaTr = mod_t_apo_arr[increment-3] - mod_t_apo_arr[i-3]
            #    deltaE = E2-E1
            
            dTr_dE_Lorb = np.abs(deltaTr)/ deltaE
            Psi_E = e_s  * (deltaPsi/Tr) * dTr_dE_Lorb *N_orb
            if np.minimum(np.abs(Psi_E) ,rosette_angle_arr_flip[i]) == np.abs(Psi_E):
                Psi_E_arr.append(Psi_E)
            else: 
                Psi_E_arr.append(rosette_angle_arr_flip[i])
            
        #writing the fits for the angles to exclude outliers in PL/PE
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


def stream_or_shell(m_sat, host_coords, sat_host_dist, peri_arr, apo_arr, coord_peri, coord_apo, time_between_apo, sm_loss, alltime_hostcoords, idx_massres_sat, order, vel_apo, t, max_loc,rosette_angle_arr, vel_peri):
    what_is_it = 0
    #print('order',order)
    #0 is intact satellite, 1 is stream, 2 is shell 
    #define the host profile
    #eventually take into account that the host is changing as well
    #print(np.shape(alltime_hostcoords))
    print('m_sat',m_sat)
    ###time_between_apo = np.abs(time_between_apo)

    print(t)
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
    print('vel_peri',vel_peri)
    print('len(peri_arr)',len(peri_arr))
    if len(peri_arr) >= 3:
        j = 0
        for rp, ra,msat,sathd in zip(peri_arr, apo_arr,m_sat,sat_host_dist):
            host_profile = Dekel(alltime_hostcoords[0,j],alltime_hostcoords[1,j],alltime_hostcoords[2,j],alltime_hostcoords[3,j],z=0.)
            r_half_host = Reff(alltime_hostcoords[4,j],host_profile.ch)
            disk_scale_length_host = 0.766421/(1.+1./flattening) * r_half_host
            disk_scale_height_host = disk_scale_length_host / flattening
            #define the host disk profile
            host_disk_profile = MN(alltime_hostcoords[0,j]*fd,disk_scale_length_host,disk_scale_height_host )
            host_stellar_disk_profile = MN(alltime_hostcoords[5,j]*fd,disk_scale_length_host,disk_scale_height_host )
            l1 = np.log10(np.linspace(10**-1,10**4,1000))
            l2 = np.log10(2*(host_profile.rhobar(np.linspace(10**-1,10**4,1000))+host_disk_profile.rhobar(np.linspace(10**-1,10**4,1000))+host_stellar_disk_profile.rhobar(np.linspace(10**-1,10**4,1000))))
            #define the total profile
            total_profile = [host_profile, host_disk_profile] ##make sure it's the stellar disk????????
            s = (msat/(3*(host_profile.M(sathd)+host_disk_profile.M(sathd))))**(1/3) #also an issue that the mass keeps decreasing below the mass resolution limit, for some sats, but does not get updated here
            coords_cart, vels_cart = rvcyltocart(coord_peri[j], vel_peri[j])
            E_new, L_new = EnergyAngMom(coords_cart, vels_cart, Phi(total_profile,sathd) , msat) #looks good
            E,L = EnergyAngMomGivenRpRa(total_profile,rp,ra)
            print('ELnew', E_new, L_new)
            E_sat.append(E_new)
            L_sat.append(L_new)
            l_s.append((np.sqrt(3)+2)*s*L_new)
            j+=1
 
        #fit_E = np.polyfit(t[max_loc[0:len(np.array(E_sat))]], np.array(E_sat), 1)
        #fit_L = np.polyfit(t[max_loc[0:len(np.array(L_sat))]], np.array(L_sat), 1)
        #mod_E_arr = fit_E[0]*t[max_loc[0:len(np.array(E_sat))]]+fit_E[1]
        #mod_L_arr = fit_L[0]*t[max_loc[0:len(np.array(L_sat))]]+fit_L[1]
        #else: 
        mod_E_arr = np.array(E_sat)
        mod_L_arr = np.array(L_sat)

        #when I define the number of orbits, I might want to exclude the ones that fall beneath 10^-3 stellar mass loss?
        #for now set the limit on the calculation by whichever array is longer 
        #if forward == True:
        range_low = 0
        range_high = len(peri_arr)-1
        skip = 1
        #else:
        #    range_low = len(peri_arr)-1
        #    range_high = 1
        #    skip = -1
            
        for i in range(range_low, range_high, skip):
            #if forward == True:
            increment = i + 1
            #increment_2 = i + 2
            #else:
            #    increment = i - 1
            #    increment_2 = i - 2
            
            z_0 = coord_apo[i][2]
            x_0 = coord_apo[i][0] * np.cos(coord_apo[i][1])
            y_0 = coord_apo[i][0] * np.sin(coord_apo[i][1])

            z_1 = coord_apo[increment][2]
            x_1 = coord_apo[increment][0] * np.cos(coord_apo[increment][1])
            y_1 = coord_apo[increment][0] * np.sin(coord_apo[increment][1])

            #z_2 = coord_apo[increment_2][2]
            #x_2 = coord_apo[increment_2][0] * np.cos(coord_apo[increment_2][1])
            #y_2 = coord_apo[increment_2][0] * np.sin(coord_apo[increment_2][1])

            coord_apo_0 = np.array([x_0,y_0,z_0])
            coord_apo_1 = np.array([x_1,y_1,z_1])
            #coord_apo_2 = np.array([x_2,y_2,z_2])

            
            deltaPsi = np.arccos(np.dot(coord_apo_0, coord_apo_1)/(np.linalg.norm(coord_apo_0)*np.linalg.norm(coord_apo_1)))#angle between successive apocenters ##wait not sure this gives me the right angle in cylindrical coordiantes ###

            deltaPsi_arr.append(deltaPsi) 
            print('deltaPsi, ',deltaPsi)
            #print('deltaPsi analytic, ',delta_Psi_isochorone_analytic(L_sat[i], M_host, scale_radius_host))
        if len(np.array(deltaPsi_arr)) > 1:

            fit_1 = np.polyfit(t[max_loc[:len(np.array(deltaPsi_arr))]], np.array(deltaPsi_arr), 1)
            #mod_deltaPsi_arr = get_huber_predictons(t[max_loc[0:len(np.array(deltaPsi_arr))]], np.array(deltaPsi_arr))
        ##   #fit_2 = interp1d(t[max_loc[0:len(np.array(deltaPsi_arr))]], np.array(deltaPsi_arr),kind = 'linear', fill_value='extrapolate')
        ##    #mod_deltaPsi_arr  = [fit_2(x).item() for x in np.array(deltaPsi_arr)]
            x = t[max_loc[0:len(np.array(deltaPsi_arr))]]
        ##    fit_t_apo = np.polyfit(t[max_loc[0:len(np.array(time_between_apo))]], time_between_apo, 1)

            #print(t[max_loc[0:len(deltaPsi_arr)]])
            #print(deltaPsi_arr)
        ##    mod_deltaPsi_arr = fit_1[0]*x+fit_1[1]
        ##    mod_t_apo_arr = fit_t_apo[0]*t[max_loc[0:len(np.array(time_between_apo))]] + fit_t_apo[1]
        
                
        #else: 
        mod_deltaPsi_arr = np.array(deltaPsi_arr)

        print('modedPsiSG',mod_deltaPsi_arr)
        #mod_t_apo_arr = t[max_loc[0:len(np.array(time_between_apo))]]
        
        mod_t_apo_arr = time_between_apo
       # print(mod_deltaPsi_arr)
        #print(mod_t_apo_arr)
        maxlocflip = []
        modtflip = []
        rosette_angle_arr_flip = []

        maxlocflip = max_loc
        modtflip = mod_t_apo_arr
        print('rosette_angle_arr',rosette_angle_arr)
        rosette_angle_arr_flip = rosette_angle_arr
        #ax[3].plot(t[maxlocflip[0:len(np.array(time_between_apo))]], modtflip)

        
        
        ##HERE##
        #if forward == True:
        range_low = 0
        range_high = len(peri_arr)-2
        skip = 1

            
        for i in range(range_low, range_high, skip):
            #if forward == True:
            increment_0 = i
            increment = i + 1

            host_profile = Dekel(alltime_hostcoords[0,i],alltime_hostcoords[1,i],alltime_hostcoords[2,i],alltime_hostcoords[3,i],z=0.)
            r_half_host = Reff(alltime_hostcoords[4,i],host_profile.ch)
            disk_scale_length_host = 0.766421/(1.+1./flattening) * r_half_host
            disk_scale_height_host = disk_scale_length_host / flattening
            #define the host disk profule
            host_disk_profile = MN(alltime_hostcoords[0,i]*fd,disk_scale_length_host,disk_scale_height_host )
            host_stellar_disk_profile = MN(alltime_hostcoords[5,i]*fd,disk_scale_length_host,disk_scale_height_host )
            l1 = np.log10(np.linspace(10**-1,10**4,1000))
            l2 = np.log10(2*(host_profile.rhobar(np.linspace(10**-1,10**4,1000))+host_disk_profile.rhobar(np.linspace(10**-1,10**4,1000))+host_stellar_disk_profile.rhobar(np.linspace(10**-1,10**4,1000))))
            #define the total profile
            total_profile = [host_profile, host_disk_profile]

            #if forward == True:
            N_orb = len(peri_arr[:increment]) #need to account for backward one 
            #else: 
            #     N_orb = len(peri_arr[increment_2:])
            #print('N_orb', N_orb)
            s = (m_sat[i]/(3*(host_profile.M(sat_host_dist[i])+host_disk_profile.M(sat_host_dist[i]))))**(1/3) # so these change in time...might need to take that into account in round 2 #also an issue that the mass keeps decreasing below the mass resolution limit, for some sats, but does not get updated here
            r_tide = sat_host_dist[i]*s
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
            #going to need E and L at different timesteps because orbit is decaying 
            #deltaPsi = mod_deltaPsi_arr[i-2]
            #deltaPsi_2 = mod_deltaPsi_arr[increment-2] 
            
            L1 = mod_L_arr[i]
            L2 = mod_L_arr[increment]
            #if forward == True:
            deltaPsi = mod_deltaPsi_arr[i]
            deltaPsi_2 = mod_deltaPsi_arr[increment] 
            d_deltaPsi_dL_Eorb = np.abs(deltaPsi - deltaPsi_2) / (L1-L2)
            #else:
            #    deltaPsi = mod_deltaPsi_arr[i-3]
            #    deltaPsi_2 = mod_deltaPsi_arr[increment-3] 
            #    d_deltaPsi_dL_Eorb = np.abs(deltaPsi_2 - deltaPsi) / (L2-L1)
            d_deltaPsi_dL_Eorb_arr.append(d_deltaPsi_dL_Eorb)
            Psi_L = l_s[i] *d_deltaPsi_dL_Eorb *N_orb
           # print('l_s',l_s[i])
           # print('d_deltaPsi_dL_Eorb',d_deltaPsi_dL_Eorb)
            Psi_L_arr.append(Psi_L)
            
            E1 = mod_E_arr[i]
            E2 = mod_E_arr[increment]
            #if forward == True:
            Tr = mod_t_apo_arr[i]
            deltaTr = mod_t_apo_arr[i] - mod_t_apo_arr[increment]
            #print('deltaTr', deltaTr)
            deltaE = E1-E2
            #print('deltaE',deltaE)
            #print('e_s',e_s)
            dTr_dE_Lorb = np.abs(deltaTr)/ deltaE
            Psi_E = e_s  * (deltaPsi/Tr) * dTr_dE_Lorb * N_orb
            if np.minimum(np.abs(Psi_E) ,rosette_angle_arr_flip[i]) == np.abs(Psi_E):
                Psi_E_arr.append(Psi_E)
            else: 
                Psi_E_arr.append(rosette_angle_arr_flip[i])
        print('Psi_E_arr_SG', Psi_E_arr) 
        print('Psi_L_arr_SG', Psi_L_arr) 
        
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
        
        if (len(Psi_L_arr) > 1) :
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

