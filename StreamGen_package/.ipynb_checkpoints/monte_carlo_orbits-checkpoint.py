#orbits
import sys
sys.path.insert(0, '/tigress/dropulic/SatGen/')
from profiles import Dekel, MN, EnergyAngMomGivenRpRa, Phi
import multiprocessing as mp
from tqdm import tqdm
import scipy
import onesat_orbit
import numpy as np
import helpers

def integrate_mc_orbits(galaxy,coordinate,sigma,coord_at, host_coords, mass, redshift_id,time_btwn_apo, ecc_est, num_mc,period_est_correction =0, forward = True,inLLcorner = False):
    num_draw = num_mc
    if inLLcorner == True:
        num_draw = 1000
    ca_list = []
    cp_list = [] 
    vels_peri_list = []
    vels_apo_list = []
    satdist_list = []
    t_list = []
    tba_list = []
    total_profile = []
    pos_vel_list = []
    
    pool = mp.Pool(mp.cpu_count()-1)
    print("cpu count",mp.cpu_count())
    results = [pool.apply_async(MC_orbits_SG, args=(galaxy,coordinate,sigma,coord_at, host_coords, mass, redshift_id,time_btwn_apo, ecc_est,period_est_correction), kwds={'forward': True, 'inLLcorner':inLLcorner}) for _ in tqdm(range(0,num_draw), position=0, leave=True)]
    #results = [pool.apply_async(MC_orbits_SG) for _ in tqdm(range(0,num_draw), position=0, leave=True)]
    print("shape results ", np.shape(results))
    print(results)
    pool.close()
    res = [f.get() for f in tqdm(results)]
    #print(np.shape(list(res[0])))
    
    for mc_i in range(0,num_draw):
        ca_list.append(res[mc_i][0])
        cp_list.append(res[mc_i][1])
        vels_peri_list.append(res[mc_i][2])
        vels_apo_list.append(res[mc_i][3])
        satdist_list.append(res[mc_i][4])
        t_list.append(res[mc_i][5])
        tba_list.append(res[mc_i][6])
        total_profile = res[mc_i][7]
        pos_vel_list.append(res[mc_i][8])

    return ca_list, cp_list, vels_peri_list, vels_apo_list, satdist_list, t_list,tba_list, total_profile, pos_vel_list, num_mc

def MC_orbits_SG(galaxy,coordinate,sigma,coord_at, host_coords, mass, redshift_id,time_btwn_apo, ecc_est,period_est_correction, forward = True, inLLcorner = False):
    #velocity in kpc/Gyr
    #total velocity dispersion in kpc/Gyr
    np.random.seed()
    coordinate = coordinate[redshift_id]
    print('coordinate',coordinate)
    print('sigma',sigma) #note sigma is still at redshift 0
    coord_init = coordinate[:3]
    vels_init  = coordinate[3:]
    print('here1')

    #want velocity in cartesian I think, so sigma can be an isotropic sphere 
    coords_cart, vels_cart = helpers.rvcyltocart(coord_init, vels_init)
    #random velocity draw
    print('here2')
    cov = np.array([[sigma**2,0,0],[0,sigma**2,0],[0,0,sigma**2]])
    random_vel_draw = np.random.multivariate_normal(vels_cart, cov,1)
    random_vel_draw = random_vel_draw.flatten()
    print('here3')
    coords_cyl, coord_rand_vel_cyl = helpers.rvcarttocyl(coords_cart, random_vel_draw)

    print('starting integrate orbit')
    timetot, Nstep, tmax, pos_vel, min_loc, max_loc, satdist, t = onesat_orbit.integrate_orbit(galaxy,coord_at, host_coords, mass, redshift_id,time_btwn_apo, ecc_est, period_est_correction = period_est_correction, forward = True, random_draw = coord_rand_vel_cyl, inLLcorner = inLLcorner)
    print('finished integrate orbit')
    all_apo, all_peri, ca, cp, tba, vels_peri, vels_apo = helpers.peri_apo_props_reintegrate(timetot, Nstep, tmax, pos_vel, min_loc, max_loc, satdist, t)
    total_profile = galaxy.get_host_profile(redshift_id)

    return ca, cp, vels_peri, vels_apo, satdist, t, tba, total_profile, pos_vel