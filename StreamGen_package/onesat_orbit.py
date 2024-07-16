#orbits
import sys
sys.path.insert(0, '/tigress/dropulic/SatGen/')
import aux
import orbit as orb
from profiles import Dekel, MN, EnergyAngMomGivenRpRa, Phi
import multiprocessing as mp
from tqdm import tqdm
import scipy
import math
import time as tt
import numpy as np
from galhalo import Reff
import helpers

def integrate_orbit(galaxy,coord_at, host_coords, mass, redshift_id,time_btwn_apo, ecc_est,period_est_correction = 0, forward = True, random_draw = None, inLLcorner = False):
    print('in integrate_orbit')
    print('coord_at', coord_at)
    print('coord_at[redshift_id]',coord_at[redshift_id])
    print(random_draw)
    print('period_est_correction', period_est_correction)
    xv_o = []
    xv_o = coord_at[redshift_id]
    z = xv_o[2]
    x = xv_o[0] * np.cos(xv_o[1])
    y = xv_o[0] * np.sin(xv_o[1])
    satdist_init = np.linalg.norm(np.array([x,y,z]))
    mass = mass[redshift_id]
    timestep = 0.01
    if ecc_est > 0.3:
        timestep = 0.001 #0.01
    if inLLcorner == True:
        timestep = timestep/ 10
        if ecc_est < 0.3:
            timestep = timestep/ 10
    #else:
    #    timestep = 0.01
    if len(time_btwn_apo) > 0:
        period_est = np.abs(time_btwn_apo[0])
        if period_est_correction !=0:
            period_est = np.abs(time_btwn_apo[period_est_correction])
    else: 
        period_est = orbital_period_estimate(satdist_init, mass)

    
    if period_est > 20:
        period_est = 20
        
    #print('period_est',period_est)
    xv_in = []
    #if forward == False:
    #    xv_in = np.array([xv_o[0],xv_o[1], xv_o[2],-1* xv_o[3], -1*xv_o[4], -1*xv_o[5]])
    #else:
    if random_draw != None:
        xv_in = np.concatenate((xv_o[:3], random_draw), axis = None)
    else:
        xv_in = xv_o
    o = []
    o = orb.orbit(xv_in)

    total_profile = galaxy.get_host_profile(redshift_id)
    satdist = []
    min_loc= []
    max_loc= []
    Nstep = []
    tmax= []
    tmin = []
    t = []
   
    # initialize orbit
    if period_est < 1.0:
        print('period less than 1 Gyr! Integrating longer to decrease noise')
        tmax = math.ceil(period_est*10) # [Gyr] 
        Nstep = math.ceil(tmax/ timestep)
    elif (period_est < 5) and (period_est > 1):
        tmax = math.ceil(period_est*5) # [Gyr] 
        Nstep = math.ceil(tmax/ timestep)
        #Nstep = math.ceil(tmax *2) 
    else:
        tmax = math.ceil(period_est) # [Gyr] 
        Nstep = math.ceil(tmax/ timestep)

    if forward == True:
        tmin =  0#-tmax#0  #-tmax#0
        tmax = tmax
    else:
        tmin = 0#tmax#0#tmax
        tmax = -tmax#-tmax#0

    t = np.linspace(tmin,tmax,Nstep+1)[0::] #[Gyr]#Nstep +1
    #print('t',t)
    start = tt.time()
    o.integrate(t,total_profile,mass)#,mass
    end = tt.time()
    timer = end - start
    #print(timer)

    time = o.t
    pos_vel = o.xvArray

    t_new = []
    satdist, min_loc, max_loc = helpers.find_apo_peri_from_pos_vel(pos_vel,  forward)
    distdiff = np.abs(satdist[0] - satdist_init)
    #print('satdist0', satdist[0])
    n = 2
    condition = False #Do I have enough pericenters?''
    if (len(min_loc[0]) < 5): condition = True #or (distdiff > 3.5)#or len(max_loc[0] < 8)
    if (timer > 120): 
        condition = False
        t_new = t
    while condition == True:
        if period_est < 1.0:
            tmax = math.ceil(period_est*10)*n # [Gyr] 
            Nstep = math.ceil(tmax/ timestep)
        elif (period_est < 5) and (period_est > 1):
            tmax = math.ceil(period_est*5)*n # [Gyr] 
            Nstep = math.ceil(tmax/ timestep)
            #fact = math.ceil(tmax *2)
            #if fact > 100: fact = 100
            #Nstep = fact*n # number of timesteps #0.037 is approximate Gyr/timestep in SatGen
        else:
            if period_est > 20:
                period_est = 20
            tmax = math.ceil(period_est)*n # [Gyr] 
            Nstep = math.ceil(tmax/ timestep)

        o = []
        o = orb.orbit(xv_in)

        if n > 5: 
            #print(str(tmax) + ' Gyr')
            condition = False
        if forward == True:
            tmin = 0#-tmax#0

            tmax = tmax
        else:
            tmin = 0#tmax#0#tmax
            tmax = -tmax#-tmax#0
        t_new = np.linspace(tmin,tmax,Nstep+1)[0::] #[Gyr]
        o.integrate(t_new,total_profile,mass)#/mass
        time = o.t
        pos_vel = o.xvArray
        #print('pos_vel',np.shape(pos_vel))
        satdist, min_loc, max_loc = helpers.find_apo_peri_from_pos_vel(pos_vel, forward)

        #print('satdist',np.shape(satdist))
        if (len(min_loc[0]) < 5) and (condition != False): 
            condition = True
        else:
            condition = False
        n = n+1
    
    min_locs = []
    max_locs = []
    if len(min_loc[0]) > len(max_loc[0]):
        #print("cond1")
        min_locs = min_loc[0][0:len(max_loc[0])]
        max_locs = max_loc[0]
    elif len(min_loc[0]) < len(max_loc[0]):
        #print("cond2")
        max_locs = max_loc[0][0:len(min_loc[0])]
        min_locs = min_loc[0]
    else:
        #print("cond3")
        max_locs = max_loc[0]
        min_locs = min_loc[0]
    min_locs = min_locs[np.where(min_locs > 4)]
    max_locs = max_locs[np.where(max_locs < (len(satdist) - 4))]

    if len(min_locs) > len(max_locs):
        #print("cond1")
        min_locs = min_locs[0:len(max_locs)]
        max_locs = max_locs
    elif len(min_locs) < len(max_locs):
        #print("cond2")
        max_locs = max_locs[0:len(min_locs)]
        min_locs = min_locs

    if len(t_new) == 0: 
        t_new = t

    return time, Nstep, tmax, pos_vel, min_locs, max_locs, satdist, t_new