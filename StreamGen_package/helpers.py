# Placeholder functions
import numpy as np
from scipy.signal import argrelmin, argrelextrema, argrelmax
import numpy.ma as ma

def peri_apo_props_reintegrate(time, Nstep, tmax, pos_vel, min_loc, max_loc, satdist, t_new):
    #print('in peri_apo_props_reintegrate')
    t = t_new
    all_apo = satdist[max_loc]
    all_peri = satdist[min_loc]
    coordinates_peri = pos_vel[min_loc,:3]
    coordinates_apo = pos_vel[max_loc,:3]
    vels_peri = pos_vel[min_loc,3:]
    vels_apo = pos_vel[max_loc,3:]
    time_between_apo = np.diff(max_loc) * np.abs(t[1] - t[0])
    
    return all_apo, all_peri, coordinates_apo, coordinates_peri, time_between_apo, vels_peri, vels_apo

def eccentricity_est(rp,ra):
    
    numerator = ra - rp 
    denominator = ra + rp
    ecc = np.divide(numerator, denominator)
    
    return ecc

def rvcyltocart(coords_cyl, vels_cyl):
    
    s = coords_cyl[0]
    phi = coords_cyl[1]
    z = coords_cyl[2]
    
    v_s = vels_cyl[0]
    v_phi = vels_cyl[1]
    v_z = vels_cyl[2]
    
    s_dot = v_s
    phi_dot = v_phi/s
    z_dot = v_z
    
    x = s*np.cos(phi)
    y = s*np.sin(phi)
    z = z
    
    x_dot = s_dot*np.cos(phi)-s*phi_dot*np.sin(phi)
    y_dot = s_dot*np.sin(phi)+s*phi_dot*np.cos(phi)
    z_dot = z_dot
    
    coords_cart = [x,y,z]
    vels_cart = [x_dot, y_dot, z_dot]
    
    return coords_cart, vels_cart

def rvcarttocyl(coords_cart, vels_cart):
    
    x  = coords_cart[0]
    y  = coords_cart[1]
    z  = coords_cart[2]
    
    v_x = vels_cart[0]
    v_y = vels_cart[1]
    v_z = vels_cart[2]
    
    rho = np.sqrt(x**2+y**2)
    theta = np.arctan2(y,x)
    
    v_rho = (x*v_x+y*v_y)/rho
    v_theta = (x*v_y-y*v_x)/rho

    coords_cyl = [rho,theta,z]
    vels_cyl = [v_rho, v_theta, v_z]
    
    return coords_cyl, vels_cyl

def EnergyAngMom(coords_cart, vels_cart, host_potential, m_sat):
    
    L =  np.linalg.norm(np.cross(coords_cart, vels_cart))
    v_sat = np.linalg.norm(vels_cart)
    E = host_potential + 0.5*v_sat**2

    return E, L

def calc_rtide(m, m_rp, rp):
    
    s = np.divide(m,(3*m_rp))**(1/3)
    r_tide = np.multipy(s,rp)
    
    return calc_rtide   

def orbital_period_estimate(r, M):
    G = 1.327 * 10**11 #km3 MSun-1 s-2
    r = r * 3.086 * 10**16 #conversion kpc to km
    T = ((4 * np.pi**2 * r**3)/(G*M))**(1/2)
    T = T / (3.16 * 10**16) #conversion from s to Gyr
    return T

def find_apo_peri_from_pos_vel(pos_vel_arr, forward = True):

    z = pos_vel_arr[:,2]
    x = pos_vel_arr[:,0] * np.cos(pos_vel_arr[:,1])
    y = pos_vel_arr[:,0] * np.sin(pos_vel_arr[:,1])
    
    satdist = np.linalg.norm(np.array([x,y,z]).T, axis = 1)
    
    min_loc = argrelmin(satdist, order = 5)
    max_loc = argrelmax(satdist, order = 5)
        
    return satdist, min_loc, max_loc
    
def get_huber_predictons(x_in,y_in):
    predictions = []
    if (len(y_in) > 1): 
        try:
            y_start = y_in
            x_start = x_in
            z_score = zscore(y_start)
            y = y_start[np.where((np.abs(z_score) < 1) & (np.abs(y_start) > 0.01))]
            x = x_start[np.where((np.abs(z_score) < 1) & (np.abs(y_start) > 0.01))]
            #y = y_start[np.abs(y_start) > 0.01]
            #x = x_start[np.abs(y_start) > 0.01]
            #standardize
            x_scaler, y_scaler = StandardScaler(), StandardScaler()
            x_train = x_scaler.fit_transform(x[..., None])
            y_train = y_scaler.fit_transform(y[..., None])
            #fit omdel
            model = HuberRegressor(epsilon=1)
            model.fit(x_train, y_train.ravel())
            #predict
            test_x = x_in
            predictions = y_scaler.inverse_transform(model.predict(x_scaler.transform(test_x[..., None])))
        except: 
            print("huber exception!")
            predictions = y_in
    else: 
        predictions = y_in
    return predictions