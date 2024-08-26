# File to load in SatGen galaxies and relevant coordinates
import numpy as np
from scipy.signal import argrelmin, argrelextrema, argrelmax
import sys, os
satgen_path = os.path.abspath(os.path.join(__file__, "./../../../SatGen/"))
sys.path.insert(0, satgen_path) #!change this to your path to SatGen to load in necessary analysis scripts
import aux
from profiles import Dekel, MN, EnergyAngMomGivenRpRa, Phi
from galhalo import Reff
import pandas as pd
import astropy
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
import astropy.units as u
import astropy.cosmology.units as cu
u.add_enabled_units(cu)

class Galaxy:
    """
    A class to represent a galaxy and its surviving satellite properties.

    Attributes:
        flattening (float): Disk flattening factor.
        fd (float): Disk fraction.
        stellar_mass_floor (float): Minimum stellar mass threshold.
        Various arrays to store subhalo and satellite properties.
    """

    def __init__(self, datadir, tree_file, disk_fraction=0.05, flattening=25., stellar_mass_floor=5e5):
        """
        Initialize the Galaxy class, load satellite data, and set up initial parameters.

        Args:
            datadir (str): Directory containing SatGen galaxy data.
            tree_file (str): Filename of the galaxy tree file.
            disk_fraction (float): Disk mass fraction. Default is 0.05.
            flattening (float): Disk flattening factor. Default is 25.
            stellar_mass_floor (float): Minimum stellar mass threshold. Default is 5e5.
        """
        file_gal = f'{datadir}/{tree_file}.npz'
        self.flattening = flattening
        self.fd = disk_fraction
        self.stellar_mass_floor = stellar_mass_floor

        print('path to galaxy', file_gal)
        print('flattening', flattening)
        print('disk fraction', disk_fraction)
        print('stellar_mass_floor', stellar_mass_floor)
        
        # Initialize arrays to hold host and subhalo properties
        self.host_coords_alltime_list = [] #host properties (virial mass, Dekel concentration, inner density slope, virial overdensity, virial radius, stellar mass) over all time
        self.idx_zaccs = [] #index at which subhalos are accreted
        self.coordinates_hold = [] # cylindrical position coordinates of subhalos
        self.velocities_hold = [] # cylindrical velocity coordinates of subhalos
        self.pericenter_locs = [] # indices of subhalo pericenters
        self.apocenter_locs = [] # indices of subhalo apocenters
        self.all_apo = [] # values of subhalo apocenters
        self.all_peri = [] # values of subhalo pericenters
        self.coordinates_peri = [] # cylindrical position coordinates of subhalo pericenters
        self.coordinates_apo = [] # cylindrical position coordinates of subhalo apocenters
        self.velocities_apo = [] # cylindrical velocity coordinates of subhalo apocenters
        self.velocities_peri = [] # cylindrical velocity coordinates of subhalo pericenters
        self.ras = [] # subhalo distances at apocenters
        self.rps = [] # subhalo distances at pericenters
        self.eccs = [] # subhalo orbital eccentricities
        self.mass_lost_alls = [] # subhalo mass loss
        self.stellar_mass_lost_alls = [] # subhalo stellar mass loss
        self.sat_distances = [] #subhalos distances over all time
        self.velocity_dispersion = [] # velocity dispersion of subhalos at z = 0
        self.s001s = [] # s001 of subhalos at z = 0
        self.rhs = [] # halo radius within which density is Delta times rhoc [kpc] for subhalos
        self.rh_halfs = [] # half-mass radii of subhalos
        self.rho_bars = [] # mean density [M_sun kpc^-3] within radius r=sqrt(R^2+z^2) for subhalos
        self.mass_ratios = [] # ratio of final stellar mass to final halo virial mass at z = 0 for subhalos
        
        # Load host and subhalo properties from file
        with np.load(file_gal) as f:
            self.coordinates_SG = f['coordinates']  # phase space coordinates [r, phi, z, vr, vphi, vz]
            vphi = f['MaxCircularVelocity']
            self.CosmicTime = f['CosmicTime']
            final_Mvir = f['mass'][:, 0]
            final_Rvir = f['VirialRadius'][:, 0]
            virial_radii = f['VirialRadius'][:, :]
            self.host_Mvir = final_Mvir[0]
            order = f['order'][:, 0]
            order_all_time = f['order'][:, :]
            self.order_variable_SG = order_all_time
            self.parentid_SG = f['ParentID']
            final_Mstar = f['StellarMass'][:, 0]
            self.host_mstar = f['StellarMass'][0, 0]
            allz_mstar = f['StellarMass'][:, :]
            allz_m = f['mass'][:, :]
            self.zacc_idx_SG = f['mass'][:, :].argmax(axis=1)
            self.host_coord = self.coordinates_SG[0, 0, :3]
            self.host_Rvir = final_Rvir[0]
            concentration = f['concentration'][:, 0]
            self.redshift = f['redshift'][:]
            self.host_conc = concentration[0]
            Dekel_conc = f['DekelConcentration'][:, 0]
            Dekel_slope = f['DekelSlope'][:, 0]
            Dekel_virialovd = f['VirialOverdensity'][:, 0]
            self.host_at_Mvir = f['mass'][0, :]
            self.host_at_dconc = f['DekelConcentration'][0, :]
            self.host_at_slope = f['DekelSlope'][0, :]
            self.host_at_vovd = f['VirialOverdensity'][0, :]
            self.host_at_rvir = f['VirialRadius'][0, :]
            self.host_at_mstar = f['StellarMass'][0, :]
            self.order3_locs_SG = np.where(order_all_time == 3)
            self.parent_zacc_SG = np.zeros_like(self.zacc_idx_SG)
            num_sat = np.arange(0, len(self.zacc_idx_SG), 1)
            
            # Correct higher-order satellites' coordinates
            self.correct_higher_order()

        # Store host coordinates and satellite properties
        self.host_coords_dekel = np.array([self.host_Mvir, Dekel_conc[0], Dekel_slope[0], Dekel_virialovd[0], self.host_Rvir, self.host_mstar])
        self.host_coords_alltime = np.array([self.host_at_Mvir, self.host_at_dconc, self.host_at_slope, self.host_at_vovd, self.host_at_rvir, self.host_at_mstar])
        self.host_coords_alltime_list.append(self.host_coords_alltime)
        
        # Create a mask for surviving satellites above the stellar mass threshold
        sm_zacc = np.max(allz_mstar, axis=1)
        surviving_sats = (final_Mvir < self.host_Mvir) & (order == 1) & (sm_zacc >= self.stellar_mass_floor)
        
        # Store satellite coordinates and velocities for surviving satellites
        self.array_s = self.coordinates_SG[surviving_sats, :, :3]
        self.vels_array_s = self.coordinates_SG[surviving_sats, :, 3:]
        self.cyl_to_cart_coord()
        
        # Store various satellite properties
        self.sat_final_mvir = np.array(final_Mvir[surviving_sats])
        self.sat_final_rvir = np.array(final_Rvir[surviving_sats])
        self.sat_final_mstar = np.array(final_Mstar[surviving_sats])
        self.Rvirs = virial_radii[surviving_sats, :]
        self.Dekel_conc_s = np.array(Dekel_conc[surviving_sats])
        self.Dekel_slope_s = np.array(Dekel_slope[surviving_sats])
        self.Dekel_virialovd_s = np.array(Dekel_virialovd[surviving_sats])
        self.allz_mstar_s = allz_mstar[surviving_sats, :]
        self.allz_m_s = allz_m[surviving_sats, :]
        self.vphi_max_s = vphi[surviving_sats, 0]
        self.num_sat_s = num_sat[surviving_sats]
        self.parentid_s = self.parentid_SG[surviving_sats, :]
        self.parentzacc_s = self.parent_zacc_SG[surviving_sats]
        
        self.properties()

    def correct_higher_order(self):
        """
        Correct the coordinates of higher-order satellites to be relative to the primary host. By default, coordinates are with respect to the satellite's immediate parent. 
        """
        iterate = 0
        for loc_i, z_i in zip(self.order3_locs_SG[0], self.order3_locs_SG[1]):
            self.parent_zacc_SG[loc_i] = self.zacc_idx_SG[self.parentid_SG[loc_i, z_i]]
            self.coordinates_SG[loc_i, z_i, :] = aux.add_cyl_vecs(self.coordinates_SG[loc_i, z_i, :], self.coordinates_SG[self.parentid_SG[loc_i, z_i], z_i, :])
            self.order_variable_SG[loc_i, z_i] = 2
            iterate += 1

        order2_locs = np.where(self.order_variable_SG == 2)
        iterate = 0
        for loc_i, z_i in zip(order2_locs[0], order2_locs[1]):
            if self.parent_zacc_SG[loc_i] == 0:
                self.parent_zacc_SG[loc_i] = self.zacc_idx_SG[self.parentid_SG[loc_i, z_i]]
            elif (self.parent_zacc_SG[loc_i] != 0) and (self.zacc_idx_SG[self.parentid_SG[loc_i, z_i]] < self.parent_zacc_SG[loc_i]):
                self.parent_zacc_SG[loc_i] = self.zacc_idx_SG[self.parentid_SG[loc_i, z_i]]
            self.coordinates_SG[loc_i, z_i, :] = aux.add_cyl_vecs(self.coordinates_SG[loc_i, z_i, :], self.coordinates_SG[self.parentid_SG[loc_i, z_i], z_i, :])
            self.order_variable_SG[loc_i, z_i] = 1
            iterate += 1

        return
            
    def cyl_to_cart_coord(self):
        """
        Convert cylindrical coordinates to Cartesian coordinates for surviving satellites.
        """
        self.z = self.array_s[:, :, 2]
        self.x = self.array_s[:, :, 0] * np.cos(self.array_s[:, :, 1])
        self.y = self.array_s[:, :, 0] * np.sin(self.array_s[:, :, 1])
        return 

            
    def properties(self):
        """
        Calculate various properties of the surviving satellites, including pericenter, apocenter, 
        velocity dispersion, and mass loss.
        """
        for sat_i in range(len(self.array_s[:, 0, 2])):
            idxzacc = self.allz_m_s[sat_i, :].argmax(axis=0)
            
            # Determine the accretion index
            if (self.parentzacc_s[sat_i] != 0) and (self.parentzacc_s[sat_i] < idxzacc):
                self.idx_zaccs.append(self.parentzacc_s[sat_i])
                idxzacc = self.parentzacc_s[sat_i]
            else:
                self.idx_zaccs.append(idxzacc)
            
            # Calculate satellite distance
            if idxzacc != 0:
                satdist_i = np.linalg.norm(np.array([self.x[sat_i, :idxzacc], self.y[sat_i, :idxzacc], self.z[sat_i, :idxzacc]]).T, axis=1)
            else:
                satdist_i = np.linalg.norm(np.array([self.x[sat_i], self.y[sat_i], self.z[sat_i]]).T, axis=1)

            # Find pericenter and apocenter locations
            min_loc = argrelmin(satdist_i[np.nonzero(satdist_i)], order=2)  # Pericenter locations
            max_loc = argrelmax(satdist_i[np.nonzero(satdist_i)], order=2)  # Apocenter locations
            min_loc = min_loc[0]  # Correct array structure
            max_loc = max_loc[0]  # Correct array structure   
            self.pericenter_locs.append(min_loc)

            min_loc_hold = []  # Holds most recent pericenter location
            max_loc_hold = []  # Holds most recent apocenter location
            
            try:
                # Try to find the most recent pericenter
                rp = satdist_i[min_loc][0]
                rp_1 = satdist_i[min_loc][-1]
                min_loc_hold = np.where(satdist_i[np.nonzero(satdist_i)] == rp_1)[0][0]
            except: 
                # If exception, use global pericenter
                rp = np.min(satdist_i[np.nonzero(satdist_i)]) 
                min_loc_hold = np.where(satdist_i[np.nonzero(satdist_i)] == rp)[0][0]

            ra = 0
            max_loc_cutoff = []

            # Determine apocenter location
            if len(max_loc) > 1:
                for max_loc_i in max_loc:
                    if max_loc_i < min_loc_hold:
                        ra_hold = satdist_i[max_loc_i]
                        if ra_hold > ra:
                            ra = ra_hold
                            max_loc_hold = max_loc_i  # Nearest local max after the first pericenter
                            max_loc_cutoff = max_loc[:max_loc_hold]
                ra = satdist_i[max_loc_cutoff][0]
                            
            if not max_loc_hold:
                ra = np.max(satdist_i[np.nonzero(satdist_i)])
                max_loc_hold = np.where(satdist_i[np.nonzero(satdist_i)] == ra)[0][0]
                max_loc_cutoff = max_loc

            # Store apocenter and pericenter information
            self.apocenter_locs.append(max_loc_cutoff)
            self.all_apo.append(satdist_i[np.nonzero(satdist_i)][max_loc_cutoff])
            self.all_peri.append(satdist_i[np.nonzero(satdist_i)][min_loc])
            self.coordinates_peri.append(self.array_s[sat_i, min_loc, :])
            self.coordinates_apo.append(self.array_s[sat_i, max_loc_cutoff, :])
            self.velocities_apo.append(self.vels_array_s[sat_i, max_loc_cutoff, :])
            self.velocities_peri.append(self.vels_array_s[sat_i, min_loc, :])
            self.ras.append(ra)
            self.rps.append(rp)
            self.eccs.append((ra - rp) / (ra + rp))
            
            #Store all coordinate information
            self.coordinates_hold.append(self.array_s[sat_i, :, :])
            self.velocities_hold.append(self.vels_array_s[sat_i, :, :])

            # Calculate satellite strucutral properties
            sat_profile = Dekel(self.sat_final_mvir[sat_i], self.Dekel_conc_s[sat_i], self.Dekel_slope_s[sat_i], self.Dekel_virialovd_s[sat_i], z=0.)
            r_half = Reff(self.sat_final_rvir[sat_i], sat_profile.ch)
            disk_scale_length = 0.766421 / (1. + 1. / self.flattening) * r_half
            disk_scale_height = disk_scale_length / self.flattening 
            sat_disk_profile = MN(self.fd * self.sat_final_mvir[sat_i], disk_scale_length, disk_scale_height)
            sat_stellar_disk_profile = MN(self.fd * self.sat_final_mstar[sat_i], disk_scale_length, disk_scale_height)
            
            self.velocity_dispersion.append(sat_profile.sigma(self.sat_final_rvir[sat_i]))
            self.s001s.append(sat_profile.s001)
            self.rhs.append(sat_profile.rh)
            self.rh_halfs.append(r_half)
            self.rho_bars.append(sat_profile.rhobar(r_half))
            self.mass_ratios.append(self.sat_final_mstar[sat_i] / self.sat_final_mvir[sat_i]) 
      
            # Calculate mass loss
            if idxzacc != 0:
                origin_mass_all = np.max(self.allz_m_s[sat_i, :idxzacc][self.allz_m_s[sat_i, :idxzacc] != -99], axis=0)
                origin_stellar_mass_all = np.max(self.allz_mstar_s[sat_i, :idxzacc][self.allz_mstar_s[sat_i, :idxzacc] != -99], axis=0)
            else:
                origin_mass_all = np.max(self.allz_m_s[sat_i][self.allz_m_s[sat_i] != -99], axis=0)
                origin_stellar_mass_all = np.max(self.allz_mstar_s[sat_i][self.allz_mstar_s[sat_i] != -99], axis=0)
                
            self.mass_lost_alls.append(np.divide(np.subtract(origin_mass_all, self.allz_m_s[sat_i, 0][self.allz_m_s[sat_i, 0] != -99]), origin_mass_all))
            self.stellar_mass_lost_alls.append(np.divide(np.subtract(origin_stellar_mass_all, self.allz_mstar_s[sat_i, 0][self.allz_mstar_s[sat_i, 0] != -99]), origin_stellar_mass_all))
            self.sat_distances.append(satdist_i)
        return
    
    def get_filtered_rows_as_dataframe(self, attributes):
        """
        Return filtered (cut to for satellites to remain inside host at z = 0 and have a pericenter greater than 4 kpc) satellite data as a pandas DataFrame based on specific attributes.

        Args:
            attributes (list): List of attributes to include in the DataFrame.

        Returns:
            pandas.DataFrame: Filtered satellite data.
        """
        rows_data = []
        
        for sat_i in range(len(self.apocenter_locs)):
            if (len(self.apocenter_locs[sat_i]) > 0 and 
                self.sat_distances[sat_i][0] < self.host_coords_dekel[4] and
                self.sat_distances[sat_i][0] > 0 and
                not any(i <= 4 for i in self.sat_distances[sat_i][np.nonzero(self.sat_distances[sat_i])])):
                
                row_data = {}
                for attr in attributes:
                    if hasattr(self, attr):
                        attr_values = getattr(self, attr)
                        row_data[attr] = attr_values[sat_i]
                    else:
                        raise ValueError(f"Attribute {attr} not found in Galaxy object.")
                
                row_data['origin_masses'] = np.max(self.allz_mstar_s[sat_i], axis=0)
                row_data['time_between_apocenters'] = np.diff(cosmo.age(self.redshift[self.apocenter_locs[sat_i]]) / u.Gyr)
                
                rows_data.append(row_data)
        
        rows_df = pd.DataFrame(rows_data)
        
        return rows_df
    
    def flatten_attributes(self):
        """
        Flatten specific attributes for easier analysis.
        """
        attributes_to_flatten = [
            'rhs', 'rho_bars', 'rh_halfs', 'mass_ratios', 'rps', 'ras', 'eccs', 'sat_final_mstars', 'sat_final_mvirs',
            'sat_final_rvirs', 'stellar_mass_lost_alls', 'mass_lost_alls',
            'idx_zaccs','allz_mstar_s','allz_m_s'
        ]
        
        for attr in attributes_to_flatten:
            if hasattr(self, attr):
                setattr(self, attr, np.array(getattr(self, attr)).flatten())
        return
    
    def get_host_profile(self, redshift_id):
        """
        Return the host profile at a specific redshift.

        Args:
            redshift_id (int): The redshift index.

        Returns:
            list: Host profile including both the halo and disk components.
        """
        host_profile = Dekel(self.host_coords_alltime[0, redshift_id], self.host_coords_alltime[1, redshift_id], self.host_coords_alltime[2, redshift_id], self.host_coords_alltime[3, redshift_id], z=0.)
        r_half_host = Reff(self.host_coords_alltime[4, redshift_id], host_profile.ch)
        disk_scale_length_host = 0.766421 / (1. + 1. / self.flattening) * r_half_host
        disk_scale_height_host = disk_scale_length_host / self.flattening
        
        host_disk_profile = MN(self.host_coords_alltime[0, redshift_id] * self.fd, disk_scale_length_host, disk_scale_height_host)
        total_profile = [host_profile, host_disk_profile]
        
        return total_profile
