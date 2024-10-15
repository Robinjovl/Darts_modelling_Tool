import os
import numpy as np
from .input_default import input_data_default

def input_data_case_4():
    input_data = input_data_default()

    input_data['case_name'] = 'case_4'

    # geometry
    input_data['frac_file'] = os.path.join('examples', 'frac_4.txt')

    # do not use cleaned mesh
    input_data['mesh_prefix'] = 'raw_lc'

    # cell sizes
    input_data['char_len'] = 30  # near fractures (characteristic length for cleaning and mesh generation) [m]
    input_data['char_len_boundary'] = 100  # grid size near grid boundaries [m]
    input_data['char_len_well'] = 5  # grid size near wells [m]
    input_data['margin'] = 400  # [m]

    # initial pressure and temperature
    # uniform initial pressure and temperature
    input_data['initial_uniform'] = True
    input_data['uniform_pressure'] = 350.  # bar
    input_data['uniform_temperature'] = 350.  # K

    # well locations
    input_data['inj_well_coords'] = [[0, 0, 25]]  # X, Y, Z (only one perforation)
    input_data['prod_well_coords'] = [[50, 50, 25]]

    # well in the matrix cells or in the fractures
    input_data['well_loc_type'] = 'wells_in_nearest_cell'

    # extrusion - number of layers by Z axis
    input_data['rsv_layers'] = 1

    input_data['z_top'] = 2000  # [m]
    input_data['height_res'] = 10  # [m]

    input_data['frac_aper'] = 1e-9  # (initial) fracture aperture [m]

    # well controls
    input_data['rate_prod'] = 0  # m3/day
    input_data['rate_inj'] = 1000  # m3/day
    input_data['delta_temp'] = 40  # inj_temp = initial_temp - delta_temp

    input_data['hcap'] = 2200.  # [kJ/m3/K]
    input_data['conduction'] = 200  # [kJ/m/day/K]

    input_data['permx'] = 100
    input_data['permy'] = 1
    input_data['permz'] = 1

    # for non-isotropic perm => grid rotation
    SHmax_azimuth = 0  # [°] from Y, clockwise
    input_data['SHmax_azimuth'] = 90 - SHmax_azimuth  # [°] from X, counter-clockwise

    return input_data

def input_data_case_4_no_conduction():
    input_data = input_data_case_4()
    input_data['case_name'] = 'case_4_no_conduction'
    input_data['conduction'] = 0  # [kJ/m/day/K]
    return input_data

def input_data_case_4_small_capacity():
    input_data = input_data_case_4()
    input_data['case_name'] = 'case_4_small_capacity'
    input_data['hcap'] = 1  # [kJ/m3/K]
    return input_data