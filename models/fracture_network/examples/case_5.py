import os
import numpy as np
from .input_default import input_data_default

def input_data_case_5():
    input_data = input_data_default()
    input_data['case_name'] = 'case_5'
    # geometry
    input_data['frac_file'] = os.path.join('examples', 'frac_5.txt')

    # use cleaned mesh
    input_data['mesh_clean'] = True  # need gmsh installed and callable from command line in order to mesh
    input_data['mesh_prefix'] = 'mergefac_0.86_clean_lc'

    # do not use cleaned mesh
    #input_data['mesh_prefix'] = 'raw_lc'

    # cell sizes
    input_data['char_len'] = 50  # near fractures (characteristic length for cleaning and mesh generation) [m]
    input_data['char_len_boundary'] = 150  # grid size near grid boundaries [m]
    input_data['margin'] = 100  # [m]

    input_data['z_top'] = 2000  # [m]
    input_data['height_res'] = 10  # [m]

    input_data['perm'] = 100 # [mD]

    # uniform initial pressure and temperature
    input_data['initial_uniform'] = True
    input_data['uniform_pressure'] = 250.  # bar
    input_data['uniform_temperature'] = 380.  # K

    input_data['well_loc_type'] = 'wells_in_frac'

    # well locations
    input_data['inj_well_coords'] = [[200, 200, 2000]]  # X, Y, Z (only one perforation)
    input_data['prod_well_coords'] = [[800, 800, 2000]]

    input_data['delta_temp'] = 40   # bars. inj_temp = initial_temp - delta_temp
    input_data['delta_p_inj']  = 20  # bars. inj_bhp = initial_pressure + delta_p_inj
    input_data['delta_p_prod'] = 20  # bars. inj_prod = initial_pressure - delta_p_prod

    return input_data