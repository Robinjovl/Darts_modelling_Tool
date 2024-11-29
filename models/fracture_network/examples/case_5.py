import os
import numpy as np
from .input_default import input_data_default

def input_data_case_5():
    idata = input_data_default()
    idata.geom['case_name'] = 'case_5'
    # geometry
    idata.geom['frac_file'] = os.path.join('examples', 'frac_5.txt')

    # use cleaned mesh
    idata.geom['mesh_clean'] = True  # need gmsh installed and callable from command line in order to mesh
    idata.geom['mesh_prefix'] = 'mergefac_0.86_clean_lc'

    # do not use cleaned mesh
    #idata.geom['mesh_prefix'] = 'raw_lc'

    # cell sizes
    idata.geom['char_len'] = 50  # near fractures (characteristic length for cleaning and mesh generation) [m]
    idata.geom['char_len_boundary'] = 150  # grid size near grid boundaries [m]
    idata.geom['margin'] = 100  # [m]

    idata.geom['z_top'] = 2000  # [m]
    idata.geom['height_res'] = 10  # [m]

    idata.rock.perm = 100  # [mD]

    # initial pressure and temperature
    idata.initial.type = 'uniform'
    idata.initial.initial_pressure = 250.  # bars
    idata.initial.initial_temperature = 380.  # K

    idata.geom['well_loc_type'] = 'wells_in_frac'

    # well locations
    idata.geom['inj_well_coords'] = [[200, 200, 2000]]  # X, Y, Z (only one perforation)
    idata.geom['prod_well_coords'] = [[800, 800, 2000]]

    wctrl = idata.well_data.controls
    wctrl.delta_temp = 40   # bars. inj_temp = initial_temp - delta_temp
    wctrl.delta_p_inj  = 20  # bars. inj_bhp = initial_pressure + delta_p_inj
    wctrl.delta_p_prod = 20  # bars. inj_prod = initial_pressure - delta_p_prod

    return idata