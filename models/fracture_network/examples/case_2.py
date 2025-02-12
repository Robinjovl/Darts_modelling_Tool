import os
from .input_default import input_data_default

def input_data_case_2():
    idata = input_data_default()

    idata.geom['case_name'] = 'case_2'

    # geometry for mesh generation
    idata.geom['margin'] = 1 # [m]
    idata.geom['frac_file'] = os.path.join('examples', 'frac_2.txt')

    # cell sizes for mesh generation
    idata.geom['char_len'] = 5  # near fractures (characteristic length for cleaning and mesh generation) [m]
    idata.geom['char_len_boundary'] = 5  # grid size near grid boundaries [m]

    # do not use cleaned mesh
    idata.geom['mesh_prefix'] = 'raw_lc'

    # well locations
    idata.geom['inj_well_coords'] = [[1, 1, 25]]  # X, Y, Z (only one perforation)
    idata.geom['prod_well_coords'] = [[99, 99, 25]]

    # well in the matrix cells or in the fractures
    idata.geom['well_loc_type'] = 'wells_in_frac'

    # well controls
    wctrl = idata.well_data.controls
    wctrl.rate_prod = None#10  # m3/day
    wctrl.rate_inj = None#5  # m3/day
    wctrl.delta_temp = 10  # inj_temp = initial_temp - delta_temp
    wctrl.delta_p = 5  # inj_bhp = initial_pressure + delta_p

    return idata