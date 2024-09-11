import os
from .input_default import input_data_default

def input_data_case_3():
    input_data = input_data_default()

    input_data['case_name'] = 'case_3'

    # geometry
    input_data['frac_file'] = os.path.join('examples', 'frac_3.txt')

    # cell sizes
    input_data['char_len'] = 10  # near fractures (characteristic length for cleaning and mesh generation) [m]
    input_data['char_len_boundary'] = 100  # grid size near grid boundaries [m]
    input_data['char_len_well'] = 50  # grid size near wells [m]

    # do not use cleaned mesh
    input_data['mesh_prefix'] = 'raw_lc'

    # principal stress, MPa
    input_data['Sh_min'] = 1
    input_data['Sh_max'] = 20
    input_data['SHmax_azimuth'] = 20  #° from X, counter-clockwize

    # well locations
    input_data['inj_well_coords'] = [[400, 400, 0],[400, -400, 0],[-400, 400, 0],[-400, -400, 0]]  # X, Y, Z (only one perforation)
    input_data['prod_well_coords'] = [[100, 0, 0], [0, 100, 0], [-100, 0, 0], [0, -100, 0]]

    input_data['frac_aper'] = 1e-2  # (initial) fracture aperture [m]

    return input_data