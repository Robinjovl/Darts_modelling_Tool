import os
from .input_default import input_data_default

def input_data_case_whitby():
    input_data = input_data_default()

    input_data['case_name'] = 'whitby'

    # geometry for mesh generation
    input_data['margin'] = 25 # [m]
    input_data['frac_file'] = os.path.join('examples', 'Whitby_raw.txt')

    # cell sizes for mesh generation
    input_data['char_len'] = 16  # near fractures (characteristic length for cleaning and mesh generation) [m]
    input_data['char_len_boundary'] = input_data['char_len']

    # use cleaned mesh
    input_data['mesh_prefix'] = 'mergefac_0.86_clean_lc'
    input_data['mesh_clean'] = True  # need gmsh installed and callable from command line in order to mesh

    # extrusion - number of layers by Z axis
    input_data['rsv_layers'] = 1

    # well locations
    input_data['inj_well_coords'] = [[400, 800, 25]]  # X, Y, Z (only one perforation)
    input_data['prod_well_coords'] = [[400, 200, 25]]

    # well in the matrix cells or in the fractures
    input_data['well_loc_type'] = 'wells_in_frac'

    input_data['poro'] = 0.2
    input_data['perm'] = 100 # [mD]

    input_data['hcap'] = 2200. # [kJ/m3/K]
    input_data['conduction'] = 181.44  # [kJ/m/day/K]

    # well controls
    input_data['rate_prod'] = None  # m3/day
    input_data['rate_inj']  = None  # m3/day
    input_data['delta_temp'] = 40  # inj_temp = initial_temp - delta_temp
    input_data['delta_p_inj']  = 50  # inj_bhp = initial_pressure + delta_p_inj
    input_data['delta_p_prod'] = 25  # inj_prod = initial_pressure - delta_p_prod

    # principal stress, MPa
    input_data['Sh_min'] = 50
    input_data['Sh_max'] = 90
    input_data['Sv'] = 120
    input_data['SHmax_azimuth'] = 0  #° from X, counter-clockwize

    # initial pressure and temperature
    # uniform initial pressure and temperature
    input_data['initial_uniform'] = True
    input_data['uniform_pressure'] = 350.  # bar
    input_data['uniform_temperature'] = 348.15  # K

    return input_data