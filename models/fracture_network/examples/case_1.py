import os
from .input_default import input_data_default

def input_data_case_1():
    idata = input_data_default()

    idata.geom['case_name'] = 'case_1'

    # geometry
    idata.geom['frac_file'] = os.path.join('examples', 'frac_1.txt')

    # do not use cleaned mesh
    idata.geom['mesh_prefix'] = 'raw_lc'

    # cell sizes
    idata.geom['char_len'] = 50  # near fractures (characteristic length for cleaning and mesh generation) [m]
    idata.geom['char_len_boundary'] = 150  # grid size near grid boundaries [m]
    idata.geom['char_len_well'] = 50  # grid size near wells [m]

    # uniform initial pressure and temperature
    idata.initial.type ='uniform'
    idata.initial.initial_pressure = 350.  # bar
    idata.initial.initial_temperature = 348.15  # K

    # well locations
    idata.geom['inj_well_coords'] = [[100, 200, 25]]  # X, Y, Z (only one perforation)
    idata.geom['prod_well_coords'] = [[800, 800, 25]]

    # well in the matrix cells or in the fractures
    idata.geom['well_loc_type'] = 'wells_in_nearest_cell'

    # extrusion - number of layers by Z axis
    idata.geom['rsv_layers'] = 3

    idata.geom['z_top'] = 2000  # [m]
    idata.geom['height_res'] = 20  # [m]

    idata.geom['frac_aper'] = 1e-3  # (initial) fracture aperture [m]

    wctrl = idata.well_data.controls
    wctrl.delta_temp = 40  # inj_temp = initial_temp - delta_temp
    wctrl.delta_p_inj  = 30  # inj_bhp = initial_pressure + delta_p_inj
    wctrl.delta_p_prod = 30  # inj_prod = initial_pressure - delta_p_prod

    #idata.geom['box_data'] = np.array([[0, 0], [0, 1000], [1000, 0], [1000, 1000]])

    return idata


