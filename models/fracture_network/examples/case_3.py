import os
from .input_default import input_data_default

def input_data_case_3():
    idata = input_data_default()

    idata.geom['case_name'] = 'case_3'

    # geometry
    idata.geom['frac_file'] = os.path.join('examples', 'frac_3.txt')

    # cell sizes
    idata.geom['char_len'] = 10  # near fractures (characteristic length for cleaning and mesh generation) [m]
    idata.geom['char_len_boundary'] = 100  # grid size near grid boundaries [m]
    idata.geom['char_len_well'] = 50  # grid size near wells [m]

    # do not use cleaned mesh
    idata.geom['mesh_prefix'] = 'raw_lc'

    # principal stress, MPa
    idata.stress['Sh_min'] = 1
    idata.stress['Sh_max'] = 20
    idata.stress['SHmax_azimuth'] = 20  #° from X, counter-clockwize

    # well locations
    idata.geom['well_coords'] = dict()
    idata.geom['well_coords']['I1'] = [400, 400, 0, 0]  # X, Y, Z1, Z2
    idata.geom['well_coords']['I2'] = [400, -400, 0, 0]  # X, Y, Z1, Z2
    idata.geom['well_coords']['I3'] = [-400, 400, 0, 0]  # X, Y, Z1, Z2
    idata.geom['well_coords']['I4'] = [-400, -400, 0, 0]  # X, Y, Z1, Z2

    idata.geom['well_coords']['P1'] = [100, 0, 0, 0]  # X, Y, Z1, Z2
    idata.geom['well_coords']['P2'] = [0, 100, 0, 0]  # X, Y, Z1, Z2
    idata.geom['well_coords']['P3'] = [-100, 0, 0, 0]  # X, Y, Z1, Z2
    idata.geom['well_coords']['P4'] = [0, -100, 0, 0]  # X, Y, Z1, Z2

    idata.geom['frac_aper'] = 1e-2  # (initial) fracture aperture [m]

    return idata
