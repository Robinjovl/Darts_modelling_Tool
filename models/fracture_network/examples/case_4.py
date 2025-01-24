import os
import numpy as np
from .input_default import input_data_default

def input_data_case_4():
    idata = input_data_default()

    idata.geom['case_name'] = 'case_4'

    # geometry
    idata.geom['frac_file'] = os.path.join('examples', 'frac_4.txt')

    # do not use cleaned mesh
    idata.geom['mesh_prefix'] = 'raw_lc'

    # cell sizes
    idata.geom['char_len'] = 30  # near fractures (characteristic length for cleaning and mesh generation) [m]
    idata.geom['char_len_boundary'] = 100  # grid size near grid boundaries [m]
    idata.geom['char_len_well'] = 5  # grid size near wells [m]
    idata.geom['margin'] = 400  # [m]

    # initial pressure and temperature
    idata.initial.type = 'uniform'
    idata.initial.initial_pressure = 350.  # bars
    idata.initial.initial_temperature = 350.  # K

    # well locations
    idata.geom['inj_well_coords'] = [[0, 0, 25]]  # X, Y, Z (only one perforation)
    idata.geom['prod_well_coords'] = [[50, 50, 25]]

    # well in the matrix cells or in the fractures
    idata.geom['well_loc_type'] = 'wells_in_nearest_cell'

    # extrusion - number of layers by Z axis
    idata.geom['rsv_layers'] = 1

    idata.geom['z_top'] = 2000  # [m]
    idata.geom['height_res'] = 10  # [m]

    idata.geom['frac_aper'] = 1e-9  # (initial) fracture aperture [m]

    # well controls
    wctrl = idata.well_data.controls
    wctrl.rate_prod = 0  # m3/day
    wctrl.rate_inj = 1000  # m3/day
    wctrl.delta_temp = 40 # inj_temp = initial_temp - delta_temp

    idata.rock.heat_capacity = 2200. # [kJ/m3/K]
    idata.rock.conductivity = 200  # [kJ/m/day/K]

    idata.rock.permx = 100  # [mD]
    idata.rock.permy = 1  # [mD]
    idata.rock.permz = 1  # [mD]

    # for non-isotropic perm => grid rotation
    SHmax_azimuth = 0  # [°] from Y, clockwise
    idata.stress['SHmax_azimuth'] = 90 - SHmax_azimuth  # [°] from X, counter-clockwise

    return idata

def input_data_case_4_no_conduction():
    idata = input_data_case_4()
    idata.geom['case_name'] = 'case_4_no_conduction'
    idata.rock.conductivity = 0  # [kJ/m/day/K]
    return idata

def input_data_case_4_small_capacity():
    idata = input_data_case_4()
    idata.geom['case_name'] = 'case_4_small_capacity'
    idata.rock.conductivity = 1  # [kJ/m/day/K]
    return idata