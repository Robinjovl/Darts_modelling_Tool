import os
from .input_default import input_data_default

def input_data_case_whitby():
    idata = input_data_default()

    idata.geom['case_name'] = 'whitby'

    # geometry for mesh generation
    idata.geom['margin'] = 25 # [m]
    idata.geom['frac_file'] = os.path.join('examples', 'Whitby_raw.txt')

    # cell sizes for mesh generation
    idata.geom['char_len'] = 16  # near fractures (characteristic length for cleaning and mesh generation) [m]
    idata.geom['char_len_boundary'] = idata.geom['char_len']

    # use cleaned mesh
    idata.geom['mesh_prefix'] = 'mergefac_0.86_clean_lc'
    idata.geom['mesh_clean'] = True  # need gmsh installed and callable from command line in order to mesh

    # extrusion - number of layers by Z axis
    idata.geom['rsv_layers'] = 1

    # well locations
    idata.geom['inj_well_coords'] = [[400, 800, 25]]  # X, Y, Z (only one perforation)
    idata.geom['prod_well_coords'] = [[400, 200, 25]]

    # well in the matrix cells or in the fractures
    idata.geom['well_loc_type'] = 'wells_in_frac'

    idata.rock.porosity = 0.2
    idata.rock.permx = 100  # [mD]

    idata.rock.heat_capacity = 2200. # [kJ/m3/K]
    idata.rock.conductivity = 181.44  # [kJ/m/day/K]

    # well controls
    wctrl = idata.well_data.controls
    wctrl.prod_rate = None  # m3/day. if None, well will work under BHP control
    wctrl.inj_rate = None   # m3/day. if None, well will work under BHP control
    wctrl.delta_temp = 40   # bars. inj_temp = initial_temp - delta_temp
    wctrl.delta_p_inj  = 50  # bars. inj_bhp = initial_pressure + delta_p_inj
    wctrl.delta_p_prod = 25  # bars. inj_prod = initial_pressure - delta_p_prod

    # principal stress, MPa
    idata.stress['Sh_min'] = 50
    idata.stress['Sh_max'] = 90
    idata.stress['Sv'] = 120
    idata.stress['SHmax_azimuth'] = 0  #° from X, counter-clockwize

    # initial pressure and temperature
    idata.initial.type = 'uniform'
    idata.initial.initial_pressure = 350.  # bars
    idata.initial.initial_temperature = 348.15   # K

    return idata