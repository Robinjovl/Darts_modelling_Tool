import numpy as np
from darts.input.input_data import InputData

def set_input_data_geom_grdecl(idata: InputData, case: str):
    geom = idata.geom  # a short name

    # idata.gridfile is defined in get_case_files (case_base.py)
    # it is grid.grdecl by default and reservoir.in for the rock properties (poro, perm) but one can redefine that here:
    #idata.gridfile =
    #idata.propfile =
    #idata.schfile =

    # properties can be replaced here as well, if needed

    if 'noburdenlayers' in case:
        # only for the thermal case (Geothermal physics):
        geom.burden_layers = 0  # the number of additional (generated on-the-fly) overburden/underburden layers

    #idata.rock.hcap_sand = 1800 # Sandstone heat capacity kJ/m3/K

    return

