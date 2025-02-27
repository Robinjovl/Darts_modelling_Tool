import numpy as np
from darts.input.input_data import InputData
from case_base import input_data_base

def input_data_case_51x51x1(idata: InputData, case: str):
    input_data_base(idata, case)

    geom = idata.geom  # a short name
    well_data = idata.well_data  # a short name

    geom.nx = 51
    geom.ny = 51
    geom.nz = 1
    geom.dx = 4000. / geom.nx
    geom.dy = geom.dx
    geom.dz = 100. / geom.nz
    geom.start_z = 2000  # top reservoir depth

    # vertical wells locations, 1-based indices
    if 'wperiodic' in case:
        well_data.add_well(name='W', loc_type='ijk', loc_ijk=(geom.nx // 2, geom.ny // 2, -1))
    else:
        well_data.add_well(name='PRD', loc_type='ijk', loc_ijk=(
        geom.nx // 2 - int(500 // geom.dx), geom.ny // 2, -1))  # I = 0.5 km to the left from the center
        well_data.add_well(name='INJ', loc_type='ijk', loc_ijk=(
        geom.nx // 2 + int(500 // geom.dx), geom.ny // 2, -1))  # I = 0.5 km to the right from the center

