import numpy as np
from darts.input.input_data import InputData
from case_base import input_data_base

def input_data_case_100x100x100(idata: InputData, case: str):
    input_data_base(idata, case)

    geom = idata.geom  # a short name
    well_data = idata.well_data  # a short name

    geom.nx = geom.ny = geom.nz = 100
    geom.dx = geom.dy = 10
    geom.dz = 1
    geom.start_z = 2000  # top reservoir depth
    geom.burden_layers = 4
    # vertical wells locations, 1-based indices
    well_data.add_well(name='PRD', loc_type='ijk', loc_ijk=(50, 20, -1))
    well_data.add_well(name='INJ', loc_type='ijk', loc_ijk=(50, 80, -1))

