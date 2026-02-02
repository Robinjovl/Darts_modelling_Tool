import numpy as np
from darts.input.input_data import InputData
from case_base import input_data_base

def input_data_case_5x3x4(idata: InputData, case: str):
    input_data_base(idata, case)

    geom = idata.geom  # a short name
    well_data = idata.well_data  # a short name

    geom.nx = 5
    geom.ny = 3
    geom.nz = 4
    geom.start_z = 1000  # top reservoir depth
    # non-uniform layers thickness
    geom.dx = np.array([500, 200, 100, 300, 500])
    geom.dy = np.array([1000, 700, 300])
    geom.dz = np.array([100, 150, 180, 120])

    # vertical wells, the 'k' index is unused (so set to -1)
    if 'wperiodic' in case:
        well_data.add_well(name='W', loc_type='ijk', loc_ijk=(geom.nx // 2, geom.ny // 2, -1))
    else:
        well_data.add_well(name='PRD', loc_type='ijk', loc_ijk=(2, 1, -1))
        well_data.add_well(name='INJ', loc_type='ijk', loc_ijk=(4, 3, -1))
        # one might use wells.add_well(name='PRD', loc_type='xyz', loc_xyz=(250.0, 500.0, 890.0))

