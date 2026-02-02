import numpy as np
from darts.input.input_data import InputData
from case_base import input_data_base

def set_fault_mult(idata: InputData):
    geom = idata.geom  # a short name
    geom.faultfile = 'faults_case_51x51x1.txt'

    fault_name = 'FLT1'
    # fault location
    i1, i2 = 25, 26
    j1, j2 = 11, 35
    # fault transmissibility multiplier
    mult = 0.
    # generate a string with fault data and write it to a file
    s = ''
    for k in range(1, idata.geom.nz + 1 + geom.burden_layers * 2):
        for j in range(j1, j2):
            s += ' '.join(map(lambda x: str(x), [fault_name, i1, j, k, i2, j, k, mult])) + '\n'
    with open(geom.faultfile, 'w') as f:
        f.writelines(s)

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

    if 'faultmult' in case:
        set_fault_mult(idata)

        dt = 365.25  # one report timestep length, [days]
        n_time_steps = 100
        idata.sim.time_steps = np.zeros(n_time_steps) + dt
