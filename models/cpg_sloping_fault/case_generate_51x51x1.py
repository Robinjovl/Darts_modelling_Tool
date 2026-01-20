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

    geom.nx = 10
    geom.ny = 10
    geom.nz = 14
    geom.dx = 0.03 / geom.nx
    geom.dy = geom.dx
    geom.dz = 0.07 / geom.nz

    idata.rock.poro = 1e-3
    idata.rock.permx = 1e-5  # mD
    idata.rock.permy = 1e-5  # mD
    idata.rock.permz = 1e-5   # mD

    # grid processing parameters
    geom.minpv = 0. # minimal pore volume threshold to set cells inactive, m^3
    geom.min_poro = 0.
    geom.min_perm = 0.
    geom.bound_volume_xy = None # lateral boundary volume, m^3
    geom.bound_volume_z_top = None # top layer boundary volume, m^3
    geom.bound_volume_z_bottom = 1000 # bottom layer boundary volume, m^3

    geom.burden_layers = 1  # the number of additional (generated on-the-fly) overburden/underburden layers
    geom.burden_init_thickness = geom.dz  # first over/under burden layer thickness, [m.]
    idata.rock.burden_prop = 1e-5  # perm and poro value for burden layers

    geom.start_z = geom.burden_init_thickness  # top reservoir depth

    idata.rock.conduction_shale = 200 # Shale conductivity kJ/m/day/K
    idata.rock.conduction_sand = 200 # Sandstone conductivity kJ/m/day/K
    idata.rock.hcap_shale = 2000 # Shale heat capacity kJ/m3/K
    idata.rock.hcap_sand = 2000 # Sandstone heat capacity kJ/m3/K

    idata.initial.type = 'depth_table'

    eps = 1e-3 # m
    # set the first layer temperature 60 degrees, the rest 100 degrees
    idata.initial.depth = [0.,
                           geom.start_z - eps,
                           geom.start_z + eps,
                           geom.start_z + 2*eps]
    idata.initial.pressure_vs_depth = [100.] * len(idata.initial.depth)
    idata.initial.temperature_vs_depth = [273.15 + 60]*2 + [273.15 + 100]*2

    dt = 10./24/60  # one report timestep length, [days]
    n_time_steps = 60*3
    idata.sim.time_steps = np.zeros(n_time_steps) + dt
