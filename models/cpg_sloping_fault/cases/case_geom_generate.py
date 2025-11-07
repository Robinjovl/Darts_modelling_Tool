import numpy as np
from darts.input.input_data import InputData

def set_input_data_geom_generate(idata: InputData, case: str):

    geom = idata.geom  # a short name
    well_data = idata.well_data  # a short name

    if 'generate_5x3x4' in case:
        geom.nx = 5
        geom.ny = 3
        geom.nz = 4
        geom.start_z = 1000  # top reservoir depth
        # non-uniform layers thickness
        geom.dx = np.array([500, 200, 100, 300, 500])
        geom.dy = np.array([1000, 700, 300])
        geom.dz = np.array([100, 150, 180, 120])

        # vertical wells, the 'k' index is unused (so set to -1)
        if 'periodic' in case:
            well_data.add_well(name='W', loc_type='ijk', loc_ijk=(geom.nx // 2, geom.ny // 2, -1))
        else:
            well_data.add_well(name='PRD', loc_type='ijk', loc_ijk=(2, 1, -1))
            well_data.add_well(name='INJ', loc_type='ijk', loc_ijk=(4, 3, -1))
            # one might use wells.add_well(name='PRD', loc_type='xyz', loc_xyz=(250.0, 500.0, 890.0))
    elif 'generate_51x51x1' in case:
        geom.nx = 51
        geom.ny = 51
        geom.nz = 1
        geom.dx = 4000. / geom.nx
        geom.dy = geom.dx
        geom.dz = 100. / geom.nz
        geom.start_z = 2000  # top reservoir depth

        # vertical wells locations, 1-based indices
        if 'periodic' in case:
            well_data.add_well(name='W', loc_type='ijk', loc_ijk=(geom.nx // 2, geom.ny // 2, -1))
        else:
            well_data.add_well(name='PRD', loc_type='ijk', loc_ijk=(
                geom.nx // 2 - int(500 // geom.dx), geom.ny // 2, -1))  # I = 0.5 km to the left from the center
            well_data.add_well(name='INJ', loc_type='ijk', loc_ijk=(
                geom.nx // 2 + int(500 // geom.dx), geom.ny // 2, -1))  # I = 0.5 km to the right from the center

        if 'faultmult' in case:

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

            # then set the fault multipliers
            set_fault_mult(idata)

            # and run longer
            dt = 365.25  # one report timestep length, [days]
            n_time_steps = 100
            idata.sim.time_steps = np.zeros(n_time_steps) + dt

    elif 'generate_100x100x1' in case:
            geom.nx = geom.ny = geom.nz = 100
            geom.dx = geom.dy = 10
            geom.dz = 1
            geom.start_z = 2000  # top reservoir depth
            geom.burden_layers = 4
            # vertical wells locations, 1-based indices
            well_data.add_well(name='PRD', loc_type='ijk', loc_ijk=(50, 20, -1))
            well_data.add_well(name='INJ', loc_type='ijk', loc_ijk=(50, 80, -1))
