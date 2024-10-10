import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from model import Model
from darts.tools.hdf5_tools import load_hdf5_to_dict

def molar_rate_operators(state):
    pc.evaluate(state)

    values = np.zeros(2)

    values[0] = pc.density[0] * pc.relperm[0] / pc.viscosity[0]
    values[1] = pc.density[1] * pc.relperm[1] / pc.viscosity[1]
    return values

def find_one_array_in_another_indices(to_find, in_array):
    indices = []
    for element in to_find:
        id = np.where(in_array == element)[0]
        if id.size > 0:
            indices.append(id[0])
    return np.array(indices, dtype=np.intp)


def calc_rates_at_perforations(data, perfs_conn_id, geometric_WI):
    # Evaluate position of block_m, block_p in stored data, for every connection
    block_m = data['static']['block_m']
    block_p = data['static']['block_p']
    cell_m = find_one_array_in_another_indices(block_m[perfs_conn_id], data['dynamic']['cell_id']) # well cells
    cell_p = find_one_array_in_another_indices(block_p[perfs_conn_id], data['dynamic']['cell_id']) # reservoir cells
    assert(cell_m.size == len(perfs_conn_id) and cell_p.size == len(perfs_conn_id))

    nt = data['dynamic']['time'].size

    # Pre-allocate data
    rates = np.zeros((nt, len(perfs_conn_id), 2))
    id_state_cell = np.zeros(len(perfs_conn_id), dtype=np.intp)

    id_pres = data['dynamic']['variable_names'].index('pressure')
    # Looping over time steps
    for i in range(nt):

        p = data['dynamic']['X'][i,:,id_pres]
        # Determine upwind cell indices for all connections
        dp = p[cell_p] - p[cell_m]
        downstream = dp < 0
        upstream = dp >= 0
        id_state_cell[downstream] = cell_m[downstream]
        id_state_cell[upstream] = cell_p[upstream]

        # Looping over perforations
        for j in range(len(perfs_conn_id)):
            state = data['dynamic']['X'][i, id_state_cell[j]]
            values = molar_rate_operators(state)
            rates[i, j] = values * geometric_WI[j] * dp[j]

    return data['dynamic']['time'], rates


# Build model
m = Model()
m.init()

num_periods = 10

# Start simulation
t = 0
for i in range(num_periods):
    dt = 365
    m.run(dt)
    t += dt
m.print_timers()
m.print_stat()

# Load C++ well data
cpp_well_data = pd.DataFrame.from_dict(m.physics.engine.time_data)
cpp_time = cpp_well_data['time'].to_numpy()

h5_well_data = os.path.join(m.output_folder, m.well_filename)
h5_well_data = load_hdf5_to_dict(h5_well_data)

# Calc Python well rates
perfs = [p for well in m.reservoir.wells for p in well.perforations]
perfs_conn_id = m.find_conn_id_for_perforation(perfs=perfs)
geometric_WI = np.array([p[2] for well in m.reservoir.wells for p in well.perforations])
# Get property container for operator calculation
property_container = m.physics.property_containers
pc = property_container[0]
Python_time, Python_rates = calc_rates_at_perforations(data=h5_well_data, perfs_conn_id=perfs_conn_id, geometric_WI=geometric_WI)

# Visualize the results
counter = 0
cpp_data_start_id = 0
for well in m.reservoir.wells:
    for perf in well.perforations:
        fig, rate = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        pattern1 = well.name + ' : p ' + str(perf[0]) + ' c 0 rate (Kmol/day)'
        pattern2 = well.name + ' : p ' + str(perf[0]) + ' c 1 rate (Kmol/day)'
        nc = 2
        cpp_rates = cpp_well_data[pattern1].to_numpy()
        new = -np.sum(Python_rates[:, counter, :], axis=1)
        rate.plot(cpp_time[cpp_data_start_id:], cpp_rates[cpp_data_start_id:],
                  color='b', marker='o', markersize=5, label='C++ code molar rates')
        rate.plot(Python_time, new, color='r', marker='*', markersize=5, label='Python code molar rates')
        rate.legend(loc='upper right', prop={'size': 18})
        rate.set_xlabel(r'time [day]', fontsize=16)
        rate.set_ylabel(r'water + steam rate [Kmol/day]', fontsize=16)
        fig.tight_layout()
        plt.savefig(well.name + '_p_' + str(perf[0]) + '_rate_cmp.png')
        counter += 1
