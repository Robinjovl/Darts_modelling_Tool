import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from model import Model
from darts.tools.hdf5_tools import load_hdf5_to_dict
from darts.engines import value_vector

def find_one_array_in_another_indices(to_find, in_array):
    indices = []
    for element in to_find:
        id = np.where(in_array == element)[0]
        if id.size > 0:
            indices.append(id[0])
    return np.array(indices, dtype=np.intp)


def calc_rates_at_perforations(data, evals, op_num, perfs_conn_id, eval_ids = None, geometric_WI = None):
    # evaluate position of block_m, block_p in stored data, for every connection
    block_m = data['static']['block_m']
    block_p = data['static']['block_p']
    cell_m = find_one_array_in_another_indices(block_m[perfs_conn_id], data['dynamic']['cell_id']) # well cells
    cell_p = find_one_array_in_another_indices(block_p[perfs_conn_id], data['dynamic']['cell_id']) # reservoir cells
    assert(cell_m.size == len(perfs_conn_id) and cell_p.size == len(perfs_conn_id))

    nt = data['dynamic']['time'].size
    n_ops = int(evals[0].__class__.__name__.split('_')[-1])
    n_dim = int(evals[0].__class__.__name__.split('_')[-2])

    # # support calculation with specific operators rather than with full
    # if eval_ids is None:
    #     eval_ids = np.arange(n_ops)

    # # support WI/CCF multipliers
    # if geometric_WI is None:
    #     geometric_WI = np.ones((len(perfs_conn_id), len(eval_ids)))

    # Pre-allocate data
    rates = np.zeros((nt, len(perfs_conn_id), len(eval_ids)))
    id_state_cell = np.zeros(len(perfs_conn_id), dtype=np.intp)

    state = value_vector()
    state.resize(n_dim)
    state_np = np.array(state, copy=False)
    values = value_vector()
    values.resize(n_ops)
    values_np = np.array(values, copy=False)

    # calculate fluxes at given time steps
    id_pres = data['dynamic']['variable_names'].index('pressure')
    for i in range(nt):

        p = data['dynamic']['X'][i,:,id_pres]
        # estimate upwind cell indices for all connections
        dp = p[cell_p] - p[cell_m]
        downstream = dp < 0
        upstream = dp >= 0
        id_state_cell[downstream] = cell_m[downstream]
        id_state_cell[upstream] = cell_p[upstream]

        # looping over connections
        for j in range(len(perfs_conn_id)):
            state_np[:] = data['dynamic']['X'][i, id_state_cell[j]]#[d['data']['pressure'][id_state_cell[j]], d['data']['temperature'][id_state_cell[j]]]
            evals[op_num[id_state_cell[j]]].evaluate(state, values)
            rates[i, j] = values_np[eval_ids] * geometric_WI[j] * dp[j]

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
    # m.save_data(type='solution', t=t+dt)
    t += dt
m.print_timers()
m.print_stat()

# Load C++ well data
cpp_well_data = pd.DataFrame.from_dict(m.physics.engine.time_data)
cpp_time = cpp_well_data['time'].to_numpy()

h5_well_data = os.path.join(m.output_folder, m.well_filename)
h5_well_data = load_hdf5_to_dict(h5_well_data)

# Calculate fluxes at all perforations
# eval = mass_evaluator(m.physics.property_containers[0])
# eval = engine_evaluator(m.physics.property_containers[0])
evals = m.op_list
eval_ids = np.arange(m.physics.n_vars, m.physics.n_vars + m.physics.n_vars * m.physics.nph)
perfs = [p for well in m.reservoir.wells for p in well.perforations]
perfs_conn_id = m.find_conn_id_for_perforation(perfs=perfs)
geometric_WI = np.array([p[2] for well in m.reservoir.wells for p in well.perforations])
# geometric_WI_d = np.array([p[3] for well in m.reservoir.wells for p in well.perforations])
geometric_WI = np.repeat(geometric_WI[:, np.newaxis], eval_ids.size, axis=1)
Python_time, Python_rates = calc_rates_at_perforations(data=h5_well_data, evals=evals, op_num=m.op_num,
                                             perfs_conn_id=perfs_conn_id, eval_ids=eval_ids, geometric_WI=geometric_WI)

# Visualize the results
counter = 0
cpp_data_start_id = 0
for well in m.reservoir.wells:
    for perf in well.perforations:
        fig, rate = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        pattern1 = well.name + ' : p ' + str(perf[0]) + ' c 0 rate (Kmol/day)'
        pattern2 = well.name + ' : p ' + str(perf[0]) + ' c 1 rate (Kmol/day)'
        nc = 2
        id_c0 = np.arange(0, Python_rates.shape[-1], nc)
        cpp_rates = cpp_well_data[pattern1].to_numpy()
        new = -np.sum(Python_rates[:, counter, id_c0], axis=1)
        rate.plot(cpp_time[cpp_data_start_id:], cpp_rates[cpp_data_start_id:],
                  color='b', marker='o', markersize=5, label='C++ code rates')
        rate.plot(Python_time, new, color='r', marker='*', markersize=5, label='Python code rates')
        rate.legend(loc='upper right', prop={'size': 18})
        rate.set_xlabel(r'time [day]', fontsize=16)
        rate.set_ylabel(r'water + steam rate [Kmol/day]', fontsize=16)
        fig.tight_layout()
        plt.savefig(well.name + '_p_' + str(perf[0]) + '_rate_cmp.png')
        counter += 1
