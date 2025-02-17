import os
import matplotlib.pyplot as plt
import numpy as np
import shutil

from darts.tools.hdf5_tools import load_hdf5_to_dict


#%% Main function

def calc_rates_at_perforations(data, perfs_conn_ids, geometric_WI, rate_type, thermal, pc):
    # Evaluate position of block_m, block_p in stored data, for every connection
    block_m = data['static']['block_m']
    block_p = data['static']['block_p']
    cell_m = find_one_array_in_another_indices(block_m[perfs_conn_ids], data['dynamic']['cell_id'])   # well cells
    cell_p = find_one_array_in_another_indices(block_p[perfs_conn_ids], data['dynamic']['cell_id'])   # reservoir cells
    assert (cell_m.size == len(perfs_conn_ids) and cell_p.size == len(perfs_conn_ids))

    nt = data['dynamic']['time'].size

    # Pre-allocate data
    if rate_type in ['phases_molar_rates', 'phases_mass_rates', 'phases_volumetric_rates']:
        rates = np.zeros((nt, len(perfs_conn_ids), pc.nph))
    elif rate_type in ['components_molar_rates', 'components_mass_rates']:
        rates = np.zeros((nt, len(perfs_conn_ids), pc.nc_fl * pc.nph))
    elif rate_type == 'heat_rate':
        if thermal:
            rates = np.zeros((nt, len(perfs_conn_ids), pc.nph))
        else:
            raise Exception('The model is isothermal, so heat rate cannot be calculated for it!')
    else:
        raise Exception("The rate type is not entered correctly or is not supported!")
    id_state_cell = np.zeros(len(perfs_conn_ids), dtype=np.intp)

    id_pres = data['dynamic']['variable_names'].index('pressure')
    # Looping over time steps
    for i in range(nt):

        p = data['dynamic']['X'][i,:,id_pres]
        # Determine upwind cell indices for all connections
        dp = p[cell_p] - p[cell_m]
        downstream = (dp < 0)
        upstream = (dp >= 0)
        id_state_cell[downstream] = cell_m[downstream]
        id_state_cell[upstream] = cell_p[upstream]

        # Looping over perforations
        for j in range(len(perfs_conn_ids)):
            state = data['dynamic']['X'][i, id_state_cell[j]]

            if rate_type == 'phases_molar_rates':
                values = phase_molar_rate_operators(state, pc)
            elif rate_type == 'phases_mass_rates':
                values = phase_mass_rate_operators(state, pc)
            elif rate_type == 'phases_volumetric_rates':
                values = phase_volumetric_rate_operators(state, pc)
            elif rate_type in ['components_molar_rates', 'components_mass_rates']:
                values = components_molar_rates_operators(state, pc)
            elif rate_type == 'heat_rate':
                values = heat_rate_operators(state, pc)

            rates[i, j] = values * geometric_WI[j] * dp[j]

    return data['dynamic']['time'], rates

#%% Operator functions
def phase_molar_rate_operators(state, pc):
    """
    This function is used for calculating molar rates of phases [kmole/day]
    """
    pc.evaluate(state)

    values = np.zeros(pc.nph)
    for j in pc.ph:
        values[j] = pc.dens_m[j] * pc.kr[j] / pc.mu[j]

    return values

def phase_mass_rate_operators(state, pc):
    """
    This function is used for calculating mass rates of phases [kg/day]
    """
    pc.evaluate(state)

    values = np.zeros(pc.nph)
    for j in pc.ph:
        values[j] = pc.dens[j] * pc.kr[j] / pc.mu[j]

    return values

def phase_volumetric_rate_operators(state, pc):
    """
    This function is used for calculating volumetric rates of phases [m3/day]
    """
    pc.evaluate(state)

    values = np.zeros(pc.nph)
    for j in pc.ph:
        values[j] = pc.kr[j] / pc.mu[j]

    return values

def components_molar_rates_operators(state, pc):
    """
    This function is used for calculating advective molar rates of components in each phase [kmole/day]
    """
    pc.evaluate(state)

    values = np.zeros(pc.nph * pc.nc_fl)
    for j in pc.ph:
        for i in range(pc.nc_fl):
            values[pc.nc_fl * j + i] = pc.x[j][i] * pc.dens_m[j] * pc.kr[j] / pc.mu[j]

    return values

def heat_rate_operators(state, pc):
    """
    This function is used for calculating advective heat rate [kJ/day]
    """
    pc.evaluate(state)
    pc.evaluate_thermal(state)

    values = np.zeros(pc.nph)
    for j in pc.ph:
        values[j] = pc.enthalpy[j] * pc.dens_m[j] * pc.kr[j] / pc.mu[j]

    return values


#%% Auxiliary functions
def find_conn_ids_for_perfs(perfs, block_m, block_p, n_res_blocks):
    res_cell_ids = [perf[1] for perf in perfs]

    perfs_conn_ids = np.nonzero(np.logical_and(np.isin(block_p, res_cell_ids), block_m >= n_res_blocks))[0]
    assert (len(perfs_conn_ids) == len(perfs) and (block_m[perfs_conn_ids] > n_res_blocks).all())
    return perfs_conn_ids

def find_one_array_in_another_indices(to_find, in_array):
    indices = []
    for element in to_find:
        id = np.where(in_array == element)[0]
        if id.size > 0:
            indices.append(id[0])
    return np.array(indices, dtype=np.intp)
