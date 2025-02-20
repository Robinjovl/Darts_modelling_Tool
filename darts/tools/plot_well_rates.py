import os
import matplotlib.pyplot as plt
import numpy as np
import shutil

from darts.tools.hdf5_tools import load_hdf5_to_dict
from darts.physics.super.property_container import PropertyContainer
# from darts.models.darts_model import DartsModel

#%% Main function
def calc_rates_at_perforations(h5_well_data: dict, perfs_conn_ids: np.ndarray, geometric_WI: np.ndarray,
                               rate_type: str, thermal: bool, pc: PropertyContainer):
    """
    Calculates different types of rates at perforations of wells

    :param h5_well_data: Well data stored in the HDF5 file
    :type h5_well_data: dict
    :param perfs_conn_ids: IDs of perforations of wells
    :type perfs_conn_ids: numpy.ndarray
    :param geometric_WI: Geometric part of well indices
    :type geometric_WI: numpy.ndarray
    :param rate_type: Types of rates to calculate
    :type rate_type: str
    :param thermal: If the model is thermal or not
    :type thermal: bool
    :param pc: An instance of the class PropertyContainer
    :type pc: PropertyContainer
    """
    # Evaluate position of block_m, block_p in stored data, for every connection
    block_m = h5_well_data['static']['block_m']
    block_p = h5_well_data['static']['block_p']
    cell_m = find_one_array_in_another_indices(block_m[perfs_conn_ids], h5_well_data['dynamic']['cell_id'])   # well cells
    cell_p = find_one_array_in_another_indices(block_p[perfs_conn_ids], h5_well_data['dynamic']['cell_id'])   # reservoir cells
    assert (cell_m.size == len(perfs_conn_ids) and cell_p.size == len(perfs_conn_ids))

    nt = h5_well_data['dynamic']['time'].size

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

    id_pres = h5_well_data['dynamic']['variable_names'].index('pressure')
    # Looping over time steps
    for i in range(nt):

        p = h5_well_data['dynamic']['X'][i,:,id_pres]
        # Determine upwind cell indices for all connections
        dp = p[cell_p] - p[cell_m]
        downstream = (dp < 0)
        upstream = (dp >= 0)
        id_state_cell[downstream] = cell_m[downstream]
        id_state_cell[upstream] = cell_p[upstream]

        # Looping over perforations
        for j in range(len(perfs_conn_ids)):
            state = h5_well_data['dynamic']['X'][i, id_state_cell[j]]

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

    return h5_well_data['dynamic']['time'], rates

#%% Operator functions
def phase_molar_rate_operators(state, pc):
    """
    This function is used for calculating molar rates of phases [kmole/day]

    :param state: State of the fluid containing the primary variables
    :type state: numpy.ndarray
    :param pc: An instance of the class PropertyContainer
    :type pc: PropertyContainer
    """
    pc.evaluate(state)

    values = np.zeros(pc.nph)
    for j in pc.ph:
        values[j] = pc.dens_m[j] * pc.kr[j] / pc.mu[j]

    return values

def phase_mass_rate_operators(state, pc):
    """
    This function is used for calculating mass rates of phases [kg/day]

    :param state: State of the fluid containing the primary variables
    :type state: numpy.ndarray
    :param pc: An instance of the class PropertyContainer
    :type pc: PropertyContainer
    """
    pc.evaluate(state)

    values = np.zeros(pc.nph)
    for j in pc.ph:
        values[j] = pc.dens[j] * pc.kr[j] / pc.mu[j]

    return values

def phase_volumetric_rate_operators(state, pc):
    """
    This function is used for calculating volumetric rates of phases under perforation conditions [m3/day]

    :param state: State of the fluid containing the primary variables
    :type state: numpy.ndarray
    :param pc: An instance of the class PropertyContainer
    :type pc: PropertyContainer
    """
    pc.evaluate(state)

    values = np.zeros(pc.nph)
    for j in pc.ph:
        values[j] = pc.kr[j] / pc.mu[j]

    return values

def components_molar_rates_operators(state, pc):
    """
    This function is used for calculating advective molar rates of components in each phase [kmole/day]

    :param state: State of the fluid containing the primary variables
    :type state: numpy.ndarray
    :param pc: An instance of the class PropertyContainer
    :type pc: PropertyContainer
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

    :param state: State of the fluid containing the primary variables
    :type state: numpy.ndarray
    :param pc: An instance of the class PropertyContainer
    :type pc: PropertyContainer
    """
    pc.evaluate(state)
    pc.evaluate_thermal(state)

    values = np.zeros(pc.nph)
    for j in pc.ph:
        values[j] = pc.enthalpy[j] * pc.dens_m[j] * pc.kr[j] / pc.mu[j]

    return values


#%% Auxiliary functions
def find_conn_ids_for_perfs(perfs, block_m, block_p, n_res_blocks):
    """
    This function finds the connection IDs of perforations

    :param perfs: List of perforations (well_block_index, reservoir_block_index, well_index, well_indexD)
    :type perfs: List
    :param block_m: block_m of the connection list
    :type block_m: numpy.ndarray
    :param block_p: block_p of the connection list
    :type block_p: numpy.ndarray
    :param n_res_blocks: Number of reservoir blocks
    :type n_res_blocks: int
    """
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
