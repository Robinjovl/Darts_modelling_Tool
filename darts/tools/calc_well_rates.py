import os
import matplotlib.pyplot as plt
import numpy as np
import shutil

from darts.tools.hdf5_tools import load_hdf5_to_dict
from darts.physics.super.property_container import PropertyContainer
# from darts.models.darts_model import DartsModel

#%% Main function
def calc_rates_at_connections(h5_well_data: dict, conn_ids: list, trans: np.ndarray,
                              thermal: bool, pc: PropertyContainer, rate_type: str):
    """
    Calculates different types of rates at perforations of wells

    :param h5_well_data: Well data stored in the HDF5 file
    :type h5_well_data: dict
    :param conn_ids: IDs of connections
    :type conn_ids: numpy.ndarray
    :param trans: Transmissibility (For perforations, it is geometric part of well index)
    :type trans: numpy.ndarray
    :param thermal: If the model is thermal or not
    :type thermal: bool
    :param pc: An instance of the class PropertyContainer()
    :type pc: PropertyContainer
    :param rate_type: Type of well rate to calculate
    :type rate_type: str
    """
    # Evaluate position of block_m, block_p in stored data, for every connection
    block_m = h5_well_data['static']['block_m']
    block_p = h5_well_data['static']['block_p']
    cell_m = find_one_array_in_another_indices(block_m[conn_ids], h5_well_data['dynamic']['cell_id'])   # well cells
    cell_p = find_one_array_in_another_indices(block_p[conn_ids], h5_well_data['dynamic']['cell_id'])   # reservoir cells
    assert (cell_m.size == len(conn_ids) and cell_p.size == len(conn_ids))

    num_ts = h5_well_data['dynamic']['time'].size

    # Pre-allocate data
    if rate_type in ['phases_molar_rates', 'phases_mass_rates', 'phases_volumetric_rates']:
        rates = np.zeros((num_ts, len(conn_ids), pc.nph))
    elif rate_type in ['components_molar_rates', 'components_mass_rates']:
        try:
            rates = np.zeros((num_ts, len(conn_ids), pc.nc_fl * pc.nph))
        except:
            rates = np.zeros((num_ts, len(conn_ids), 1 * pc.nph))

    elif rate_type == 'advective_heat_rate':
        if thermal:
            rates = np.zeros((num_ts, len(conn_ids), pc.nph))
        else:
            raise Exception('The model is isothermal, so advective heat rate cannot be calculated for it!')
    else:
        raise Exception("The rate type is not entered correctly or is not supported!")
    id_state_cell = np.zeros(len(conn_ids), dtype=np.intp)

    id_pres = h5_well_data['dynamic']['variable_names'].index('pressure')
    # Looping over time steps
    for i in range(num_ts):

        p = h5_well_data['dynamic']['X'][i,:,id_pres]
        # Determine upwind cell indices for all connections
        dp = p[cell_p] - p[cell_m]
        downstream = (dp < 0)
        upstream = (dp >= 0)
        id_state_cell[downstream] = cell_m[downstream]
        id_state_cell[upstream] = cell_p[upstream]

        # Looping over perforations
        for j in range(len(conn_ids)):
            state = h5_well_data['dynamic']['X'][i, id_state_cell[j]]

            if rate_type == 'phases_molar_rates':
                values = phase_molar_rate_operators(state, pc)
            elif rate_type == 'phases_mass_rates':
                values = phase_mass_rate_operators(state, pc)
            elif rate_type == 'phases_volumetric_rates':
                values = phase_volumetric_rate_operators(state, pc)
            elif rate_type in ['components_molar_rates']:
                values = components_molar_rates_operators(state, pc)
            elif rate_type == 'components_mass_rates':
                values = components_mass_rates_operators(state, pc)
            elif rate_type == 'advective_heat_rate':
                values = heat_rate_operators(state, pc)
            else:
                raise Exception("Rate type is entered incorrectly!")

            rates[i, j] = values * trans[j] * dp[j]

    return rates

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
        try:
            values[j] = pc.dens_m[j] * pc.kr[j] / pc.mu[j] # compositional
        except:
            values[j] = pc.dens_m[j] * pc.relperm[j] / pc.viscosity[j]

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
        try:
            values[j] = pc.dens_m[j] * pc.kr[j] / pc.mu[j]
        except:
            values[j] = pc.density[j] * pc.relperm[j] / pc.viscosity[j]


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
        try:
            values[j] = pc.kr[j] / pc.mu[j]
        except:
            values[j] = pc.relperm[j] / pc.viscosity[j]

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

    try:
        values = np.zeros(pc.nph * pc.nc_fl)
    except:
        values = np.zeros(pc.nph * 1)

    for j in pc.ph:
        try:
            for i in range(pc.nc_fl):
                values[pc.nc_fl * j + i] = pc.x[j][i] * pc.dens_m[j] * pc.kr[j] / pc.mu[j]
        except:
            for i in range(1):
                values[1 * j + i] = pc.dens_m[j] * pc.relperm[j] / pc.viscosity[j]

    return values

def components_mass_rates_operators(state, pc):
    """
    This function is used for calculating advective mass rates of components in each phase [kg/day]

    :param state: State of the fluid containing the primary variables
    :type state: numpy.ndarray
    :param pc: An instance of the class PropertyContainer
    :type pc: PropertyContainer
    """
    pc.evaluate(state)

    values = np.zeros(pc.nph * pc.nc_fl)
    for j in pc.ph:
        for i in range(pc.nc_fl):
            try:
                values[pc.nc_fl * j + i] = pc.x[j][i] * pc.dens_m[j] * pc.Mw[i] * pc.kr[j] / pc.mu[j]
            except:
                values[1 * j + i] = pc.dens_m[j] * pc.Mw[i] * pc.relperm[j] / pc.viscosity[j]

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

def find_conn_ids_for_wellhead_conns():
    return None

def find_one_array_in_another_indices(to_find, in_array):
    indices = []
    for element in to_find:
        id = np.where(in_array == element)[0]
        if id.size > 0:
            indices.append(id[0])
    return np.array(indices, dtype=np.intp)
