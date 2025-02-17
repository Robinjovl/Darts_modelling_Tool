import os
import matplotlib.pyplot as plt
import numpy as np
import shutil

from darts.tools.hdf5_tools import load_hdf5_to_dict
from darts.physics.super.property_container import PropertyContainer
# from darts.models.darts_model import DartsModel

# def plot_well_rates(types_of_well_rates: list, m: DartsModel):
#     """
#     Plots the following types of rates for each perforation and each well:
#     'phases_molar_rates'
#     'phases_mass_rates'
#     'phases_volumetric_rates'
#
#     'components_molar_rates'
#     'components_mass_rates'
#
#     'heat_rate'
#
#     :param types_of_well_rates: Type or types of well rates the user wants to plot
#     :type types_of_well_rates: list
#     :param m: An instance of DartsModel
#     :type m: DartsModel
#     """
#     h5_well_data_address = os.path.join(m.output_folder, m.well_filename)
#     h5_well_data = load_hdf5_to_dict(h5_well_data_address)
#
#     # Calculate well rates
#     perfs = [p for well in m.reservoir.wells for p in well.perforations]
#     block_m = np.array(m.reservoir.mesh.block_m, copy=False)
#     block_p = np.array(m.reservoir.mesh.block_p, copy=False)
#     perfs_conn_ids = find_conn_ids_for_perfs(perfs, block_m, block_p, m.reservoir.mesh.n_res_blocks)
#     # Get well indices for each perforation
#     geometric_WI = np.array([p[2] for well in m.reservoir.wells for p in well.perforations])
#     # Get property container for operators calculations
#     property_container = m.physics.property_containers
#     pc = property_container[0]
#
#     # Create new folders in which well rates output will be stored
#     main_dir = os.path.join(m.output_folder, 'output_well_rates')
#     if not os.path.exists(main_dir):
#         os.mkdir(main_dir)
#     elif os.path.exists(main_dir):
#         shutil.rmtree(main_dir)
#         os.mkdir(main_dir)
#     for well in m.reservoir.wells:
#         well_dir = os.path.join(main_dir, 'well_' + well.name)
#         os.mkdir(well_dir)
#         for perf in well.perforations:
#             perf_dir = os.path.join(main_dir, 'well_' + well.name, 'perf_' + str(perf[0]))
#             os.mkdir(perf_dir)
#
#     for rate_type in types_of_well_rates:
#         if rate_type == 'heat_rate' and not m.physics.thermal:
#             continue
#
#         time, Python_rates = calc_rates_at_perforations(h5_well_data, perfs_conn_ids, geometric_WI, rate_type, m, pc)
#
#         """""""""  Plot well rates over time """""""""
#         """ Rates for each perforation """
#         perf_counter = 0
#         for well in m.reservoir.wells:
#             for perf in well.perforations:
#                 if rate_type in ['phases_molar_rates', 'phases_mass_rates', 'phases_volumetric_rates']:
#                     for phase in range(pc.nph):
#                         phase_molar_rate_for_perf = - Python_rates[:, perf_counter, phase]
#
#                         plt.figure(figsize=(12, 6))
#                         plt.plot(time, phase_molar_rate_for_perf, color='r', marker='o', markersize=5)
#                         plt.xlabel('time [day]', fontsize=16)
#
#                         if rate_type == 'phases_molar_rates':
#                             plt.ylabel(pc.phases_name[phase] + ' molar rate [kmol/day]', fontsize=16)
#                             perf_dir = os.path.join(main_dir, 'well_' + well.name, 'perf_' + str(perf[0]))
#                             plt.savefig(perf_dir + '/well_' + well.name + '_perf_' + str(perf[0]) + '_molar_rate_' + pc.phases_name[phase] + '.png')
#                         elif rate_type == 'phases_mass_rates':
#                             plt.ylabel(pc.phases_name[phase] + ' mass rate [kg/day]', fontsize=16)
#                             perf_dir = os.path.join(main_dir, 'well_' + well.name, 'perf_' + str(perf[0]))
#                             plt.savefig(perf_dir + '/well_' + well.name + '_perf_' + str(perf[0]) + '_mass_rate_' + pc.phases_name[phase] + '.png')
#                         elif rate_type == 'phases_volumetric_rates':
#                             plt.ylabel(pc.phases_name[phase] + ' volumetric rate [m$^3$/day]', fontsize=16)
#                             perf_dir = os.path.join(main_dir, 'well_' + well.name, 'perf_' + str(perf[0]))
#                             plt.savefig(perf_dir + '/well_' + well.name + '_perf_' + str(perf[0]) + '_volumetric_rate_' + pc.phases_name[phase] + '.png')
#
#                 elif rate_type in ['components_molar_rates', 'components_mass_rates']:
#                     for component in range(pc.nc_fl):
#                         if rate_type == 'components_molar_rates':
#                             component_molar_rate_for_perf = - np.sum(Python_rates[:, perf_counter, component::pc.nc_fl], axis=1)
#                         elif rate_type == 'components_mass_rates':
#                             component_mass_rate_for_perf = - np.sum(Python_rates[:, perf_counter, component::pc.nc_fl], axis=1) * pc.Mw[component]
#
#                         plt.figure(figsize=(12, 6))
#                         if rate_type == 'components_molar_rates':
#                             plt.plot(time, component_molar_rate_for_perf, color='r', marker='o', markersize=5)
#                         elif rate_type == 'components_mass_rates':
#                             plt.plot(time, component_mass_rate_for_perf, color='r', marker='o', markersize=5)
#
#                         plt.xlabel('time [day]', fontsize=16)
#
#                         if rate_type == 'components_molar_rates':
#                             plt.ylabel(pc.components_name[component] + ' molar rate [kmole/day]', fontsize=16)
#                             perf_dir = os.path.join(main_dir, 'well_' + well.name, 'perf_' + str(perf[0]))
#                             plt.savefig(perf_dir + '/well_' + well.name + '_perf_' + str(perf[0]) + '_molar_rate_' + pc.components_name[component] + '.png')
#                         elif rate_type == 'components_mass_rates':
#                             plt.ylabel(pc.components_name[component] + ' mass rate [kg/day]', fontsize=16)
#                             perf_dir = os.path.join(main_dir, 'well_' + well.name, 'perf_' + str(perf[0]))
#                             plt.savefig(perf_dir + '/well_' + well.name + '_perf_' + str(perf[0]) + '_mass_rate_' + pc.components_name[component] + '.png')
#
#                 elif rate_type == 'heat_rate':
#                     heat_rate_of_all_phases_for_perf = - np.sum(Python_rates[:, perf_counter, :], axis=1)
#
#                     plt.figure(figsize=(12, 6))
#                     plt.plot(time, heat_rate_of_all_phases_for_perf, color='r', marker='o', markersize=5)
#                     plt.xlabel('time [day]', fontsize=16)
#                     plt.ylabel('heat rate [kJ/day]', fontsize=16)
#                     perf_dir = os.path.join(main_dir, 'well_' + well.name, 'perf_' + str(perf[0]))
#                     plt.savefig(perf_dir + '/well_' + well.name + '_perf_' + str(perf[0]) + '_heat_rate.png')
#
#                 perf_counter += 1
#
#         """ Total rates for each well """
#         if rate_type == 'phases_molar_rates':
#             perf_counter = 0
#             for well in m.reservoir.wells:
#                 for phase in range(pc.nph):
#                     phase_molar_rate_for_well = 0
#                     for perf in well.perforations:
#                         phase_molar_rate_for_well += - Python_rates[:, perf_counter, phase]
#
#                         perf_counter += 1
#                     perf_counter -= len(well.perforations)
#
#                     plt.figure(figsize=(12, 6))
#                     plt.plot(time, phase_molar_rate_for_well, color='r', marker='o', markersize=5)
#                     plt.xlabel('time [day]', fontsize=16)
#                     plt.ylabel(pc.phases_name[phase] + ' molar rate [kmol/day]', fontsize=16)
#                     well_dir = os.path.join(main_dir, 'well_' + well.name)
#                     plt.savefig(well_dir + '/well_' + well.name + '_molar_rate_' + pc.phases_name[phase] + '.png')
#
#                 perf_counter += len(well.perforations)
#
#         elif rate_type == 'phases_mass_rates':
#             perf_counter = 0
#             for well in m.reservoir.wells:
#
#                 for phase in range(pc.nph):
#                     phase_mass_rate_for_well = 0
#                     for perf in well.perforations:
#                         phase_mass_rate_for_well += - Python_rates[:, perf_counter, phase]
#
#                         perf_counter += 1
#                     perf_counter -= len(well.perforations)
#
#                     plt.figure(figsize=(12, 6))
#                     plt.plot(time, phase_mass_rate_for_well, color='r', marker='o', markersize=5)
#                     plt.xlabel('time [day]', fontsize=16)
#                     plt.ylabel(pc.phases_name[phase] + ' mass rate [kg/day]', fontsize=16)
#                     well_dir = os.path.join(main_dir, 'well_' + well.name)
#                     plt.savefig(well_dir + '/well_' + well.name + '_mass_rate_' + pc.phases_name[phase] + '.png')
#
#                 perf_counter += len(well.perforations)
#
#         elif rate_type == 'phases_volumetric_rates':
#             perf_counter = 0
#             for well in m.reservoir.wells:
#
#                 for phase in range(pc.nph):
#                     phase_volumetric_rate_for_well = 0
#                     for perf in well.perforations:
#                         phase_volumetric_rate_for_well += - Python_rates[:, perf_counter, phase]
#
#                         perf_counter += 1
#                     perf_counter -= len(well.perforations)
#
#                     plt.figure(figsize=(12, 6))
#                     plt.plot(time, phase_volumetric_rate_for_well, color='r', marker='o', markersize=5)
#                     plt.xlabel('time [day]', fontsize=16)
#                     plt.ylabel(pc.phases_name[phase] + ' volumetric rate [m$^3$/day]', fontsize=16)
#                     well_dir = os.path.join(main_dir, 'well_' + well.name)
#                     plt.savefig(well_dir + '/well_' + well.name + '_volumetric_rate_' + pc.phases_name[phase] + '.png')
#
#                 perf_counter += len(well.perforations)
#
#         elif rate_type == 'components_molar_rates':
#             perf_counter = 0
#             for well in m.reservoir.wells:
#                 for component in range(pc.nc_fl):
#                     component_molar_rate_for_well = 0
#                     for perf in well.perforations:
#                         component_molar_rate_for_well += - np.sum(Python_rates[:, perf_counter, component::pc.nc_fl], axis=1)
#
#                         perf_counter += 1
#                     perf_counter -= len(well.perforations)
#
#                     plt.figure(figsize=(12, 6))
#                     plt.plot(time, component_molar_rate_for_well, color='r', marker='o', markersize=5)
#                     plt.xlabel('time [day]', fontsize=16)
#                     plt.ylabel(pc.components_name[component] + ' molar rate [kmole/day]', fontsize=16)
#                     well_dir = os.path.join(main_dir, 'well_' + well.name)
#                     plt.savefig(well_dir + '/well_' + well.name + '_molar_rate_' + pc.components_name[component] + '.png')
#
#                 perf_counter += len(well.perforations)
#
#         elif rate_type == 'components_mass_rates':
#             perf_counter = 0
#             for well in m.reservoir.wells:
#                 for component in range(pc.nc_fl):
#                     component_mass_rate_for_well = 0
#                     for perf in well.perforations:
#                         component_mass_rate_for_well += - np.sum(Python_rates[:, perf_counter, component::pc.nc_fl], axis=1) * pc.Mw[component]
#
#                         perf_counter += 1
#                     perf_counter -= len(well.perforations)
#
#                     plt.figure(figsize=(12, 6))
#                     plt.plot(time, component_mass_rate_for_well, color='r', marker='o', markersize=5)
#                     plt.xlabel('time [day]', fontsize=16)
#                     plt.ylabel(pc.components_name[component] + ' mass rate [kg/day]', fontsize=16)
#                     well_dir = os.path.join(main_dir, 'well_' + well.name)
#                     plt.savefig(well_dir + '/well_' + well.name + '_mass_rate_' + pc.components_name[component] + '.png')
#
#                 perf_counter += len(well.perforations)
#
#         elif rate_type == 'heat_rate':
#             perf_counter = 0
#             for well in m.reservoir.wells:
#
#                 heat_rate_for_well = 0
#                 for phase in range(pc.nph):
#                     for perf in well.perforations:
#                         heat_rate_for_well += - Python_rates[:, perf_counter, phase]
#
#                         perf_counter += 1
#                     perf_counter -= len(well.perforations)
#
#                 plt.figure(figsize=(12, 6))
#                 plt.plot(time, heat_rate_for_well, color='r', marker='o', markersize=5)
#                 plt.xlabel('time [day]', fontsize=16)
#                 plt.ylabel('heat rate [kJ/day]', fontsize=16)
#                 well_dir = os.path.join(main_dir, 'well_' + well.name)
#                 plt.savefig(well_dir + '/well_' + well.name + '_heat_rate.png')
#
#                 perf_counter += len(well.perforations)

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
