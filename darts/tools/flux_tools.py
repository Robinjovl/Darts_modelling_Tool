import os
import numpy as np

from darts.engines import value_vector, ms_well
from darts.tools.hdf5_tools import load_hdf5_to_dict

def calc_connection_fluxes(m, conn_ids, flux_eval = None, eval_ids = None) -> np.ndarray:
    """
    Function to calculate fluxes over connections with given (array of) connection indices
    :param m: Darts model
    :param conn_ids: array of connection indices
    :param flux_eval: array of flux evaluators for all regions
    :param eval_ids: array of indices of evaluator output responsible for fluxes (between 0 and n_ops)
    :return: array of fluxes
    :rtype: np.ndarray
    """

    # extract the block_m -> block_p indices of connected cells
    cell_m = np.array(m.reservoir.mesh.block_m, copy=False)[conn_ids] # well cells
    cell_p = np.array(m.reservoir.mesh.block_p, copy=False)[conn_ids] # reservoir cells
    tran = np.array(m.reservoir.mesh.tran, copy=False)[conn_ids]
    tranD = np.array(m.reservoir.mesh.tranD, copy=False)[conn_ids]

    if flux_eval is None:
        flux_eval = m.op_list

    if len(flux_eval) == 1:
        op_num = np.zeros(m.op_num.size, dtype=np.intp)
    else:
        op_num = m.op_num

    n_conns = np.array(conn_ids).size
    name_split = flux_eval[0].__class__.__name__.split('_')
    n_ops = int(name_split[-1])
    n_state = int(name_split[-2])
    n_vars = m.physics.n_vars # note that n_state is not always equal to n_vars
    p_var = 0

    if eval_ids is None:
        eval_ids = np.arange(n_ops)

    # initialize arrays
    rates = np.zeros((n_conns, *np.array(eval_ids).shape))
    id_state_cell = np.zeros(n_conns, dtype=np.intp)
    state = value_vector()
    state.resize(n_state)
    state_np = np.array(state, copy=False)
    values = value_vector()
    values.resize(n_ops)
    values_np = np.array(values, copy=False)

    # calculate pressure difference
    X = np.array(m.physics.engine.X, copy=False)
    p_diff = X[n_vars * cell_p + p_var] - X[n_vars * cell_m + p_var]

    # calculate upwind cell
    down = p_diff < 0
    up = p_diff >= 0
    id_state_cell[down] = cell_m[down]
    id_state_cell[up] = cell_p[up]

    # calculate
    for i, conn_id in enumerate(conn_ids):
        state_np[:] = X[n_vars * id_state_cell[i] + p_var:n_vars * id_state_cell[i] + p_var + n_state]
        flux_eval[op_num[id_state_cell[i]]].evaluate(state, values)
        rates[i] = values_np[eval_ids] * tran[i] * p_diff[i]

    return rates

def get_molar_well_rates(m) -> dict:
    """
    Calculate molar well rate for every component (including temperature) for every well
    for all the timesteps from the saved well data in *.h5
    :param m: Darts model
    :return: dictionary with arrays of fluxes
    :rtype: dict
    """
    n_vars = m.physics.n_vars

    # load new well data
    new_filename = os.path.join(m.output_folder, m.well_filename)
    new_data = load_hdf5_to_dict(new_filename)
    new_data_cell_id = new_data['dynamic']['cell_id']
    new_data_var_id = np.concatenate([np.arange(i * n_vars, i * n_vars + n_vars) for i in new_data_cell_id])

    # calculate new rates at all timesteps
    molar_rate = {}
    for i in range(new_data['dynamic']['time'].size):
        # substitute solution with data from file, for current timestep
        np.array(m.physics.engine.X, copy=False)[new_data_var_id] = new_data['dynamic']['X'][i].flatten()
        # calculate new rates for every well
        for well in m.reservoir.wells:
            if well.name not in molar_rate:
                molar_rate[well.name] = []

            molar_rate[well.name].append(get_molar_well_rate(m, well))

    for well in m.reservoir.wells:
        molar_rate[well.name] = np.array(molar_rate[well.name])

    return molar_rate

def get_molar_well_rate(m, well: ms_well) -> np.ndarray:
    """
    Calculate molar well rate for every component (including temperature) for a given well
    :param m: Darts model
    :param well: well
    :type well: ms_well
    :return: array of fluxes
    :rtype: np.ndarray
    """

    # indices of flux mobility multipliers in op_list output
    eval_ids = np.vstack(
        [np.arange(m.physics.n_vars + i, m.physics.n_vars * (m.physics.nph + 1), \
                   m.physics.n_vars) for i in range(m.physics.nc + m.physics.thermal)])
    # calculate fluxes
    rates = calc_connection_fluxes(m=m, conn_ids=[m.well_head_conn_id[well.name]],
                                        flux_eval=m.op_list,
                                        eval_ids=eval_ids)
    # sum fluxes over phases
    return np.sum(rates[0], axis=-1)

def get_molar_well_rate_profile(m, well: ms_well) -> np.ndarray:
    """
    Calculate molar well rate for every component (including temperature) for all perforations in a given well
    :param m: Darts model
    :param well: well
    :type well: ms_well
    :return: array of fluxes
    :rtype: np.ndarray
    """

    # indices of flux mobility multipliers in op_list output
    eval_ids = np.vstack(
        [np.arange(m.physics.n_vars + i, m.physics.n_vars * (m.physics.nph + 1), \
                   m.physics.n_vars) for i in range(m.physics.nc + m.physics.thermal)])
    # calculate fluxes
    rates = calc_connection_fluxes(m=m, conn_ids=m.well_perf_conn_ids[well.name],
                                        flux_eval=m.op_list,
                                        eval_ids=eval_ids)
    # sum fluxes over phases
    return np.sum(rates, axis=-1)

def get_phase_volumetric_well_rates(m) -> dict:
    """
    Calculate volumetric well rate for every fluid phase for every well
    for all the timesteps from the saved well data in *.h5
    :param m: Darts model
    :return: dictionary with arrays of fluxes
    :rtype: dict
    """

    n_vars = m.physics.n_vars

    # load new well data
    new_filename = os.path.join(m.output_folder, m.well_filename)
    new_data = load_hdf5_to_dict(new_filename)
    new_data_cell_id = new_data['dynamic']['cell_id']
    new_data_var_id = np.concatenate([np.arange(i * n_vars, i * n_vars + n_vars) for i in new_data_cell_id])

    # calculate new rates at all timesteps
    vol_rate = {}
    for i in range(new_data['dynamic']['time'].size):
        # substitute solution with data from file, for current timestep
        np.array(m.physics.engine.X, copy=False)[new_data_var_id] = new_data['dynamic']['X'][i].flatten()
        # calculate new rates for every well
        for well in m.reservoir.wells:
            if well.name not in vol_rate:
                vol_rate[well.name] = []

            vol_rate[well.name].append(get_phase_volumetric_well_rate(m, well))

    for well in m.reservoir.wells:
        vol_rate[well.name] = np.array(vol_rate[well.name])

    return vol_rate

def get_phase_volumetric_well_rate(m, well: ms_well) -> np.ndarray:
    """
    Calculate volumetric well rate for every fluid phase for a given well
    :param m: Darts model
    :param well: well
    :type well: ms_well
    :return: array of fluxes
    :rtype: np.ndarray
    """

    # calculate fluxes
    rates = calc_connection_fluxes(m=m, conn_ids=[m.well_head_conn_id[well.name]],
                                        flux_eval=[m.physics.rate_itor])
    return rates[0]

def get_phase_volumetric_well_rate_profile(m, well: ms_well) -> np.ndarray:
    """
    Calculate volumetric well rate for every fluid phase for all perforations in a given well
    :param m: Darts model
    :param well: well
    :type well: ms_well
    :return: array of fluxes
    :rtype: np.ndarray
    """

    # calculate fluxes
    rates = calc_connection_fluxes(m=m, conn_ids=m.well_perf_conn_ids[well.name],
                                        flux_eval=[m.physics.rate_itor])
    return rates

def get_mass_well_rates(m) -> dict:
    """
    Calculate mass well rate for every well and for all the timesteps from the saved well data in *.h5
    :param m: Darts model
    :return: dictionary with arrays of fluxes
    :rtype: dict
    """

    n_vars = m.physics.n_vars

    # load new well data
    new_filename = os.path.join(m.output_folder, m.well_filename)
    new_data = load_hdf5_to_dict(new_filename)
    new_data_cell_id = new_data['dynamic']['cell_id']
    new_data_var_id = np.concatenate([np.arange(i * n_vars, i * n_vars + n_vars) for i in new_data_cell_id])

    # calculate new rates at all timesteps
    mass_rate = {}
    for i in range(new_data['dynamic']['time'].size):
        # substitute solution with data from file, for current timestep
        np.array(m.physics.engine.X, copy=False)[new_data_var_id] = new_data['dynamic']['X'][i].flatten()
        # calculate new rates for every well
        for well in m.reservoir.wells:
            if well.name not in mass_rate:
                mass_rate[well.name] = []

            mass_rate[well.name].append(get_mass_well_rate(m, well))

    for well in m.reservoir.wells:
        mass_rate[well.name] = np.array(mass_rate[well.name])

    return mass_rate

def get_mass_well_rate(m, well: ms_well) -> np.ndarray:
    """
    Calculate mass well rate for a given well
    :param m: Darts model
    :param well: well
    :type well: ms_well
    :return: array of fluxes
    :rtype: np.ndarray
    """

    n_ops = m.physics.mass_flux_operators[next(iter(m.physics.mass_flux_operators))].n_ops
    eval_ids = np.arange(n_ops)
    # calculate fluxes
    rates = calc_connection_fluxes(m=m, conn_ids=[m.well_head_conn_id[well.name]],
                                        flux_eval=m.physics.mass_flux_itor,
                                        eval_ids=eval_ids)
    return rates[0]

def get_mass_well_rate_profile(m, well: ms_well) -> np.ndarray:
    """
    Calculate mass well rate for all perforations in a given well
    :param m: Darts model
    :param well: well
    :type well: ms_well
    :return: array of fluxes
    :rtype: np.ndarray
    """

    n_ops = m.physics.mass_flux_operators[next(iter(m.physics.mass_flux_operators))].n_ops
    eval_ids = np.arange(n_ops)
    # calculate fluxes
    rates = calc_connection_fluxes(m=m, conn_ids=m.well_perf_conn_ids[well.name],
                                        flux_eval=m.physics.mass_flux_itor,
                                        eval_ids=eval_ids)
    return rates