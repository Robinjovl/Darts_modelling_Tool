import os

import numpy as np
import pandas as pd

from darts.models.darts_model import DartsModel
from darts.tools.hdf5_tools import load_hdf5_to_dict


def save_dfm_well_props(
    well_name: str,
    coupled_model: DartsModel,
):
    """
    Store the primary variables and phase properties of the well segments of the specified DFM well in a pickle file
    located in the output folder

    :param well_name: Name of the well the properties of which will be saved
    :type well_name: str
    :param coupled_model: The instance of the DartsModel
    :type coupled_model: DartsModel

    """
    # Find the index of the well in the cpp well list
    iw = None
    for idx, w in enumerate(coupled_model.reservoir.wells):
        if w.name == well_name:
            iw = idx
            break
    if iw is None:
        raise ValueError("The specified well name is not found!")

    # Get the time array, well primary variables, and other well info
    h5_well_file_path = coupled_model.well_filepath
    h5_well_dict = load_hdf5_to_dict(h5_well_file_path)

    # Get well geometry info
    num_segments = coupled_model.reservoir.wells[iw].num_segments
    cell_id = h5_well_dict["dynamic"]["cell_id"]
    well_head_idx = coupled_model.reservoir.wells[iw].well_head_idx
    well_bottom_idx = coupled_model.reservoir.wells[iw].well_bottom_idx
    well_segments_idxs = np.arange(well_head_idx, well_bottom_idx + 1)
    well_segments_idxs_in_well_h5 = coupled_model.output.find_values_in_an_array(
        well_segments_idxs, cell_id
    )

    pc = coupled_model.physics.property_containers[0]

    # Get phase indices
    n_mobile_phases = coupled_model.wells[well_name].n_mobile_phases
    g_idx = coupled_model.wells[well_name].g_idx
    if n_mobile_phases == 2:
        l_idx = coupled_model.wells[well_name].l_idx
    elif n_mobile_phases == 3:
        la_idx = coupled_model.wells[well_name].la_idx
        lb_idx = coupled_model.wells[well_name].lb_idx

    # Preallocate arrays for primary vars and phase properties
    p = np.zeros(num_segments)
    z = np.zeros((num_segments, pc.nc))
    T = np.zeros(num_segments)

    sG = np.zeros(num_segments)
    if n_mobile_phases == 2:
        sL = np.zeros(num_segments)
    elif n_mobile_phases == 3:
        sL_a = np.zeros(num_segments)
        sL_b = np.zeros(num_segments)
    rhoG = np.zeros(num_segments)
    if n_mobile_phases == 2:
        rhoL = np.zeros(num_segments)
    elif n_mobile_phases == 3:
        rhoL_a = np.zeros(num_segments)
        rhoL_b = np.zeros(num_segments)
    miuG = np.zeros(num_segments)
    if n_mobile_phases == 2:
        miuL = np.zeros(num_segments)
    elif n_mobile_phases == 3:
        miuL_a = np.zeros(num_segments)
        miuL_b = np.zeros(num_segments)
    xG = np.zeros((num_segments, pc.nc_fl))
    if n_mobile_phases == 2:
        xL = np.zeros((num_segments, pc.nc_fl))
    elif n_mobile_phases == 3:
        xL_a = np.zeros((num_segments, pc.nc_fl))
        xL_b = np.zeros((num_segments, pc.nc_fl))

    # Initialize an empty DataFrame to store the primary variables and phase props
    data_frame = pd.DataFrame()

    # For phase velocity calculations
    coupled_model.wells[well_name].is_first_first_iter = True
    iter_counter = 0
    flag = 1

    time = h5_well_dict["dynamic"]["time"]
    X_well_h5 = h5_well_dict["dynamic"]["X"]
    time_from_zero = np.insert(time, 0, 0.0)
    time_step_sizes = np.diff(time_from_zero)
    for i, dt in enumerate(time_step_sizes):
        for j in range(num_segments):
            state = X_well_h5[i, well_segments_idxs_in_well_h5[j], :]
            p[j] = state[0]

            # Evaluate temperature for when the primary vars are PH and evaluate phase props
            pc.evaluate(state)

            if not pc.thermal:
                z_full = state[1:]
                z_full = np.append(z_full, 1 - sum(state[1:]))
                z[j, :] = z_full
            elif pc.thermal:
                z_full = state[1:-1]
                z_full = np.append(z_full, 1 - sum(state[1:-1]))
                z[j, :] = z_full
                if (
                    coupled_model.physics.state_spec
                    == coupled_model.physics.StateSpecification.PT
                ):
                    T[j] = state[-1]
                elif (
                    coupled_model.physics.state_spec
                    == coupled_model.physics.StateSpecification.PH
                ):
                    T[j] = pc.temperature

            xG[j, :] = pc.x[g_idx, :]
            if n_mobile_phases == 2:
                xL[j, :] = pc.x[l_idx, :]
            elif n_mobile_phases == 3:
                xL_a[j, :] = pc.x[la_idx, :]
                xL_b[j, :] = pc.x[lb_idx, :]
            sG[j] = pc.sat[g_idx]
            if n_mobile_phases == 2:
                sL[j] = pc.sat[l_idx]
            elif n_mobile_phases == 3:
                sL_a[j] = pc.sat[la_idx]
                sL_b[j] = pc.sat[lb_idx]
            rhoG[j] = pc.dens[g_idx]
            if n_mobile_phases == 2:
                rhoL[j] = pc.dens[l_idx]
            elif n_mobile_phases == 3:
                rhoL_a[j] = pc.dens[la_idx]
                rhoL_b[j] = pc.dens[lb_idx]
            miuG[j] = pc.mu[g_idx]
            if n_mobile_phases == 2:
                miuL[j] = pc.mu[l_idx]
            elif n_mobile_phases == 3:
                miuL_a[j] = pc.mu[la_idx]
                miuL_b[j] = pc.mu[lb_idx]

        # Save phase velocities
        if i == 0:
            initial_conditions = coupled_model.wells[
                well_name
            ].initial_conditions.initial_conditions_vector
            Xn_ms_well = initial_conditions
            X_ms_well = X_well_h5[i, well_segments_idxs_in_well_h5, :].flatten()
        else:
            Xn_ms_well = X_well_h5[i - 1, well_segments_idxs_in_well_h5, :].flatten()
            X_ms_well = X_well_h5[i, well_segments_idxs_in_well_h5, :].flatten()
        phase_velocities = coupled_model.wells[well_name].eval_phase_vels(
            Xn_ms_well, X_ms_well, dt, time_from_zero[i], iter_counter, flag
        )
        # phase_velocities = np.zeros((num_segments - 1) * 2)
        mid = int(len(phase_velocities) / 2)
        vG = phase_velocities[:mid]
        vL = phase_velocities[mid:]
        # Make velocity variables have the same size as the other props
        vG = np.append(vG, np.nan)
        vL = np.append(vL, np.nan)

        if n_mobile_phases == 2:
            ts_primary_vars_and_phases_props = [
                p.copy(),
                z.copy(),
                T.copy(),
                xG.copy(),
                xL.copy(),
                sG.copy(),
                sL.copy(),
                rhoG.copy(),
                rhoL.copy(),
                miuG.copy(),
                miuL.copy(),
                vG.copy(),
                vL.copy(),
            ]
            data_frame = pd.concat(
                [
                    data_frame,
                    pd.DataFrame(
                        list(zip(*ts_primary_vars_and_phases_props, strict=False)),
                        columns=[
                            "pressure",
                            "Overall mole fractions",
                            "temperature",
                            "xG",
                            "xL",
                            "sG",
                            "sL",
                            "rhoG",
                            "rhoL",
                            "miuG",
                            "miuL",
                            "vG",
                            "vL",
                        ],
                    ),
                ]
            )
        elif n_mobile_phases == 3:
            ts_primary_vars_and_phases_props = [
                p.copy(),
                z.copy(),
                T.copy(),
                xG.copy(),
                xL_a.copy(),
                xL_b.copy(),
                sG.copy(),
                sL_a.copy(),
                sL_b.copy(),
                rhoG.copy(),
                rhoL_a.copy(),
                rhoL_b.copy(),
                miuG.copy(),
                miuL_a.copy(),
                miuL_b.copy(),
                vG.copy(),
                vL.copy(),
            ]
            data_frame = pd.concat(
                [
                    data_frame,
                    pd.DataFrame(
                        list(zip(*ts_primary_vars_and_phases_props, strict=False)),
                        columns=[
                            "pressure",
                            "Overall mole fractions",
                            "temperature",
                            "xG",
                            "xL_a",
                            "xL_b",
                            "sG",
                            "sL_a",
                            "sL_b",
                            "rhoG",
                            "rhoL_a",
                            "rhoL_b",
                            "miuG",
                            "miuL_a",
                            "miuL_b",
                            "vG",
                            "vL",
                        ],
                    ),
                ]
            )

    # Save the data frame in a pickle file
    file_name = f"dfm_well_props_{well_name}.pkl"
    file_path = os.path.join(coupled_model.output.output_folder, file_name)
    data_frame.to_pickle(file_path)
