import os

import numpy as np
import pandas as pd

from darts.models.darts_model import DartsModel
from darts.tools.hdf5_tools import load_hdf5_to_dict


def get_full_overall_composition(state, pc):
    composition_state = state[1:-1] if pc.thermal else state[1:]
    return np.append(composition_state, 1.0 - np.sum(composition_state))


def copy_property_value(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    return value


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
    output_prop_names = list(pc.output_props)

    # Initialize an empty DataFrame to store the primary variables and phase props
    data_frame_parts = []

    # For phase velocity calculations
    coupled_model.wells[well_name].is_first_first_iter = True
    iter_counter = 0
    flag = 1

    time = h5_well_dict["dynamic"]["time"]
    X_well_h5 = h5_well_dict["dynamic"]["X"]
    time_from_zero = np.insert(time, 0, 0.0)
    time_step_sizes = np.diff(time_from_zero)
    for i, dt in enumerate(time_step_sizes):
        ts_data = {
            "pressure": np.zeros(num_segments),
            "z": [None] * num_segments,
        }
        output_prop_data = {
            prop_name: [None] * num_segments for prop_name in output_prop_names
        }
        temperature = np.zeros(num_segments)

        for j in range(num_segments):
            state = X_well_h5[i, well_segments_idxs_in_well_h5[j], :]

            # Evaluate temperature for when the primary vars are PH and evaluate phase props.
            pc.evaluate(state)

            ts_data["pressure"][j] = state[0]
            ts_data["z"][j] = get_full_overall_composition(state, pc)
            temperature[j] = pc.temperature

            for prop_name in output_prop_names:
                output_prop_data[prop_name][j] = copy_property_value(
                    pc.output_props[prop_name]()
                )

        # Keep temperature available for existing well-property plots even if the property container
        # does not expose it in output_props.
        ts_data.update(output_prop_data)
        if "temperature" not in ts_data:
            ts_data["temperature"] = temperature

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
        ts_data["vG"] = np.append(vG, np.nan)
        ts_data["vL"] = np.append(vL, np.nan)

        data_frame_parts.append(pd.DataFrame(ts_data))

    data_frame = pd.concat(data_frame_parts, ignore_index=True)

    # Save the data frame in a pickle file
    file_name = f"dfm_well_props_{well_name}.pkl"
    file_path = os.path.join(coupled_model.output_folder, file_name)
    data_frame.to_pickle(file_path)
