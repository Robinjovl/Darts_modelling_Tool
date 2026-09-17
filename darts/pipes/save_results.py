import copy
import os

import numpy as np
import pandas as pd

from darts.models.darts_model import DartsModel
from darts.tools.hdf5_tools import load_hdf5_to_dict


def _get_full_overall_composition(state, pc):
    composition_state = state[1:-1] if pc.thermal else state[1:]
    return np.append(composition_state, 1.0 - np.sum(composition_state))


def _copy_property_value(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _append_unique(items, new_items):
    for item in new_items:
        if item not in items:
            items.append(item)


def _get_phase_rate_prop_names(phase_names):
    return [
        f"phase_{rate_type}_rate_{phase_name}"
        for rate_type in ("molar", "mass", "volumetric")
        for phase_name in phase_names
    ]


def _flatten_property_values(values):
    flattened_values = values.reshape(-1)
    try:
        return flattened_values.astype(float)
    except (TypeError, ValueError):
        return flattened_values


def _get_requested_well_properties(
    coupled_model: DartsModel,
    output_properties,
    include_overall_composition,
    include_phase_velocities,
    include_phase_rates,
):
    pc = coupled_model.physics.property_containers[0]
    primary_prop_names = list(coupled_model.physics.vars)
    secondary_prop_names = list(pc.output_props)
    phase_rate_prop_names = _get_phase_rate_prop_names(pc.phases_name)

    if output_properties is None:
        requested_props = []
        _append_unique(requested_props, primary_prop_names)
        _append_unique(requested_props, secondary_prop_names)
    else:
        if isinstance(output_properties, str):
            requested_props = [output_properties]
        else:
            requested_props = list(output_properties)
        if not all(isinstance(prop, str) for prop in requested_props):
            raise TypeError("All entries in output_properties must be strings.")

    if output_properties is None and "temperature" not in requested_props:
        requested_props.append("temperature")

    known_props = set(primary_prop_names)
    known_props.update(secondary_prop_names)
    known_props.update(["temperature", "z", "vG", "vL"])
    known_props.update(phase_rate_prop_names)
    unknown_props = [prop for prop in requested_props if prop not in known_props]
    if unknown_props:
        raise KeyError("Unknown well output properties: " + ", ".join(unknown_props))

    # Requesting a derived property by name implies enabling its evaluation.
    include_overall_composition |= "z" in requested_props
    include_phase_velocities |= any(prop in requested_props for prop in ("vG", "vL"))
    include_phase_rates |= any(
        prop in requested_props for prop in phase_rate_prop_names
    )

    primary_prop_idxs = {
        prop: primary_prop_names.index(prop)
        for prop in requested_props
        if prop in primary_prop_names
    }
    requested_secondary_props = [
        prop
        for prop in requested_props
        if prop in secondary_prop_names and prop not in primary_prop_idxs
    ]

    return (
        primary_prop_idxs,
        requested_secondary_props,
        include_overall_composition,
        include_phase_velocities,
        include_phase_rates,
    )


def _pad_interface_values(interface_values, num_segments):
    segment_values = np.full(num_segments, np.nan, dtype=float)
    segment_values[: len(interface_values)] = interface_values
    return segment_values


def _get_upstream_interface_phase_props(segment_props, phase_velocity_matrix):
    num_interfaces, num_phases = phase_velocity_matrix.shape
    interface_props = np.zeros((num_interfaces, num_phases), dtype=float)

    for i in range(num_interfaces):
        for j in range(num_phases):
            if phase_velocity_matrix[i, j] >= 0:
                interface_props[i, j] = segment_props[i, j]
            else:
                interface_props[i, j] = segment_props[i + 1, j]

    return interface_props


def _evaluate_segment_phase_props(states, pc):
    num_segments = states.shape[0]
    saturations = np.zeros((num_segments, pc.nph), dtype=float)
    mass_densities = np.zeros((num_segments, pc.nph), dtype=float)
    molar_densities = np.zeros((num_segments, pc.nph), dtype=float)

    for i, state in enumerate(states):
        pc.evaluate(state)
        saturations[i, :] = pc.sat
        mass_densities[i, :] = pc.dens
        molar_densities[i, :] = pc.dens_m

    return saturations, mass_densities, molar_densities


def _get_phase_velocity_matrix(pipe, phase_velocities):
    num_interfaces = pipe.geometry.num_interfaces
    nph = pipe.physics.nph
    phase_velocity_matrix = np.zeros((num_interfaces, nph), dtype=float)

    vG = phase_velocities[:num_interfaces]
    vL = phase_velocities[num_interfaces:]
    phase_velocity_matrix[:, pipe.g_idx] = vG
    if pipe.n_mobile_phases == 2:
        phase_velocity_matrix[:, pipe.l_idx] = vL
    elif pipe.n_mobile_phases == 3:
        phase_velocity_matrix[:, pipe.la_idx] = vL
        phase_velocity_matrix[:, pipe.lb_idx] = vL

    return phase_velocity_matrix


def _get_phase_rates(pipe, states, phase_velocities, pc):
    saturations, mass_densities, molar_densities = _evaluate_segment_phase_props(
        states, pc
    )
    phase_velocity_matrix = _get_phase_velocity_matrix(pipe, phase_velocities)
    interface_saturations = _get_upstream_interface_phase_props(
        saturations, phase_velocity_matrix
    )
    interface_mass_densities = _get_upstream_interface_phase_props(
        mass_densities, phase_velocity_matrix
    )
    interface_molar_densities = _get_upstream_interface_phase_props(
        molar_densities, phase_velocity_matrix
    )

    volumetric_rates = (
        pipe.geometry.pipe_internal_A * interface_saturations * phase_velocity_matrix
    )
    mass_rates = volumetric_rates * interface_mass_densities
    molar_rates = volumetric_rates * interface_molar_densities

    return {
        "molar": molar_rates,
        "mass": mass_rates,
        "volumetric": volumetric_rates,
    }


def save_dfm_well_props(
    well_name: str,
    coupled_model: DartsModel,
    output_properties=None,
    *,
    include_overall_composition: bool = False,
    include_phase_velocities: bool = False,
    include_phase_rates: bool = False,
):
    """
    Store selected primary variables and phase properties of the well segments of the specified DFM well in a pickle
    file located in the output folder.

    :param well_name: Name of the well the properties of which will be saved
    :type well_name: str
    :param coupled_model: The instance of the DartsModel
    :type coupled_model: DartsModel
    :param output_properties: Optional property names to save. If omitted, primary
                              variables and PropertyContainer.output_props are saved.
    :param include_overall_composition: If True, save the full overall composition
                                        vector in column "z".
    :param include_phase_velocities: If True, evaluate and save vG and vL
                                     [m/day].
    :param include_phase_rates: If True, evaluate and save phase molar rates
                                [kmol/day], phase mass rates [kg/day], and phase
                                volumetric rates [m3/day].

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

    time = h5_well_dict["dynamic"]["time"]
    X_well_h5 = h5_well_dict["dynamic"]["X"]
    X_well_segments = X_well_h5[:, well_segments_idxs_in_well_h5, :]
    num_timesteps = len(time)
    pc = coupled_model.physics.property_containers[0]

    (
        primary_prop_idxs,
        output_prop_names,
        include_overall_composition,
        include_phase_velocities,
        include_phase_rates,
    ) = _get_requested_well_properties(
        coupled_model,
        output_properties,
        include_overall_composition,
        include_phase_velocities,
        include_phase_rates,
    )

    data = {
        prop_name: X_well_segments[:, :, prop_idx].reshape(-1)
        for prop_name, prop_idx in primary_prop_idxs.items()
    }

    if include_overall_composition:
        data["z"] = [
            _get_full_overall_composition(X_well_segments[i, j, :], pc)
            for i in range(num_timesteps)
            for j in range(num_segments)
        ]

    should_evaluate_secondary_props = bool(output_prop_names)
    if should_evaluate_secondary_props:
        output_prop_data = {
            prop_name: np.empty((num_timesteps, num_segments), dtype=object)
            for prop_name in output_prop_names
        }

        output_prop_getters = {
            prop_name: pc.output_props[prop_name] for prop_name in output_prop_names
        }
        for i in range(num_timesteps):
            for j in range(num_segments):
                state = X_well_segments[i, j, :]
                pc.evaluate(state)

                for prop_name, get_prop_value in output_prop_getters.items():
                    output_prop_data[prop_name][i, j] = _copy_property_value(
                        get_prop_value()
                    )

        for prop_name, prop_values in output_prop_data.items():
            data[prop_name] = _flatten_property_values(prop_values)

    if include_phase_velocities or include_phase_rates:
        pipe = coupled_model.wells[well_name]
        iter_counter = 0
        flag = 1
        time_from_zero = np.insert(time, 0, 0.0)
        time_step_sizes = np.diff(time_from_zero)
        if include_phase_velocities:
            vG_data = np.empty((num_timesteps, num_segments), dtype=float)
            vL_data = np.empty((num_timesteps, num_segments), dtype=float)
        if include_phase_rates:
            phase_rate_data = {
                prop_name: np.empty((num_timesteps, num_segments), dtype=float)
                for prop_name in _get_phase_rate_prop_names(pc.phases_name)
            }

        # The replay below mutates the live Pipe used by the running simulation
        # (reset_pipe_state / eval_phase_vels / accept_pipe_state). Snapshot the
        # state that persists across accepted time steps (the accepted-state
        # container, the first-iteration flag, and the attributes
        # _load_accepted_pipe_state restores) and put it back afterwards so
        # saving results mid-simulation is side-effect free.
        replayed_pipe_attrs = (
            "is_first_first_iter",
            "_accepted_pipe_state",
            "iter_phases_props",
            "rhoM_face",
            "rhoM_vM",
            "vM",
            "vG",
            "vL",
        )
        pipe_state_backup = {
            attr: copy.deepcopy(getattr(pipe, attr))
            for attr in replayed_pipe_attrs
            if hasattr(pipe, attr)
        }
        try:
            pipe.reset_pipe_state()
            for i, dt in enumerate(time_step_sizes):
                if i == 0:
                    Xn_ms_well = coupled_model.wells[
                        well_name
                    ].initial_conditions.initial_conditions_vector
                else:
                    Xn_ms_well = X_well_segments[i - 1, :, :].flatten()
                X_ms_well = X_well_segments[i, :, :].flatten()

                phase_velocities = pipe.eval_phase_vels(
                    Xn_ms_well, X_ms_well, dt, time_from_zero[i], iter_counter, flag
                )
                num_interfaces = pipe.geometry.num_interfaces
                if include_phase_velocities:
                    vG_data[i, :] = _pad_interface_values(
                        phase_velocities[:num_interfaces], num_segments
                    )
                    vL_data[i, :] = _pad_interface_values(
                        phase_velocities[num_interfaces:], num_segments
                    )
                if include_phase_rates:
                    phase_rates = _get_phase_rates(
                        pipe, X_well_segments[i, :, :], phase_velocities, pc
                    )
                    for rate_type, rates in phase_rates.items():
                        for phase_idx, phase_name in enumerate(pc.phases_name):
                            prop_name = f"phase_{rate_type}_rate_{phase_name}"
                            phase_rate_data[prop_name][i, :] = _pad_interface_values(
                                rates[:, phase_idx], num_segments
                            )
                pipe.accept_pipe_state()
        finally:
            for attr in replayed_pipe_attrs:
                if attr in pipe_state_backup:
                    setattr(pipe, attr, pipe_state_backup[attr])
                elif hasattr(pipe, attr):
                    delattr(pipe, attr)

        if include_phase_velocities:
            data["vG"] = vG_data.reshape(-1)
            data["vL"] = vL_data.reshape(-1)
        if include_phase_rates:
            for prop_name, prop_values in phase_rate_data.items():
                data[prop_name] = prop_values.reshape(-1)

    data_frame = pd.DataFrame(data)

    # Save the data frame in a pickle file
    file_name = f"dfm_well_props_{well_name}.pkl"
    file_path = os.path.join(coupled_model.output_folder, file_name)
    data_frame.to_pickle(file_path)
