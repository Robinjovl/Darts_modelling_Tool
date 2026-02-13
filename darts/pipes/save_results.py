import os

import numpy as np
import pandas as pd


def save_segments_primary_vars_and_phase_props(h5_well_data, coupled_model):
    """
    Store the primary variables and phase properties of well segments in a pickle file

    :param h5_well_data: Dictionary of well data loaded from the HDF5 file
    :type h5_well_data: dict
    :param coupled_model: The instance of the DartsModel
    :type coupled_model: DartsModel
    """
    property_container = coupled_model.physics.property_containers[0]

    num_perfs = len(coupled_model.reservoir.wells[0].perforations)
    num_segments = len(h5_well_data["dynamic"]["X"][0, :, 0]) - num_perfs

    p = np.zeros(num_segments)
    z = np.zeros((num_segments, property_container.nc))
    T = np.zeros(num_segments)

    sG = np.zeros(num_segments)
    if property_container.nph == 2:
        sL = np.zeros(num_segments)
    elif property_container.nph == 3:
        sL_a = np.zeros(num_segments)
        sL_b = np.zeros(num_segments)
    rhoG = np.zeros(num_segments)
    if property_container.nph == 2:
        rhoL = np.zeros(num_segments)
    elif property_container.nph == 3:
        rhoL_a = np.zeros(num_segments)
        rhoL_b = np.zeros(num_segments)
    miuG = np.zeros(num_segments)
    if property_container.nph == 2:
        miuL = np.zeros(num_segments)
    elif property_container.nph == 3:
        miuL_a = np.zeros(num_segments)
        miuL_b = np.zeros(num_segments)
    xG = np.zeros((num_segments, property_container.nc))
    if property_container.nph == 2:
        xL = np.zeros((num_segments, property_container.nc))
    elif property_container.nph == 3:
        xL_a = np.zeros((num_segments, property_container.nc))
        xL_b = np.zeros((num_segments, property_container.nc))
    # Initialize an empty DataFrame to store the primary variables and phase props
    data_frame = pd.DataFrame()

    # For phase velocity calculations
    next(iter(coupled_model.wells.values())).is_first_first_iter = True
    iter_counter = 0
    flag = 1

    time = h5_well_data["dynamic"]["time"]
    time_from_zero = np.insert(time, 0, 0.0)
    time_step_sizes = np.diff(time_from_zero)
    for i, dt in enumerate(time_step_sizes):
        for j in range(num_segments):
            state = h5_well_data["dynamic"]["X"][i, j + num_perfs, :]
            p[j] = state[0]

            # Evaluate temperature for when the primary vars are PH and evaluate phase props
            property_container.evaluate(state)

            if not coupled_model.physics.property_containers[0].thermal:
                z_full = state[1:]
                z_full = np.append(z_full, 1 - sum(state[1:]))
                z[j, :] = z_full
            elif coupled_model.physics.property_containers[0].thermal:
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
                    T[j] = property_container.temperature

            # xG[j,:] = property_container.x[1,:]
            xG[j, :] = property_container.x[0, :]
            if property_container.nph == 2:
                xL[j, :] = property_container.x[1, :]
            elif property_container.nph == 3:
                # xL_a[j, :] = property_container.x[0, :]
                xL_a[j, :] = property_container.x[1, :]
                xL_b[j, :] = property_container.x[2, :]
            # sG[j] = property_container.sat[1]
            sG[j] = property_container.sat[0]
            if property_container.nph == 2:
                sL[j] = property_container.sat[1]
            elif property_container.nph == 3:
                # sL_a[j] = property_container.sat[0]
                sL_a[j] = property_container.sat[1]
                sL_b[j] = property_container.sat[2]
            # rhoG[j] = property_container.dens[1]
            rhoG[j] = property_container.dens[0]
            if property_container.nph == 2:
                rhoL[j] = property_container.dens[1]
            elif property_container.nph == 3:
                # rhoL_a[j] = property_container.dens[0]
                rhoL_a[j] = property_container.dens[1]
                rhoL_b[j] = property_container.dens[2]
            # miuG[j] = property_container.mu[1]
            miuG[j] = property_container.mu[0]
            if property_container.nph == 2:
                miuL[j] = property_container.mu[1]
            elif property_container.nph == 3:
                # miuL_a[j] = property_container.mu[0]
                miuL_a[j] = property_container.mu[1]
                miuL_b[j] = property_container.mu[2]

        # Save phase velocities
        if i == 0:
            initial_conditions = next(
                iter(coupled_model.wells.values())
            ).initial_conditions.initial_conditions_vector
            Xn_ms_well = initial_conditions
            X_ms_well = h5_well_data["dynamic"]["X"][i, num_perfs:, :].flatten()
        else:
            Xn_ms_well = h5_well_data["dynamic"]["X"][i - 1, num_perfs:, :].flatten()
            X_ms_well = h5_well_data["dynamic"]["X"][i, num_perfs:, :].flatten()
        phase_velocities = next(iter(coupled_model.wells.values())).eval_phase_vels(
            Xn_ms_well, X_ms_well, dt, time_from_zero[i], iter_counter, flag
        )
        # phase_velocities = np.zeros((num_segments - 1) * 2)
        mid = int(len(phase_velocities) / 2)
        vG = phase_velocities[:mid]
        vL = phase_velocities[mid:]
        # Make velocity variables have the same size as the other props
        vG = np.append(vG, np.nan)
        vL = np.append(vL, np.nan)

        if property_container.nph == 2:
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
                            "Pressure",
                            "Overall mole fractions",
                            "Temperature",
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
        elif property_container.nph == 3:
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
                            "Pressure",
                            "Overall mole fractions",
                            "Temperature",
                            "xG",
                            "xL_a",
                            "xL_a",
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

    data_frame.to_pickle(
        os.path.join(
            coupled_model.output.output_folder,
            "well_primary_vars_and_phase_props.pkl",
        )
    )
