import numpy as np
import pandas as pd

from darts.models.darts_model import DartsModel

def save_segments_primary_vars_and_phase_props(h5_well_data, coupled_model):
    """
    Stores the primary variables and phase properties of well segments in a pickle file

    :param h5_well_data: Dictionary of well data loaded from the HDF5 file
    :type h5_well_data: dict
    :param coupled_model: The instance of the DartsModel
    :type coupled_model: DartsModel
    """
    property_container = coupled_model.physics.property_containers[0]

    num_perfs = len(coupled_model.reservoir.wells[0].perforations)
    num_segments = len(h5_well_data["dynamic"]["X"][0,:,0]) - num_perfs

    p = np.zeros(num_segments)
    z = np.zeros((num_segments, property_container.nc - 1))
    T = np.zeros((num_segments))

    sG = np.zeros(num_segments)
    sL = np.zeros(num_segments)
    rhoG = np.zeros(num_segments)
    rhoL = np.zeros(num_segments)
    miuG = np.zeros(num_segments)
    miuL = np.zeros(num_segments)
    xG = np.zeros((num_segments, property_container.nc))
    xL = np.zeros((num_segments, property_container.nc))
    # Initialize an empty DataFrame to store the primary variables and phase props
    data_frame = pd.DataFrame()

    # For phase velocity calculations
    next(iter(coupled_model.wells.values())).is_first_first_iter = True
    iter_counter = 0
    flag = 1

    time = h5_well_data['dynamic']['time']
    time_step_sizes = np.diff(np.insert(time, 0, 0))
    for i, dt in enumerate(time_step_sizes):
        for j in range(num_segments):
            state = h5_well_data["dynamic"]["X"][i,j + num_perfs,:]
            p[j] = state[0]

            if not coupled_model.physics.property_containers[0].thermal:
                z[j, :] = state[1:]
            elif coupled_model.physics.property_containers[0].thermal:
                z[j, :] = state[1:-1]
                T[j] = state[-1]

            property_container.evaluate(state)
            xG[j,:] = property_container.x[0,:]
            xL[j,:] = property_container.x[1,:]
            sG[j] = property_container.sat[0]
            sL[j] = property_container.sat[1]
            rhoG[j] = property_container.dens[0]
            rhoL[j] = property_container.dens[1]
            miuG[j] = property_container.mu[0]
            miuL[j] = property_container.mu[1]

        # Save phase velocities
        if i == 0:
            initial_conditions = np.column_stack([coupled_model.wells['I1'].initial_conditions[key] for key in coupled_model.wells['I1'].initial_conditions]).ravel()
            Xn_ms_well = initial_conditions
            X_ms_well = h5_well_data["dynamic"]["X"][i,num_perfs:,:].flatten()
        else:
            Xn_ms_well = h5_well_data["dynamic"]["X"][i-1, num_perfs:, :].flatten()
            X_ms_well = h5_well_data["dynamic"]["X"][i, num_perfs:, :].flatten()
        phase_velocities = next(iter(coupled_model.wells.values())).evaluate_phase_velocities(Xn_ms_well, X_ms_well,
                                                                                              dt, iter_counter, flag)
        mid = int(len(phase_velocities) / 2)
        vG = phase_velocities[:mid]
        vL = phase_velocities[mid:]
        # Make velocity variables have the same size as the other props
        vG = np.append(vG, np.nan)
        vL = np.append(vL, np.nan)

        ts_primary_vars_and_phases_props = [p.copy(), z.copy(), T.copy(), xG.copy(), xL.copy(),
                                            sG.copy(), sL.copy(), rhoG.copy(), rhoL.copy(),
                                            miuG.copy(), miuL.copy(), vG.copy(), vL.copy()]

        data_frame = pd.concat([data_frame, pd.DataFrame(list(zip(*ts_primary_vars_and_phases_props)),
                                                         columns=["Pressure", "Overall mole fractions", "Temperature",
                                                                  "xG", "xL", "sG", "sL", "rhoG", "rhoL", "miuG", "miuL"
                                                                  , "vG", "vL"])])

    data_frame.to_pickle("output/stored_primary_vars_and_phase_props.pkl")
