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
    num_time_steps = len(h5_well_data["dynamic"]["X"][:,0,0])

    p = np.zeros(num_segments)
    z = np.zeros((num_segments, property_container.nc - 1))

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

    for i in range(num_time_steps):
        for j in range(num_segments):
            state = h5_well_data["dynamic"]["X"][i,j + num_perfs,:]
            p[j] = state[0]
            z[j,:] = state[1:]

            property_container.evaluate(state)
            xG[j,:] = property_container.x[0,:]
            xL[j,:] = property_container.x[1,:]
            sG[j] = property_container.sat[0]
            sL[j] = property_container.sat[1]
            rhoG[j] = property_container.dens[0]
            rhoL[j] = property_container.dens[1]
            miuG[j] = property_container.mu[0]
            miuL[j] = property_container.mu[1]

        ts_primary_vars_and_phases_props = [p.copy(), z.copy(), xG.copy(), xL.copy(),
                                            sG.copy(), sL.copy(), rhoG.copy(), rhoL.copy(),
                                            miuG.copy(), miuL.copy()]
        data_frame = pd.concat([data_frame, pd.DataFrame(list(zip(*ts_primary_vars_and_phases_props)),
                                                         columns=["Pressure", "Overall mole fractions", "xG", "xL",
                                                                  "sG", "sL", "rhoG", "rhoL", "miuG", "miuL"])])
    data_frame.to_pickle("output/stored_primary_vars_and_phase_props.pkl")
