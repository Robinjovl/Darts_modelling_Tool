import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import pickle

from darts.models.darts_model import DartsModel

def visualize_results_heat_maps(primary_vars_and_phase_props_file_address: str, h5_well_data: dict, coupled_model: DartsModel):
    # y axis: standard direction and segment depths in meters
    y_axis_convention = "standard"
    y_axis = "segment_depth"

    # y axis: T2Well direction and segment indices
    # y_axis_convention = "T2Well"
    # y_axis = "segment_index"

    # x_axis = "time_step_index"
    x_axis = "simulation_time"

    well_geom = next(iter(coupled_model.wells_geometry.values()))

    if y_axis == "segment_depth":
        measured_depths_segments = (sum(well_geom.segments_lengths) - well_geom.z)
        true_vertical_depths_segments = measured_depths_segments * np.cos(well_geom.inclination_angle_radian)
        if y_axis_convention == "standard":
            true_vertical_depths_segments = true_vertical_depths_segments[::-1]

    if x_axis == "simulation_time":
        dt = h5_well_data["dynamic"]["time"]
        simulation_time = np.cumsum(dt)

    # Get components names
    components_names = coupled_model.physics.property_containers[0].components_name
    num_components = len(components_names)
    num_segments = well_geom.num_segments

    #%% Component/components overall mole fraction profiles
    #
    # for c, comp_name in enumerate(components_names[:-1]):
    #     # Initialize component c mole fraction matrix
    #     component_c_mole_fraction_matrix = np.zeros((num_segments, len(time_steps)))
    #
    #     # Fill component c mole fraction matrix
    #     for i, time_step in enumerate(time_steps):
    #         component_c_mole_fraction_profile = primary_variables_df[time_step][num_segments * (c + 1):num_segments * (c + 2)]
    #         component_c_mole_fraction_matrix[:, i] = component_c_mole_fraction_profile
    #
    #     # Initialize the plot
    #     fig, ax = plt.subplots(figsize=(12, 6))
    #
    #     # Create the heatmap
    #     cmap = plt.get_cmap('jet')
    #     if x_axis == "time_step_index" and y_axis == "segment_index":
    #         cax = ax.pcolormesh(range(len(time_steps)), range(num_segments), component_c_mole_fraction_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)
    #
    #         # Set the y-axis ticks
    #         ax.yaxis.set_major_locator(MultipleLocator(1))
    #
    #         # Add axes labels
    #         ax.set_xlabel('Time step [-]', fontsize=14)
    #         ax.set_ylabel('Segment index [-]', fontsize=14)
    #
    #     elif x_axis == "simulation_time" and y_axis == "segment_index":
    #         cax = ax.pcolormesh(simulation_time, range(num_segments), component_c_mole_fraction_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)
    #
    #         # Set the y-axis ticks
    #         ax.yaxis.set_major_locator(MultipleLocator(1))
    #
    #         # Add axes labels
    #         ax.set_xlabel('Simulation time [second]', fontsize=14)
    #         ax.set_ylabel('Segment index [-]', fontsize=14)
    #
    #     elif x_axis == "time_step_index" and y_axis == "segment_depth":
    #         cax = ax.pcolormesh(range(len(time_steps)), true_vertical_depths_segments, component_c_mole_fraction_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)
    #
    #         # Add axes labels
    #         ax.set_xlabel('Time step [-]', fontsize=14)
    #         ax.set_ylabel('TVD [meter]', fontsize=14)
    #
    #     elif x_axis == "simulation_time" and y_axis == "segment_depth":
    #         cax = ax.pcolormesh(simulation_time, true_vertical_depths_segments, component_c_mole_fraction_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)
    #
    #         # Add axes labels
    #         ax.set_xlabel('Simulation time [second]', fontsize=14)
    #         ax.set_ylabel('TVD [meter]', fontsize=14)
    #
    #     if y_axis_convention == "standard":
    #         # Reverse the y-axis
    #         ax.invert_yaxis()
    #
    #     # Add title
    #     ax.set_title(comp_name + " overall mole fraction profile along the wellbore over time", fontsize=14, fontweight='bold')
    #
    #     # Add a colorbar
    #     cbar = fig.colorbar(cax, ax=ax)
    #     cbar.set_label(comp_name + ' overall mole fraction [-]', fontsize=14)
    #
    #     plt.tight_layout()
    #     plt.show()

    # Load primary vars and phase props
    data_frame = pd.read_pickle(primary_vars_and_phase_props_file_address)

    num_ts = int(len(data_frame["sG"]) / num_segments)   # Initial conditions of sG is not stored.

    #%% Pressure profile

    # Initialize the pressure matrix
    p_matrix = np.zeros((num_segments, num_ts))

    # Fill the pressure matrix
    for ts_counter in range(num_ts):
        p = data_frame["Pressure"][ts_counter * num_segments:(ts_counter + 1) * num_segments]
        p_matrix[:, ts_counter] = p

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))


    # Create the heatmap
    cmap = plt.get_cmap('jet')
    if x_axis == "time_step_index" and y_axis == "segment_index":
        cax = ax.pcolormesh(range(num_ts), range(num_segments), p_matrix, cmap=cmap, shading='auto')

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel('Time step [-]', fontsize=14)
        ax.set_ylabel('Segment index [-]', fontsize=14)

    elif x_axis == "simulation_time" and y_axis == "segment_index":
        cax = ax.pcolormesh(simulation_time, range(num_segments), p_matrix, cmap=cmap, shading='auto')

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel('Simulation time [second]', fontsize=14)
        ax.set_ylabel('Segment index [-]', fontsize=14)

    elif x_axis == "time_step_index" and y_axis == "segment_depth":
        cax = ax.pcolormesh(range(num_ts), true_vertical_depths_segments, p_matrix, cmap=cmap, shading='auto')

        # Add axes labels
        ax.set_xlabel('Time step [-]', fontsize=14)
        ax.set_ylabel('TVD [meter]', fontsize=14)

    elif x_axis == "simulation_time" and y_axis == "segment_depth":
        cax = ax.pcolormesh(simulation_time, true_vertical_depths_segments, p_matrix, cmap=cmap, shading='auto')

        # Add axes labels
        ax.set_xlabel('Simulation time [second]', fontsize=14)
        ax.set_ylabel('TVD [meter]', fontsize=14)

    if y_axis_convention == "standard":
        # Reverse the y-axis
        ax.invert_yaxis()


    # Add title
    ax.set_title('Pressure profile along the wellbore over time', fontsize=14, fontweight='bold')

    # Add a colorbar to show the pressure values
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Pressure [bar]', fontsize=14)

    plt.tight_layout()
    plt.show()

    # %% Overall mole fraction profiles

    for comp_idx in range(num_components):
        # Initialize the overall mole fraction matrix
        z_c_matrix = np.zeros((num_segments, num_ts))

        # Fill the overall mole fraction matrix
        for ts_counter in range(num_ts):
            z = data_frame["Overall mole fractions"][ts_counter * num_segments:(ts_counter + 1) * num_segments]
            z = z.tolist()
            z_c = np.zeros(num_segments)
            for segment_idx in range(num_segments):
                try:
                    z_c[segment_idx] = z[segment_idx][comp_idx]
                except:
                    z_c[segment_idx] = 1 - sum(z[segment_idx])
            z_c_matrix[:, ts_counter] = z_c

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create the heatmap
        cmap = plt.get_cmap('jet')
        if x_axis == "time_step_index" and y_axis == "segment_index":
            cax = ax.pcolormesh(range(num_ts), range(num_segments), z_c_matrix, cmap=cmap, shading='auto')

            # Set the y-axis ticks
            ax.yaxis.set_major_locator(MultipleLocator(1))

            # Add axes labels
            ax.set_xlabel('Time step [-]', fontsize=14)
            ax.set_ylabel('Segment index [-]', fontsize=14)

        elif x_axis == "simulation_time" and y_axis == "segment_index":
            cax = ax.pcolormesh(simulation_time, range(num_segments), z_c_matrix, cmap=cmap, shading='auto')

            # Set the y-axis ticks
            ax.yaxis.set_major_locator(MultipleLocator(1))

            # Add axes labels
            ax.set_xlabel('Simulation time [second]', fontsize=14)
            ax.set_ylabel('Segment index [-]', fontsize=14)

        elif x_axis == "time_step_index" and y_axis == "segment_depth":
            cax = ax.pcolormesh(range(num_ts), true_vertical_depths_segments, z_c_matrix, cmap=cmap, shading='auto')

            # Add axes labels
            ax.set_xlabel('Time step [-]', fontsize=14)
            ax.set_ylabel('TVD [meter]', fontsize=14)

        elif x_axis == "simulation_time" and y_axis == "segment_depth":
            cax = ax.pcolormesh(simulation_time, true_vertical_depths_segments, z_c_matrix, cmap=cmap, shading='auto')

            # Add axes labels
            ax.set_xlabel('Simulation time [second]', fontsize=14)
            ax.set_ylabel('TVD [meter]', fontsize=14)

        if y_axis_convention == "standard":
            # Reverse the y-axis
            ax.invert_yaxis()


        # Add title
        ax.set_title('Profile of overall mole fraction of ' + components_names[comp_idx] + ' along the wellbore over time', fontsize=14, fontweight='bold')

        # Add a colorbar to show the temperature values
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label(components_names[comp_idx] + ' overall mole fraction [-]', fontsize=14)

        plt.tight_layout()
        plt.show()


    #%% Temperature profile

    if coupled_model.physics.thermal is True:
        # Initialize the temperature matrix
        T_matrix = np.zeros((num_segments, num_ts))

        # Fill the temperature matrix
        for ts_counter in range(num_ts):
            T = data_frame["Temperature"][ts_counter * num_segments:(ts_counter + 1) * num_segments]
            T_matrix[:, ts_counter] = T

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create the heatmap
        cmap = plt.get_cmap('jet')
        if x_axis == "time_step_index" and y_axis == "segment_index":
            cax = ax.pcolormesh(range(num_ts), range(num_segments), T_matrix, cmap=cmap, shading='auto')

            # Set the y-axis ticks
            ax.yaxis.set_major_locator(MultipleLocator(1))

            # Add axes labels
            ax.set_xlabel('Time step [-]', fontsize=14)
            ax.set_ylabel('Segment index [-]', fontsize=14)

        elif x_axis == "simulation_time" and y_axis == "segment_index":
            cax = ax.pcolormesh(simulation_time, range(num_segments), T_matrix, cmap=cmap, shading='auto')

            # Set the y-axis ticks
            ax.yaxis.set_major_locator(MultipleLocator(1))

            # Add axes labels
            ax.set_xlabel('Simulation time [second]', fontsize=14)
            ax.set_ylabel('Segment index [-]', fontsize=14)

        elif x_axis == "time_step_index" and y_axis == "segment_depth":
            cax = ax.pcolormesh(range(num_ts), true_vertical_depths_segments, T_matrix, cmap=cmap, shading='auto')

            # Add axes labels
            ax.set_xlabel('Time step [-]', fontsize=14)
            ax.set_ylabel('TVD [meter]', fontsize=14)

        elif x_axis == "simulation_time" and y_axis == "segment_depth":
            cax = ax.pcolormesh(simulation_time, true_vertical_depths_segments, T_matrix, cmap=cmap, shading='auto')

            # Add axes labels
            ax.set_xlabel('Simulation time [second]', fontsize=14)
            ax.set_ylabel('TVD [meter]', fontsize=14)

        if y_axis_convention == "standard":
            # Reverse the y-axis
            ax.invert_yaxis()


        # Add title
        ax.set_title('Temperature profile along the wellbore over time', fontsize=14, fontweight='bold')

        # Add a colorbar to show the temperature values
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label('Temperature [\u00B0C]', fontsize=14)

        plt.tight_layout()
        plt.show()

    #%% Gas saturation profile

    # Initialize the gas saturation matrix
    sG_matrix = np.zeros((num_segments, num_ts))

    # Fill the gas saturation matrix
    for ts_counter in range(num_ts):
        sG = data_frame["sG"][ts_counter * num_segments:(ts_counter + 1) * num_segments]
        sG_matrix[:, ts_counter] = sG

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))


    # Create the heatmap
    cmap = plt.get_cmap('jet')
    if x_axis == "time_step_index" and y_axis == "segment_index":
        cax = ax.pcolormesh(range(num_ts), range(num_segments), sG_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel('Time step [-]', fontsize=14)
        ax.set_ylabel('Segment index [-]', fontsize=14)

    elif x_axis == "simulation_time" and y_axis == "segment_index":
        cax = ax.pcolormesh(simulation_time, range(num_segments), sG_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel('Simulation time [second]', fontsize=14)
        ax.set_ylabel('Segment index [-]', fontsize=14)

    elif x_axis == "time_step_index" and y_axis == "segment_depth":
        cax = ax.pcolormesh(range(num_ts), true_vertical_depths_segments, sG_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

        # Add axes labels
        ax.set_xlabel('Time step [-]', fontsize=14)
        ax.set_ylabel('TVD [meter]', fontsize=14)

    elif x_axis == "simulation_time" and y_axis == "segment_depth":
        cax = ax.pcolormesh(simulation_time, true_vertical_depths_segments, sG_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

        # Add axes labels
        ax.set_xlabel('Simulation time [second]', fontsize=14)
        ax.set_ylabel('TVD [meter]', fontsize=14)

    if y_axis_convention == "standard":
        # Reverse the y-axis
        ax.invert_yaxis()


    # Add title
    ax.set_title('Gas saturation profile along the wellbore over time', fontsize=14, fontweight='bold')

    # Add a colorbar to show the gas saturation values
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Gas saturation [-]', fontsize=14)

    plt.tight_layout()
    plt.show()


    #%% Profile/profiles of components mole fractions in the gaseous phase

    for c, comp_name in enumerate(components_names):
        # Initialize the xG_mole_c matrix
        xG_mole_c_matrix = np.zeros((num_segments, num_ts))

        # Fill the xG_mole_c matrix
        for ts_counter in range(num_ts):
            xG = data_frame["xG"][ts_counter * num_segments:(ts_counter + 1) * num_segments]
            xG_c = np.array([x[c] for x in xG])
            xG_mole_c_matrix[:, ts_counter] = xG_c

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create the heatmap
        cmap = plt.get_cmap('jet')
        if x_axis == "time_step_index" and y_axis == "segment_index":
            cax = ax.pcolormesh(range(num_ts), range(num_segments), xG_mole_c_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

            # Set the y-axis ticks
            ax.yaxis.set_major_locator(MultipleLocator(1))

            # Add axes labels
            ax.set_xlabel('Time step [-]', fontsize=14)
            ax.set_ylabel('Segment index [-]', fontsize=14)

        elif x_axis == "simulation_time" and y_axis == 'segment_index':
            cax = ax.pcolormesh(simulation_time, range(num_segments), xG_mole_c_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

            # Set the y-axis ticks
            ax.yaxis.set_major_locator(MultipleLocator(1))

            # Add axes labels
            ax.set_xlabel('Simulation time [second]', fontsize=14)
            ax.set_ylabel('Segment index [-]', fontsize=14)

        elif x_axis == "time_step_index" and y_axis == 'segment_depth':
            cax = ax.pcolormesh(range(num_ts), true_vertical_depths_segments, xG_mole_c_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

            # Add axes labels
            ax.set_xlabel('Time step [-]', fontsize=14)
            ax.set_ylabel('TVD [meter]', fontsize=14)

        elif x_axis == "simulation_time" and y_axis == 'segment_depth':
            cax = ax.pcolormesh(simulation_time, true_vertical_depths_segments, xG_mole_c_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

            # Add axes labels
            ax.set_xlabel('Simulation time [second]', fontsize=14)
            ax.set_ylabel('TVD [meter]', fontsize=14)


        if y_axis_convention == "standard":
            # Reverse the y-axis
            ax.invert_yaxis()

        # Add title
        ax.set_title('Profile of ' + comp_name + ' mole fraction in the gaseous phase along the wellbore over time', fontsize=14, fontweight='bold')

        # Add a colorbar to show the xG_mole_CO2 values
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label(comp_name + ' mole fraction in the gaseous phase [-]', fontsize=14)

        plt.tight_layout()
        plt.show()


    #%% Profile/profiles of components mole fractions in the liquid phase

    for c, comp_name in enumerate(components_names):
        # Initialize the xL_mole_c matrix
        xL_mole_c_matrix = np.zeros((num_segments, num_ts))

        # Fill the xL_mole_c matrix
        for ts_counter in range(num_ts):
            xL = data_frame["xL"][ts_counter * num_segments:(ts_counter + 1) * num_segments]
            xL_c = np.array([x[c] for x in xL])
            xL_mole_c_matrix[:, ts_counter] = xL_c

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create the heatmap
        cmap = plt.get_cmap('jet')
        if x_axis == "time_step_index" and y_axis == "segment_index":
            cax = ax.pcolormesh(range(num_ts), range(num_segments), xL_mole_c_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

            # Set the y-axis ticks
            ax.yaxis.set_major_locator(MultipleLocator(1))

            # Add axes labels
            ax.set_xlabel('Time step [-]', fontsize=14)
            ax.set_ylabel('Segment index [-]', fontsize=14)

        elif x_axis == "simulation_time" and y_axis == 'segment_index':
            cax = ax.pcolormesh(simulation_time, range(num_segments), xL_mole_c_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

            # Set the y-axis ticks
            ax.yaxis.set_major_locator(MultipleLocator(1))

            # Add axes labels
            ax.set_xlabel('Simulation time [second]', fontsize=14)
            ax.set_ylabel('Segment index [-]', fontsize=14)

        elif x_axis == "time_step_index" and y_axis == 'segment_depth':
            cax = ax.pcolormesh(range(num_ts), true_vertical_depths_segments, xL_mole_c_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

            # Add axes labels
            ax.set_xlabel('Time step [-]', fontsize=14)
            ax.set_ylabel('TVD [meter]', fontsize=14)

        elif x_axis == "simulation_time" and y_axis == 'segment_depth':
            cax = ax.pcolormesh(simulation_time, true_vertical_depths_segments, xL_mole_c_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

            # Add axes labels
            ax.set_xlabel('Simulation time [second]', fontsize=14)
            ax.set_ylabel('TVD [meter]', fontsize=14)


        if y_axis_convention == "standard":
            # Reverse the y-axis
            ax.invert_yaxis()

        # Add title
        ax.set_title('Profile of ' + comp_name + ' mole fraction in the liquid phase along the wellbore over time', fontsize=14, fontweight='bold')

        # Add a colorbar to show the xG_mole_CO2 values
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label(comp_name + ' mole fraction in the liquid phase [-]', fontsize=14)

        plt.tight_layout()
        plt.show()


    #%% Gas density profile

    # Initialize the gas density matrix
    rhoG_matrix = np.zeros((num_segments, num_ts))

    # Fill the gas density matrix
    for ts_counter in range(num_ts):
        rhoG = data_frame["rhoG"][ts_counter * num_segments:(ts_counter + 1) * num_segments]
        rhoG_matrix[:, ts_counter] = rhoG

    # Apply a mask to hide values equal to zero
    threshold = 0  # Set your threshold here
    rhoG_matrix_masked = np.ma.masked_where(rhoG_matrix == threshold, rhoG_matrix)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))


    # Create the heatmap
    cmap = plt.get_cmap('jet')
    if x_axis == "time_step_index" and y_axis == "segment_index":
        cax = ax.pcolormesh(range(num_ts), range(num_segments), rhoG_matrix_masked, cmap=cmap, shading='auto')

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel('Time step [-]', fontsize=14)
        ax.set_ylabel('Segment index [-]', fontsize=14)

    elif x_axis == "simulation_time" and y_axis == "segment_index":
        cax = ax.pcolormesh(simulation_time, range(num_segments), rhoG_matrix_masked, cmap=cmap, shading='auto')

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel('Simulation time [second]', fontsize=14)
        ax.set_ylabel('Segment index [-]', fontsize=14)

    elif x_axis == "time_step_index" and y_axis == "segment_depth":
        cax = ax.pcolormesh(range(num_ts), true_vertical_depths_segments, rhoG_matrix_masked, cmap=cmap, shading='auto')

        # Add axes labels
        ax.set_xlabel('Time step [-]', fontsize=14)
        ax.set_ylabel('TVD [meter]', fontsize=14)

    elif x_axis == "simulation_time" and y_axis == "segment_depth":
        cax = ax.pcolormesh(simulation_time, true_vertical_depths_segments, rhoG_matrix_masked, cmap=cmap, shading='auto')

        # Add axes labels
        ax.set_xlabel('Simulation time [second]', fontsize=14)
        ax.set_ylabel('TVD [meter]', fontsize=14)

    if y_axis_convention == "standard":
        # Reverse the y-axis
        ax.invert_yaxis()


    # Add title
    ax.set_title('Gas density profile along the wellbore over time', fontsize=14, fontweight='bold')

    # Add a colorbar to show the gas density values
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Gas density [kg/m$^3$]', fontsize=14)

    plt.tight_layout()
    plt.show()


    #%% Liquid density profile

    # Initialize the liquid density matrix
    rhoL_matrix = np.zeros((num_segments, num_ts))

    # Fill the liquid density matrix
    for ts_counter in range(num_ts):
        rhoL = data_frame["rhoL"][ts_counter * num_segments:(ts_counter + 1) * num_segments]
        rhoL_matrix[:, ts_counter] = rhoL

    # Apply a mask to hide values equal to zero
    threshold = 0  # Set your threshold here
    rhoL_matrix_masked = np.ma.masked_where(rhoL_matrix == threshold, rhoL_matrix)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))


    # Create the heatmap
    cmap = plt.get_cmap('jet')
    if x_axis == "time_step_index" and y_axis == "segment_index":
        cax = ax.pcolormesh(range(num_ts), range(num_segments), rhoL_matrix_masked, cmap=cmap, shading='auto')

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel('Time step [-]', fontsize=14)
        ax.set_ylabel('Segment index [-]', fontsize=14)

    elif x_axis == "simulation_time" and y_axis == "segment_index":
        cax = ax.pcolormesh(simulation_time, range(num_segments), rhoL_matrix_masked, cmap=cmap, shading='auto')

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel('Simulation time [second]', fontsize=14)
        ax.set_ylabel('Segment index [-]', fontsize=14)

    elif x_axis == "time_step_index" and y_axis == "segment_depth":
        cax = ax.pcolormesh(range(num_ts), true_vertical_depths_segments, rhoL_matrix_masked, cmap=cmap, shading='auto')

        # Add axes labels
        ax.set_xlabel('Time step [-]', fontsize=14)
        ax.set_ylabel('TVD [meter]', fontsize=14)

    elif x_axis == "simulation_time" and y_axis == "segment_depth":
        cax = ax.pcolormesh(simulation_time, true_vertical_depths_segments, rhoL_matrix_masked, cmap=cmap, shading='auto')

        # Add axes labels
        ax.set_xlabel('Simulation time [second]', fontsize=14)
        ax.set_ylabel('TVD [meter]', fontsize=14)

    if y_axis_convention == "standard":
        # Reverse the y-axis
        ax.invert_yaxis()

    # Add title
    ax.set_title('Liquid density profile along the wellbore over time', fontsize=14, fontweight='bold')

    # Add a colorbar to show the liquid density values
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Liquid density [kg/m$^3$]', fontsize=14)

    plt.tight_layout()
    plt.show()

    #%% Gas viscosity profile

    # Initialize the gas viscosity matrix
    miuG_matrix = np.zeros((num_segments, num_ts))

    # Fill the liquid density matrix
    for ts_counter in range(num_ts):
        miuG = data_frame["miuG"][ts_counter * num_segments:(ts_counter + 1) * num_segments]
        miuG_matrix[:, ts_counter] = miuG

    # Apply a mask to hide values equal to zero
    threshold = 0  # Set your threshold here
    miuG_matrix_masked = np.ma.masked_where(miuG_matrix == threshold, miuG_matrix)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))


    # Create the heatmap
    cmap = plt.get_cmap('jet')
    if x_axis == "time_step_index" and y_axis == "segment_index":
        cax = ax.pcolormesh(range(num_ts), range(num_segments), miuG_matrix_masked, cmap=cmap, shading='auto')

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel('Time step [-]', fontsize=14)
        ax.set_ylabel('Segment index [-]', fontsize=14)

    elif x_axis == "simulation_time" and y_axis == "segment_index":
        cax = ax.pcolormesh(simulation_time, range(num_segments), miuG_matrix_masked, cmap=cmap, shading='auto')

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel('Simulation time [second]', fontsize=14)
        ax.set_ylabel('Segment index [-]', fontsize=14)

    elif x_axis == "time_step_index" and y_axis == "segment_depth":
        cax = ax.pcolormesh(range(num_ts), true_vertical_depths_segments, miuG_matrix_masked, cmap=cmap, shading='auto')

        # Add axes labels
        ax.set_xlabel('Time step [-]', fontsize=14)
        ax.set_ylabel('TVD [meter]', fontsize=14)

    elif x_axis == "simulation_time" and y_axis == "segment_depth":
        cax = ax.pcolormesh(simulation_time, true_vertical_depths_segments, miuG_matrix_masked, cmap=cmap, shading='auto')

        # Add axes labels
        ax.set_xlabel('Simulation time [second]', fontsize=14)
        ax.set_ylabel('TVD [meter]', fontsize=14)

    if y_axis_convention == "standard":
        # Reverse the y-axis
        ax.invert_yaxis()

    # Add title
    ax.set_title('Gas viscosity profile along the wellbore over time', fontsize=14, fontweight='bold')

    # Add a colorbar to show the liquid density values
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Gas viscosity [cP]', fontsize=14)

    plt.tight_layout()
    plt.show()

    #%% Liquid viscosity profile

    # Initialize the liquid viscosity matrix
    miuL_matrix = np.zeros((num_segments, num_ts))

    # Fill the liquid density matrix
    for ts_counter in range(num_ts):
        miuL = data_frame["miuL"][ts_counter * num_segments:(ts_counter + 1) * num_segments]
        miuL_matrix[:, ts_counter] = miuL

    # Apply a mask to hide values equal to zero
    threshold = 0  # Set your threshold here
    miuL_matrix_masked = np.ma.masked_where(miuL_matrix == threshold, miuL_matrix)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))


    # Create the heatmap
    cmap = plt.get_cmap('jet')
    if x_axis == "time_step_index" and y_axis == "segment_index":
        cax = ax.pcolormesh(range(num_ts), range(num_segments), miuL_matrix_masked, cmap=cmap, shading='auto')

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel('Time step [-]', fontsize=14)
        ax.set_ylabel('Segment index [-]', fontsize=14)

    elif x_axis == "simulation_time" and y_axis == "segment_index":
        cax = ax.pcolormesh(simulation_time, range(num_segments), miuL_matrix_masked, cmap=cmap, shading='auto')

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel('Simulation time [second]', fontsize=14)
        ax.set_ylabel('Segment index [-]', fontsize=14)

    elif x_axis == "time_step_index" and y_axis == "segment_depth":
        cax = ax.pcolormesh(range(num_ts), true_vertical_depths_segments, miuL_matrix_masked, cmap=cmap, shading='auto')

        # Add axes labels
        ax.set_xlabel('Time step [-]', fontsize=14)
        ax.set_ylabel('TVD [meter]', fontsize=14)

    elif x_axis == "simulation_time" and y_axis == "segment_depth":
        cax = ax.pcolormesh(simulation_time, true_vertical_depths_segments, miuL_matrix_masked, cmap=cmap, shading='auto')

        # Add axes labels
        ax.set_xlabel('Simulation time [second]', fontsize=14)
        ax.set_ylabel('TVD [meter]', fontsize=14)

    if y_axis_convention == "standard":
        # Reverse the y-axis
        ax.invert_yaxis()

    # Add title
    ax.set_title('Liquid viscosity profile along the wellbore over time', fontsize=14, fontweight='bold')

    # Add a colorbar to show the liquid density values
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Liquid viscosity [$cP$]', fontsize=14)

    plt.tight_layout()
    plt.show()


    #%% Gas velocity profile

    # # Initialize the gas velocity matrix
    # num_interfaces = num_segments - 1
    # vG_matrix = np.zeros((num_interfaces, num_ts))
    #
    # # Fill the gas velocity matrix
    # for ts_counter in range(num_ts):
    #     vG = data_frame["vG"][ts_counter * num_segments:(ts_counter + 1) * num_segments]
    #     vG_matrix[:, ts_counter] = vG[:-1]
    #
    # # Apply a mask to hide some values
    # vG_matrix_masked = np.ma.masked_where(vG_matrix == 0, vG_matrix)
    #
    # # Initialize the plot
    # fig, ax = plt.subplots(figsize=(12, 6))
    #
    #
    # # Create the heatmap
    # cmap = plt.get_cmap('jet')
    # if x_axis == "time_step_index" and y_axis == "segment_index":
    #     cax = ax.pcolormesh(range(num_ts), range(num_interfaces), vG_matrix_masked, cmap=cmap, shading='auto')
    #
    #     # Set the y-axis ticks
    #     ax.yaxis.set_major_locator(MultipleLocator(1))
    #
    #     # Add axes labels
    #     ax.set_xlabel('Time step [-]', fontsize=14)
    #     ax.set_ylabel('Interface index [-]', fontsize=14)
    #
    # elif x_axis == "simulation_time" and y_axis == "segment_index":
    #     cax = ax.pcolormesh(simulation_time, range(num_interfaces), vG_matrix_masked, cmap=cmap, shading='auto')
    #
    #     # Set the y-axis ticks
    #     ax.yaxis.set_major_locator(MultipleLocator(1))
    #
    #     # Add axes labels
    #     ax.set_xlabel('Simulation time [second]', fontsize=14)
    #     ax.set_ylabel('Interface index [-]', fontsize=14)
    #
    # elif x_axis == "time_step_index" and y_axis == "segment_depth":
    #     cax = ax.pcolormesh(range(num_ts), true_vertical_depths_interfaces, vG_matrix_masked, cmap=cmap, shading='auto')
    #
    #     # Add axes labels
    #     ax.set_xlabel('Time step [-]', fontsize=14)
    #     ax.set_ylabel('TVD [meter]', fontsize=14)
    #
    # elif x_axis == "simulation_time" and y_axis == "segment_depth":
    #     cax = ax.pcolormesh(simulation_time, true_vertical_depths_interfaces, vG_matrix_masked, cmap=cmap, shading='auto')
    #
    #     # Add axes labels
    #     ax.set_xlabel('Simulation time [second]', fontsize=14)
    #     ax.set_ylabel('TVD [meter]', fontsize=14)
    #
    # if y_axis_convention == "standard":
    #     # Reverse the y-axis
    #     ax.invert_yaxis()
    #
    #
    # # Add title
    # ax.set_title('Gas velocity profile along the wellbore over time', fontsize=14, fontweight='bold')
    #
    # # Add a colorbar to show the gas velocity values
    # cbar = fig.colorbar(cax, ax=ax)
    # cbar.set_label('Gas velocity [m/s]', fontsize=14)
    #
    # plt.tight_layout()
    # plt.show()


    #%% Liquid velocity profile

    # # Initialize the gas velocity matrix
    # num_interfaces = num_segments - 1
    # vL_matrix = np.zeros((num_interfaces, num_ts))
    #
    # # Fill the liquid velocity matrix
    # for ts_counter in range(num_ts):
    #     vL = data_frame["vL"][ts_counter * num_segments:(ts_counter + 1) * num_segments]
    #     vL_matrix[:, ts_counter] = vL[:-1]
    #
    # # Apply a mask to hide some values
    # vL_matrix_masked = np.ma.masked_where(vL_matrix == 0, vL_matrix)
    #
    # # Initialize the plot
    # fig, ax = plt.subplots(figsize=(12, 6))
    #
    #
    # # Create the heatmap
    # cmap = plt.get_cmap('jet')
    # if x_axis == "time_step_index" and y_axis == "segment_index":
    #     cax = ax.pcolormesh(range(num_ts), range(num_interfaces), vL_matrix_masked, cmap=cmap, shading='auto')
    #
    #     # Set the y-axis ticks
    #     ax.yaxis.set_major_locator(MultipleLocator(1))
    #
    #     # Add axes labels
    #     ax.set_xlabel('Time step [-]', fontsize=14)
    #     ax.set_ylabel('Interface index [-]', fontsize=14)
    #
    # elif x_axis == "simulation_time" and y_axis == "segment_index":
    #     cax = ax.pcolormesh(simulation_time, range(num_interfaces), vL_matrix_masked, cmap=cmap, shading='auto')
    #
    #     # Set the y-axis ticks
    #     ax.yaxis.set_major_locator(MultipleLocator(1))
    #
    #     # Add axes labels
    #     ax.set_xlabel('Simulation time [second]', fontsize=14)
    #     ax.set_ylabel('Interface index [-]', fontsize=14)
    #
    # elif x_axis == "time_step_index" and y_axis == "segment_depth":
    #     cax = ax.pcolormesh(range(num_ts), true_vertical_depths_interfaces, vL_matrix_masked, cmap=cmap, shading='auto')
    #
    #     # Add axes labels
    #     ax.set_xlabel('Time step [-]', fontsize=14)
    #     ax.set_ylabel('TVD [meter]', fontsize=14)
    #
    # elif x_axis == "simulation_time" and y_axis == "segment_depth":
    #     cax = ax.pcolormesh(simulation_time, true_vertical_depths_interfaces, vL_matrix_masked, cmap=cmap, shading='auto')
    #
    #     # Add axes labels
    #     ax.set_xlabel('Simulation time [second]', fontsize=14)
    #     ax.set_ylabel('TVD [meter]', fontsize=14)
    #
    # if y_axis_convention == "standard":
    #     # Reverse the y-axis
    #     ax.invert_yaxis()
    #
    #
    # # Add title
    # ax.set_title('Liquid velocity profile along the wellbore over time', fontsize=14, fontweight='bold')
    #
    # # Add a colorbar to show the liquid velocity values
    # cbar = fig.colorbar(cax, ax=ax)
    # cbar.set_label('Liquid velocity [m/s]', fontsize=14)
    #
    # plt.tight_layout()
    # plt.show()
