import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from darts.models.darts_model import DartsModel
from darts.tools.hdf5_tools import load_hdf5_to_dict


def plot_heat_map_pcolormesh(
    well_name: str,
    coupled_model: DartsModel,
    max_ts_idx: int = None,
    x_axis: str = "simulated_time",
    y_axis: str = "segments_MD",
    cmap_color: str = "jet",
    save_as: str = "pdf",
    show_plot: bool = True,
    font_size: float = 14,
    with_title: bool = True,
):
    """
    Plot property profiles over time using pcolormesh for the specified well

    :param well_name: Name of the well the properties of which will be plotted
    :type well_name: str
    :param coupled_model: An instance of DartsModel
    :type coupled_model: DartsModel
    :param max_ts_idx: If specified, the heat map will be shown until the specified maximum time step index. If not
    specified, the heat map will be shown for all the time steps.
    :type max_ts_idx: int
    :param x_axis: "simulated_time" or "time_step_index"
    :type x_axis: str
    :param y_axis: "segments_MD" or "segments_TVD" or "segment_index"
    :type y_axis: str
    :param save_as: The extension of the image files that will be saved
    :type save_as: str
    :param show_plot: Whether or not to show the plot
    :type show_plot: bool
    :param font_size: Size of the fonts
    :type font_size: float
    :param with_title: If you want the figure to have a title or not
    :type with_title: bool
    """
    output_folder_name = f'heat_maps_pcolormesh_{well_name}'
    main_dir = os.path.join(coupled_model.output_folder, output_folder_name)

    # Reset directory
    if os.path.exists(main_dir):
        shutil.rmtree(main_dir)
    os.makedirs(main_dir)

    # Well HDF5 file is used here to get the time step sizes
    h5_well_file_path = coupled_model.well_filepath
    h5_well_dict = load_hdf5_to_dict(h5_well_file_path)

    # Get well geometry info
    well_geom = coupled_model.wells[well_name].geometry
    segments_MD = well_geom.z
    interfaces_MD = well_geom.z_interfaces
    if y_axis == "segments_TVD":
        segments_TVD = well_geom.TVD_segments
        interfaces_TVD = well_geom.TVD_interfaces
    num_segments = well_geom.num_segments
    num_interfaces = well_geom.num_interfaces

    # Get physics info
    pc = coupled_model.physics.property_containers[0]
    components_names = pc.components_name
    num_components = len(components_names)

    # Load primary vars and phase props
    well_props_file_path = os.path.join(
        coupled_model.output.output_folder, f"dfm_well_props_{well_name}.pkl"
    )
    data_frame = pd.read_pickle(well_props_file_path)

    num_ts = int(
        len(data_frame["sG"]) / num_segments
    )  # Initial conditions of sG is not stored.
    if max_ts_idx is None:
        max_ts_idx = num_ts
    assert max_ts_idx <= num_ts, (
        f"max_ts_idx is larger than the total number of time steps, which is {num_ts}!"
    )

    if x_axis == "simulated_time":
        # Convert days to seconds
        simulated_time = h5_well_dict["dynamic"]["time"] * 24 * 60 * 60
        # Apply the user-specified time-step index range
        simulated_time = simulated_time[:max_ts_idx]

    time_step_idx_range = range(max_ts_idx)
    num_selected_ts = len(time_step_idx_range)

    if x_axis == "time_step_index":
        x = time_step_idx_range
        x_label = "Time step [-]"
    elif x_axis == "simulated_time":
        x = simulated_time
        x_label = "Simulated time [second]"

    if y_axis == "segment_index":
        y_segments = range(num_segments)
        y_segments_label = "Segment index [-]"
        y_interfaces = range(num_interfaces)
        y_interfaces_label = "Interface index [-]"
    elif y_axis == "segments_MD":
        y_segments = segments_MD
        y_segments_label = "Segment MD [meter]"
        y_interfaces = interfaces_MD
        y_interfaces_label = "Interface MD [meter]"
    elif y_axis == "segments_TVD":
        y_segments = segments_TVD
        y_segments_label = "Segment TVD [meter]"
        y_interfaces = interfaces_TVD
        y_interfaces_label = "Interface TVD [meter]"

    # %% Pressure profile

    # Use figure counter for name of the saved figure
    figure_counter = 0
    # Initialize the pressure matrix
    p_matrix = np.zeros((num_segments, num_selected_ts))

    # Fill the pressure matrix
    for ts_counter in time_step_idx_range:
        p = data_frame["Pressure"][
            ts_counter * num_segments : (ts_counter + 1) * num_segments
        ]
        p_matrix[:, ts_counter] = p

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create the heatmap
    cax = ax.pcolormesh(x, y_segments, p_matrix, cmap=cmap_color, shading="auto")

    # Set the y-axis ticks
    if y_axis == "segment_index":
        ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_segments_label, fontsize=font_size)

    ax.tick_params(axis="both", labelsize=font_size)  # Set the font size of tick labels

    # Reverse the y-axis
    ax.invert_yaxis()

    # Add title
    if with_title:
        ax.set_title(
            "Pressure profile along the wellbore over time",
            fontsize=font_size,
            fontweight="bold",
        )

    # Add a colorbar to show the pressure values
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Pressure [bar]", fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

    plt.tight_layout()
    file_address = os.path.join(main_dir, f"{figure_counter}- Pressure.{save_as}")
    plt.savefig(file_address)
    if show_plot:
        plt.show()

    plt.close(fig)

    # %% Overall mole fraction profiles

    for comp_idx in range(num_components):
        # Update figure counter for name of the saved figure
        figure_counter += 1
        # Initialize the overall mole fraction matrix
        z_c_matrix = np.zeros((num_segments, num_selected_ts))

        # Fill the overall mole fraction matrix
        for ts_counter in time_step_idx_range:
            z = data_frame["Overall mole fractions"][
                ts_counter * num_segments : (ts_counter + 1) * num_segments
            ]
            z = z.tolist()
            z_c = np.zeros(num_segments)
            for segment_idx in range(num_segments):
                z_c[segment_idx] = z[segment_idx][comp_idx]
            z_c_matrix[:, ts_counter] = z_c

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create the heatmap
        cax = ax.pcolormesh(x, y_segments, z_c_matrix, cmap=cmap_color, shading="auto")

        # Set the y-axis ticks
        if y_axis == "segment_index":
            ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_segments_label, fontsize=font_size)

        ax.tick_params(
            axis="both", labelsize=font_size
        )  # Set the font size of tick labels

        # Reverse the y-axis
        ax.invert_yaxis()

        # Add title
        if with_title:
            ax.set_title(
                "Profile of overall mole fraction of "
                + components_names[comp_idx]
                + " along the wellbore over time",
                fontsize=font_size,
                fontweight="bold",
            )

        # Add a colorbar to show the overall mole fraction values
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label(
            components_names[comp_idx] + " overall mole fraction [-]",
            fontsize=font_size,
        )
        cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

        plt.tight_layout()
        file_address = os.path.join(
            main_dir,
            f"{figure_counter}- {components_names[comp_idx]} overall mole fraction.{save_as}",
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

        plt.close(fig)

    # %% Temperature profile

    # Update figure counter for name of the saved figure
    figure_counter += 1
    if coupled_model.physics.property_containers[0].thermal:
        # Initialize the temperature matrix
        T_matrix = np.zeros((num_segments, num_selected_ts))

        # Fill the temperature matrix
        for ts_counter in time_step_idx_range:
            T = (
                data_frame["Temperature"][
                    ts_counter * num_segments : (ts_counter + 1) * num_segments
                ]
                - 273.15
            )
            T_matrix[:, ts_counter] = T

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create the heatmap
        cax = ax.pcolormesh(x, y_segments, T_matrix, cmap=cmap_color, shading="auto")

        # Set the y-axis ticks
        if y_axis == "segment_index":
            ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_segments_label, fontsize=font_size)

        ax.tick_params(
            axis="both", labelsize=font_size
        )  # Set the font size of tick labels

        # Reverse the y-axis
        ax.invert_yaxis()

        # Add title
        if with_title:
            ax.set_title(
                "Temperature profile along the wellbore over time",
                fontsize=font_size,
                fontweight="bold",
            )

        # Add a colorbar to show the temperature values
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label("Temperature [\u00b0C]", fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

        plt.tight_layout()
        file_address = os.path.join(
            main_dir,
            f"{figure_counter}- Temperature.{save_as}",
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

        plt.close(fig)

    # %% Gas saturation profile

    # Update figure counter for name of the saved figure
    figure_counter += 1
    # Initialize the gas saturation matrix
    sG_matrix = np.zeros((num_segments, num_selected_ts))

    # Fill the gas saturation matrix
    for ts_counter in time_step_idx_range:
        sG = data_frame["sG"][
            ts_counter * num_segments : (ts_counter + 1) * num_segments
        ]
        sG_matrix[:, ts_counter] = sG

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create the heatmap
    cax = ax.pcolormesh(
        x, y_segments, sG_matrix, cmap=cmap_color, shading="auto", vmin=0, vmax=1
    )

    # Set the y-axis ticks
    if y_axis == "segment_index":
        ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_segments_label, fontsize=font_size)

    ax.tick_params(axis="both", labelsize=font_size)  # Set the font size of tick labels

    # Reverse the y-axis
    ax.invert_yaxis()

    # Add title
    if with_title:
        ax.set_title(
            "Gas saturation profile along the wellbore over time",
            fontsize=font_size,
            fontweight="bold",
        )

    # Add a colorbar to show the gas saturation values
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Gas saturation [-]", fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

    plt.tight_layout()
    file_address = os.path.join(
        main_dir,
        f"{figure_counter}- Gas saturation.{save_as}",
    )
    plt.savefig(file_address)
    if show_plot:
        plt.show()

    plt.close(fig)

    # %% Liquid L_a saturation profile

    if pc.nph == 3:
        # Update figure counter for name of the saved figure
        figure_counter += 1
        # Initialize the liquid L_a saturation matrix
        sL_a_matrix = np.zeros((num_segments, num_selected_ts))

        # Fill the liquid L_a saturation matrix
        for ts_counter in time_step_idx_range:
            sL_a = data_frame["sL_a"][
                ts_counter * num_segments : (ts_counter + 1) * num_segments
            ]
            sL_a_matrix[:, ts_counter] = sL_a

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create the heatmap
        cax = ax.pcolormesh(x, y_segments, sL_a_matrix, cmap=cmap_color, shading="auto")

        # Set the y-axis ticks
        if y_axis == "segment_index":
            ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_segments_label, fontsize=font_size)

        ax.tick_params(
            axis="both", labelsize=font_size
        )  # Set the font size of tick labels

        # Reverse the y-axis
        ax.invert_yaxis()

        # Add title
        if with_title:
            ax.set_title(
                "Liquid L_a saturation profile along the wellbore over time",
                fontsize=font_size,
                fontweight="bold",
            )

        # Add a colorbar to show the liquid L_a saturation values
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label("Liquid L_a saturation [-]", fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

        plt.tight_layout()
        file_address = os.path.join(
            main_dir,
            f"{figure_counter}- Liquid L_a saturation.{save_as}",
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

        plt.close(fig)

    # %% Liquid L_b saturation profile

    if pc.nph == 3:
        # Update figure counter for name of the saved figure
        figure_counter += 1
        # Initialize the liquid L_b saturation matrix
        sL_b_matrix = np.zeros((num_segments, num_selected_ts))

        # Fill the liquid L_b saturation matrix
        for ts_counter in time_step_idx_range:
            sL_b = data_frame["sL_b"][
                ts_counter * num_segments : (ts_counter + 1) * num_segments
            ]
            sL_b_matrix[:, ts_counter] = sL_b

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create the heatmap
        cax = ax.pcolormesh(x, y_segments, sL_b_matrix, cmap=cmap_color, shading="auto")

        # Set the y-axis ticks
        if y_axis == "segment_index":
            ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_segments_label, fontsize=font_size)

        ax.tick_params(
            axis="both", labelsize=font_size
        )  # Set the font size of tick labels

        # Reverse the y-axis
        ax.invert_yaxis()

        # Add title
        if with_title:
            ax.set_title(
                "Liquid L_b saturation profile along the wellbore over time",
                fontsize=font_size,
                fontweight="bold",
            )

        # Add a colorbar to show the liquid L_b saturation values
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label("Liquid L_b saturation [-]", fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

        plt.tight_layout()
        file_address = os.path.join(
            main_dir,
            f"{figure_counter}- Liquid L_b saturation.{save_as}",
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

        plt.close(fig)

    # %% Profile/profiles of components mole fractions in the gaseous phase

    for c, comp_name in enumerate(components_names):
        # Update figure counter for name of the saved figure
        figure_counter += 1
        # Initialize the xG_mole_c matrix
        xG_mole_c_matrix = np.zeros((num_segments, num_selected_ts))

        # Fill the xG_mole_c matrix
        for ts_counter in time_step_idx_range:
            xG = data_frame["xG"][
                ts_counter * num_segments : (ts_counter + 1) * num_segments
            ]
            xG_c = np.array([x[c] for x in xG])
            xG_mole_c_matrix[:, ts_counter] = xG_c

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create the heatmap
        cax = ax.pcolormesh(
            x,
            y_segments,
            xG_mole_c_matrix,
            cmap=cmap_color,
            shading="auto",
            vmin=0,
            vmax=1,
        )

        # Set the y-axis ticks
        if y_axis == "segment_index":
            ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_segments_label, fontsize=font_size)

        ax.tick_params(
            axis="both", labelsize=font_size
        )  # Set the font size of tick labels

        # Reverse the y-axis
        ax.invert_yaxis()

        # Add title
        if with_title:
            ax.set_title(
                "Profile of "
                + comp_name
                + " mole fraction in the gaseous phase along the wellbore over time",
                fontsize=font_size,
                fontweight="bold",
            )

        # Add a colorbar to show the xG_mole values
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label(
            comp_name + " mole fraction in the gaseous phase [-]", fontsize=font_size
        )
        cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

        plt.tight_layout()
        file_address = os.path.join(
            main_dir,
            f"{figure_counter}- {comp_name} mole fraction in the gaseous phase.{save_as}",
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

        plt.close(fig)

    # %% Profile/profiles of components mole fractions in the liquid phase

    if pc.nph == 2:
        for c, comp_name in enumerate(components_names):
            # Update figure counter for name of the saved figure
            figure_counter += 1
            # Initialize the xL_mole_c matrix
            xL_mole_c_matrix = np.zeros((num_segments, num_selected_ts))

            # Fill the xL_mole_c matrix
            for ts_counter in time_step_idx_range:
                xL = data_frame["xL"][
                    ts_counter * num_segments : (ts_counter + 1) * num_segments
                ]
                xL_c = np.array([x[c] for x in xL])
                xL_mole_c_matrix[:, ts_counter] = xL_c

            # Initialize the plot
            fig, ax = plt.subplots(figsize=(12, 6))

            # Create the heatmap
            cax = ax.pcolormesh(
                x,
                y_segments,
                xL_mole_c_matrix,
                cmap=cmap_color,
                shading="auto",
                vmin=0,
                vmax=1,
            )

            # Set the y-axis ticks
            if y_axis == "segment_index":
                ax.yaxis.set_major_locator(MultipleLocator(1))

            # Add axes labels
            ax.set_xlabel(x_label, fontsize=font_size)
            ax.set_ylabel(y_segments_label, fontsize=font_size)

            ax.tick_params(
                axis="both", labelsize=font_size
            )  # Set the font size of tick labels

            # Reverse the y-axis
            ax.invert_yaxis()

            # Add title
            if with_title:
                ax.set_title(
                    "Profile of "
                    + comp_name
                    + " mole fraction in the liquid phase along the wellbore over time",
                    fontsize=font_size,
                    fontweight="bold",
                )

            # Add a colorbar to show the xL_mole values
            cbar = fig.colorbar(cax, ax=ax)
            cbar.set_label(
                comp_name + " mole fraction in the liquid phase [-]", fontsize=font_size
            )
            cbar.ax.tick_params(
                labelsize=font_size
            )  # Set tick font size of the colorbar

            plt.tight_layout()
            file_address = os.path.join(
                main_dir,
                f"{figure_counter}- {comp_name} mole fraction in the liquid phase.{save_as}",
            )
            plt.savefig(file_address)
            if show_plot:
                plt.show()

            plt.close(fig)

        # %% Profile/profiles of components mole fractions in the liquid phase L_a

        if pc.nph == 3:
            for c, comp_name in enumerate(components_names):
                # Update figure counter for name of the saved figure
                figure_counter += 1
                # Initialize the xL_a_mole_c matrix
                xL_a_mole_c_matrix = np.zeros((num_segments, num_selected_ts))

                # Fill the xL_a_mole_c matrix
                for ts_counter in time_step_idx_range:
                    xL_a = data_frame["xL_a"][
                        ts_counter * num_segments : (ts_counter + 1) * num_segments
                    ]
                    xL_a_c = np.array([x[c] for x in xL_a])
                    xL_a_mole_c_matrix[:, ts_counter] = xL_a_c

                # Initialize the plot
                fig, ax = plt.subplots(figsize=(12, 6))

                # Create the heatmap
                cax = ax.pcolormesh(
                    x,
                    y_segments,
                    xL_a_mole_c_matrix,
                    cmap=cmap_color,
                    shading="auto",
                    vmin=0,
                    vmax=1,
                )

                # Set the y-axis ticks
                if y_axis == "segment_index":
                    ax.yaxis.set_major_locator(MultipleLocator(1))

                # Add axes labels
                ax.set_xlabel(x_label, fontsize=font_size)
                ax.set_ylabel(y_segments_label, fontsize=font_size)

                ax.tick_params(
                    axis="both", labelsize=font_size
                )  # Set the font size of tick labels

                # Reverse the y-axis
                ax.invert_yaxis()

                # Add title
                if with_title:
                    ax.set_title(
                        "Profile of "
                        + comp_name
                        + " mole fraction in the liquid phase L_a along the wellbore over time",
                        fontsize=font_size,
                        fontweight="bold",
                    )

                # Add a colorbar to show the xL_a_mole values
                cbar = fig.colorbar(cax, ax=ax)
                cbar.set_label(
                    comp_name + " mole fraction in the liquid phase L_a [-]",
                    fontsize=font_size,
                )
                cbar.ax.tick_params(
                    labelsize=font_size
                )  # Set tick font size of the colorbar

                plt.tight_layout()
                file_address = os.path.join(
                    main_dir,
                    f"{figure_counter}- {comp_name} mole fraction in the liquid phase L_a.{save_as}",
                )
                plt.savefig(file_address)
                if show_plot:
                    plt.show()

                plt.close(fig)

        # %% Profile/profiles of components mole fractions in the liquid phase L_b

        if pc.nph == 3:
            for c, comp_name in enumerate(components_names):
                # Update figure counter for name of the saved figure
                figure_counter += 1
                # Initialize the xL_b_mole_c matrix
                xL_b_mole_c_matrix = np.zeros((num_segments, num_selected_ts))

                # Fill the xL_b_mole_c matrix
                for ts_counter in time_step_idx_range:
                    xL_b = data_frame["xL_b"][
                        ts_counter * num_segments : (ts_counter + 1) * num_segments
                    ]
                    xL_b_c = np.array([x[c] for x in xL_b])
                    xL_b_mole_c_matrix[:, ts_counter] = xL_b_c

                # Initialize the plot
                fig, ax = plt.subplots(figsize=(12, 6))

                # Create the heatmap
                cax = ax.pcolormesh(
                    x,
                    y_segments,
                    xL_b_mole_c_matrix,
                    cmap=cmap_color,
                    shading="auto",
                    vmin=0,
                    vmax=1,
                )

                # Set the y-axis ticks
                if y_axis == "segment_index":
                    ax.yaxis.set_major_locator(MultipleLocator(1))

                # Add axes labels
                ax.set_xlabel(x_label, fontsize=font_size)
                ax.set_ylabel(y_segments_label, fontsize=font_size)

                ax.tick_params(
                    axis="both", labelsize=font_size
                )  # Set the font size of tick labels

                # Reverse the y-axis
                ax.invert_yaxis()

                # Add title
                if with_title:
                    ax.set_title(
                        "Profile of "
                        + comp_name
                        + " mole fraction in the liquid phase L_b along the wellbore over time",
                        fontsize=font_size,
                        fontweight="bold",
                    )

                # Add a colorbar to show the xL_b_mole values
                cbar = fig.colorbar(cax, ax=ax)
                cbar.set_label(
                    comp_name + " mole fraction in the liquid phase L_b [-]",
                    fontsize=font_size,
                )
                cbar.ax.tick_params(
                    labelsize=font_size
                )  # Set tick font size of the colorbar

                plt.tight_layout()
                file_address = os.path.join(
                    main_dir,
                    f"{figure_counter}- {comp_name} mole fraction in the liquid phase L_b.{save_as}",
                )
                plt.savefig(file_address)
                if show_plot:
                    plt.show()

                plt.close(fig)

    # %% Gas density profile

    # Update figure counter for name of the saved figure
    figure_counter += 1
    # Initialize the gas density matrix
    rhoG_matrix = np.zeros((num_segments, num_selected_ts))

    # Fill the gas density matrix
    for ts_counter in time_step_idx_range:
        rhoG = data_frame["rhoG"][
            ts_counter * num_segments : (ts_counter + 1) * num_segments
        ]
        rhoG_matrix[:, ts_counter] = rhoG

    # Apply a mask to hide values equal to zero
    threshold = 0  # Set your threshold here
    rhoG_matrix_masked = np.ma.masked_where(rhoG_matrix == threshold, rhoG_matrix)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create the heatmap
    cax = ax.pcolormesh(
        x, y_segments, rhoG_matrix_masked, cmap=cmap_color, shading="auto"
    )

    # Set the y-axis ticks
    if y_axis == "segment_index":
        ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_segments_label, fontsize=font_size)

    ax.tick_params(axis="both", labelsize=font_size)  # Set the font size of tick labels

    # Reverse the y-axis
    ax.invert_yaxis()

    # Add title
    if with_title:
        ax.set_title(
            "Gas density profile along the wellbore over time",
            fontsize=font_size,
            fontweight="bold",
        )

    # Add a colorbar to show the gas density values
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Gas density [kg/m$^3$]", fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

    plt.tight_layout()
    file_address = os.path.join(main_dir, f"{figure_counter}- Gas density.{save_as}")
    plt.savefig(file_address)
    if show_plot:
        plt.show()

    plt.close(fig)

    # %% Liquid density profile

    if pc.nph == 2:
        # Update figure counter for name of the saved figure
        figure_counter += 1
        # Initialize the liquid density matrix
        rhoL_matrix = np.zeros((num_segments, num_selected_ts))

        # Fill the liquid density matrix
        for ts_counter in time_step_idx_range:
            rhoL = data_frame["rhoL"][
                ts_counter * num_segments : (ts_counter + 1) * num_segments
            ]
            rhoL_matrix[:, ts_counter] = rhoL

        # Apply a mask to hide values equal to zero
        threshold = 0  # Set your threshold here
        rhoL_matrix_masked = np.ma.masked_where(rhoL_matrix == threshold, rhoL_matrix)

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create the heatmap
        cax = ax.pcolormesh(
            x, y_segments, rhoL_matrix_masked, cmap=cmap_color, shading="auto"
        )

        # Set the y-axis ticks
        if y_axis == "segment_index":
            ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_segments_label, fontsize=font_size)

        ax.tick_params(
            axis="both", labelsize=font_size
        )  # Set the font size of tick labels

        # Reverse the y-axis
        ax.invert_yaxis()

        # Add title
        if with_title:
            ax.set_title(
                "Liquid density profile along the wellbore over time",
                fontsize=font_size,
                fontweight="bold",
            )

        # Add a colorbar to show the liquid density values
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label("Liquid density [kg/m$^3$]", fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

        plt.tight_layout()
        file_address = os.path.join(
            main_dir,
            f"{figure_counter}- Liquid density.{save_as}",
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

        plt.close(fig)

    # %% Liquid L_a density profile

    if pc.nph == 3:
        # Update figure counter for name of the saved figure
        figure_counter += 1
        # Initialize the liquid L_a density matrix
        rhoL_a_matrix = np.zeros((num_segments, num_selected_ts))

        # Fill the liquid L_a density matrix
        for ts_counter in time_step_idx_range:
            rhoL_a = data_frame["rhoL_a"][
                ts_counter * num_segments : (ts_counter + 1) * num_segments
            ]
            rhoL_a_matrix[:, ts_counter] = rhoL_a

        # Apply a mask to hide values equal to zero
        threshold = 0  # Set your threshold here
        rhoL_a_matrix_masked = np.ma.masked_where(
            rhoL_a_matrix == threshold, rhoL_a_matrix
        )

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create the heatmap
        cax = ax.pcolormesh(
            x, y_segments, rhoL_a_matrix_masked, cmap=cmap_color, shading="auto"
        )

        # Set the y-axis ticks
        if y_axis == "segment_index":
            ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_segments_label, fontsize=font_size)

        ax.tick_params(
            axis="both", labelsize=font_size
        )  # Set the font size of tick labels

        # Reverse the y-axis
        ax.invert_yaxis()

        # Add title
        if with_title:
            ax.set_title(
                "Liquid L_a density profile along the wellbore over time",
                fontsize=font_size,
                fontweight="bold",
            )

        # Add a colorbar to show the liquid L_a density values
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label("Liquid L_a density [kg/m$^3$]", fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

        plt.tight_layout()
        file_address = os.path.join(
            main_dir,
            f"{figure_counter}- Liquid L_a density.{save_as}",
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

        plt.close(fig)

    # %% Liquid L_b density profile

    if pc.nph == 3:
        # Update figure counter for name of the saved figure
        figure_counter += 1
        # Initialize the liquid L_b density matrix
        rhoL_b_matrix = np.zeros((num_segments, num_selected_ts))

        # Fill the liquid L_b density matrix
        for ts_counter in time_step_idx_range:
            rhoL_b = data_frame["rhoL_b"][
                ts_counter * num_segments : (ts_counter + 1) * num_segments
            ]
            rhoL_b_matrix[:, ts_counter] = rhoL_b

        # Apply a mask to hide values equal to zero
        threshold = 0  # Set your threshold here
        rhoL_b_matrix_masked = np.ma.masked_where(
            rhoL_b_matrix == threshold, rhoL_b_matrix
        )

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create the heatmap
        cax = ax.pcolormesh(
            x, y_segments, rhoL_b_matrix_masked, cmap=cmap_color, shading="auto"
        )

        # Set the y-axis ticks
        if y_axis == "segment_index":
            ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_segments_label, fontsize=font_size)

        ax.tick_params(
            axis="both", labelsize=font_size
        )  # Set the font size of tick labels

        # Reverse the y-axis
        ax.invert_yaxis()

        # Add title
        if with_title:
            ax.set_title(
                "Liquid L_b density profile along the wellbore over time",
                fontsize=font_size,
                fontweight="bold",
            )

        # Add a colorbar to show the liquid L_b density values
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label("Liquid L_b density [kg/m$^3$]", fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

        plt.tight_layout()
        file_address = os.path.join(
            main_dir,
            f"{figure_counter}- Liquid L_b density.{save_as}",
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

        plt.close(fig)

    # %% Gas viscosity profile

    # Update figure counter for name of the saved figure
    figure_counter += 1
    # Initialize the gas viscosity matrix
    miuG_matrix = np.zeros((num_segments, num_selected_ts))

    # Fill the liquid density matrix
    for ts_counter in time_step_idx_range:
        miuG = data_frame["miuG"][
            ts_counter * num_segments : (ts_counter + 1) * num_segments
        ]
        miuG_matrix[:, ts_counter] = miuG

    # Apply a mask to hide values equal to zero
    threshold = 0  # Set your threshold here
    miuG_matrix_masked = np.ma.masked_where(miuG_matrix == threshold, miuG_matrix)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create the heatmap
    cax = ax.pcolormesh(
        x, y_segments, miuG_matrix_masked, cmap=cmap_color, shading="auto"
    )

    # Set the y-axis ticks
    if y_axis == "segment_index":
        ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_segments_label, fontsize=font_size)

    ax.tick_params(axis="both", labelsize=font_size)  # Set the font size of tick labels

    # Reverse the y-axis
    ax.invert_yaxis()

    # Add title
    if with_title:
        ax.set_title(
            "Gas viscosity profile along the wellbore over time",
            fontsize=font_size,
            fontweight="bold",
        )

    # Add a colorbar to show the gas viscosity values
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Gas viscosity [cP]", fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

    plt.tight_layout()
    file_address = os.path.join(main_dir, f"{figure_counter}- Gas viscosity.{save_as}")
    plt.savefig(file_address)
    if show_plot:
        plt.show()

    plt.close(fig)

    # %% Liquid viscosity profile

    if pc.nph == 2:
        # Update figure counter for name of the saved figure
        figure_counter += 1
        # Initialize the liquid viscosity matrix
        miuL_matrix = np.zeros((num_segments, num_selected_ts))

        # Fill the liquid density matrix
        for ts_counter in time_step_idx_range:
            miuL = data_frame["miuL"][
                ts_counter * num_segments : (ts_counter + 1) * num_segments
            ]
            miuL_matrix[:, ts_counter] = miuL

        # Apply a mask to hide values equal to zero
        threshold = 0  # Set your threshold here
        miuL_matrix_masked = np.ma.masked_where(miuL_matrix == threshold, miuL_matrix)

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create the heatmap
        cax = ax.pcolormesh(
            x, y_segments, miuL_matrix_masked, cmap=cmap_color, shading="auto"
        )

        # Set the y-axis ticks
        if y_axis == "segment_index":
            ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_segments_label, fontsize=font_size)

        ax.tick_params(
            axis="both", labelsize=font_size
        )  # Set the font size of tick labels

        # Reverse the y-axis
        ax.invert_yaxis()

        # Add title
        if with_title:
            ax.set_title(
                "Liquid viscosity profile along the wellbore over time",
                fontsize=font_size,
                fontweight="bold",
            )

        # Add a colorbar to show the liquid viscosity values
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label("Liquid viscosity [$cP$]", fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

        plt.tight_layout()
        file_address = os.path.join(
            main_dir,
            f"{figure_counter}- Liquid viscosity.{save_as}",
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

        plt.close(fig)

    # %% Liquid L_a viscosity profile

    if pc.nph == 3:
        # Update figure counter for name of the saved figure
        figure_counter += 1
        # Initialize the liquid L_a viscosity matrix
        miuL_a_matrix = np.zeros((num_segments, num_selected_ts))

        # Fill the liquid L_a density matrix
        for ts_counter in time_step_idx_range:
            miuL_a = data_frame["miuL_a"][
                ts_counter * num_segments : (ts_counter + 1) * num_segments
            ]
            miuL_a_matrix[:, ts_counter] = miuL_a

        # Apply a mask to hide values equal to zero
        threshold = 0  # Set your threshold here
        miuL_a_matrix_masked = np.ma.masked_where(
            miuL_a_matrix == threshold, miuL_a_matrix
        )

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create the heatmap
        cax = ax.pcolormesh(
            x, y_segments, miuL_a_matrix_masked, cmap=cmap_color, shading="auto"
        )

        # Set the y-axis ticks
        if y_axis == "segment_index":
            ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_segments_label, fontsize=font_size)

        ax.tick_params(
            axis="both", labelsize=font_size
        )  # Set the font size of tick labels

        # Reverse the y-axis
        ax.invert_yaxis()

        # Add title
        if with_title:
            ax.set_title(
                "Liquid L_a viscosity profile along the wellbore over time",
                fontsize=font_size,
                fontweight="bold",
            )

        # Add a colorbar to show the liquid L_a viscosity values
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label("Liquid L_a viscosity [$cP$]", fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

        plt.tight_layout()
        file_address = os.path.join(
            main_dir,
            f"{figure_counter}- Liquid L_a viscosity.{save_as}",
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

        plt.close(fig)

    # %% Liquid L_b viscosity profile

    if pc.nph == 3:
        # Update figure counter for name of the saved figure
        figure_counter += 1
        # Initialize the liquid L_b viscosity matrix
        miuL_b_matrix = np.zeros((num_segments, num_selected_ts))

        # Fill the liquid L_b density matrix
        for ts_counter in time_step_idx_range:
            miuL_b = data_frame["miuL_b"][
                ts_counter * num_segments : (ts_counter + 1) * num_segments
            ]
            miuL_b_matrix[:, ts_counter] = miuL_b

        # Apply a mask to hide values equal to zero
        threshold = 0  # Set your threshold here
        miuL_b_matrix_masked = np.ma.masked_where(
            miuL_b_matrix == threshold, miuL_b_matrix
        )

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create the heatmap
        cax = ax.pcolormesh(
            x, y_segments, miuL_b_matrix_masked, cmap=cmap_color, shading="auto"
        )

        # Set the y-axis ticks
        if y_axis == "segment_index":
            ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_segments_label, fontsize=font_size)

        ax.tick_params(
            axis="both", labelsize=font_size
        )  # Set the font size of tick labels

        # Reverse the y-axis
        ax.invert_yaxis()

        # Add title
        if with_title:
            ax.set_title(
                "Liquid L_b viscosity profile along the wellbore over time",
                fontsize=font_size,
                fontweight="bold",
            )

        # Add a colorbar to show the liquid L_b viscosity values
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label("Liquid L_b viscosity [$cP$]", fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

        plt.tight_layout()
        file_address = os.path.join(
            main_dir,
            f"{figure_counter}- Liquid L_b viscosity.{save_as}",
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

        plt.close(fig)

    # %% Gas velocity profile

    # Update figure counter for name of the saved figure
    figure_counter += 1
    # Initialize the gas velocity matrix
    vG_matrix = np.zeros((num_interfaces, num_selected_ts))

    # Fill the gas velocity matrix
    for ts_counter in time_step_idx_range:
        vG = data_frame["vG"][
            ts_counter * num_segments : (ts_counter + 1) * num_segments
        ]
        vG_matrix[:, ts_counter] = vG[:-1] / (24 * 60 * 60)  # convert m/day to m/s

    # Apply a mask to hide values equal to zero
    threshold = 0  # Set your threshold here
    vG_matrix_masked = np.ma.masked_where(vG_matrix == threshold, vG_matrix)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create the heatmap
    cax = ax.pcolormesh(
        x, y_interfaces, vG_matrix_masked, cmap=cmap_color, shading="auto"
    )

    # Set the y-axis ticks
    if y_axis == "segment_index":
        ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_interfaces_label, fontsize=font_size)

    ax.tick_params(axis="both", labelsize=font_size)  # Set the font size of tick labels

    # Reverse the y-axis
    ax.invert_yaxis()

    # Add title
    if with_title:
        ax.set_title(
            "Gas velocity profile along the wellbore over time",
            fontsize=font_size,
            fontweight="bold",
        )

    # Add a colorbar to show the gas velocity values
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Gas velocity [m/s]", fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

    plt.tight_layout()
    file_address = os.path.join(main_dir, f"{figure_counter}- Gas velocity.{save_as}")
    plt.savefig(file_address)
    if show_plot:
        plt.show()

    plt.close(fig)

    # %% Liquid velocity profile

    # Update figure counter for name of the saved figure
    figure_counter += 1
    # Initialize the liquid velocity matrix
    vL_matrix = np.zeros((num_interfaces, num_selected_ts))

    # Fill the liquid velocity matrix
    for ts_counter in time_step_idx_range:
        vL = data_frame["vL"][
            ts_counter * num_segments : (ts_counter + 1) * num_segments
        ]
        vL_matrix[:, ts_counter] = vL[:-1] / (24 * 60 * 60)  # convert m/day to m/s

    # Apply a mask to hide values equal to zero
    threshold = 0  # Set your threshold here
    vL_matrix_masked = np.ma.masked_where(vL_matrix == threshold, vL_matrix)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create the heatmap
    cax = ax.pcolormesh(
        x, y_interfaces, vL_matrix_masked, cmap=cmap_color, shading="auto"
    )

    # Set the y-axis ticks
    if y_axis == "segment_index":
        ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_interfaces_label, fontsize=font_size)

    ax.tick_params(axis="both", labelsize=font_size)  # Set the font size of tick labels

    # Reverse the y-axis
    ax.invert_yaxis()

    # Add title
    if with_title:
        ax.set_title(
            "Liquid velocity profile along the wellbore over time",
            fontsize=font_size,
            fontweight="bold",
        )

    # Add a colorbar to show the liquid velocity values
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Liquid velocity [m/s]", fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

    plt.tight_layout()
    file_address = os.path.join(
        main_dir,
        f"{figure_counter}- Liquid velocity.{save_as}",
    )
    plt.savefig(file_address)
    if show_plot:
        plt.show()

    plt.close(fig)
