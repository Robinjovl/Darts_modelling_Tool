import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MultipleLocator

from darts.models.darts_model import DartsModel


def plot_heat_map_contourf(
    primary_vars_and_phase_props_file_address: str,
    h5_well_data: dict,
    coupled_model: DartsModel,
    min_ts_idx: int = 0,
    max_ts_idx: int = None,
    x_axis: str = "simulation_time",
    y_axis: str = "segments_MD",
    cmap_color: str = "jet",
    save_as: str = 'pdf',
    show_plot: bool = True,
    y_axis_tick_interval=50.0,
    n_cmap_bins_p: int = 10,
    n_cmap_bins_comp: int = 10,
    n_cmap_bins_t: int = 10,
    n_cmap_bins_s: int = 10,
    n_cmap_bins_rho: int = 10,
    n_cmap_bins_miu: int = 10,
    n_cmap_bins_v: int = 10,
    font_size: float = 14,
    with_title: bool = True,
    with_logarithmic_x_axis: bool = False,
):
    """
    :param primary_vars_and_phase_props_file_address: Address of the pickle file in which primary variables and phase
    properties of well segments are stored
    :type primary_vars_and_phase_props_file_address: str
    :param h5_well_data: HDF5 file containing well solution. It's used here to get the time step sizes
    :type h5_well_data: dict
    :param coupled_model: An instance of DartsModel
    :type coupled_model: DartsModel
    :param min_ts_idx: If specified, the heat map will be shown from the specified minimum time step index. If not
    specified, the heat map will be shown from the zeroth time step.
    :type min_ts_idx: int
    :param max_ts_idx: If specified, the heat map will be shown until the specified maximum time step index. If not
    specified, the heat map will be shown till the last time step.
    :type max_ts_idx: int
    :param x_axis: "simulation_time" or "time_step_index"
    :type x_axis: str
    :param y_axis: "segments_MD" or "segments_TVD" or "segment_index"
    :param save_as: The extension of the image files that will be saved
    :type save_as: str
    :param show_plot: Whether or not to show the plot
    :type show_plot: bool
    :param y_axis_tick_interval: The interval of the ticks of the y axis.
    :type y_axis_tick_interval: float
    :param n_cmap_bins_p: Number of bins of the colorbar and colormap of pressure
    :type n_cmap_bins_p: int
    :param n_cmap_bins_comp: Number of bins of the colorbar and colormap of composition
    :type n_cmap_bins_comp: int
    :param n_cmap_bins_t: Number of bins of the colorbar and colormap of temperature
    :type n_cmap_bins_t: int
    :param n_cmap_bins_s: Number of bins of the colorbar and colormap of saturation
    :type n_cmap_bins_s: int
    :param n_cmap_bins_rho: Number of bins of the colorbar and colormap of density
    :type n_cmap_bins_rho: int
    :param n_cmap_bins_miu: Number of bins of the colorbar and colormap of viscosity
    :type n_cmap_bins_miu: int
    :param n_cmap_bins_v: Number of bins of the colorbar and colormap of velocity
    :type n_cmap_bins_v: int
    :param font_size: Size of the fonts
    :type font_size: float
    :param with_title: If you want the figure to have a title or not
    :type with_title: bool
    :param with_logarithmic_x_axis: Whether or not to have the logarithmic x-axis
    :type with_logarithmic_x_axis: bool
    """
    main_dir = os.path.join(coupled_model.output_folder, 'heat_maps_contourf')

    # Reset_directory
    if os.path.exists(main_dir):
        shutil.rmtree(main_dir)
    os.makedirs(main_dir)

    # This line gets the geometry object of the first well (by insertion order) from the wells_geometry dictionary
    # and assigns it to well_geom.
    well_geom = next(iter(coupled_model.wells.values())).geometry

    segments_MD = well_geom.z
    interfaces_MD = well_geom.z_interfaces
    if y_axis == "segments_TVD":
        segments_TVD = well_geom.TVD_segments
        interfaces_TVD = well_geom.TVD_interfaces

    # Get components names
    pc = coupled_model.physics.property_containers[0]
    components_names = pc.components_name
    num_components = len(components_names)
    num_segments = well_geom.num_segments
    num_interfaces = num_segments - 1

    # Load primary vars and phase props
    data_frame = pd.read_pickle(primary_vars_and_phase_props_file_address)
    num_ts = int(
        len(data_frame["sG"]) / num_segments
    )  # Initial conditions of sG is not stored.
    if max_ts_idx is None:
        max_ts_idx = num_ts
    assert max_ts_idx <= num_ts, (
        f"max_ts_idx is larger than the total number of time steps, which is {num_ts}!"
    )
    assert min_ts_idx < num_ts, (
        f"min_ts_idx is equal to or larger than the total number of time steps, which is {num_ts}!"
    )
    assert min_ts_idx < max_ts_idx, (
        f"min_ts_idx is equal to or larger than max_ts_idx, which is {max_ts_idx}!"
    )

    if x_axis == "simulation_time":
        simulation_time = (
            h5_well_data["dynamic"]["time"] * 24 * 60 * 60
        )  # convert days to seconds
        # Apply the user-specified time-step index range
        simulation_time = simulation_time[min_ts_idx:max_ts_idx]

    time_step_idx_range = range(min_ts_idx, max_ts_idx)
    num_selected_ts = len(time_step_idx_range)

    if x_axis == "time_step_index":
        x = time_step_idx_range
        x_label = 'Time step [-]'
    elif x_axis == "simulation_time":
        x = simulation_time
        x_label = 'Simulation time [second]'

    if y_axis == "segment_index":
        y_segments = range(num_segments)
        y_segments_label = 'Segment index [-]'
        y_interfaces = range(num_interfaces)
        y_interfaces_label = 'Interface index [-]'
    elif y_axis == "segments_MD":
        y_segments = segments_MD
        y_segments_label = "Segment MD [meter]"
        y_interfaces = interfaces_MD
        y_interfaces_label = 'Interface MD [meter]'
    elif y_axis == "segments_TVD":
        y_segments = segments_TVD
        y_segments_label = "Segment TVD [meter]"
        y_interfaces = interfaces_TVD
        y_interfaces_label = 'Interface TVD [meter]'

    # %% Pressure profile

    # Use figure counter for name of the saved figure
    figure_counter = 0
    # Initialize the pressure matrix
    p_matrix = np.zeros((num_segments, num_selected_ts))

    # Fill the pressure matrix
    for ts_idx, ts_counter in enumerate(time_step_idx_range):
        p = data_frame["Pressure"][
            ts_counter * num_segments : (ts_counter + 1) * num_segments
        ]
        p_matrix[:, ts_idx] = p

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create a discrete colorbar and colormap
    pmin, pmax = np.min(p_matrix), np.max(p_matrix)
    levels = np.linspace(pmin, pmax, n_cmap_bins_p + 1)
    cmap = plt.get_cmap(cmap_color, n_cmap_bins_p)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # Region‐based fill
    cf = ax.contourf(
        x,
        y_segments,
        p_matrix,
        levels=levels,
        cmap=cmap,
        norm=norm,
        # extend='both'  # if you want arrows at the ends
    )

    # Overlay the exact same contour lines
    _cs = ax.contour(x, y_segments, p_matrix, levels=levels, colors='k', linewidths=0.7)
    # ax.clabel(_cs, fmt='%.0f')  # if you want labels on the lines

    # Create the colorbar
    cbar = fig.colorbar(
        cf,
        ax=ax,
        boundaries=levels,
        ticks=levels,
        spacing='proportional',
    )
    cbar.set_label('Pressure [bar]', fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(y_axis_tick_interval))

    # Reverse the y-axis
    ax.invert_yaxis()

    # Add axes labels
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_segments_label, fontsize=font_size)

    # Make the x-axis logarithmic
    if with_logarithmic_x_axis:
        ax.set_xscale('log')

    ax.tick_params(axis='both', labelsize=font_size)  # Set the font size of tick labels

    # Add title
    if with_title:
        ax.set_title(
            'Pressure profile along the wellbore over time',
            fontsize=font_size,
            fontweight='bold',
        )

    plt.tight_layout()
    file_address = os.path.join(main_dir, f"{figure_counter}- Pressure." + save_as)
    plt.savefig(file_address)
    if show_plot:
        plt.show()

    # %% Overall mole fraction profiles

    for comp_idx in range(num_components):
        # Update figure counter for name of the saved figure
        figure_counter += 1
        # Initialize the overall mole fraction matrix
        z_c_matrix = np.zeros((num_segments, num_selected_ts))

        # Fill the overall mole fraction matrix
        for ts_idx, ts_counter in enumerate(time_step_idx_range):
            z = data_frame["Overall mole fractions"][
                ts_counter * num_segments : (ts_counter + 1) * num_segments
            ]
            z = z.tolist()
            z_c = np.zeros(num_segments)
            for segment_idx in range(num_segments):
                z_c[segment_idx] = z[segment_idx][comp_idx]
            z_c_matrix[:, ts_idx] = z_c

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create a discrete colorbar and colormap
        z_c_min, z_c_max = (
            np.nanmin(z_c_matrix),
            np.nanmax(z_c_matrix),
        )  # Using np.nanmin or np.nanmax because if nan exists in the matrix, np.min and np.max return nan as min and max, which we don't want.
        if z_c_min == z_c_max:
            z_c_min = 0.0
            z_c_max = 1.0
        levels = np.linspace(z_c_min, z_c_max, n_cmap_bins_comp + 1)
        cmap = plt.get_cmap(cmap_color, n_cmap_bins_comp)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        # Region‐based fill
        cf = ax.contourf(
            x,
            y_segments,
            z_c_matrix,
            levels=levels,
            cmap=cmap,
            norm=norm,
            # extend='both'  # if you want arrows at the ends
        )

        # Overlay the exact same contour lines
        _cs = ax.contour(
            x, y_segments, z_c_matrix, levels=levels, colors='k', linewidths=0.7
        )
        # ax.clabel(_cs, fmt='%.0f')  # if you want labels on the lines

        # Create the colorbar
        cbar = fig.colorbar(
            cf,
            ax=ax,
            boundaries=levels,
            ticks=levels,
            spacing='proportional',
        )
        cbar.set_label(
            components_names[comp_idx] + ' overall mole fraction [-]',
            fontsize=font_size,
        )
        cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(y_axis_tick_interval))

        # Reverse the y-axis
        ax.invert_yaxis()

        # Add axes labels
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_segments_label, fontsize=font_size)

        # Make the x-axis logarithmic
        if with_logarithmic_x_axis:
            ax.set_xscale('log')

        ax.tick_params(
            axis='both', labelsize=font_size
        )  # Set the font size of tick labels

        # Add title
        if with_title:
            ax.set_title(
                'Profile of overall mole fraction of '
                + components_names[comp_idx]
                + ' along the wellbore over time',
                fontsize=font_size,
                fontweight='bold',
            )

        plt.tight_layout()
        file_address = os.path.join(
            main_dir,
            f"{figure_counter}- {components_names[comp_idx]} overall mole fraction."
            + save_as,
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

    # %% Temperature profile

    # Update figure counter for name of the saved figure
    figure_counter += 1
    if coupled_model.physics.property_containers[0].thermal:
        # Initialize the temperature matrix
        T_matrix = np.zeros((num_segments, num_selected_ts))

        # Fill the temperature matrix
        for ts_idx, ts_counter in enumerate(time_step_idx_range):
            T = (
                data_frame["Temperature"][
                    ts_counter * num_segments : (ts_counter + 1) * num_segments
                ]
                - 273.15
            )
            T_matrix[:, ts_idx] = T

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create a discrete colorbar and colormap
        t_min, t_max = np.min(T_matrix), np.max(T_matrix)
        levels = np.linspace(t_min, t_max, n_cmap_bins_t + 1)
        cmap = plt.get_cmap(cmap_color, n_cmap_bins_t)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        # Region‐based fill
        cf = ax.contourf(
            x,
            y_segments,
            T_matrix,
            levels=levels,
            cmap=cmap,
            norm=norm,
            # extend='both'  # if you want arrows at the ends
        )

        # Overlay the exact same contour lines
        _cs = ax.contour(
            x, y_segments, T_matrix, levels=levels, colors='k', linewidths=0.7
        )
        # ax.clabel(_cs, fmt='%.0f')  # if you want labels on the lines

        # Create the colorbar
        cbar = fig.colorbar(
            cf,
            ax=ax,
            boundaries=levels,
            ticks=levels,
            spacing='proportional',
        )
        cbar.set_label('Temperature [\u00b0C]', fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(y_axis_tick_interval))

        # Reverse the y-axis
        ax.invert_yaxis()

        # Add axes labels
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_segments_label, fontsize=font_size)

        # Make the x-axis logarithmic
        if with_logarithmic_x_axis:
            ax.set_xscale('log')

        ax.tick_params(
            axis='both', labelsize=font_size
        )  # Set the font size of tick labels

        # Add title
        if with_title:
            ax.set_title(
                'Temperature profile along the wellbore over time',
                fontsize=font_size,
                fontweight='bold',
            )

        plt.tight_layout()
        file_address = os.path.join(
            main_dir, f"{figure_counter}- Temperature." + save_as
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

    # %% Gas saturation profile

    # Update figure counter for name of the saved figure
    figure_counter += 1
    # Initialize the gas saturation matrix
    sG_matrix = np.zeros((num_segments, num_selected_ts))

    # Fill the gas saturation matrix
    for ts_idx, ts_counter in enumerate(time_step_idx_range):
        sG = data_frame["sG"][
            ts_counter * num_segments : (ts_counter + 1) * num_segments
        ]
        sG_matrix[:, ts_idx] = sG

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create a discrete colorbar and colormap
    # sg_min, sg_max = 0, 1
    sg_min, sg_max = np.min(sG_matrix), np.max(sG_matrix)
    levels = np.linspace(sg_min, sg_max, n_cmap_bins_s + 1)
    cmap = plt.get_cmap(cmap_color, n_cmap_bins_s)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # Region‐based fill
    cf = ax.contourf(
        x,
        y_segments,
        sG_matrix,
        levels=levels,
        cmap=cmap,
        norm=norm,
        # extend='both'  # if you want arrows at the ends
    )

    # Overlay the exact same contour lines
    _cs = ax.contour(
        x, y_segments, sG_matrix, levels=levels, colors='k', linewidths=0.7
    )
    # ax.clabel(_cs, fmt='%.0f')  # if you want labels on the lines

    # Create the colorbar
    cbar = fig.colorbar(
        cf,
        ax=ax,
        boundaries=levels,
        ticks=levels,
        spacing='proportional',
    )
    cbar.set_label('Gas saturation [-]', fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(y_axis_tick_interval))

    # Reverse the y-axis
    ax.invert_yaxis()

    # Add axes labels
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_segments_label, fontsize=font_size)

    # Make the x-axis logarithmic
    if with_logarithmic_x_axis:
        ax.set_xscale('log')

    ax.tick_params(axis='both', labelsize=font_size)  # Set the font size of tick labels

    # Add title
    if with_title:
        ax.set_title(
            'Gas saturation profile along the wellbore over time',
            fontsize=font_size,
            fontweight='bold',
        )

    plt.tight_layout()
    file_address = os.path.join(
        main_dir, f"{figure_counter}- Gas saturation." + save_as
    )
    plt.savefig(file_address)
    if show_plot:
        plt.show()

    # %% Liquid L_a saturation profile

    if pc.nph == 3:
        # Update figure counter for name of the saved figure
        figure_counter += 1
        # Initialize the liquid L_a saturation matrix
        sL_a_matrix = np.zeros((num_segments, num_selected_ts))

        # Fill the liquid L_a saturation matrix
        for ts_idx, ts_counter in enumerate(time_step_idx_range):
            sL_a = data_frame["sL_a"][
                ts_counter * num_segments : (ts_counter + 1) * num_segments
            ]
            sL_a_matrix[:, ts_idx] = sL_a

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create a discrete colorbar and colormap
        # sla_min, sla_max = 0, 1
        sla_min, sla_max = np.min(sL_a_matrix), np.max(sL_a_matrix)
        levels = np.linspace(sla_min, sla_max, n_cmap_bins_s + 1)
        cmap = plt.get_cmap(cmap_color, n_cmap_bins_s)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        # Region‐based fill
        cf = ax.contourf(
            x,
            y_segments,
            sL_a_matrix,
            levels=levels,
            cmap=cmap,
            norm=norm,
            # extend='both'  # if you want arrows at the ends
        )

        # Overlay the exact same contour lines
        _cs = ax.contour(
            x, y_segments, sL_a_matrix, levels=levels, colors='k', linewidths=0.7
        )
        # ax.clabel(_cs, fmt='%.0f')  # if you want labels on the lines

        # Create the colorbar
        cbar = fig.colorbar(
            cf,
            ax=ax,
            boundaries=levels,
            ticks=levels,
            spacing='proportional',
        )
        cbar.set_label('Liquid L_a saturation [-]', fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(y_axis_tick_interval))

        # Reverse the y-axis
        ax.invert_yaxis()

        # Add axes labels
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_segments_label, fontsize=font_size)

        # Make the x-axis logarithmic
        if with_logarithmic_x_axis:
            ax.set_xscale('log')

        ax.tick_params(
            axis='both', labelsize=font_size
        )  # Set the font size of tick labels

        # Add title
        if with_title:
            ax.set_title(
                'Liquid L_a saturation profile along the wellbore over time',
                fontsize=font_size,
                fontweight='bold',
            )

        plt.tight_layout()
        file_address = os.path.join(
            coupled_model.output_folder,
            f"{figure_counter}- Liquid L_a saturation." + save_as,
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

    # %% Liquid L_b saturation profile

    if pc.nph == 3:
        # Update figure counter for name of the saved figure
        figure_counter += 1
        # Initialize the liquid L_b saturation matrix
        sL_b_matrix = np.zeros((num_segments, num_selected_ts))

        # Fill the liquid L_b saturation matrix
        for ts_idx, ts_counter in enumerate(time_step_idx_range):
            sL_b = data_frame["sL_b"][
                ts_counter * num_segments : (ts_counter + 1) * num_segments
            ]
            sL_b_matrix[:, ts_idx] = sL_b

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create a discrete colorbar and colormap
        # slb_min, slb_max = 0, 1
        slb_min, slb_max = np.min(sL_b_matrix), np.max(sL_b_matrix)
        levels = np.linspace(slb_min, slb_max, n_cmap_bins_s + 1)
        cmap = plt.get_cmap(cmap_color, n_cmap_bins_s)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        # Region‐based fill
        cf = ax.contourf(
            x,
            y_segments,
            sL_b_matrix,
            levels=levels,
            cmap=cmap,
            norm=norm,
            # extend='both'  # if you want arrows at the ends
        )

        # Overlay the exact same contour lines
        _cs = ax.contour(
            x, y_segments, sL_b_matrix, levels=levels, colors='k', linewidths=0.7
        )
        # ax.clabel(_cs, fmt='%.0f')  # if you want labels on the lines

        # Create the colorbar
        cbar = fig.colorbar(
            cf,
            ax=ax,
            boundaries=levels,
            ticks=levels,
            spacing='proportional',
        )
        cbar.set_label('Liquid L_b saturation [-]', fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(y_axis_tick_interval))

        # Reverse the y-axis
        ax.invert_yaxis()

        # Add axes labels
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_segments_label, fontsize=font_size)

        # Make the x-axis logarithmic
        if with_logarithmic_x_axis:
            ax.set_xscale('log')

        ax.tick_params(
            axis='both', labelsize=font_size
        )  # Set the font size of tick labels

        # Add title
        if with_title:
            ax.set_title(
                'Liquid L_b saturation profile along the wellbore over time',
                fontsize=font_size,
                fontweight='bold',
            )

        plt.tight_layout()
        file_address = os.path.join(
            main_dir, f"{figure_counter}- Liquid L_b saturation." + save_as
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

    # %% Profile/profiles of components mole fractions in the gaseous phase

    for c, comp_name in enumerate(components_names):
        # Update figure counter for name of the saved figure
        figure_counter += 1
        # Initialize the xG_mole_c matrix
        xG_mole_c_matrix = np.zeros((num_segments, num_selected_ts))

        # Fill the xG_mole_c matrix
        for ts_idx, ts_counter in enumerate(time_step_idx_range):
            xG = data_frame["xG"][
                ts_counter * num_segments : (ts_counter + 1) * num_segments
            ]
            xG_c = np.array([x[c] for x in xG])
            xG_mole_c_matrix[:, ts_idx] = xG_c

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create a discrete colorbar and colormap
        x_min, x_max = (
            np.nanmin(xG_mole_c_matrix),
            np.nanmax(xG_mole_c_matrix),
        )  # Using np.nanmin or np.nanmax because if nan exists in the matrix, np.min and np.max return nan as min and max, which we don't want.
        if x_min == x_max:
            x_min = 0.0
            x_max = 1.0
        levels = np.linspace(x_min, x_max, n_cmap_bins_comp + 1)
        cmap = plt.get_cmap(cmap_color, n_cmap_bins_comp)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        # Region‐based fill
        cf = ax.contourf(
            x,
            y_segments,
            xG_mole_c_matrix,
            levels=levels,
            cmap=cmap,
            norm=norm,
            # extend='both'  # if you want arrows at the ends
        )

        # Overlay the exact same contour lines
        _cs = ax.contour(
            x, y_segments, xG_mole_c_matrix, levels=levels, colors='k', linewidths=0.7
        )
        # ax.clabel(_cs, fmt='%.0f')  # if you want labels on the lines

        # Create the colorbar
        cbar = fig.colorbar(
            cf,
            ax=ax,
            boundaries=levels,
            ticks=levels,
            spacing='proportional',
        )
        cbar.set_label(
            comp_name + ' mole fraction in the gaseous phase [-]', fontsize=font_size
        )
        cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(y_axis_tick_interval))

        # Reverse the y-axis
        ax.invert_yaxis()

        # Add axes labels
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_segments_label, fontsize=font_size)

        # Make the x-axis logarithmic
        if with_logarithmic_x_axis:
            ax.set_xscale('log')

        ax.tick_params(
            axis='both', labelsize=font_size
        )  # Set the font size of tick labels

        # Add title
        if with_title:
            ax.set_title(
                'Profile of '
                + comp_name
                + ' mole fraction in the gaseous phase along the wellbore over time',
                fontsize=font_size,
                fontweight='bold',
            )

        plt.tight_layout()
        file_address = os.path.join(
            main_dir,
            f"{figure_counter}- {comp_name} mole fraction in the gaseous phase."
            + save_as,
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

    # %% Profile/profiles of components mole fractions in the liquid phase

    if pc.nph == 2:
        for c, comp_name in enumerate(components_names):
            # Update figure counter for name of the saved figure
            figure_counter += 1
            # Initialize the xL_mole_c matrix
            xL_mole_c_matrix = np.zeros((num_segments, num_selected_ts))

            # Fill the xL_mole_c matrix
            for ts_idx, ts_counter in enumerate(time_step_idx_range):
                xL = data_frame["xL"][
                    ts_counter * num_segments : (ts_counter + 1) * num_segments
                ]
                xL_c = np.array([x[c] for x in xL])
                xL_mole_c_matrix[:, ts_idx] = xL_c

            # Initialize the plot
            fig, ax = plt.subplots(figsize=(12, 6))

            # Create a discrete colorbar and colormap
            x_min, x_max = (
                np.nanmin(xL_mole_c_matrix),
                np.nanmax(xL_mole_c_matrix),
            )  # Using np.nanmin because if nan exists in the matrix, np.min and np.max return nan as min and max
            if x_min == x_max:
                x_min = 0.0
                x_max = 1.0
            levels = np.linspace(x_min, x_max, n_cmap_bins_comp + 1)
            cmap = plt.get_cmap(cmap_color, n_cmap_bins_comp)
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

            # Region‐based fill
            cf = ax.contourf(
                x,
                y_segments,
                xL_mole_c_matrix,
                levels=levels,
                cmap=cmap,
                norm=norm,
                # extend='both'  # if you want arrows at the ends
            )

            # Overlay the exact same contour lines
            _cs = ax.contour(
                x,
                y_segments,
                xL_mole_c_matrix,
                levels=levels,
                colors='k',
                linewidths=0.7,
            )
            # ax.clabel(_cs, fmt='%.0f')  # if you want labels on the lines

            # Create the colorbar
            cbar = fig.colorbar(
                cf,
                ax=ax,
                boundaries=levels,
                ticks=levels,
                spacing='proportional',
            )
            cbar.set_label(
                comp_name + ' mole fraction in the liquid phase [-]', fontsize=font_size
            )
            cbar.ax.tick_params(
                labelsize=font_size
            )  # Set tick font size of the colorbar

            # Set the y-axis ticks
            ax.yaxis.set_major_locator(MultipleLocator(y_axis_tick_interval))

            # Reverse the y-axis
            ax.invert_yaxis()

            # Add axes labels
            ax.set_xlabel(x_label, fontsize=font_size)
            ax.set_ylabel(y_segments_label, fontsize=font_size)

            # Make the x-axis logarithmic
            if with_logarithmic_x_axis:
                ax.set_xscale('log')

            ax.tick_params(
                axis='both', labelsize=font_size
            )  # Set the font size of tick labels

            # Add title
            if with_title:
                ax.set_title(
                    'Profile of '
                    + comp_name
                    + ' mole fraction in the liquid phase along the wellbore over time',
                    fontsize=font_size,
                    fontweight='bold',
                )

            plt.tight_layout()
            file_address = os.path.join(
                main_dir,
                f"{figure_counter}- {comp_name} mole fraction in the liquid phase."
                + save_as,
            )
            plt.savefig(file_address)
            if show_plot:
                plt.show()

        # %% Profile/profiles of components mole fractions in the liquid phase L_a

        if pc.nph == 3:
            for c, comp_name in enumerate(components_names):
                # Update figure counter for name of the saved figure
                figure_counter += 1
                # Initialize the xL_a_mole_c matrix
                xL_a_mole_c_matrix = np.zeros((num_segments, num_selected_ts))

                # Fill the xL_a_mole_c matrix
                for ts_idx, ts_counter in enumerate(time_step_idx_range):
                    xL_a = data_frame["xL_a"][
                        ts_counter * num_segments : (ts_counter + 1) * num_segments
                    ]
                    xL_a_c = np.array([x[c] for x in xL_a])
                    xL_a_mole_c_matrix[:, ts_idx] = xL_a_c

                # Initialize the plot
                fig, ax = plt.subplots(figsize=(12, 6))

                # Create a discrete colorbar and colormap
                x_min, x_max = (
                    np.nanmin(xL_a_mole_c_matrix),
                    np.nanmax(xL_a_mole_c_matrix),
                )  # Using np.nanmin because if nan exists in the matrix, np.min and np.max return nan as min and max
                if x_min == x_max:
                    x_min = 0.0
                    x_max = 1.0
                levels = np.linspace(x_min, x_max, n_cmap_bins_comp + 1)
                cmap = plt.get_cmap(cmap_color, n_cmap_bins_comp)
                norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

                # Region‐based fill
                cf = ax.contourf(
                    x,
                    y_segments,
                    xL_a_mole_c_matrix,
                    levels=levels,
                    cmap=cmap,
                    norm=norm,
                    # extend='both'  # if you want arrows at the ends
                )

                # Overlay the exact same contour lines
                _cs = ax.contour(
                    x,
                    y_segments,
                    xL_a_mole_c_matrix,
                    levels=levels,
                    colors='k',
                    linewidths=0.7,
                )
                # ax.clabel(_cs, fmt='%.0f')  # if you want labels on the lines

                # Create the colorbar
                cbar = fig.colorbar(
                    cf,
                    ax=ax,
                    boundaries=levels,
                    ticks=levels,
                    spacing='proportional',
                )
                cbar.set_label(
                    comp_name + ' mole fraction in the liquid phase L_a [-]',
                    fontsize=font_size,
                )
                cbar.ax.tick_params(
                    labelsize=font_size
                )  # Set tick font size of the colorbar

                # Set the y-axis ticks
                ax.yaxis.set_major_locator(MultipleLocator(y_axis_tick_interval))

                # Reverse the y-axis
                ax.invert_yaxis()

                # Add axes labels
                ax.set_xlabel(x_label, fontsize=font_size)
                ax.set_ylabel(y_segments_label, fontsize=font_size)

                # Make the x-axis logarithmic
                if with_logarithmic_x_axis:
                    ax.set_xscale('log')

                ax.tick_params(
                    axis='both', labelsize=font_size
                )  # Set the font size of tick labels

                # Add title
                if with_title:
                    ax.set_title(
                        'Profile of '
                        + comp_name
                        + ' mole fraction in the liquid phase L_a along the wellbore over time',
                        fontsize=font_size,
                        fontweight='bold',
                    )

                plt.tight_layout()
                file_address = os.path.join(
                    main_dir,
                    f"{figure_counter}- {comp_name} mole fraction in the liquid phase L_a."
                    + save_as,
                )
                plt.savefig(file_address)
                if show_plot:
                    plt.show()

        # %% Profile/profiles of components mole fractions in the liquid phase L_b

        if pc.nph == 3:
            for c, comp_name in enumerate(components_names):
                # Update figure counter for name of the saved figure
                figure_counter += 1
                # Initialize the xL_b_mole_c matrix
                xL_b_mole_c_matrix = np.zeros((num_segments, num_selected_ts))

                # Fill the xL_b_mole_c matrix
                for ts_idx, ts_counter in enumerate(time_step_idx_range):
                    xL_b = data_frame["xL_b"][
                        ts_counter * num_segments : (ts_counter + 1) * num_segments
                    ]
                    xL_b_c = np.array([x[c] for x in xL_b])
                    xL_b_mole_c_matrix[:, ts_idx] = xL_b_c

                # Initialize the plot
                fig, ax = plt.subplots(figsize=(12, 6))

                # Create a discrete colorbar and colormap
                x_min, x_max = (
                    np.nanmin(xL_b_mole_c_matrix),
                    np.nanmax(xL_b_mole_c_matrix),
                )  # Using np.nanmin because if nan exists in the matrix, np.min and np.max return nan as min and max
                if x_min == x_max:
                    x_min = 0.0
                    x_max = 1.0
                levels = np.linspace(x_min, x_max, n_cmap_bins_comp + 1)
                cmap = plt.get_cmap(cmap_color, n_cmap_bins_comp)
                norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

                # Region‐based fill
                cf = ax.contourf(
                    x,
                    y_segments,
                    xL_b_mole_c_matrix,
                    levels=levels,
                    cmap=cmap,
                    norm=norm,
                    # extend='both'  # if you want arrows at the ends
                )

                # Overlay the exact same contour lines
                _cs = ax.contour(
                    x,
                    y_segments,
                    xL_b_mole_c_matrix,
                    levels=levels,
                    colors='k',
                    linewidths=0.7,
                )
                # ax.clabel(_cs, fmt='%.0f')  # if you want labels on the lines

                # Create the colorbar
                cbar = fig.colorbar(
                    cf,
                    ax=ax,
                    boundaries=levels,
                    ticks=levels,
                    spacing='proportional',
                )
                cbar.set_label(
                    comp_name + ' mole fraction in the liquid phase L_b [-]',
                    fontsize=font_size,
                )
                cbar.ax.tick_params(
                    labelsize=font_size
                )  # Set tick font size of the colorbar

                # Set the y-axis ticks
                ax.yaxis.set_major_locator(MultipleLocator(y_axis_tick_interval))

                # Reverse the y-axis
                ax.invert_yaxis()

                # Add axes labels
                ax.set_xlabel(x_label, fontsize=font_size)
                ax.set_ylabel(y_segments_label, fontsize=font_size)

                # Make the x-axis logarithmic
                if with_logarithmic_x_axis:
                    ax.set_xscale('log')

                ax.tick_params(
                    axis='both', labelsize=font_size
                )  # Set the font size of tick labels

                # Add title
                if with_title:
                    ax.set_title(
                        'Profile of '
                        + comp_name
                        + ' mole fraction in the liquid phase L_b along the wellbore over time',
                        fontsize=font_size,
                        fontweight='bold',
                    )

                plt.tight_layout()
                file_address = os.path.join(
                    main_dir,
                    f"{figure_counter}- {comp_name} mole fraction in the liquid phase L_b."
                    + save_as,
                )
                plt.savefig(file_address)
                if show_plot:
                    plt.show()

    # %% Gas density profile

    # Update figure counter for name of the saved figure
    figure_counter += 1
    # Initialize the gas density matrix
    rhoG_matrix = np.zeros((num_segments, num_selected_ts))

    # Fill the gas density matrix
    for ts_idx, ts_counter in enumerate(time_step_idx_range):
        rhoG = data_frame["rhoG"][
            ts_counter * num_segments : (ts_counter + 1) * num_segments
        ]
        rhoG_matrix[:, ts_idx] = rhoG

    # Apply a mask to hide values equal to zero
    threshold = 0  # Set your threshold here
    rhoG_matrix_masked = np.ma.masked_where(rhoG_matrix == threshold, rhoG_matrix)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create a discrete colorbar and colormap
    rhog_min, rhog_max = np.min(rhoG_matrix_masked), np.max(rhoG_matrix_masked)
    levels = np.linspace(rhog_min, rhog_max, n_cmap_bins_rho + 1)
    cmap = plt.get_cmap(cmap_color, n_cmap_bins_rho)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # Region‐based fill
    cf = ax.contourf(
        x,
        y_segments,
        rhoG_matrix_masked,
        levels=levels,
        cmap=cmap,
        norm=norm,
        # extend='both'  # if you want arrows at the ends
    )

    # Overlay the exact same contour lines
    _cs = ax.contour(
        x, y_segments, rhoG_matrix_masked, levels=levels, colors='k', linewidths=0.7
    )
    # ax.clabel(_cs, fmt='%.0f')  # if you want labels on the lines

    # Create the colorbar
    cbar = fig.colorbar(
        cf,
        ax=ax,
        boundaries=levels,
        ticks=levels,
        spacing='proportional',
    )
    cbar.set_label('Gas density [kg/m$^3$]', fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(y_axis_tick_interval))

    # Reverse the y-axis
    ax.invert_yaxis()

    # Add axes labels
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_segments_label, fontsize=font_size)

    # Make the x-axis logarithmic
    if with_logarithmic_x_axis:
        ax.set_xscale('log')

    ax.tick_params(axis='both', labelsize=font_size)  # Set the font size of tick labels

    # Add title
    if with_title:
        ax.set_title(
            'Gas density profile along the wellbore over time',
            fontsize=font_size,
            fontweight='bold',
        )

    plt.tight_layout()
    file_address = os.path.join(main_dir, f"{figure_counter}- Gas density." + save_as)
    plt.savefig(file_address)
    if show_plot:
        plt.show()

    # %% Liquid density profile

    if pc.nph == 2:
        # Update figure counter for name of the saved figure
        figure_counter += 1
        # Initialize the liquid density matrix
        rhoL_matrix = np.zeros((num_segments, num_selected_ts))

        # Fill the liquid density matrix
        for ts_idx, ts_counter in enumerate(time_step_idx_range):
            rhoL = data_frame["rhoL"][
                ts_counter * num_segments : (ts_counter + 1) * num_segments
            ]
            rhoL_matrix[:, ts_idx] = rhoL

        # Apply a mask to hide values equal to zero
        threshold = 0  # Set your threshold here
        rhoL_matrix_masked = np.ma.masked_where(rhoL_matrix == threshold, rhoL_matrix)

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create a discrete colorbar and colormap
        rhol_min, rhol_max = np.min(rhoL_matrix_masked), np.max(rhoL_matrix_masked)
        levels = np.linspace(rhol_min, rhol_max, n_cmap_bins_rho + 1)
        cmap = plt.get_cmap(cmap_color, n_cmap_bins_rho)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        # Region‐based fill
        cf = ax.contourf(
            x,
            y_segments,
            rhoL_matrix_masked,
            levels=levels,
            cmap=cmap,
            norm=norm,
            # extend='both'  # if you want arrows at the ends
        )

        # Overlay the exact same contour lines
        _cs = ax.contour(
            x, y_segments, rhoL_matrix_masked, levels=levels, colors='k', linewidths=0.7
        )
        # ax.clabel(_cs, fmt='%.0f')  # if you want labels on the lines

        # Create the colorbar
        cbar = fig.colorbar(
            cf,
            ax=ax,
            boundaries=levels,
            ticks=levels,
            spacing='proportional',
        )
        cbar.set_label('Liquid density [kg/m$^3$]', fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(y_axis_tick_interval))

        # Reverse the y-axis
        ax.invert_yaxis()

        # Add axes labels
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_segments_label, fontsize=font_size)

        # Make the x-axis logarithmic
        if with_logarithmic_x_axis:
            ax.set_xscale('log')

        ax.tick_params(
            axis='both', labelsize=font_size
        )  # Set the font size of tick labels

        # Add title
        if with_title:
            ax.set_title(
                'Liquid density profile along the wellbore over time',
                fontsize=font_size,
                fontweight='bold',
            )

        plt.tight_layout()
        file_address = os.path.join(
            main_dir, f"{figure_counter}- Liquid density." + save_as
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

    # %% Liquid L_a density profile

    if pc.nph == 3:
        # Update figure counter for name of the saved figure
        figure_counter += 1
        # Initialize the liquid L_a density matrix
        rhoL_a_matrix = np.zeros((num_segments, num_selected_ts))

        # Fill the liquid L_a density matrix
        for ts_idx, ts_counter in enumerate(time_step_idx_range):
            rhoL_a = data_frame["rhoL_a"][
                ts_counter * num_segments : (ts_counter + 1) * num_segments
            ]
            rhoL_a_matrix[:, ts_idx] = rhoL_a

        # Apply a mask to hide values equal to zero
        threshold = 0  # Set your threshold here
        rhoL_a_matrix_masked = np.ma.masked_where(
            rhoL_a_matrix == threshold, rhoL_a_matrix
        )

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create a discrete colorbar and colormap
        rhola_min, rhola_max = (
            np.min(rhoL_a_matrix_masked),
            np.max(rhoL_a_matrix_masked),
        )
        levels = np.linspace(rhola_min, rhola_max, n_cmap_bins_rho + 1)
        cmap = plt.get_cmap(cmap_color, n_cmap_bins_rho)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        # Region‐based fill
        cf = ax.contourf(
            x,
            y_segments,
            rhoL_a_matrix_masked,
            levels=levels,
            cmap=cmap,
            norm=norm,
            # extend='both'  # if you want arrows at the ends
        )

        # Overlay the exact same contour lines
        _cs = ax.contour(
            x,
            y_segments,
            rhoL_a_matrix_masked,
            levels=levels,
            colors='k',
            linewidths=0.7,
        )
        # ax.clabel(_cs, fmt='%.0f')  # if you want labels on the lines

        # Create the colorbar
        cbar = fig.colorbar(
            cf,
            ax=ax,
            boundaries=levels,
            ticks=levels,
            spacing='proportional',
        )
        cbar.set_label('Liquid L_a density [kg/m$^3$]', fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(y_axis_tick_interval))

        # Reverse the y-axis
        ax.invert_yaxis()

        # Add axes labels
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_segments_label, fontsize=font_size)

        # Make the x-axis logarithmic
        if with_logarithmic_x_axis:
            ax.set_xscale('log')

        ax.tick_params(
            axis='both', labelsize=font_size
        )  # Set the font size of tick labels

        # Add title
        if with_title:
            ax.set_title(
                'Liquid L_a density profile along the wellbore over time',
                fontsize=font_size,
                fontweight='bold',
            )

        plt.tight_layout()
        file_address = os.path.join(
            main_dir, f"{figure_counter}- Liquid L_a density." + save_as
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

    # %% Liquid L_b density profile

    if pc.nph == 3:
        # Update figure counter for name of the saved figure
        figure_counter += 1
        # Initialize the liquid L_b density matrix
        rhoL_b_matrix = np.zeros((num_segments, num_selected_ts))

        # Fill the liquid L_b density matrix
        for ts_idx, ts_counter in enumerate(time_step_idx_range):
            rhoL_b = data_frame["rhoL_b"][
                ts_counter * num_segments : (ts_counter + 1) * num_segments
            ]
            rhoL_b_matrix[:, ts_idx] = rhoL_b

        # Apply a mask to hide values equal to zero
        threshold = 0  # Set your threshold here
        rhoL_b_matrix_masked = np.ma.masked_where(
            rhoL_b_matrix == threshold, rhoL_b_matrix
        )

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create a discrete colorbar and colormap
        rholb_min, rholb_max = (
            np.min(rhoL_b_matrix_masked),
            np.max(rhoL_b_matrix_masked),
        )
        levels = np.linspace(rholb_min, rholb_max, n_cmap_bins_rho + 1)
        cmap = plt.get_cmap(cmap_color, n_cmap_bins_rho)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        # Region‐based fill
        cf = ax.contourf(
            x,
            y_segments,
            rhoL_b_matrix_masked,
            levels=levels,
            cmap=cmap,
            norm=norm,
            # extend='both'  # if you want arrows at the ends
        )

        # Overlay the exact same contour lines
        _cs = ax.contour(
            x,
            y_segments,
            rhoL_b_matrix_masked,
            levels=levels,
            colors='k',
            linewidths=0.7,
        )
        # ax.clabel(_cs, fmt='%.0f')  # if you want labels on the lines

        # Create the colorbar
        cbar = fig.colorbar(
            cf,
            ax=ax,
            boundaries=levels,
            ticks=levels,
            spacing='proportional',
        )
        cbar.set_label('Liquid L_b density [kg/m$^3$]', fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(y_axis_tick_interval))

        # Reverse the y-axis
        ax.invert_yaxis()

        # Add axes labels
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_segments_label, fontsize=font_size)

        # Make the x-axis logarithmic
        if with_logarithmic_x_axis:
            ax.set_xscale('log')

        ax.tick_params(
            axis='both', labelsize=font_size
        )  # Set the font size of tick labels

        # Add title
        if with_title:
            ax.set_title(
                'Liquid L_b density profile along the wellbore over time',
                fontsize=font_size,
                fontweight='bold',
            )

        plt.tight_layout()
        file_address = os.path.join(
            main_dir, f"{figure_counter}- Liquid L_b density." + save_as
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

    # %% Gas viscosity profile

    # Update figure counter for name of the saved figure
    figure_counter += 1
    # Initialize the gas viscosity matrix
    miuG_matrix = np.zeros((num_segments, num_selected_ts))

    # Fill the gas viscosity matrix
    for ts_idx, ts_counter in enumerate(time_step_idx_range):
        miuG = data_frame["miuG"][
            ts_counter * num_segments : (ts_counter + 1) * num_segments
        ]
        miuG_matrix[:, ts_idx] = miuG

    # Apply a mask to hide values equal to zero
    threshold = 0  # Set your threshold here
    miuG_matrix_masked = np.ma.masked_where(miuG_matrix == threshold, miuG_matrix)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create a discrete colorbar and colormap
    miug_min, miug_max = np.min(miuG_matrix_masked), np.max(miuG_matrix_masked)
    levels = np.linspace(miug_min, miug_max, n_cmap_bins_miu + 1)
    cmap = plt.get_cmap(cmap_color, n_cmap_bins_rho)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # Region‐based fill
    cf = ax.contourf(
        x,
        y_segments,
        miuG_matrix_masked,
        levels=levels,
        cmap=cmap,
        norm=norm,
        # extend='both'  # if you want arrows at the ends
    )

    # Overlay the exact same contour lines
    _cs = ax.contour(
        x, y_segments, miuG_matrix_masked, levels=levels, colors='k', linewidths=0.7
    )
    # ax.clabel(_cs, fmt='%.0f')  # if you want labels on the lines

    # Create the colorbar
    cbar = fig.colorbar(
        cf,
        ax=ax,
        boundaries=levels,
        ticks=levels,
        spacing='proportional',
    )
    cbar.set_label('Gas viscosity [cP]', fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(y_axis_tick_interval))

    # Reverse the y-axis
    ax.invert_yaxis()

    # Add axes labels
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_segments_label, fontsize=font_size)

    # Make the x-axis logarithmic
    if with_logarithmic_x_axis:
        ax.set_xscale('log')

    ax.tick_params(axis='both', labelsize=font_size)  # Set the font size of tick labels

    # Add title
    if with_title:
        ax.set_title(
            'Gas viscosity profile along the wellbore over time',
            fontsize=font_size,
            fontweight='bold',
        )

    plt.tight_layout()
    file_address = os.path.join(main_dir, f"{figure_counter}- Gas viscosity." + save_as)
    plt.savefig(file_address)
    if show_plot:
        plt.show()

    # %% Liquid viscosity profile

    if pc.nph == 2:
        # Update figure counter for name of the saved figure
        figure_counter += 1
        # Initialize the liquid viscosity matrix
        miuL_matrix = np.zeros((num_segments, num_selected_ts))

        # Fill the liquid viscosity matrix
        for ts_idx, ts_counter in enumerate(time_step_idx_range):
            miuL = data_frame["miuL"][
                ts_counter * num_segments : (ts_counter + 1) * num_segments
            ]
            miuL_matrix[:, ts_idx] = miuL

        # Apply a mask to hide values equal to zero
        threshold = 0  # Set your threshold here
        miuL_matrix_masked = np.ma.masked_where(miuL_matrix == threshold, miuL_matrix)

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create a discrete colorbar and colormap
        miul_min, miul_max = np.min(miuL_matrix_masked), np.max(miuL_matrix_masked)
        levels = np.linspace(miul_min, miul_max, n_cmap_bins_miu + 1)
        cmap = plt.get_cmap(cmap_color, n_cmap_bins_rho)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        # Region‐based fill
        cf = ax.contourf(
            x,
            y_segments,
            miuL_matrix_masked,
            levels=levels,
            cmap=cmap,
            norm=norm,
            # extend='both'  # if you want arrows at the ends
        )

        # Overlay the exact same contour lines
        _cs = ax.contour(
            x, y_segments, miuL_matrix_masked, levels=levels, colors='k', linewidths=0.7
        )
        # ax.clabel(_cs, fmt='%.0f')  # if you want labels on the lines

        # Create the colorbar
        cbar = fig.colorbar(
            cf,
            ax=ax,
            boundaries=levels,
            ticks=levels,
            spacing='proportional',
        )
        cbar.set_label('Liquid viscosity [$cP$]', fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(y_axis_tick_interval))

        # Reverse the y-axis
        ax.invert_yaxis()

        # Add axes labels
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_segments_label, fontsize=font_size)

        # Make the x-axis logarithmic
        if with_logarithmic_x_axis:
            ax.set_xscale('log')

        ax.tick_params(
            axis='both', labelsize=font_size
        )  # Set the font size of tick labels

        # Add title
        if with_title:
            ax.set_title(
                'Liquid viscosity profile along the wellbore over time',
                fontsize=font_size,
                fontweight='bold',
            )

        plt.tight_layout()
        file_address = os.path.join(
            main_dir, f"{figure_counter}- Liquid viscosity." + save_as
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

    # %% Liquid L_a viscosity profile

    if pc.nph == 3:
        # Update figure counter for name of the saved figure
        figure_counter += 1
        # Initialize the liquid L_a viscosity matrix
        miuL_a_matrix = np.zeros((num_segments, num_selected_ts))

        # Fill the liquid L_a viscosity matrix
        for ts_idx, ts_counter in enumerate(time_step_idx_range):
            miuL_a = data_frame["miuL_a"][
                ts_counter * num_segments : (ts_counter + 1) * num_segments
            ]
            miuL_a_matrix[:, ts_idx] = miuL_a

        # Apply a mask to hide values equal to zero
        threshold = 0  # Set your threshold here
        miuL_a_matrix_masked = np.ma.masked_where(
            miuL_a_matrix == threshold, miuL_a_matrix
        )

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create a discrete colorbar and colormap
        miula_min, miula_max = (
            np.min(miuL_a_matrix_masked),
            np.max(miuL_a_matrix_masked),
        )
        levels = np.linspace(miula_min, miula_max, n_cmap_bins_miu + 1)
        cmap = plt.get_cmap(cmap_color, n_cmap_bins_rho)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        # Region‐based fill
        cf = ax.contourf(
            x,
            y_segments,
            miuL_a_matrix_masked,
            levels=levels,
            cmap=cmap,
            norm=norm,
            # extend='both'  # if you want arrows at the ends
        )

        # Overlay the exact same contour lines
        _cs = ax.contour(
            x,
            y_segments,
            miuL_a_matrix_masked,
            levels=levels,
            colors='k',
            linewidths=0.7,
        )
        # ax.clabel(_cs, fmt='%.0f')  # if you want labels on the lines

        # Create the colorbar
        cbar = fig.colorbar(
            cf,
            ax=ax,
            boundaries=levels,
            ticks=levels,
            spacing='proportional',
        )
        cbar.set_label('Liquid L_a viscosity [$cP$]', fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(y_axis_tick_interval))

        # Reverse the y-axis
        ax.invert_yaxis()

        # Add axes labels
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_segments_label, fontsize=font_size)

        # Make the x-axis logarithmic
        if with_logarithmic_x_axis:
            ax.set_xscale('log')

        ax.tick_params(
            axis='both', labelsize=font_size
        )  # Set the font size of tick labels

        # Add title
        if with_title:
            ax.set_title(
                'Liquid L_a viscosity profile along the wellbore over time',
                fontsize=font_size,
                fontweight='bold',
            )

        plt.tight_layout()
        file_address = os.path.join(
            main_dir, f"{figure_counter}- Liquid L_a viscosity." + save_as
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

    # %% Liquid L_b viscosity profile

    if pc.nph == 3:
        # Update figure counter for name of the saved figure
        figure_counter += 1
        # Initialize the liquid L_b viscosity matrix
        miuL_b_matrix = np.zeros((num_segments, num_selected_ts))

        # Fill the liquid L_b viscosity matrix
        for ts_idx, ts_counter in enumerate(time_step_idx_range):
            miuL_b = data_frame["miuL_b"][
                ts_counter * num_segments : (ts_counter + 1) * num_segments
            ]
            miuL_b_matrix[:, ts_idx] = miuL_b

        # Apply a mask to hide values equal to zero
        threshold = 0  # Set your threshold here
        miuL_b_matrix_masked = np.ma.masked_where(
            miuL_b_matrix == threshold, miuL_b_matrix
        )

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create a discrete colorbar and colormap
        miulb_min, miulb_max = (
            np.min(miuL_b_matrix_masked),
            np.max(miuL_b_matrix_masked),
        )
        levels = np.linspace(miulb_min, miulb_max, n_cmap_bins_miu + 1)
        cmap = plt.get_cmap(cmap_color, n_cmap_bins_rho)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        # Region‐based fill
        cf = ax.contourf(
            x,
            y_segments,
            miuL_b_matrix_masked,
            levels=levels,
            cmap=cmap,
            norm=norm,
            # extend='both'  # if you want arrows at the ends
        )

        # Overlay the exact same contour lines
        _cs = ax.contour(
            x,
            y_segments,
            miuL_b_matrix_masked,
            levels=levels,
            colors='k',
            linewidths=0.7,
        )
        # ax.clabel(_cs, fmt='%.0f')  # if you want labels on the lines

        # Create the colorbar
        cbar = fig.colorbar(
            cf,
            ax=ax,
            boundaries=levels,
            ticks=levels,
            spacing='proportional',
        )
        cbar.set_label('Liquid L_b viscosity [$cP$]', fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(y_axis_tick_interval))

        # Reverse the y-axis
        ax.invert_yaxis()

        # Add axes labels
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_segments_label, fontsize=font_size)

        # Make the x-axis logarithmic
        if with_logarithmic_x_axis:
            ax.set_xscale('log')

        ax.tick_params(
            axis='both', labelsize=font_size
        )  # Set the font size of tick labels

        # Add title
        if with_title:
            ax.set_title(
                'Liquid L_b viscosity profile along the wellbore over time',
                fontsize=font_size,
                fontweight='bold',
            )

        plt.tight_layout()
        file_address = os.path.join(
            main_dir, f"{figure_counter}- Liquid L_b viscosity." + save_as
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

    # %% Gas velocity profile

    # Update figure counter for name of the saved figure
    figure_counter += 1
    # Initialize the gas velocity matrix
    vG_matrix = np.zeros((num_interfaces, num_selected_ts))

    # Fill the gas velocity matrix
    for ts_idx, ts_counter in enumerate(time_step_idx_range):
        vG = data_frame["vG"][
            ts_counter * num_segments : (ts_counter + 1) * num_segments
        ]
        vG_matrix[:, ts_idx] = vG[:-1] / (24 * 60 * 60)  # convert m/day to m/s

    # Apply a mask to hide values equal to zero
    threshold = 0  # Set your threshold here
    vG_matrix_masked = np.ma.masked_where(vG_matrix == threshold, vG_matrix)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create a discrete colorbar and colormap
    vg_min, vg_max = np.min(vG_matrix_masked), np.max(vG_matrix_masked)
    levels = np.linspace(vg_min, vg_max, n_cmap_bins_v + 1)
    cmap = plt.get_cmap(cmap_color, n_cmap_bins_rho)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # Region‐based fill
    cf = ax.contourf(
        x,
        y_interfaces,
        vG_matrix_masked,
        levels=levels,
        cmap=cmap,
        norm=norm,
        # extend='both'  # if you want arrows at the ends
    )

    # Overlay the exact same contour lines
    _cs = ax.contour(
        x, y_interfaces, vG_matrix_masked, levels=levels, colors='k', linewidths=0.7
    )
    # ax.clabel(_cs, fmt='%.0f')  # if you want labels on the lines

    # Create the colorbar
    cbar = fig.colorbar(
        cf,
        ax=ax,
        boundaries=levels,
        ticks=levels,
        spacing='proportional',
    )
    cbar.set_label('Gas velocity [m/s]', fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(y_axis_tick_interval))

    # Reverse the y-axis
    ax.invert_yaxis()

    # Add axes labels
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_interfaces_label, fontsize=font_size)

    ax.tick_params(axis='both', labelsize=font_size)  # Set the font size of tick labels

    # Add title
    if with_title:
        ax.set_title(
            'Gas velocity profile along the wellbore over time',
            fontsize=font_size,
            fontweight='bold',
        )

    plt.tight_layout()
    file_address = os.path.join(main_dir, f"{figure_counter}- Gas velocity." + save_as)
    plt.savefig(file_address)
    if show_plot:
        plt.show()

    # %% Liquid velocity profile

    # Update figure counter for name of the saved figure
    figure_counter += 1
    # Initialize the liquid velocity matrix
    vL_matrix = np.zeros((num_interfaces, num_selected_ts))

    # Fill the liquid velocity matrix
    for ts_idx, ts_counter in enumerate(time_step_idx_range):
        vL = data_frame["vL"][
            ts_counter * num_segments : (ts_counter + 1) * num_segments
        ]
        vL_matrix[:, ts_idx] = vL[:-1] / (24 * 60 * 60)  # convert m/day to m/s

    # Apply a mask to hide values equal to zero
    threshold = 0  # Set your threshold here
    vL_matrix_masked = np.ma.masked_where(vL_matrix == threshold, vL_matrix)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create a discrete colorbar and colormap
    vl_min, vl_max = np.min(vL_matrix_masked), np.max(vL_matrix_masked)
    levels = np.linspace(vl_min, vl_max, n_cmap_bins_v + 1)
    cmap = plt.get_cmap(cmap_color, n_cmap_bins_rho)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # Region‐based fill
    cf = ax.contourf(
        x,
        y_interfaces,
        vL_matrix_masked,
        levels=levels,
        cmap=cmap,
        norm=norm,
        # extend='both'  # if you want arrows at the ends
    )

    # Overlay the exact same contour lines
    _cs = ax.contour(
        x, y_interfaces, vL_matrix_masked, levels=levels, colors='k', linewidths=0.7
    )
    # ax.clabel(_cs, fmt='%.0f')  # if you want labels on the lines

    # Create the colorbar
    cbar = fig.colorbar(
        cf,
        ax=ax,
        boundaries=levels,
        ticks=levels,
        spacing='proportional',
    )
    cbar.set_label('Liquid velocity [m/s]', fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(y_axis_tick_interval))

    # Reverse the y-axis
    ax.invert_yaxis()

    # Add axes labels
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_interfaces_label, fontsize=font_size)

    ax.tick_params(axis='both', labelsize=font_size)  # Set the font size of tick labels

    # Add title
    if with_title:
        ax.set_title(
            'Liquid velocity profile along the wellbore over time',
            fontsize=font_size,
            fontweight='bold',
        )

    plt.tight_layout()
    file_address = os.path.join(
        main_dir, f"{figure_counter}- Liquid velocity." + save_as
    )
    plt.savefig(file_address)
    if show_plot:
        plt.show()
