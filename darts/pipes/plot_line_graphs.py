import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from darts.models.darts_model import DartsModel


def plot_line_graphs(
    primary_vars_and_phase_props_file_address: str,
    h5_well_data: dict,
    coupled_model: DartsModel,
    time_step_increment: int = 1,
    show_plot: bool = True,
):
    """
    :param primary_vars_and_phase_props_file_address: Address of the pickle file in which primary variables and phase
    properties of well segments are stored
    :param h5_well_data: HDF5 file containing well solution. It's used here to get the time step sizes
    :param coupled_model: An instance of DartsModel
    :param show_plot: Whether or not to show the plot
    :type show_plot: bool
    """
    main_dir = os.path.join(coupled_model.output_folder, "line_graphs")

    # Reset_directory
    if os.path.exists(main_dir):
        shutil.rmtree(main_dir)
    os.makedirs(main_dir)

    # Load primary vars and phase props
    data_frame = pd.read_pickle(primary_vars_and_phase_props_file_address)

    # This line gets the geometry object of the first well (by insertion order) from the wells_geometry dictionary and assigns it to well_geom.
    well_geom = next(iter(coupled_model.wells.values())).geometry
    num_segments = well_geom.num_segments

    components_names = coupled_model.physics.property_containers[0].components_name
    num_components = len(components_names)

    simulation_times = (
        h5_well_data["dynamic"]["time"] * 24 * 60 * 60
    )  # convert days to seconds
    num_ts = len(simulation_times)
    list_of_time_steps = range(0, num_ts, time_step_increment)

    # Create a colormap
    cmap = plt.colormaps.get_cmap(
        "jet"
    )  # You can use other colormaps like 'plasma', 'inferno', etc.
    num_lines = num_ts  # Number of time steps you are plotting
    colors = cmap(np.linspace(0, 1, num_lines))  # Create a color gradient

    # %% Pressure profile

    # Use figure counter for name of the saved figure
    figure_counter = 0
    # Initialize the plot
    plt.figure(figsize=(12, 6))

    for ts_counter in list_of_time_steps:
        pressure_profile = data_frame["Pressure"][
            ts_counter * num_segments : (ts_counter + 1) * num_segments
        ]
        # plt.plot(pressure_profile, list(range(num_segments)), label=ts_counter)
        plt.plot(
            pressure_profile, list(range(num_segments)), color=colors[ts_counter]
        )  # without legend

    # Reverse the y-axis
    plt.gca().invert_yaxis()
    plt.ylim(num_segments - 1, 0)

    # Set the y-axis ticks
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))
    # plt.gca().xaxis.set_major_locator(MultipleLocator(0.01))

    # Set x-axis limits
    # plt.xlim(4, 12)

    # Add labels and legend
    plt.xlabel("Pressure [bar]", fontsize=14)
    plt.ylabel("Segment index", fontsize=14)
    plt.title(
        "Pressure profile/profiles along the wellbore", fontsize=14, fontweight="bold"
    )
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
    # plt.tight_layout(rect=[0, 0, 0.99, 1])  # Adjust the size of the axes to make space for the legend
    # plt.legend(loc='upper right')

    plt.tight_layout()
    file_address = os.path.join(main_dir, f"{figure_counter}- Pressure.png")
    plt.savefig(file_address)
    if show_plot:
        plt.show()

    # %% Component/components overall mole fraction profile

    for comp_idx in range(num_components):
        # Update figure counter for name of the saved figure
        figure_counter += 1
        # Initialize the plot
        plt.figure(figsize=(12, 6))

        for ts_counter in list_of_time_steps:
            z_profile = data_frame["Overall mole fractions"][
                ts_counter * num_segments : (ts_counter + 1) * num_segments
            ]
            z_profile = z_profile.tolist()
            z_c_profile = np.zeros(num_segments)
            for segment_idx in range(num_segments):
                try:
                    z_c_profile[segment_idx] = z_profile[segment_idx][comp_idx]
                except:
                    z_c_profile[segment_idx] = 1 - sum(z_profile[segment_idx])
            plt.plot(z_c_profile, list(range(num_segments)), color=colors[ts_counter])

        # Reverse the y-axis
        plt.gca().invert_yaxis()
        plt.ylim(num_segments - 1, 0)

        # Set the y-axis ticks
        plt.gca().yaxis.set_major_locator(MultipleLocator(1))
        # Set the x-axis ticks
        # plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))

        # Set x-axis limits
        # plt.xlim(4, 12)

        # Add labels and legend
        plt.xlabel(
            components_names[comp_idx] + " overall mole fraction [-]", fontsize=14
        )
        plt.ylabel("Segment index", fontsize=14)
        plt.title(
            components_names[comp_idx]
            + " overall mole fraction profile/profiles along the wellbore",
            fontsize=14,
            fontweight="bold",
        )
        # plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
        # plt.tight_layout(rect=[0, 0, 0.99, 1])  # Adjust the size of the axes to make space for the legend
        # plt.legend(loc='upper right')

        plt.tight_layout()
        file_address = os.path.join(
            main_dir,
            f"{figure_counter}- {components_names[comp_idx]} overall mole fraction.png",
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

    # %% Temperature profile

    # Update figure counter for name of the saved figure
    figure_counter += 1
    # Temperature profile is plotted if the system is non-isothermal.
    if coupled_model.physics.property_containers[0].thermal:
        # Initialize the plot
        plt.figure(figsize=(12, 6))

        for ts_counter in list_of_time_steps:
            temp_profile = (
                data_frame["Temperature"][
                    ts_counter * num_segments : (ts_counter + 1) * num_segments
                ]
                - 273.15
            )
            # plt.plot(temp_profile, list(range(num_segments)), label=ts_counter)
            plt.plot(
                temp_profile, list(range(num_segments)), color=colors[ts_counter]
            )  # without legend

        # Reverse the y-axis
        plt.gca().invert_yaxis()
        plt.ylim(num_segments - 1, 0)

        # Set the y-axis ticks
        plt.gca().yaxis.set_major_locator(MultipleLocator(1))
        # plt.gca().xaxis.set_major_locator(MultipleLocator(0.01))

        # Set x-axis limits
        # plt.xlim(4, 12)

        # Add labels and legend
        plt.xlabel("Temperature [\u00b0C]", fontsize=14)
        plt.ylabel("Segment index", fontsize=14)
        plt.title(
            "Temperature profile/profiles along the wellbore",
            fontsize=14,
            fontweight="bold",
        )
        # plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
        # plt.tight_layout(rect=[0, 0, 0.99, 1])  # Adjust the size of the axes to make space for the legend
        # plt.legend(loc='upper right')

        plt.tight_layout()
        file_address = os.path.join(main_dir, f"{figure_counter}- Temperature.png")
        plt.savefig(file_address)
        if show_plot:
            plt.show()

    # %% Gas saturation profile

    # Update figure counter for name of the saved figure
    figure_counter += 1
    # Initialize the plot
    plt.figure(figsize=(12, 6))

    for ts_counter in list_of_time_steps:
        sG_profile = data_frame["sG"][
            ts_counter * num_segments : (ts_counter + 1) * num_segments
        ]
        plt.plot(sG_profile, list(range(num_segments)), color=colors[ts_counter])

    # Reverse the y-axis
    plt.gca().invert_yaxis()
    plt.ylim(num_segments - 1, 0)

    # Set the y-axis ticks
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))

    plt.xlabel("Gas saturation [-]", fontsize=14)
    plt.ylabel("Segment index", fontsize=14)

    plt.title(
        "Gas saturation profile/profiles along the wellbore",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    file_address = os.path.join(main_dir, f"{figure_counter}- Gas saturation.png")
    plt.savefig(file_address)
    if show_plot:
        plt.show()

    # %% Profile/profiles of components mole fractions in the gaseous phase

    for c, comp_name in enumerate(components_names):
        # Update figure counter for name of the saved figure
        figure_counter += 1
        # Initialize the plot
        plt.figure(figsize=(12, 6))

        for ts_counter in list_of_time_steps:
            xG_profile = data_frame["xG"][
                ts_counter * num_segments : (ts_counter + 1) * num_segments
            ]
            xG_c_profile = np.array([x[c] for x in xG_profile])
            plt.plot(xG_c_profile, list(range(num_segments)), color=colors[ts_counter])

        # Reverse the y-axis
        plt.gca().invert_yaxis()
        plt.ylim(num_segments - 1, 0)

        # Set the y-axis ticks
        plt.gca().yaxis.set_major_locator(MultipleLocator(1))

        plt.xlabel(comp_name + " mole fraction in the gaseous phase [-]", fontsize=14)
        plt.ylabel("Segment index", fontsize=14)

        plt.title(
            comp_name
            + " mole fraction in the gaseous phase profile/profiles along the wellbore",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()
        file_address = os.path.join(
            main_dir,
            f"{figure_counter}- {comp_name} mole fraction in the gaseous phase.png",
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

    # %% Profile/profiles of components mole fractions in the liquid phase

    for c, comp_name in enumerate(components_names):
        # Update figure counter for name of the saved figure
        figure_counter += 1
        # Initialize the plot
        plt.figure(figsize=(12, 6))

        for ts_counter in list_of_time_steps:
            xL_profile = data_frame["xL"][
                ts_counter * num_segments : (ts_counter + 1) * num_segments
            ]
            xL_c_profile = np.array([x[c] for x in xL_profile])
            plt.plot(xL_c_profile, list(range(num_segments)), color=colors[ts_counter])

        # Reverse the y-axis
        plt.gca().invert_yaxis()
        plt.ylim(num_segments - 1, 0)

        # Set the y-axis ticks
        plt.gca().yaxis.set_major_locator(MultipleLocator(1))

        plt.xlabel(comp_name + " mole fraction in the liquid phase [-]", fontsize=14)
        plt.ylabel("Segment index", fontsize=14)

        plt.title(
            comp_name
            + " mole fraction in the liquid phase profile/profiles along the wellbore",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()
        file_address = os.path.join(
            main_dir,
            f"{figure_counter}- {comp_name} mole fraction in the liquid phase.png",
        )
        plt.savefig(file_address)
        if show_plot:
            plt.show()

    # %% Gas density profile

    # Update figure counter for name of the saved figure
    figure_counter += 1
    # Initialize the plot
    plt.figure(figsize=(12, 6))

    for ts_counter in list_of_time_steps:
        rhoG_profile = data_frame["rhoG"][
            ts_counter * num_segments : (ts_counter + 1) * num_segments
        ]

        # Apply a mask to hide values equal to zero
        threshold = 0  # Set your threshold here
        rhoG_profile_masked = np.ma.masked_where(
            rhoG_profile == threshold, rhoG_profile
        )

        plt.plot(
            rhoG_profile_masked, list(range(num_segments)), color=colors[ts_counter]
        )

    # Reverse the y-axis
    plt.gca().invert_yaxis()
    plt.ylim(num_segments - 1, 0)

    # Set the y-axis ticks
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))

    plt.xlabel("Gas density [kg/m$^3$]", fontsize=14)
    plt.ylabel("Segment index", fontsize=14)

    plt.title(
        "Gas density profile/profiles along the wellbore",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    file_address = os.path.join(main_dir, f"{figure_counter}- Gas density.png")
    plt.savefig(file_address)
    if show_plot:
        plt.show()

    # %% Liquid density profile

    # Update figure counter for name of the saved figure
    figure_counter += 1
    # Initialize the plot
    plt.figure(figsize=(12, 6))

    for ts_counter in list_of_time_steps:
        rhoL_profile = data_frame["rhoL"][
            ts_counter * num_segments : (ts_counter + 1) * num_segments
        ]

        # Apply a mask to hide values equal to zero
        threshold = 0  # Set your threshold here
        rhoL_profile_masked = np.ma.masked_where(
            rhoL_profile == threshold, rhoL_profile
        )

        plt.plot(
            rhoL_profile_masked, list(range(num_segments)), color=colors[ts_counter]
        )

    # Reverse the y-axis
    plt.gca().invert_yaxis()
    plt.ylim(num_segments - 1, 0)

    # Set the y-axis ticks
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))

    plt.xlabel("Liquid density [kg/m$^3$]", fontsize=14)
    plt.ylabel("Segment index", fontsize=14)

    plt.title(
        "Liquid density profile/profiles along the wellbore",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    file_address = os.path.join(main_dir, f"{figure_counter}- Liquid density.png")
    plt.savefig(file_address)
    if show_plot:
        plt.show()

    # %% Gas viscosity profile

    # Update figure counter for name of the saved figure
    figure_counter += 1
    # Initialize the plot
    plt.figure(figsize=(12, 6))

    for ts_counter in list_of_time_steps:
        miuG_profile = data_frame["miuG"][
            ts_counter * num_segments : (ts_counter + 1) * num_segments
        ]

        # Apply a mask to hide values equal to zero
        threshold = 0  # Set your threshold here
        miuG_profile_masked = np.ma.masked_where(
            miuG_profile == threshold, miuG_profile
        )

        plt.plot(
            miuG_profile_masked, list(range(num_segments)), color=colors[ts_counter]
        )

    # Reverse the y-axis
    plt.gca().invert_yaxis()
    plt.ylim(num_segments - 1, 0)

    # Set the y-axis ticks
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))

    plt.xlabel("Gas viscosity [cP]", fontsize=14)
    plt.ylabel("Segment index", fontsize=14)

    plt.title(
        "Gas viscosity profile/profiles along the wellbore",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    file_address = os.path.join(main_dir, f"{figure_counter}- Gas viscosity.png")
    plt.savefig(file_address)
    if show_plot:
        plt.show()

    # %% Liquid viscosity profile

    # Update figure counter for name of the saved figure
    figure_counter += 1
    # Initialize the plot
    plt.figure(figsize=(12, 6))

    for ts_counter in list_of_time_steps:
        miuL_profile = data_frame["miuL"][
            ts_counter * num_segments : (ts_counter + 1) * num_segments
        ]

        # Apply a mask to hide values equal to zero
        threshold = 0  # Set your threshold here
        miuL_profile_masked = np.ma.masked_where(
            miuL_profile == threshold, miuL_profile
        )

        plt.plot(
            miuL_profile_masked, list(range(num_segments)), color=colors[ts_counter]
        )

    # Reverse the y-axis
    plt.gca().invert_yaxis()
    plt.ylim(num_segments - 1, 0)

    # Set the y-axis ticks
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))

    plt.xlabel("Liquid viscosity [cP]", fontsize=14)
    plt.ylabel("Segment index", fontsize=14)

    plt.title(
        "Liquid viscosity profile/profiles along the wellbore",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    file_address = os.path.join(main_dir, f"{figure_counter}- Liquid viscosity.png")
    plt.savefig(file_address)
    if show_plot:
        plt.show()
