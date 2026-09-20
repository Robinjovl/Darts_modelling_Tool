import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from darts.models.darts_model import DartsModel
from darts.tools.hdf5_tools import load_hdf5_to_dict


def plot_line_graphs(
    well_name: str,
    coupled_model: DartsModel,
    time_step_increment: int = 1,
    show_plot: bool = True,
):
    """
    Plot property profiles over time using line graphs for the specified well

    :param well_name: Name of the well the properties of which will be plotted
    :type well_name: str
    :param coupled_model: An instance of DartsModel
    :param show_plot: Whether or not to show the plot
    :type show_plot: bool
    """
    output_folder_name = f'line_graphs_{well_name}'
    main_dir = os.path.join(coupled_model.output_folder, output_folder_name)

    # Reset directory
    if os.path.exists(main_dir):
        shutil.rmtree(main_dir)
    os.makedirs(main_dir)

    # Well HDF5 file is used here to get the time step sizes
    h5_well_file_path = coupled_model.well_filepath
    h5_well_dict = load_hdf5_to_dict(h5_well_file_path)

    # Load primary vars and phase props
    well_props_file_path = os.path.join(
        coupled_model.output_folder, f"dfm_well_props_{well_name}.pkl"
    )
    data_frame = pd.read_pickle(well_props_file_path)

    # Get well geometry info
    well_geom = coupled_model.wells[well_name].geometry
    num_segments = well_geom.num_segments
    num_interfaces = well_geom.num_interfaces

    # Get physics info
    pc = coupled_model.physics.property_containers[0]
    components_names = pc.components_name
    fluid_components_names = pc.components_name[: pc.nc_fl]
    num_components = len(components_names)
    n_mobile_phases = coupled_model.wells[well_name].n_mobile_phases

    # Convert days to seconds
    simulated_time = h5_well_dict["dynamic"]["time"] * 24 * 60 * 60
    num_ts = len(simulated_time)
    list_of_time_steps = range(0, num_ts, time_step_increment)

    # Create a colormap (other options: 'plasma', 'inferno', etc.)
    cmap = plt.colormaps.get_cmap("jet")
    num_lines = num_ts  # Number of time steps you are plotting
    colors = cmap(np.linspace(0, 1, num_lines))  # Create a color gradient

    # %% Pressure profile

    # Use figure counter for name of the saved figure
    figure_counter = 0
    # Initialize the plot
    plt.figure(figsize=(12, 6))

    for ts_counter in list_of_time_steps:
        pressure_profile = data_frame["pressure"][
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

    plt.close()

    # %% Component/components overall mole fraction profile

    if "z" in data_frame.columns:
        for comp_idx in range(num_components):
            # Update figure counter for name of the saved figure
            figure_counter += 1
            # Initialize the plot
            plt.figure(figsize=(12, 6))

            for ts_counter in list_of_time_steps:
                z_profile = data_frame["z"][
                    ts_counter * num_segments : (ts_counter + 1) * num_segments
                ]
                z_profile = z_profile.tolist()
                z_c_profile = np.zeros(num_segments)
                for segment_idx in range(num_segments):
                    try:
                        z_c_profile[segment_idx] = z_profile[segment_idx][comp_idx]
                    except:
                        z_c_profile[segment_idx] = 1 - sum(z_profile[segment_idx])
                plt.plot(
                    z_c_profile, list(range(num_segments)), color=colors[ts_counter]
                )

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

            plt.close()

    # %% Temperature profile

    # Temperature profile is plotted if the system is non-isothermal.
    if coupled_model.physics.property_containers[0].thermal:
        # Update figure counter for name of the saved figure
        figure_counter += 1

        # Initialize the plot
        plt.figure(figsize=(12, 6))

        for ts_counter in list_of_time_steps:
            temp_profile = (
                data_frame["temperature"][
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

        plt.close()

    # %% Gas volume fraction profile

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

    plt.xlabel("Gas volume fraction [-]", fontsize=14)
    plt.ylabel("Segment index", fontsize=14)

    plt.title(
        "Gas volume fraction profile/profiles along the wellbore",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    file_address = os.path.join(main_dir, f"{figure_counter}- Gas volume fraction.png")
    plt.savefig(file_address)
    if show_plot:
        plt.show()

    plt.close()

    def plot_phase_mass_fraction_profiles(phase_name, phase_description):
        nonlocal figure_counter
        for comp_name in fluid_components_names:
            # Update figure counter for name of the saved figure
            figure_counter += 1
            # Initialize the plot
            plt.figure(figsize=(12, 6))

            prop_name = f"x{comp_name}_in_{phase_name}_mass"
            for ts_counter in list_of_time_steps:
                x_profile = data_frame[prop_name][
                    ts_counter * num_segments : (ts_counter + 1) * num_segments
                ]
                plt.plot(x_profile, list(range(num_segments)), color=colors[ts_counter])

            # Reverse the y-axis
            plt.gca().invert_yaxis()
            plt.ylim(num_segments - 1, 0)

            # Set the y-axis ticks
            plt.gca().yaxis.set_major_locator(MultipleLocator(1))

            plt.xlabel(
                comp_name + f" mass fraction in the {phase_description} phase [-]",
                fontsize=14,
            )
            plt.ylabel("Segment index", fontsize=14)

            plt.title(
                comp_name
                + f" mass fraction in the {phase_description} phase profile/profiles along the wellbore",
                fontsize=14,
                fontweight="bold",
            )

            plt.tight_layout()
            file_address = os.path.join(
                main_dir,
                f"{figure_counter}- {comp_name} mass fraction in the {phase_description} phase.png",
            )
            plt.savefig(file_address)
            if show_plot:
                plt.show()

            plt.close()

    # %% Profile/profiles of components mass fractions in phases
    plot_phase_mass_fraction_profiles("G", "G")
    if n_mobile_phases == 2:
        plot_phase_mass_fraction_profiles("L", "L")
    elif n_mobile_phases == 3:
        plot_phase_mass_fraction_profiles("L_a", "L_a")
        plot_phase_mass_fraction_profiles("L_b", "L_b")

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

    plt.close()

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

    plt.close()

    # %% Gas viscosity profile

    # Update figure counter for name of the saved figure
    figure_counter += 1
    # Initialize the plot
    plt.figure(figsize=(12, 6))

    for ts_counter in list_of_time_steps:
        muG_profile = data_frame["muG"][
            ts_counter * num_segments : (ts_counter + 1) * num_segments
        ]

        # Apply a mask to hide values equal to zero
        threshold = 0  # Set your threshold here
        muG_profile_masked = np.ma.masked_where(muG_profile == threshold, muG_profile)

        plt.plot(
            muG_profile_masked, list(range(num_segments)), color=colors[ts_counter]
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

    plt.close()

    # %% Liquid viscosity profile

    # Update figure counter for name of the saved figure
    figure_counter += 1
    # Initialize the plot
    plt.figure(figsize=(12, 6))

    for ts_counter in list_of_time_steps:
        muL_profile = data_frame["muL"][
            ts_counter * num_segments : (ts_counter + 1) * num_segments
        ]

        # Apply a mask to hide values equal to zero
        threshold = 0  # Set your threshold here
        muL_profile_masked = np.ma.masked_where(muL_profile == threshold, muL_profile)

        plt.plot(
            muL_profile_masked, list(range(num_segments)), color=colors[ts_counter]
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

    plt.close()

    def plot_interface_profile(prop_name, x_label, title, file_name):
        nonlocal figure_counter
        if prop_name not in data_frame.columns:
            return

        figure_counter += 1
        plt.figure(figsize=(12, 6))

        for ts_counter in list_of_time_steps:
            profile = data_frame[prop_name][
                ts_counter * num_segments : (ts_counter + 1) * num_segments
            ].to_numpy(dtype=float)
            profile = profile[:-1] / (24 * 60 * 60)
            plt.plot(profile, list(range(num_interfaces)), color=colors[ts_counter])

        plt.gca().invert_yaxis()
        plt.ylim(num_interfaces - 1, 0)
        plt.gca().yaxis.set_major_locator(MultipleLocator(1))

        plt.xlabel(x_label, fontsize=14)
        plt.ylabel("Interface index", fontsize=14)
        plt.title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()
        file_address = os.path.join(main_dir, f"{figure_counter}- {file_name}.png")
        plt.savefig(file_address)
        if show_plot:
            plt.show()

        plt.close()

    # %% Phase velocity profiles

    plot_interface_profile(
        "vG",
        "Gas velocity [m/s]",
        "Gas velocity profile/profiles along the wellbore",
        "Gas velocity",
    )
    plot_interface_profile(
        "vL",
        "Liquid velocity [m/s]",
        "Liquid velocity profile/profiles along the wellbore",
        "Liquid velocity",
    )

    # %% Phase rate profiles

    PHASE_RATE_PLOT_SPECS = (
        ("molar", "molar", "kmol/s"),
        ("mass", "mass", "kg/s"),
        ("volumetric", "volumetric", "m$^3$/s"),
    )
    for rate_type, rate_label, unit in PHASE_RATE_PLOT_SPECS:
        for phase_name in pc.phases_name:
            prop_name = f"phase_{rate_type}_rate_{phase_name}"
            if phase_name == "G":
                phase_display = "Gas"
            elif phase_name == "L":
                phase_display = "Liquid"
            else:
                phase_display = phase_name
            plot_interface_profile(
                prop_name,
                f"{phase_display} {rate_label} rate [{unit}]",
                f"{phase_display} {rate_label} rate profile/profiles along the wellbore",
                f"{phase_display} {rate_label} rate",
            )
