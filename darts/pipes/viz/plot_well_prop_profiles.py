import os
import shutil

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from darts.models.darts_model import DartsModel
from darts.tools.hdf5_tools import load_hdf5_to_dict


def plot_well_prop_profiles(
    coupled_model: DartsModel,
):
    """
    Plot well property profiles at certain time steps. Please note that currently, the function
    is limited to report_step_labels and report_step_times used below.

    :param coupled_model: An instance of DartsModel
    :type coupled_model: DartsModel
    """
    main_dir = os.path.join(coupled_model.output_folder, 'well_prop_profiles')

    # Reset_directory
    if os.path.exists(main_dir):
        shutil.rmtree(main_dir)
    os.makedirs(main_dir)

    # Load primary vars and phase props
    primary_vars_and_phase_props_file_path = os.path.join(
        coupled_model.output.output_folder, "well_primary_vars_and_phase_props.pkl"
    )
    data_frame = pd.read_pickle(primary_vars_and_phase_props_file_path)

    # Well HDF5 file is used here to get the time step sizes
    h5_well_file_path = coupled_model.well_filepath
    h5_well_dict = load_hdf5_to_dict(h5_well_file_path)
    simulated_time = h5_well_dict["dynamic"]["time"]

    # This line gets the geometry object of the first well (by insertion order) from the wells_geometry dictionary
    # and assigns it to well_geom.
    well_geom = next(iter(coupled_model.wells.values())).geometry

    # Select the last num_segments cells (wellbore segments)
    num_segments = well_geom.num_segments
    segment_depths = well_geom.z

    report_step_labels = [
        "Initial conditions",
        "1 minute",
        "2 minutes",
        "3 minutes",
        "5 minutes",
        "10 minutes",
        "20 minutes",
        "30 minutes",
        "50 minutes",
        "1 hour",
        "2 hours",
        "3 hours",
        "5 hours",
        "10 hours",
        "1 day",
        "2 days",
        "3 days",
        "5 days",
        "10 days",
        "20 days",
        "30 days",
        "50 days",
        "100 days",
        "200 days",
        "365 days",
    ]
    report_step_times = [
        0.0,
        1 / 24 / 60,  # 1 minute
        1 / 24 / 30 - 1 / 24 / 60,  # 2 minute
        1 / 24 / 20 - 1 / 24 / 30,  # 3 minute
        1 / 24 / 12 - 1 / 24 / 20,  # 5 minute
        1 / 24 / 6 - 1 / 24 / 12,  # 10 minute
        1 / 24 / 3 - 1 / 24 / 6,  # 20 minute
        1 / 24 / 2 - 1 / 24 / 3,  # 30 minute
        1 / 24 / 6 * 5 - 1 / 24 / 2,  # 50 minute
        1 / 24 - 1 / 24 / 6 * 5,  # 1 hour
        2 / 24 - 1 / 24,  # 2 hour
        3 / 24 - 2 / 24,  # 3 hour
        4.99 / 24 - 3 / 24,  # 5 hour
        9.98 / 24 - 4.99 / 24,  # 10 hour
        1 - 9.98 / 24,  # 1 day
        2 - 1,  # 2 day
        3 - 2,  # 3 day
        5 - 3,  # 5 day
        10 - 5,  # 10 day
        19.99 - 10,  # 20 day
        30.002 - 19.99,  # 30 day
        49.99 - 30.002,  # 50 day
        100.02 - 49.99,  # 100 day
        200.001 - 100.02,  # 200 day
        365 - 200.001,  # 365 day
    ]

    report_step_times = np.cumsum(report_step_times)
    report_indices = [
        np.where(np.isclose(simulated_time, a))[0][0] for a in report_step_times
    ]

    # Generate a colormap for the report steps
    cmap = matplotlib.colormaps['jet']
    colors = [cmap(i / len(report_step_labels)) for i in range(len(report_step_labels))]

    # Define markers and line styles
    markers = ['o', 's', 'd', '^', 'v', 'x', '*']
    line_styles = ['-', '--', '-.', ':']

    # Font size adjustments
    font_size_labels = 16
    # font_size_title = 18
    font_size_ticks = 12

    # %% Pressure
    # Create a figure and a single set of axes
    plt.figure(figsize=(10, 6))

    for idx, report_index in enumerate(report_indices):
        pressure = data_frame["Pressure"][
            report_index * num_segments : (report_index + 1) * num_segments
        ]
        color = colors[idx]  # Assign color from the colormap
        marker = markers[idx % len(markers)]  # Cycle through markers
        line_style = line_styles[idx % len(line_styles)]  # Cycle through line styles
        plt.plot(
            pressure,
            segment_depths,
            marker=marker,
            linestyle=line_style,
            color=color,
            label=report_step_labels[idx],
        )

    # Add labels, title, and grid
    # plt.title("Profile of pressure along the wellbore", fontsize=font_size_title, pad=15)
    plt.xlabel("Pressure [bar]", fontsize=font_size_labels, labelpad=10)
    # plt.ylabel("Segment index [-]", fontsize=font_size_labels, labelpad=10)
    plt.ylabel("Well segment depth [m]", fontsize=font_size_labels, labelpad=10)
    plt.yticks(fontsize=font_size_ticks)
    plt.gca().invert_yaxis()  # Invert y-axis for proper orientation
    # plt.grid(True)
    plt.grid(linestyle='--')  # Add dashed grid lines
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Move the x-axis to the top
    plt.gca().xaxis.set_label_position('top')  # Move x-axis label to the top
    plt.gca().xaxis.tick_top()  # Move x-axis ticks to the top
    plt.xticks(fontsize=font_size_ticks)

    # Position the legend outside the plot, with multiple columns
    plt.legend(
        fontsize=9,
        loc='upper left',
        bbox_to_anchor=(1.05, 1.05),  # Position legend outside
        ncol=1,  # Single or multiple columns for compactness
        title="Report steps",
        title_fontsize=12,
    ).get_frame().set_edgecolor('black')  # Optional: Add a border
    plt.gca().legend_.set_frame_on(True)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(os.path.join(main_dir, "wellbore_pressure_profiles.pdf"), format='pdf')
    plt.savefig(os.path.join(main_dir, "wellbore_pressure_profiles.svg"), format='svg')
    plt.show()

    # %% z_c
    # Create a figure and a single set of axes
    plt.figure(figsize=(10, 6))

    for idx, report_index in enumerate(report_indices):
        z = data_frame["Overall mole fractions"][
            report_index * num_segments : (report_index + 1) * num_segments
        ]
        z = z.tolist()
        z_c = np.zeros(num_segments)
        for segment_idx in range(num_segments):
            z_c[segment_idx] = z[segment_idx][0]  # Only for the first component

        color = colors[idx]  # Assign color from the colormap
        marker = markers[idx % len(markers)]  # Cycle through markers
        line_style = line_styles[idx % len(line_styles)]  # Cycle through line styles
        plt.plot(
            z_c,
            segment_depths,
            marker=marker,
            linestyle=line_style,
            color=color,
            label=report_step_labels[idx],
        )

    # Add labels, title, and grid
    # plt.title("Profile of overall mole fraction of CO2 along the wellbore", fontsize=font_size_title, pad=15)
    plt.xlabel(
        "Overall mole fraction of CO$_2$ [-]", fontsize=font_size_labels, labelpad=10
    )
    # plt.ylabel("Segment index [-]", fontsize=font_size_labels, labelpad=10)
    plt.ylabel("Well segment depth [m]", fontsize=font_size_labels, labelpad=10)
    plt.yticks(fontsize=font_size_ticks)
    plt.gca().invert_yaxis()  # Invert y-axis for proper orientation
    # plt.grid(True)
    plt.grid(linestyle='--')  # Add dashed grid lines
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Move the x-axis to the top
    plt.gca().xaxis.set_label_position('top')  # Move x-axis label to the top
    plt.gca().xaxis.tick_top()  # Move x-axis ticks to the top
    plt.xticks(fontsize=font_size_ticks)

    # Position the legend outside the plot, with multiple columns
    plt.legend(
        fontsize=9,
        loc='upper left',
        bbox_to_anchor=(1.05, 1.05),  # Position legend outside
        ncol=1,  # Single or multiple columns for compactness
        title="Report steps",
        title_fontsize=12,
    ).get_frame().set_edgecolor('black')  # Optional: Add a border
    plt.gca().legend_.set_frame_on(True)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(
        os.path.join(main_dir, "wellbore_CO2_overall_mole_fraction_profiles.pdf"),
        format='pdf',
    )
    plt.savefig(
        os.path.join(main_dir, "wellbore_CO2_overall_mole_fraction_profiles.svg"),
        format='svg',
    )
    plt.show()

    # %% Temperature
    # Create a figure and a single set of axes
    plt.figure(figsize=(10, 6))

    for idx, report_index in enumerate(report_indices):
        temp = data_frame["Temperature"][
            report_index * num_segments : (report_index + 1) * num_segments
        ]
        color = colors[idx]  # Assign color from the colormap
        marker = markers[idx % len(markers)]  # Cycle through markers
        line_style = line_styles[idx % len(line_styles)]  # Cycle through line styles
        plt.plot(
            temp - 273.15,
            segment_depths,
            marker=marker,
            linestyle=line_style,
            color=color,
            label=report_step_labels[idx],
        )

    # Add labels, title, and grid
    plt.xlabel("Temperature [\u00b0C]", fontsize=font_size_labels, labelpad=10)
    # plt.ylabel("Segment index [-]", fontsize=font_size_labels, labelpad=10)
    plt.ylabel("Well segment depth [m]", fontsize=font_size_labels, labelpad=10)
    plt.yticks(fontsize=font_size_ticks)
    plt.gca().invert_yaxis()  # Invert y-axis for proper orientation
    # plt.grid(True)
    plt.grid(linestyle='--')  # Add dashed grid lines
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Move the x-axis to the top
    plt.gca().xaxis.set_label_position('top')  # Move x-axis label to the top
    plt.gca().xaxis.tick_top()  # Move x-axis ticks to the top
    plt.xticks(fontsize=font_size_ticks)

    # Position the legend outside the plot, with multiple columns
    plt.legend(
        fontsize=9,
        loc='upper left',
        bbox_to_anchor=(1.05, 1.05),  # Position legend outside
        ncol=1,  # Single or multiple columns for compactness
        title="Report steps",
        title_fontsize=12,
    ).get_frame().set_edgecolor('black')  # Optional: Add a border
    plt.gca().legend_.set_frame_on(True)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(
        os.path.join(main_dir, "wellbore_temperature_profiles.pdf"), format='pdf'
    )
    plt.savefig(
        os.path.join(main_dir, "wellbore_temperature_profiles.svg"), format='svg'
    )
    plt.show()

    # %% Gas saturation
    # Create a figure and a single set of axes
    plt.figure(figsize=(10, 6))

    for idx, report_index in enumerate(report_indices):
        sG = data_frame["sG"][
            report_index * num_segments : (report_index + 1) * num_segments
        ]
        color = colors[idx]  # Assign color from the colormap
        marker = markers[idx % len(markers)]  # Cycle through markers
        line_style = line_styles[idx % len(line_styles)]  # Cycle through line styles
        plt.plot(
            sG,
            segment_depths,
            marker=marker,
            linestyle=line_style,
            color=color,
            label=report_step_labels[idx],
        )

    # Add labels, title, and grid
    plt.xlabel("Gas saturation [-]", fontsize=font_size_labels, labelpad=10)
    # plt.ylabel("Segment index [-]", fontsize=font_size_labels, labelpad=10)
    plt.ylabel("Well segment depth [m]", fontsize=font_size_labels, labelpad=10)
    plt.yticks(fontsize=font_size_ticks)
    plt.gca().invert_yaxis()  # Invert y-axis for proper orientation
    # plt.grid(True)
    plt.grid(linestyle='--')  # Add dashed grid lines
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Move the x-axis to the top
    plt.gca().xaxis.set_label_position('top')  # Move x-axis label to the top
    plt.gca().xaxis.tick_top()  # Move x-axis ticks to the top
    plt.xticks(fontsize=font_size_ticks)

    # Position the legend outside the plot, with multiple columns
    plt.legend(
        fontsize=9,
        loc='upper left',
        bbox_to_anchor=(1.05, 1.05),  # Position legend outside
        ncol=1,  # Single or multiple columns for compactness
        title="Report steps",
        title_fontsize=12,
    ).get_frame().set_edgecolor('black')  # Optional: Add a border
    plt.gca().legend_.set_frame_on(True)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(os.path.join(main_dir, "wellbore_sG_profiles.pdf"), format='pdf')
    plt.savefig(os.path.join(main_dir, "wellbore_sG_profiles.svg"), format='svg')
    plt.show()
