"""
This script can be used to plot the desired property, which is stored in well_primary_vars_and_phase_props.pkl,
for the desired wellbore segment, e.g., 0 for the wellhead and num_segments - 1 for the bottom-hole, over time for
different scenarios saved in different output folders each of which containing the following two files:
    - well_data.h5
    - well_primary_vars_and_phase_props.pkl

As an example, you can use this script to plot BHP or BHT vs time for different scenarios.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from darts.tools.hdf5_tools import load_hdf5_to_dict

""" Input """
min_time_step_idx = 10  # This can be used to avoid plotting very small time steps
num_segments = 41

scenarios_labels = ["110", "200", "500", "1000"]
legend_labels = [
    "OBL resolution = 110",
    "OBL resolution = 200",
    "OBL resolution = 500",
    "OBL resolution = 1000",
]

property_key = "Pressure"
desired_well_segment_idx = (
    num_segments - 1
)  # For bottomhole = num_segments - 1, for wellhead = 0
y_label = "BHP [bar]"
y_min = 10
y_max = 45
y_tick_increment = 5
output_name = "BHP_time_series_obl_resolution_sens_ana"

# # Note that the temperature stored is in Kelvin
# property_key = "Temperature"
# desired_well_segment_idx = (
#     num_segments - 1
# )  # For bottomhole = num_segments - 1, for wellhead = 0
# y_min = -35
# y_max = 80
# y_tick_increment = 10
# y_label = "BHT [\u00b0C]"
# output_name = "BHT_time_series_obl_resolution_sens_ana"


""" Main code """
list_of_simulated_time = []
list_of_property_time_series = []

for scenario in scenarios_labels:
    output_folder = f"output_{scenario}"

    well_data_file_path = os.path.join(output_folder, "well_data.h5")
    h5_well_data = load_hdf5_to_dict(well_data_file_path)
    simulated_time = h5_well_data["dynamic"]["time"] * 24 * 60 * 60

    primary_vars_and_phase_props_file_address = os.path.join(
        output_folder, "well_primary_vars_and_phase_props.pkl"
    )

    # Load primary vars and phase props
    data_frame = pd.read_pickle(primary_vars_and_phase_props_file_address)
    property_time_series = data_frame[property_key][desired_well_segment_idx]

    list_of_simulated_time += [simulated_time]
    list_of_property_time_series += [property_time_series]


# Global style for consistency
plt.rcParams.update(
    {
        "font.size": 12,  # base font
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.0,  # axis spine thickness
        "xtick.major.size": 6,  # tick length
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,  # embed TrueType for Illustrator/Indesign
        "ps.fonttype": 42,
    }
)

# Linestyle/marker combos for grayscale-friendly distinction
linestyles = ["-", "--", "-.", ":", "-", "--"]
markers = [
    "o",
    "s",
    "D",
    "^",
    "v",
    "None",
]  # 'None' if you want one line without markers

fig, ax = plt.subplots(figsize=(8, 5))

for idx in range(len(scenarios_labels)):
    ax.semilogx(
        list_of_simulated_time[idx][min_time_step_idx:],
        list_of_property_time_series[idx][min_time_step_idx:],
        linestyle='',
        # linestyle=linestyles[idx % len(linestyles)],
        marker=markers[idx % len(markers)],
        linewidth=2.0,
        markersize=5,
        label=legend_labels[idx],
    )

# X axis formatting
x_min = list_of_simulated_time[0][min_time_step_idx]
x_max = max(list_of_simulated_time[-1])
ax.set_xlim(x_min, x_max)

ax.set_ylim(y_min, y_max)
ax.set_yticks(np.arange(y_min, y_max + y_tick_increment, y_tick_increment))

# Subtle grid on x and y
ax.grid(True, which="major", axis="x", linestyle="--", alpha=0.3)
ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.3)

# Labels
ax.set_xlabel("Simulated time [second]", labelpad=6)
ax.set_ylabel(y_label, labelpad=6)

# Legend: compact, outside or inside depending on space
leg = ax.legend(frameon=False, loc="best", handlelength=3)
if leg.get_title() is not None:
    leg.get_title().set_fontsize(12)

""" Add a zoomed inset, zoomed-in view or inset axis"""
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
#
# # Create inset axis (smaller plot inside the main one)
# axins = inset_axes(ax, width="30%", height="30%", loc="lower left",
#                    bbox_to_anchor=(0.3, 0.35, 0.9, 0.9),  # position relative to main axis
#                    bbox_transform=ax.transAxes)
#
# # Plot same curves inside inset
# for idx in range(len(scenarios_labels)):
#     axins.plot(list_of_simulated_time[idx], list_of_property_time_series[idx], linestyle=linestyles[idx % len(linestyles)], marker=markers[idx % len(markers)], linewidth=2.0, markersize=4)
#
# # Set zoomed-in region
# axins.set_xlim(700, 900)
# axins.set_ylim(28, 33)   # For pressure
# # axins.set_ylim(-7, -2)   # For temperature
#
# # Optional: smaller tick labels
# axins.tick_params(axis='both', which='major', labelsize=9)
#
# # Draw a box around zoom area on main plot
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
""" End the zoomed inset"""

fig.tight_layout()
fig.savefig(output_name + ".pdf")
plt.show()
