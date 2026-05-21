"""
Figure 6
This script can be used to plot the desired reservoir property at different reporting times (solution files)
within an axes.
All the reservoir solutions need to be saved in a single file: all_solutions.csv
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

num_res_cells = 1000  # number of reservoir cells to keep in matrices

# Fast load; infer low-memory dtypes
df = pd.read_csv("all_solutions.csv", low_memory=False)
print(df.shape, df.columns.tolist())

# Which timesteps / how many cells?
print("Timesteps:", sorted(df["Timestep"].unique()))
print("Cells:", df["CellID"].nunique())

# Ensure stable ordering
tvals = np.array(sorted(df["Timestep"].unique()))
# cvals = np.array(sorted(df["CellID"].unique()))
cvals = np.arange(num_res_cells)  # we only want 0..249
# nT, nC = len(tvals), len(cvals)
nT, nC = len(tvals), num_res_cells

# Make an index to pivot quickly
df_idx = df.set_index(["Timestep", "CellID"]).sort_index()


def to_matrix(column):
    """
    Return a (nT, num_res_cells) matrix for the given column (trimmed).
    """
    full_idx = pd.MultiIndex.from_product(
        [tvals, sorted(df["CellID"].unique())], names=["Timestep", "CellID"]
    )
    s = df_idx[column].reindex(full_idx)
    matrix_full = s.values.reshape(len(tvals), -1)  # all cells
    return matrix_full[:, :num_res_cells]  # keep only the first num_res_cells


# Plot reservoir property vs radial distance

# Get the desired property matrix
# property_matrix = to_matrix("pressure")       # shape (nT, nC)
# y_label = "Pressure [bar]"
# output_name = "reservoir_pressure"
# x_min = 4e-2
# x_max = 1e3
# y_min = 10
# y_max = 30
# y_tick_increment = 2

# property_matrix = to_matrix("temperature") - 273.15       # shape (nT, nC)
# y_label = "Temperature [\u00B0C]"
# output_name = "reservoir_temperature"
# x_min = 4e-2
# x_max = 1e3
# y_min = -20
# y_max = 80
# y_tick_increment = 10

property_matrix = to_matrix("sat_LCO2")  # shape (nT, nC)
y_label = "Liquid saturation [-]"
output_name = "reservoir_liquid_saturation"
x_min = 4e-2
x_max = 1e3
y_min = 0
y_max = 1
y_tick_increment = 0.1

# distance = df["X"][:num_res_cells]
#
# # Times corresponding to rows in P
# times = ["Initial conditions", "1 min", "2 min", "3 min", "5 min", "10 min"]
#
# plt.figure(figsize=(8, 5))
#
# for i in range(property_matrix.shape[0]):   # loop over rows
#     plt.plot(distance, property_matrix[i, :], label=times[i])
#
# plt.xscale("log")  # make x-axis logarithmic
#
# plt.xlabel("Radial distance [m]", fontsize=18)
# plt.ylabel(y_label, fontsize=18)
#
# plt.xlim([x_min, x_max])
# plt.ylim([y_min, y_max])
#
# plt.xticks(fontsize=14)
# plt.yticks(np.arange(10, 31, 2), fontsize=18)
#
# plt.legend(fontsize=12)
# plt.grid(True, which="both", linestyle="--", alpha=0.7)
# plt.savefig(output_name + ".pdf")
# plt.tight_layout()
# plt.show()


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

distance = df["X"][:num_res_cells]

times = ["Initial conditions", "1 min", "2 min", "3 min", "5 min", "10 min"]

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

for i in range(property_matrix.shape[0]):
    ax.plot(
        distance,
        property_matrix[i, :],
        linestyle=linestyles[i % len(linestyles)],
        marker=markers[i % len(markers)],
        linewidth=2.0,
        markersize=4,
        label=times[i],
    )

# Logarithmic x-axis with tidy ticks
ax.set_xscale("log")
ax.set_xlim(x_min, x_max)

# Major ticks at powers of 10; lighter minor ticks in between
ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
ax.xaxis.set_minor_locator(
    LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=50)
)
ax.xaxis.set_minor_formatter(
    NullFormatter()
)  # keep minor ticks unlabeled for cleanliness

# Y axis formatting
ax.set_ylim(y_min, y_max)
ax.set_yticks(np.arange(y_min, y_max + y_tick_increment, y_tick_increment))

# Labels
ax.set_xlabel("Radial distance [m]", labelpad=6)
ax.set_ylabel(y_label, labelpad=6)

# Legend: compact, outside or inside depending on space
leg = ax.legend(title="Time", ncol=2, frameon=False, loc="upper right", handlelength=3)
if leg.get_title() is not None:
    leg.get_title().set_fontsize(12)

# Subtle grid only on y
ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.3)
ax.grid(False, axis="x")  # keep log x clean

fig.tight_layout()
# Save vector
fig.savefig(output_name + ".pdf", bbox_inches="tight")
plt.show()


""" Plot phase density or viscosity """
fig = plt.figure(figsize=(8, 5))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

gas_matrix = to_matrix("rho_gas")
liq_matrix = to_matrix("rho_LCO2")
y_label_1 = "Gas density [kg/m$^3$]"
y_label_2 = "Liquid density [kg/m$^3$]"
output_name = "reservoir_density"
x_min = 4e-2
x_max = 1e3
y_min_gas, y_max_gas = 10, 90
y_min_liq, y_max_liq = -5, 1050

# gas_matrix = to_matrix("miu_gas")
# liq_matrix = to_matrix("miu_LCO2")
# y_label_1 = "Gas viscosity [cP]"
# y_label_2 = "Liquid viscosity [cP]"
# output_name = "reservoir_viscosity"
# x_min = 4e-2
# x_max = 1e3
# y_min_gas, y_max_gas = 0.013, 0.02
# y_min_liq, y_max_liq = 0, 0.135

assert gas_matrix.shape == liq_matrix.shape
distance = df["X"][:num_res_cells]
nT = gas_matrix.shape[0]
try:
    assert len(times) == nT
except Exception:
    times = [f"t{i}" for i in range(nT)]

# Colors shared across phases (same color = same time)
colors = mpl.rcParams['axes.prop_cycle'].by_key().get('color') or [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

markers = ['o', '^', 'D', 'v', 's', '>']

# Gas (left y): solid + filled markers
for i in range(nT):
    c = colors[i % len(colors)]
    ax1.plot(
        distance,
        gas_matrix[i, :],
        linestyle='-',
        marker=markers[i % len(markers)],
        linewidth=2.0,  # match linewidth
        markersize=4.0,  # match markersize
        color=c,
        label=f"Gas — {times[i]}",
        zorder=3,
    )

# Liquid (right y): dashed + hollow markers
for i in range(nT):
    c = colors[i % len(colors)]
    ax2.plot(
        distance,
        liq_matrix[i, :],
        linestyle='--',
        marker=markers[i % len(markers)],
        linewidth=2.0,  # match linewidth
        markersize=4.0,  # match markersize
        color=c,
        markerfacecolor='white',
        markeredgecolor=c,
        markeredgewidth=1.0,  # keep crisp but consistent
        label=f"Liquid — {times[i]}",
        zorder=2,
    )

# Axes formatting
ax1.set_xscale("log")
ax1.set_xlim(x_min, x_max)
ax1.xaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
ax1.xaxis.set_minor_locator(
    LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=50)
)
ax1.xaxis.set_minor_formatter(NullFormatter())

ax1.set_xlabel("Radial distance [m]", labelpad=6)
ax1.set_ylabel(y_label_1, labelpad=6)
ax2.set_ylabel(y_label_2, labelpad=6)

# y axes limits
ax1.set_ylim(y_min_gas, y_max_gas)  # left axis (Gas)
ax2.set_ylim(y_min_liq, y_max_liq)  # right axis (Liquid)

# Subtle grid only on y (to match the other figure)
ax1.grid(True, which="major", axis="y", linestyle="--", alpha=0.3)
ax1.grid(False, axis="x")

# Legends inside the main axes (right side)
phase_handles = [
    Line2D(
        [0],
        [0],
        linestyle='-',
        marker='o',
        linewidth=2,
        markersize=4,
        color='black',
        label='Gas',
    ),
    Line2D(
        [0],
        [0],
        linestyle='--',
        marker='o',
        linewidth=2,
        markersize=4,
        color='black',
        markerfacecolor='white',
        markeredgecolor='black',
        label='Liquid',
    ),
]

# # Phase legend on the right, slightly lower
# leg_phase = ax1.legend(
#     handles=phase_handles, title="Phase",
#     loc="upper right",
#     bbox_to_anchor=(0.98, 0.75),
#     frameon=False, borderaxespad=0.0
# )
#
# # Must draw before reading bbox
# fig.canvas.draw()
# bb = leg_phase.get_window_extent()
# bb_ax = bb.transformed(ax1.transAxes.inverted())
#
# # Time legend directly below Phase
# time_handles = [
#     Line2D([0], [0], linestyle='-', linewidth=2.0, color=colors[i % len(colors)], label=times[i])
#     for i in range(len(times))
# ]
# gap = 0.02
# leg_time = ax1.legend(
#     handles=time_handles, title="Time",
#     loc="upper right",
#     bbox_to_anchor=(0.98, bb_ax.y0 - gap),
#     ncol=1, frameon=False, handlelength=3.0,  # handlelength to match
#     borderaxespad=0.0, labelspacing=0.5
# )

# Phase legend at top right of the figure
phase_top = 0.98  # was 0.75
leg_phase = ax1.legend(
    handles=phase_handles,
    title="Phase",
    loc="upper right",
    bbox_to_anchor=(0.98, phase_top),
    frameon=False,
    borderaxespad=0.0,
)

# draw, then place Time directly underneath Phase
fig.canvas.draw()
bb_phase = leg_phase.get_window_extent().transformed(ax1.transAxes.inverted())

# Time legend directly below Phase
time_handles = [
    Line2D(
        [0],
        [0],
        linestyle='-',
        linewidth=2.0,
        color=colors[i % len(colors)],
        label=times[i],
    )
    for i in range(len(times))
]

gap = 0.02  # vertical gap between the two legends
leg_time = ax1.legend(
    handles=time_handles,
    title="Time",
    loc="upper right",
    bbox_to_anchor=(0.98, bb_phase.y0 - gap),
    ncol=1,
    frameon=False,
    handlelength=3.0,
    borderaxespad=0.0,
)

# Keep Phase legend visible
ax1.add_artist(leg_phase)

if leg_phase.get_title():
    leg_phase.get_title().set_fontsize(12)
if leg_time.get_title():
    leg_time.get_title().set_fontsize(12)

# Save / Show
fig.tight_layout()
fig.savefig(output_name + ".pdf", bbox_inches="tight")
plt.show()
