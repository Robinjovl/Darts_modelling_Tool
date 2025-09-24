"""
Figure 16-b
This script can be used to plot the desired reservoir property for different scenarios (solution files) at final
conditions within an axes with zoomed inset.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

num_res_cells = 1000  # number of reservoir cells

df_0 = pd.read_csv("11_bar_base_case.csv")
df_1 = pd.read_csv("15_bar.csv")
df_2 = pd.read_csv("20_bar.csv")

distance = df_0["X"][:num_res_cells]
x_min = 4e-2
x_max = 1e3 + 100


property_0 = df_0["temperature"][:num_res_cells] - 273.15  # initial property profile
property_1 = df_1["temperature"][:num_res_cells] - 273.15
property_2 = df_2["temperature"][:num_res_cells] - 273.15
y_label = "Reservoir temperature [\u00b0C]"
output_name = "reservoir_temperature"
y_min = -10
y_max = 80
y_tick_increment = 10


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

ax.plot(
    distance,
    property_0,
    linestyle=linestyles[0],
    marker=markers[0],
    linewidth=2.0,
    markersize=4,
    label=r"$P_\mathrm{res}$ = 11 bar (base-case scenario)",
)
ax.plot(
    distance,
    property_1,
    linestyle=linestyles[1],
    marker=markers[1],
    linewidth=2.0,
    markersize=4,
    label=r"$P_\mathrm{res}$ = 15 bar",
)
ax.plot(
    distance,
    property_2,
    linestyle=linestyles[2],
    marker=markers[2],
    linewidth=2.0,
    markersize=4,
    label=r"$P_\mathrm{res}$ = 20 bar",
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

# Subtle grid only on y
ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.3)
ax.grid(False, axis="x")  # keep log x clean

# Labels
ax.set_xlabel("Radial distance [m]", labelpad=6)
ax.set_ylabel(y_label, labelpad=6)

# Legend: compact, outside or inside depending on space
leg = ax.legend(
    frameon=False,
    loc="upper right",
    bbox_to_anchor=(1, 1 - 0.078),  # brings the legend about 1 cm lower
    handlelength=3,
)
if leg.get_title() is not None:
    leg.get_title().set_fontsize(12)

""" Add a zoomed inset, zoomed-in view or inset axis"""

# Create inset axis (smaller plot inside the main one)
axins = inset_axes(
    ax,
    width="30%",
    height="30%",
    loc="lower left",
    bbox_to_anchor=(0.25, 0.15, 0.9, 0.9),  # position relative to main axis
    bbox_transform=ax.transAxes,
)

# Plot same curves inside inset
axins.plot(
    distance,
    property_0,
    linestyle=linestyles[0],
    marker=markers[0],
    linewidth=2.0,
    markersize=4,
)
axins.plot(
    distance,
    property_1,
    linestyle=linestyles[1],
    marker=markers[1],
    linewidth=2.0,
    markersize=4,
)
axins.plot(
    distance,
    property_2,
    linestyle=linestyles[2],
    marker=markers[2],
    linewidth=2.0,
    markersize=4,
)

# Set zoomed-in region
axins.set_xlim(4e-2, 0.25)
axins.set_ylim(-10, 1)  # adjust y-limits to highlight the differences
axins.xaxis.set_major_locator(MultipleLocator(0.05))
axins.yaxis.set_major_locator(MultipleLocator(2))

# Optional: smaller tick labels
axins.tick_params(axis='both', which='major', labelsize=9)

# Draw a box around zoom area on main plot
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
""" End the zoomed inset"""

fig.tight_layout()
fig.savefig(output_name + ".pdf")
plt.show()
