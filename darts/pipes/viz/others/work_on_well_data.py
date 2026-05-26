"""
Figure 16-a
This script can be used to plot the desired well property profile for different scenarios
(dfm_well_props_{well_name}.pkl files) at final conditions within an axes.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_0 = pd.read_pickle("11_bar_base_case.pkl")
df_1 = pd.read_pickle("15_bar.pkl")
df_2 = pd.read_pickle("20_bar.pkl")

num_segments = 41
well_length = num_segments * 50
depth = np.linspace(0 + 25, well_length - 25, num_segments)
y_min = 0
y_max = well_length

property_0 = df_0["Temperature"][-41::] - 273.15  # initial property profile
property_1 = df_1["Temperature"][-41::] - 273.15
property_2 = df_2["Temperature"][-41::] - 273.15
x_label = "Wellbore temperature [\u00b0C]"
output_name = "well_temperature"
x_min = -7
x_max = 0
# y_tick_increment = 1


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
    property_0,
    depth,
    linestyle=linestyles[0],
    marker=markers[0],
    linewidth=2.0,
    markersize=4,
    label=r"$P_\mathrm{res}$ = 11 bar (base-case scenario)",
)
ax.plot(
    property_1,
    depth,
    linestyle=linestyles[0],
    marker=markers[1],
    linewidth=2.0,
    markersize=4,
    label=r"$P_\mathrm{res}$ = 15 bar",
)
ax.plot(
    property_2,
    depth,
    linestyle=linestyles[0],
    marker=markers[2],
    linewidth=2.0,
    markersize=4,
    label=r"$P_\mathrm{res}$ = 20 bar",
)

# X axis formatting
ax.set_xlim(x_min, x_max)
# ax.set_xticks(np.arange(x_min, x_max+y_tick_increment, y_tick_increment))

ax.set_ylim(y_min, y_max)

# Subtle grid on x and y
ax.grid(True, which="major", axis="x", linestyle="--", alpha=0.3)
ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.3)

# Labels
ax.set_xlabel(x_label, labelpad=6)
ax.set_ylabel("Segment MD [m]", labelpad=6)

# Reverse y-axis
ax.invert_yaxis()

# Put x-ticks and label at the top
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# Legend: compact, outside or inside depending on space
leg = ax.legend(frameon=False, loc="upper right", handlelength=3)
if leg.get_title() is not None:
    leg.get_title().set_fontsize(12)

fig.tight_layout()
fig.savefig(output_name + ".pdf")
plt.show()
