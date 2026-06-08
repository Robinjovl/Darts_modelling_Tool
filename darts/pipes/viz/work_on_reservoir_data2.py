"""
Figure 14
This script can be used to plot the desired reservoir property for different scenarios (solution files) at initial
conditions and final conditions within an axes.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter

num_res_cells = 1000  # number of reservoir cells

df_0 = pd.read_csv("initial_conditions.csv")
df_1 = pd.read_csv("500mDm.csv")
df_2 = pd.read_csv("5000mDm_base_case.csv")
df_3 = pd.read_csv("10000mDm.csv")
df_4 = pd.read_csv("20000mDm.csv")

distance = df_0["X"][:num_res_cells]
x_min = 4e-2
x_max = 1e3 + 100

# property_0 = df_0["pressure"][:num_res_cells]   # initial property profile
# property_1 = df_1["pressure"][:num_res_cells]
# property_2 = df_2["pressure"][:num_res_cells]
# property_3 = df_3["pressure"][:num_res_cells]
# property_4 = df_4["pressure"][:num_res_cells]
# y_label = "Reservoir pressure [bar]"
# output_name = "reservoir_pressure"
# y_min = 10
# y_max = 75
# y_tick_increment = 5

property_0 = df_0["temperature"][:num_res_cells] - 273.15  # initial property profile
property_1 = df_1["temperature"][:num_res_cells] - 273.15
property_2 = df_2["temperature"][:num_res_cells] - 273.15
property_3 = df_3["temperature"][:num_res_cells] - 273.15
property_4 = df_4["temperature"][:num_res_cells] - 273.15
y_label = "Reservoir temperature [\u00b0C]"
output_name = "reservoir_temperature"
y_min = -30
y_max = 80
y_tick_increment = 10

# property_0 = df_0["sat_gas"][:num_res_cells]   # initial property profile
# property_1 = df_1["sat_gas"][:num_res_cells]
# property_2 = df_2["sat_gas"][:num_res_cells]
# property_3 = df_3["sat_gas"][:num_res_cells]
# property_4 = df_4["sat_gas"][:num_res_cells]
# y_label = "Reservoir gas saturation [-]"
# output_name = "reservoir_gas_saturation"
# y_min = -0.1
# y_max = 1.1
# y_tick_increment = 0.1

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
    label="Initial conditions",
)
ax.plot(
    distance,
    property_1,
    linestyle=linestyles[1],
    marker=markers[1],
    linewidth=2.0,
    markersize=4,
    label=r"Kh = $5 \times 10^2 \ \mathrm{mD} \cdot \mathrm{m}$",
)
ax.plot(
    distance,
    property_2,
    linestyle=linestyles[2],
    marker=markers[2],
    linewidth=2.0,
    markersize=4,
    label=r"Kh = $5 \times 10^3 \ \mathrm{mD} \cdot \mathrm{m}$",
)
ax.plot(
    distance,
    property_3,
    linestyle=linestyles[3],
    marker=markers[3],
    linewidth=2.0,
    markersize=4,
    label=r"Kh = $1 \times 10^4 \ \mathrm{mD} \cdot \mathrm{m}$",
)
ax.plot(
    distance,
    property_4,
    linestyle=linestyles[4],
    marker=markers[4],
    linewidth=2.0,
    markersize=4,
    label=r"Kh = $2 \times 10^4 \ \mathrm{mD} \cdot \mathrm{m}$",
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
ax.set_yticks(np.arange(y_min, y_max, y_tick_increment))

# Subtle grid only on y
ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.3)
ax.grid(False, axis="x")  # keep log x clean

# Labels
ax.set_xlabel("Radial distance [m]", labelpad=6)
ax.set_ylabel(y_label, labelpad=6)

# Legend: compact, outside or inside depending on space
leg = ax.legend(
    # title="Time",
    # ncol=2,
    frameon=False,
    loc="upper right",
    handlelength=3,
)
if leg.get_title() is not None:
    leg.get_title().set_fontsize(12)


fig.tight_layout()
fig.savefig(output_name + ".pdf")
plt.show()
