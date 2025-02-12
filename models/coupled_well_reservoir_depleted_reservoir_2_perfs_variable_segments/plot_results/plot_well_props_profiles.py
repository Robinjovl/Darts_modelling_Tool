import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from matplotlib.cm import get_cmap

# Load the HDF5 file
file_path = "solution.h5"  # Ensure the file is in the same directory as this script

# Open the file and extract the data
with h5py.File(file_path, 'r') as h5_file:
    # Extract datasets
    X = h5_file['dynamic/X'][:]  # Dynamic variable data
    segment_id = h5_file['dynamic/cell_id'][:]  # Cell IDs
    time = h5_file['dynamic/time'][:]  # Time steps

# Select the last num_segments cells (wellbore segments)
num_segments = 65
well_segments = segment_id[-num_segments:]
# For constant segment length
# segment_length = 50
# segments_depths = np.arange(num_segments) * segment_length + segment_length/2
# For variable segment length
segments_lengths = np.concatenate((50 * np.ones(55), 25 * np.ones(10)))
segments_depths = np.cumsum(segments_lengths) - segments_lengths / 2
# Last num_segments cell IDs
well_indices = np.arange(len(segment_id))[-num_segments:]  # Indices of the last num_segments cells

time_step_labels = ["Initial conditions", "1 minute", "2 minutes", "3 minutes", "5 minutes", "10 minutes",
                    "20 minutes", "30 minutes", "50 minutes", "1 hour", "2 hours", "3 hours", "5 hours", "10 hours",
                    "1 day", "2 days", "3 days", "5 days", "10 days", "20 days", "30 days"]

# Font size adjustments
font_size_labels = 16
font_size_title = 18
font_size_ticks = 12


#%% Pressure
# Create a figure and a single set of axes
plt.figure(figsize=(10, 6))

# Generate a colormap for the time steps
cmap = get_cmap('jet')
colors = [cmap(i / len(time_step_labels)) for i in range(len(time_step_labels))]

# Define markers and line styles
markers = ['o', 's', 'd', '^', 'v', 'x', '*']
line_styles = ['-', '--', '-.', ':']

for idx, time_step_label in enumerate(time_step_labels):
    co2_data = X[idx, well_indices, 0]
    color = colors[idx]  # Assign color from the colormap
    marker = markers[idx % len(markers)]  # Cycle through markers
    line_style = line_styles[idx % len(line_styles)]  # Cycle through line styles
    plt.plot(
        co2_data, segments_depths,
        marker=marker,
        linestyle=line_style,
        color=color,
        label=time_step_label
    )

# Add labels, title, and grid
# plt.title("Profile of pressure along the wellbore", fontsize=font_size_title, pad=15)
plt.xlabel("Pressure [bar]", fontsize=font_size_labels, labelpad=10)
# plt.ylabel("Segment index [-]", fontsize=font_size_labels, labelpad=10)
plt.ylabel("Well segment depth [m]", fontsize=font_size_labels, labelpad=10)
plt.yticks(fontsize=font_size_ticks)
plt.gca().invert_yaxis()  # Invert y-axis for proper orientation
# plt.grid(True)
plt.grid(linestyle='--')   # Add dashed grid lines
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
plt.savefig("wellbore_pressure_profiles")
plt.show()


#%% zCO2
# Create a figure and a single set of axes
plt.figure(figsize=(10, 6))

# Generate a colormap for the time steps
cmap = get_cmap('jet')
colors = [cmap(i / len(time_step_labels)) for i in range(len(time_step_labels))]

# Define markers and line styles
markers = ['o', 's', 'd', '^', 'v', 'x', '*']
line_styles = ['-', '--', '-.', ':']

for idx, time_step_label in enumerate(time_step_labels):
    co2_data = X[idx, well_indices, 1]
    color = colors[idx]  # Assign color from the colormap
    marker = markers[idx % len(markers)]  # Cycle through markers
    line_style = line_styles[idx % len(line_styles)]  # Cycle through line styles
    plt.plot(
        co2_data, segments_depths,
        marker=marker,
        linestyle=line_style,
        color=color,
        label=time_step_label
    )

# Add labels, title, and grid
# plt.title("Profile of overall mole fraction of CO2 along the wellbore", fontsize=font_size_title, pad=15)
plt.xlabel("Overall mole fraction of CO$_2$ [-]", fontsize=font_size_labels, labelpad=10)
# plt.ylabel("Segment index [-]", fontsize=font_size_labels, labelpad=10)
plt.ylabel("Well segment depth [m]", fontsize=font_size_labels, labelpad=10)
plt.yticks(fontsize=font_size_ticks)
plt.gca().invert_yaxis()  # Invert y-axis for proper orientation
# plt.grid(True)
plt.grid(linestyle='--')   # Add dashed grid lines
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
plt.savefig("wellbore_CO2_overall_mole_fraction_profiles")
plt.show()


#%% zCO2
# Create a figure and a single set of axes
plt.figure(figsize=(10, 6))

# Generate a colormap for the time steps
cmap = get_cmap('jet')
colors = [cmap(i / len(time_step_labels)) for i in range(len(time_step_labels))]

# Define markers and line styles
markers = ['o', 's', 'd', '^', 'v', 'x', '*']
line_styles = ['-', '--', '-.', ':']

for idx, time_step_label in enumerate(time_step_labels):
    co2_data = X[idx, well_indices, 2]
    color = colors[idx]  # Assign color from the colormap
    marker = markers[idx % len(markers)]  # Cycle through markers
    line_style = line_styles[idx % len(line_styles)]  # Cycle through line styles
    plt.plot(
        co2_data, segments_depths,
        marker=marker,
        linestyle=line_style,
        color=color,
        label=time_step_label
    )

# Add labels, title, and grid
# plt.title("Profile of overall mole fraction of CH4 along the wellbore", fontsize=font_size_title, pad=15)
plt.xlabel("Overall mole fraction of CH$_4$ [-]", fontsize=font_size_labels, labelpad=10)
# plt.ylabel("Segment index [-]", fontsize=font_size_labels, labelpad=10)
plt.ylabel("Well segment depth [m]", fontsize=font_size_labels, labelpad=10)
plt.yticks(fontsize=font_size_ticks)
plt.gca().invert_yaxis()  # Invert y-axis for proper orientation
# plt.grid(True)
plt.grid(linestyle='--')   # Add dashed grid lines
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
plt.savefig("wellbore_CH4_overall_mole_fraction_profiles")
plt.show()
