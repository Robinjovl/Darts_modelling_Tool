import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

# Load the HDF5 file
file_path = "solution.h5"  # Ensure the file is in the same directory as this script

# Open the file and extract the data
with h5py.File(file_path, 'r') as h5_file:
    # Extract datasets
    X = h5_file['dynamic/X'][:]  # Dynamic variable data
    cell_id = h5_file['dynamic/cell_id'][:]  # Cell IDs
    time = h5_file['dynamic/time'][:]  # Time steps

# Select the last 20 cells (wellbore segments)
last_20_cells = cell_id[-20:]  # Last 20 cell IDs
last_20_indices = np.arange(len(cell_id))[-20:]  # Indices of the last 20 cells

time_step = 1   # 0 initial conditions, 1 and 2 are identical, and are the final conditions

# Font size adjustments
font_size_labels = 16
font_size_title = 18
font_size_ticks = 12

# Plot 1
plt.figure(figsize=(10, 6))
pressure_data = X[time_step, last_20_indices, 0]
plt.plot(pressure_data, last_20_cells, 'o-')
plt.title("Profile of pressure along the wellbore", fontsize=font_size_title, pad=15)
plt.xlabel("Pressure [bar]", fontsize=font_size_labels, labelpad=10)
plt.ylabel("Segment index [-]", fontsize=font_size_labels, labelpad=10)
plt.xticks(fontsize=font_size_ticks)
plt.yticks(fontsize=font_size_ticks)
plt.gca().invert_yaxis()  # Invert y-axis for proper orientation
plt.grid(True)
plt.tight_layout()  # Make the plot layout tight
plt.show()

# Plot 2
plt.figure(figsize=(10, 6))
co2_data = X[time_step, last_20_indices, 1]
plt.plot(co2_data, last_20_cells, 'o-')
plt.title("Profile of overall mole fraction of CO2 along the wellbore", fontsize=font_size_title, pad=15)
plt.xlabel("Overall mole fraction of CO2 [-]", fontsize=font_size_labels, labelpad=10)
plt.ylabel("Segment index [-]", fontsize=font_size_labels, labelpad=10)
plt.xticks(fontsize=font_size_ticks)
plt.yticks(fontsize=font_size_ticks)
plt.gca().invert_yaxis()  # Invert y-axis for proper orientation
plt.grid(True)
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.6f'))
plt.tight_layout()  # Make the plot layout tight
plt.show()

# Plot 3
plt.figure(figsize=(10, 6))
methane_data = X[time_step, last_20_indices, 2]
plt.plot(methane_data, last_20_cells, 'o-')
plt.title("Profile of overall mole fraction of CH4 along the wellbore", fontsize=font_size_title, pad=15)
plt.xlabel("Overall mole fraction of CH4 [-]", fontsize=font_size_labels, labelpad=10)
plt.ylabel("Segment index [-]", fontsize=font_size_labels, labelpad=10)
plt.xticks(fontsize=font_size_ticks)
plt.yticks(fontsize=font_size_ticks)
plt.gca().invert_yaxis()  # Invert y-axis for proper orientation
plt.grid(True)
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.6f'))
plt.tight_layout()  # Make the plot layout tight
plt.show()
