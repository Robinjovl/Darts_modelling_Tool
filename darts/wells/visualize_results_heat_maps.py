# TODO: This file is copied from the standalone well simulator and it should be adjusted to the coupled model.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import pickle

from units import *
from library import components_molecular_weights


# y axis: standard direction and segment depths in meters
y_axis_convention = "standard"
y_axis = "segment_depth"

# y axis: T2Well direction and segment indices
# y_axis_convention = "T2Well"
# y_axis = "segment_index"


# x_axis = "time_step_index"
x_axis = "simulation_time"

with open('stored_dt_pipe_geometry_other_info.pkl', 'rb') as file:
    _, pipe_geom, _ = pickle.load(file)

if y_axis == "segment_depth":
    # Load the PipeModel instance from the pickle file

    measured_depths_segments = sum(pipe_geom.segments_lengths) - pipe_geom.z
    true_vertical_depths_segments = measured_depths_segments * np.cos(pipe_geom.inclination_angle_radian)
    if y_axis_convention == "standard":
        true_vertical_depths_segments = true_vertical_depths_segments
    elif y_axis_convention == "T2Well":
        true_vertical_depths_segments = true_vertical_depths_segments[::-1]

    measured_depths_interfaces = sum(pipe_geom.segments_lengths) - pipe_geom.z_interfaces
    true_vertical_depths_interfaces = measured_depths_interfaces * np.cos(pipe_geom.inclination_angle_radian)
    if y_axis_convention == "standard":
        true_vertical_depths_interfaces = true_vertical_depths_interfaces
    elif y_axis_convention == "T2Well":
        true_vertical_depths_interfaces = true_vertical_depths_interfaces[::-1]

if x_axis == "simulation_time":
    # Load dt from the pickle file
    with open('stored_dt_pipe_geometry_other_info.pkl', 'rb') as file:
        dt, _, _ = pickle.load(file)

    simulation_time = np.concatenate(([0], np.cumsum(dt)))

# Load components names from the pickle file
with open('stored_dt_pipe_geometry_other_info.pkl', 'rb') as file:
    _, _, components_names = pickle.load(file)

# Load primary variables
primary_variables_df = pd.read_pickle('stored_primary_variables.pkl')

num_segments = pipe_geom.num_segments
time_steps = primary_variables_df.columns


#%% Component/components overall mole fraction profiles

for c, comp_name in enumerate(components_names[:-1]):
    # Initialize component c mole fraction matrix
    component_c_mole_fraction_matrix = np.zeros((num_segments, len(time_steps)))

    # Fill component c mole fraction matrix
    for i, time_step in enumerate(time_steps):
        component_c_mole_fraction_profile = primary_variables_df[time_step][num_segments * (c + 1):num_segments * (c + 2)]
        component_c_mole_fraction_matrix[:, i] = component_c_mole_fraction_profile

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create the heatmap
    cmap = plt.get_cmap('jet')
    if x_axis == "time_step_index" and y_axis == "segment_index":
        cax = ax.pcolormesh(range(len(time_steps)), range(num_segments), component_c_mole_fraction_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel('Time step [-]', fontsize=14)
        ax.set_ylabel('Segment index [-]', fontsize=14)

    elif x_axis == "simulation_time" and y_axis == "segment_index":
        cax = ax.pcolormesh(simulation_time, range(num_segments), component_c_mole_fraction_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel('Simulation time [second]', fontsize=14)
        ax.set_ylabel('Segment index [-]', fontsize=14)

    elif x_axis == "time_step_index" and y_axis == "segment_depth":
        cax = ax.pcolormesh(range(len(time_steps)), true_vertical_depths_segments, component_c_mole_fraction_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

        # Add axes labels
        ax.set_xlabel('Time step [-]', fontsize=14)
        ax.set_ylabel('TVD [meter]', fontsize=14)

    elif x_axis == "simulation_time" and y_axis == "segment_depth":
        cax = ax.pcolormesh(simulation_time, true_vertical_depths_segments, component_c_mole_fraction_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

        # Add axes labels
        ax.set_xlabel('Simulation time [second]', fontsize=14)
        ax.set_ylabel('TVD [meter]', fontsize=14)

    if y_axis_convention == "standard":
        # Reverse the y-axis
        ax.invert_yaxis()

    # Add title
    ax.set_title(comp_name + " overall mole fraction profile along the wellbore over time", fontsize=14, fontweight='bold')

    # Add a colorbar
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label(comp_name + ' overall mole fraction [-]', fontsize=14)

    plt.tight_layout()
    plt.show()

# Load phase props
phase_props_df = pd.read_pickle('stored_phase_props.pkl')

num_segments = max(phase_props_df.index) + 1
num_ts = int(len(phase_props_df["sG"]) / num_segments)   # Initial conditions of sG is not stored.
simulation_time = np.delete(simulation_time, 0)

#%% Pressure profile

# Initialize the pressure matrix
p_matrix = np.zeros((num_segments, num_ts))

# Fill the pressure matrix
for ts_counter in range(num_ts):
    p = phase_props_df["Pressure"][ts_counter * num_segments:(ts_counter + 1) * num_segments]
    p_matrix[:, ts_counter] = p

# Initialize the plot
fig, ax = plt.subplots(figsize=(12, 6))


# Create the heatmap
cmap = plt.get_cmap('jet')
if x_axis == "time_step_index" and y_axis == "segment_index":
    cax = ax.pcolormesh(range(num_ts), range(num_segments), p_matrix, cmap=cmap, shading='auto')

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel('Time step [-]', fontsize=14)
    ax.set_ylabel('Segment index [-]', fontsize=14)

elif x_axis == "simulation_time" and y_axis == "segment_index":
    cax = ax.pcolormesh(simulation_time, range(num_segments), p_matrix, cmap=cmap, shading='auto')

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel('Simulation time [second]', fontsize=14)
    ax.set_ylabel('Segment index [-]', fontsize=14)

elif x_axis == "time_step_index" and y_axis == "segment_depth":
    cax = ax.pcolormesh(range(num_ts), true_vertical_depths_segments, p_matrix, cmap=cmap, shading='auto')

    # Add axes labels
    ax.set_xlabel('Time step [-]', fontsize=14)
    ax.set_ylabel('TVD [meter]', fontsize=14)

elif x_axis == "simulation_time" and y_axis == "segment_depth":
    cax = ax.pcolormesh(simulation_time, true_vertical_depths_segments, p_matrix, cmap=cmap, shading='auto')

    # Add axes labels
    ax.set_xlabel('Simulation time [second]', fontsize=14)
    ax.set_ylabel('TVD [meter]', fontsize=14)

if y_axis_convention == "standard":
    # Reverse the y-axis
    ax.invert_yaxis()


# Add title
ax.set_title('Pressure profile along the wellbore over time', fontsize=14, fontweight='bold')

# Add a colorbar to show the pressure values
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('Pressure [bar]', fontsize=14)

plt.tight_layout()
plt.show()


#%% Overall mole fraction profiles

# Initialize the composition matrix
z_matrix = np.zeros((num_segments, num_ts, num_components))

# Fill the composition matrix
for ts_counter in range(num_ts):
    z = phase_props_df["Overall mole fractions"][ts_counter * num_segments:(ts_counter + 1) * num_segments]
    z_matrix[:, ts_counter] = z

# Initialize the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Create the heatmap
cmap = plt.get_cmap('jet')
if x_axis == "time_step_index" and y_axis == "segment_index":
    cax = ax.pcolormesh(range(num_ts), range(num_segments), z_matrix, cmap=cmap, shading='auto')

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel('Time step [-]', fontsize=14)
    ax.set_ylabel('Segment index [-]', fontsize=14)

elif x_axis == "simulation_time" and y_axis == "segment_index":
    cax = ax.pcolormesh(simulation_time, range(num_segments), z_matrix, cmap=cmap, shading='auto')

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel('Simulation time [second]', fontsize=14)
    ax.set_ylabel('Segment index [-]', fontsize=14)

elif x_axis == "time_step_index" and y_axis == "segment_depth":
    cax = ax.pcolormesh(range(num_ts), true_vertical_depths_segments, z_matrix, cmap=cmap, shading='auto')

    # Add axes labels
    ax.set_xlabel('Time step [-]', fontsize=14)
    ax.set_ylabel('TVD [meter]', fontsize=14)

elif x_axis == "simulation_time" and y_axis == "segment_depth":
    cax = ax.pcolormesh(simulation_time, true_vertical_depths_segments, z_matrix, cmap=cmap, shading='auto')

    # Add axes labels
    ax.set_xlabel('Simulation time [second]', fontsize=14)
    ax.set_ylabel('TVD [meter]', fontsize=14)

if y_axis_convention == "standard":
    # Reverse the y-axis
    ax.invert_yaxis()


# Add title
ax.set_title('Temperature profile along the wellbore over time', fontsize=14, fontweight='bold')

# Add a colorbar to show the temperature values
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('Temperature [\u00B0C]', fontsize=14)

plt.tight_layout()
plt.show()


#%% Temperature profile

if thermal is True:
    # Initialize the temperature matrix
    T_matrix = np.zeros((num_segments, num_ts))

    # Fill the temperature matrix
    for ts_counter in range(num_ts):
        T = phase_props_df["Temperature"][ts_counter * num_segments:(ts_counter + 1) * num_segments]
        T_matrix[:, ts_counter] = T

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create the heatmap
    cmap = plt.get_cmap('jet')
    if x_axis == "time_step_index" and y_axis == "segment_index":
        cax = ax.pcolormesh(range(num_ts), range(num_segments), T_matrix, cmap=cmap, shading='auto')

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel('Time step [-]', fontsize=14)
        ax.set_ylabel('Segment index [-]', fontsize=14)

    elif x_axis == "simulation_time" and y_axis == "segment_index":
        cax = ax.pcolormesh(simulation_time, range(num_segments), T_matrix, cmap=cmap, shading='auto')

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel('Simulation time [second]', fontsize=14)
        ax.set_ylabel('Segment index [-]', fontsize=14)

    elif x_axis == "time_step_index" and y_axis == "segment_depth":
        cax = ax.pcolormesh(range(num_ts), true_vertical_depths_segments, T_matrix, cmap=cmap, shading='auto')

        # Add axes labels
        ax.set_xlabel('Time step [-]', fontsize=14)
        ax.set_ylabel('TVD [meter]', fontsize=14)

    elif x_axis == "simulation_time" and y_axis == "segment_depth":
        cax = ax.pcolormesh(simulation_time, true_vertical_depths_segments, T_matrix, cmap=cmap, shading='auto')

        # Add axes labels
        ax.set_xlabel('Simulation time [second]', fontsize=14)
        ax.set_ylabel('TVD [meter]', fontsize=14)

    if y_axis_convention == "standard":
        # Reverse the y-axis
        ax.invert_yaxis()


    # Add title
    ax.set_title('Temperature profile along the wellbore over time', fontsize=14, fontweight='bold')

    # Add a colorbar to show the temperature values
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Temperature [\u00B0C]', fontsize=14)

    plt.tight_layout()
    plt.show()

#%% Gas saturation profile

# Initialize the gas saturation matrix
sG_matrix = np.zeros((num_segments, num_ts))

# Fill the gas saturation matrix
for ts_counter in range(num_ts):
    sG = phase_props_df["sG"][ts_counter * num_segments:(ts_counter + 1) * num_segments]
    sG_matrix[:, ts_counter] = sG

# Initialize the plot
fig, ax = plt.subplots(figsize=(12, 6))


# Create the heatmap
cmap = plt.get_cmap('jet')
if x_axis == "time_step_index" and y_axis == "segment_index":
    cax = ax.pcolormesh(range(num_ts), range(num_segments), sG_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel('Time step [-]', fontsize=14)
    ax.set_ylabel('Segment index [-]', fontsize=14)

elif x_axis == "simulation_time" and y_axis == "segment_index":
    cax = ax.pcolormesh(simulation_time, range(num_segments), sG_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel('Simulation time [second]', fontsize=14)
    ax.set_ylabel('Segment index [-]', fontsize=14)

elif x_axis == "time_step_index" and y_axis == "segment_depth":
    cax = ax.pcolormesh(range(num_ts), true_vertical_depths_segments, sG_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

    # Add axes labels
    ax.set_xlabel('Time step [-]', fontsize=14)
    ax.set_ylabel('TVD [meter]', fontsize=14)

elif x_axis == "simulation_time" and y_axis == "segment_depth":
    cax = ax.pcolormesh(simulation_time, true_vertical_depths_segments, sG_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

    # Add axes labels
    ax.set_xlabel('Simulation time [second]', fontsize=14)
    ax.set_ylabel('TVD [meter]', fontsize=14)

if y_axis_convention == "standard":
    # Reverse the y-axis
    ax.invert_yaxis()


# Add title
ax.set_title('Gas saturation profile along the wellbore over time', fontsize=14, fontweight='bold')

# Add a colorbar to show the gas saturation values
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('Gas saturation [-]', fontsize=14)

plt.tight_layout()
plt.show()


#%% Profile/profiles of components mole fractions in the gaseous phase

for c, comp_name in enumerate(components_names):
    # Initialize the xG_mole_c matrix
    xG_mole_c_matrix = np.zeros((num_segments, num_ts))

    # Fill the xG_mole_c matrix
    for ts_counter in range(num_ts):
        xG_mass = phase_props_df["xG_mass"][ts_counter * num_segments:(ts_counter + 1) * num_segments]
        xG_mass_c = np.array([x[c] for x in xG_mass])
        total_moles_in_gas_per_segment = [sum(xG_mass[segment][comp_index] / components_molecular_weights[component_name]
                                          for comp_index, component_name in enumerate(components_names))
                                          for segment in range(num_segments)]
        with np.errstate(invalid='ignore'):   # This suppresses the error for division by zeros in the following line
            xG_mole_c = (xG_mass_c / components_molecular_weights[comp_name]) / total_moles_in_gas_per_segment
        xG_mole_c_matrix[:, ts_counter] = xG_mole_c

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create the heatmap
    cmap = plt.get_cmap('jet')
    if x_axis == "time_step_index" and y_axis == "segment_index":
        cax = ax.pcolormesh(range(num_ts), range(num_segments), xG_mole_c_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel('Time step [-]', fontsize=14)
        ax.set_ylabel('Segment index [-]', fontsize=14)

    elif x_axis == "simulation_time" and y_axis == 'segment_index':
        cax = ax.pcolormesh(simulation_time, range(num_segments), xG_mole_c_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel('Simulation time [second]', fontsize=14)
        ax.set_ylabel('Segment index [-]', fontsize=14)

    elif x_axis == "time_step_index" and y_axis == 'segment_depth':
        cax = ax.pcolormesh(range(num_ts), true_vertical_depths_segments, xG_mole_c_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

        # Add axes labels
        ax.set_xlabel('Time step [-]', fontsize=14)
        ax.set_ylabel('TVD [meter]', fontsize=14)

    elif x_axis == "simulation_time" and y_axis == 'segment_depth':
        cax = ax.pcolormesh(simulation_time, true_vertical_depths_segments, xG_mole_c_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

        # Add axes labels
        ax.set_xlabel('Simulation time [second]', fontsize=14)
        ax.set_ylabel('TVD [meter]', fontsize=14)


    if y_axis_convention == "standard":
        # Reverse the y-axis
        ax.invert_yaxis()

    # Add title
    ax.set_title('Profile of ' + comp_name + ' mole fraction in the gaseous phase along the wellbore over time', fontsize=14, fontweight='bold')

    # Add a colorbar to show the xG_mole_CO2 values
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label(comp_name + ' mole fraction in the gaseous phase [-]', fontsize=14)

    plt.tight_layout()
    plt.show()


#%% Profile/profiles of components mole fractions in the liquid phase

for c, comp_name in enumerate(components_names):
    # Initialize the xL_mole_c matrix
    xL_mole_c_matrix = np.zeros((num_segments, num_ts))

    # Fill the xL_mole_c matrix
    for ts_counter in range(num_ts):
        xL_mass = phase_props_df["xL_mass"][ts_counter * num_segments:(ts_counter + 1) * num_segments]
        xL_mass_c = np.array([x[c] for x in xL_mass])
        total_moles_in_liquid_per_segment = [sum(xL_mass[segment][comp_index] / components_molecular_weights[component_name]
                                             for comp_index, component_name in enumerate(components_names))
                                             for segment in range(num_segments)]
        with np.errstate(invalid='ignore'):  # This suppresses the error for division by zeros in the following line
            xL_mole_c = (xL_mass_c / components_molecular_weights[comp_name]) / total_moles_in_liquid_per_segment
        xL_mole_c_matrix[:, ts_counter] = xL_mole_c

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create the heatmap
    cmap = plt.get_cmap('jet')
    if x_axis == "time_step_index" and y_axis == "segment_index":
        cax = ax.pcolormesh(range(num_ts), range(num_segments), xL_mole_c_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel('Time step [-]', fontsize=14)
        ax.set_ylabel('Segment index [-]', fontsize=14)

    elif x_axis == "simulation_time" and y_axis == 'segment_index':
        cax = ax.pcolormesh(simulation_time, range(num_segments), xL_mole_c_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add axes labels
        ax.set_xlabel('Simulation time [second]', fontsize=14)
        ax.set_ylabel('Segment index [-]', fontsize=14)

    elif x_axis == "time_step_index" and y_axis == 'segment_depth':
        cax = ax.pcolormesh(range(num_ts), true_vertical_depths_segments, xL_mole_c_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

        # Add axes labels
        ax.set_xlabel('Time step [-]', fontsize=14)
        ax.set_ylabel('TVD [meter]', fontsize=14)

    elif x_axis == "simulation_time" and y_axis == 'segment_depth':
        cax = ax.pcolormesh(simulation_time, true_vertical_depths_segments, xL_mole_c_matrix, cmap=cmap, shading='auto', vmin=0, vmax=1)

        # Add axes labels
        ax.set_xlabel('Simulation time [second]', fontsize=14)
        ax.set_ylabel('TVD [meter]', fontsize=14)


    if y_axis_convention == "standard":
        # Reverse the y-axis
        ax.invert_yaxis()

    # Add title
    ax.set_title('Profile of ' + comp_name + ' mole fraction in the liquid phase along the wellbore over time', fontsize=14, fontweight='bold')

    # Add a colorbar to show the xG_mole_CO2 values
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label(comp_name + ' mole fraction in the liquid phase [-]', fontsize=14)

    plt.tight_layout()
    plt.show()


#%% Gas density profile

# Initialize the gas density matrix
rhoG_matrix = np.zeros((num_segments, num_ts))

# Fill the gas density matrix
for ts_counter in range(num_ts):
    rhoG = phase_props_df["rhoG"][ts_counter * num_segments:(ts_counter + 1) * num_segments]
    rhoG_matrix[:, ts_counter] = rhoG

# Apply a mask to hide values equal to or below a certain threshold
threshold = 0  # Set your threshold here
rhoG_matrix_masked = np.ma.masked_where(rhoG_matrix <= threshold, rhoG_matrix)

# Initialize the plot
fig, ax = plt.subplots(figsize=(12, 6))


# Create the heatmap
cmap = plt.get_cmap('jet')
if x_axis == "time_step_index" and y_axis == "segment_index":
    cax = ax.pcolormesh(range(num_ts), range(num_segments), rhoG_matrix_masked, cmap=cmap, shading='auto')

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel('Time step [-]', fontsize=14)
    ax.set_ylabel('Segment index [-]', fontsize=14)

elif x_axis == "simulation_time" and y_axis == "segment_index":
    cax = ax.pcolormesh(simulation_time, range(num_segments), rhoG_matrix_masked, cmap=cmap, shading='auto')

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel('Simulation time [second]', fontsize=14)
    ax.set_ylabel('Segment index [-]', fontsize=14)

elif x_axis == "time_step_index" and y_axis == "segment_depth":
    cax = ax.pcolormesh(range(num_ts), true_vertical_depths_segments, rhoG_matrix_masked, cmap=cmap, shading='auto')

    # Add axes labels
    ax.set_xlabel('Time step [-]', fontsize=14)
    ax.set_ylabel('TVD [meter]', fontsize=14)

elif x_axis == "simulation_time" and y_axis == "segment_depth":
    cax = ax.pcolormesh(simulation_time, true_vertical_depths_segments, rhoG_matrix_masked, cmap=cmap, shading='auto')

    # Add axes labels
    ax.set_xlabel('Simulation time [second]', fontsize=14)
    ax.set_ylabel('TVD [meter]', fontsize=14)

if y_axis_convention == "standard":
    # Reverse the y-axis
    ax.invert_yaxis()


# Add title
ax.set_title('Gas density profile along the wellbore over time', fontsize=14, fontweight='bold')

# Add a colorbar to show the gas density values
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('Gas density [kg/m$^3$]', fontsize=14)

plt.tight_layout()
plt.show()


#%% Liquid density profile

# Initialize the liquid density matrix
rhoL_matrix = np.zeros((num_segments, num_ts))

# Fill the liquid density matrix
for ts_counter in range(num_ts):
    rhoL = phase_props_df["rhoL"][ts_counter * num_segments:(ts_counter + 1) * num_segments]
    rhoL_matrix[:, ts_counter] = rhoL

# Apply a mask to hide values equal to or below a certain threshold
threshold = 0  # Set your threshold here
rhoL_matrix_masked = np.ma.masked_where(rhoL_matrix <= threshold, rhoL_matrix)

# Initialize the plot
fig, ax = plt.subplots(figsize=(12, 6))


# Create the heatmap
cmap = plt.get_cmap('jet')
if x_axis == "time_step_index" and y_axis == "segment_index":
    cax = ax.pcolormesh(range(num_ts), range(num_segments), rhoL_matrix_masked, cmap=cmap, shading='auto')

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel('Time step [-]', fontsize=14)
    ax.set_ylabel('Segment index [-]', fontsize=14)

elif x_axis == "simulation_time" and y_axis == "segment_index":
    cax = ax.pcolormesh(simulation_time, range(num_segments), rhoL_matrix_masked, cmap=cmap, shading='auto')

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel('Simulation time [second]', fontsize=14)
    ax.set_ylabel('Segment index [-]', fontsize=14)

elif x_axis == "time_step_index" and y_axis == "segment_depth":
    cax = ax.pcolormesh(range(num_ts), true_vertical_depths_segments, rhoL_matrix_masked, cmap=cmap, shading='auto')

    # Add axes labels
    ax.set_xlabel('Time step [-]', fontsize=14)
    ax.set_ylabel('TVD [meter]', fontsize=14)

elif x_axis == "simulation_time" and y_axis == "segment_depth":
    cax = ax.pcolormesh(simulation_time, true_vertical_depths_segments, rhoL_matrix_masked, cmap=cmap, shading='auto')

    # Add axes labels
    ax.set_xlabel('Simulation time [second]', fontsize=14)
    ax.set_ylabel('TVD [meter]', fontsize=14)

if y_axis_convention == "standard":
    # Reverse the y-axis
    ax.invert_yaxis()

# Add title
ax.set_title('Liquid density profile along the wellbore over time', fontsize=14, fontweight='bold')

# Add a colorbar to show the liquid density values
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('Liquid density [kg/m$^3$]', fontsize=14)

plt.tight_layout()
plt.show()

#%% Gas viscosity profile

# Initialize the gas viscosity matrix
miuG_matrix = np.zeros((num_segments, num_ts))

# Fill the liquid density matrix
for ts_counter in range(num_ts):
    miuG = phase_props_df["miuG"][ts_counter * num_segments:(ts_counter + 1) * num_segments]
    miuG_matrix[:, ts_counter] = miuG * 1e3

# Apply a mask to hide values equal to or below a certain threshold
threshold = 0  # Set your threshold here
miuG_matrix_masked = np.ma.masked_where(miuG_matrix <= threshold, miuG_matrix)

# Initialize the plot
fig, ax = plt.subplots(figsize=(12, 6))


# Create the heatmap
cmap = plt.get_cmap('jet')
if x_axis == "time_step_index" and y_axis == "segment_index":
    cax = ax.pcolormesh(range(num_ts), range(num_segments), miuG_matrix_masked, cmap=cmap, shading='auto')

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel('Time step [-]', fontsize=14)
    ax.set_ylabel('Segment index [-]', fontsize=14)

elif x_axis == "simulation_time" and y_axis == "segment_index":
    cax = ax.pcolormesh(simulation_time, range(num_segments), miuG_matrix_masked, cmap=cmap, shading='auto')

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel('Simulation time [second]', fontsize=14)
    ax.set_ylabel('Segment index [-]', fontsize=14)

elif x_axis == "time_step_index" and y_axis == "segment_depth":
    cax = ax.pcolormesh(range(num_ts), true_vertical_depths_segments, miuG_matrix_masked, cmap=cmap, shading='auto')

    # Add axes labels
    ax.set_xlabel('Time step [-]', fontsize=14)
    ax.set_ylabel('TVD [meter]', fontsize=14)

elif x_axis == "simulation_time" and y_axis == "segment_depth":
    cax = ax.pcolormesh(simulation_time, true_vertical_depths_segments, miuG_matrix_masked, cmap=cmap, shading='auto')

    # Add axes labels
    ax.set_xlabel('Simulation time [second]', fontsize=14)
    ax.set_ylabel('TVD [meter]', fontsize=14)

if y_axis_convention == "standard":
    # Reverse the y-axis
    ax.invert_yaxis()

# Add title
ax.set_title('Gas viscosity profile along the wellbore over time', fontsize=14, fontweight='bold')

# Add a colorbar to show the liquid density values
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('Gas viscosity [cP]', fontsize=14)

plt.tight_layout()
plt.show()

#%% Liquid viscosity profile

# Initialize the liquid viscosity matrix
miuL_matrix = np.zeros((num_segments, num_ts))

# Fill the liquid density matrix
for ts_counter in range(num_ts):
    miuL = phase_props_df["miuL"][ts_counter * num_segments:(ts_counter + 1) * num_segments]
    miuL_matrix[:, ts_counter] = miuL * 1e3

# Apply a mask to hide values equal to or below a certain threshold
threshold = 0  # Set your threshold here
miuL_matrix_masked = np.ma.masked_where(miuL_matrix <= threshold, miuL_matrix)

# Initialize the plot
fig, ax = plt.subplots(figsize=(12, 6))


# Create the heatmap
cmap = plt.get_cmap('jet')
if x_axis == "time_step_index" and y_axis == "segment_index":
    cax = ax.pcolormesh(range(num_ts), range(num_segments), miuL_matrix_masked, cmap=cmap, shading='auto')

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel('Time step [-]', fontsize=14)
    ax.set_ylabel('Segment index [-]', fontsize=14)

elif x_axis == "simulation_time" and y_axis == "segment_index":
    cax = ax.pcolormesh(simulation_time, range(num_segments), miuL_matrix_masked, cmap=cmap, shading='auto')

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel('Simulation time [second]', fontsize=14)
    ax.set_ylabel('Segment index [-]', fontsize=14)

elif x_axis == "time_step_index" and y_axis == "segment_depth":
    cax = ax.pcolormesh(range(num_ts), true_vertical_depths_segments, miuL_matrix_masked, cmap=cmap, shading='auto')

    # Add axes labels
    ax.set_xlabel('Time step [-]', fontsize=14)
    ax.set_ylabel('TVD [meter]', fontsize=14)

elif x_axis == "simulation_time" and y_axis == "segment_depth":
    cax = ax.pcolormesh(simulation_time, true_vertical_depths_segments, miuL_matrix_masked, cmap=cmap, shading='auto')

    # Add axes labels
    ax.set_xlabel('Simulation time [second]', fontsize=14)
    ax.set_ylabel('TVD [meter]', fontsize=14)

if y_axis_convention == "standard":
    # Reverse the y-axis
    ax.invert_yaxis()

# Add title
ax.set_title('Liquid viscosity profile along the wellbore over time', fontsize=14, fontweight='bold')

# Add a colorbar to show the liquid density values
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('Liquid viscosity [$cP$]', fontsize=14)

plt.tight_layout()
plt.show()


#%% Gas velocity profile

# Initialize the gas velocity matrix
num_interfaces = num_segments - 1
vG_matrix = np.zeros((num_interfaces, num_ts))

# Fill the gas velocity matrix
for ts_counter in range(num_ts):
    vG = phase_props_df["vG"][ts_counter * num_segments:(ts_counter + 1) * num_segments]
    vG_matrix[:, ts_counter] = vG[:-1]

# Apply a mask to hide some values
vG_matrix_masked = np.ma.masked_where(vG_matrix == 0, vG_matrix)

# Initialize the plot
fig, ax = plt.subplots(figsize=(12, 6))


# Create the heatmap
cmap = plt.get_cmap('jet')
if x_axis == "time_step_index" and y_axis == "segment_index":
    cax = ax.pcolormesh(range(num_ts), range(num_interfaces), vG_matrix_masked, cmap=cmap, shading='auto')

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel('Time step [-]', fontsize=14)
    ax.set_ylabel('Interface index [-]', fontsize=14)

elif x_axis == "simulation_time" and y_axis == "segment_index":
    cax = ax.pcolormesh(simulation_time, range(num_interfaces), vG_matrix_masked, cmap=cmap, shading='auto')

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel('Simulation time [second]', fontsize=14)
    ax.set_ylabel('Interface index [-]', fontsize=14)

elif x_axis == "time_step_index" and y_axis == "segment_depth":
    cax = ax.pcolormesh(range(num_ts), true_vertical_depths_interfaces, vG_matrix_masked, cmap=cmap, shading='auto')

    # Add axes labels
    ax.set_xlabel('Time step [-]', fontsize=14)
    ax.set_ylabel('TVD [meter]', fontsize=14)

elif x_axis == "simulation_time" and y_axis == "segment_depth":
    cax = ax.pcolormesh(simulation_time, true_vertical_depths_interfaces, vG_matrix_masked, cmap=cmap, shading='auto')

    # Add axes labels
    ax.set_xlabel('Simulation time [second]', fontsize=14)
    ax.set_ylabel('TVD [meter]', fontsize=14)

if y_axis_convention == "standard":
    # Reverse the y-axis
    ax.invert_yaxis()


# Add title
ax.set_title('Gas velocity profile along the wellbore over time', fontsize=14, fontweight='bold')

# Add a colorbar to show the gas velocity values
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('Gas velocity [m/s]', fontsize=14)

plt.tight_layout()
plt.show()


#%% Liquid velocity profile

# Initialize the gas velocity matrix
num_interfaces = num_segments - 1
vL_matrix = np.zeros((num_interfaces, num_ts))

# Fill the liquid velocity matrix
for ts_counter in range(num_ts):
    vL = phase_props_df["vL"][ts_counter * num_segments:(ts_counter + 1) * num_segments]
    vL_matrix[:, ts_counter] = vL[:-1]

# Apply a mask to hide some values
vL_matrix_masked = np.ma.masked_where(vL_matrix == 0, vL_matrix)

# Initialize the plot
fig, ax = plt.subplots(figsize=(12, 6))


# Create the heatmap
cmap = plt.get_cmap('jet')
if x_axis == "time_step_index" and y_axis == "segment_index":
    cax = ax.pcolormesh(range(num_ts), range(num_interfaces), vL_matrix_masked, cmap=cmap, shading='auto')

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel('Time step [-]', fontsize=14)
    ax.set_ylabel('Interface index [-]', fontsize=14)

elif x_axis == "simulation_time" and y_axis == "segment_index":
    cax = ax.pcolormesh(simulation_time, range(num_interfaces), vL_matrix_masked, cmap=cmap, shading='auto')

    # Set the y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Add axes labels
    ax.set_xlabel('Simulation time [second]', fontsize=14)
    ax.set_ylabel('Interface index [-]', fontsize=14)

elif x_axis == "time_step_index" and y_axis == "segment_depth":
    cax = ax.pcolormesh(range(num_ts), true_vertical_depths_interfaces, vL_matrix_masked, cmap=cmap, shading='auto')

    # Add axes labels
    ax.set_xlabel('Time step [-]', fontsize=14)
    ax.set_ylabel('TVD [meter]', fontsize=14)

elif x_axis == "simulation_time" and y_axis == "segment_depth":
    cax = ax.pcolormesh(simulation_time, true_vertical_depths_interfaces, vL_matrix_masked, cmap=cmap, shading='auto')

    # Add axes labels
    ax.set_xlabel('Simulation time [second]', fontsize=14)
    ax.set_ylabel('TVD [meter]', fontsize=14)

if y_axis_convention == "standard":
    # Reverse the y-axis
    ax.invert_yaxis()


# Add title
ax.set_title('Liquid velocity profile along the wellbore over time', fontsize=14, fontweight='bold')

# Add a colorbar to show the liquid velocity values
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('Liquid velocity [m/s]', fontsize=14)

plt.tight_layout()
plt.show()
