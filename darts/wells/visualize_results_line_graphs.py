# TODO: This file is copied from the standalone well simulator and it should be adjusted to the coupled model.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import pickle

from units import *
from library import components_molecular_weights

#%% Load primary variables from the pickle file
primary_variables_df = pd.read_pickle('stored_primary_variables.pkl')

time_steps = primary_variables_df.columns

selected_time_steps_for_primary_vars = time_steps[0::5]   # Plot every 5 time steps

# Create a colormap
cmap = plt.colormaps.get_cmap('jet')  # You can use other colormaps like 'plasma', 'inferno', etc.
num_lines = len(selected_time_steps_for_primary_vars)  # Number of time steps you are plotting
colors = cmap(np.linspace(0, 1, num_lines))  # Create a color gradient

#%% Load phase props from the pickle file
# Load phase props
phase_props_df = pd.read_pickle('stored_phase_props.pkl')

num_segments = max(phase_props_df.index) + 1
num_ts = int(len(phase_props_df["sG"]) / num_segments)   # Initial conditions of phase props, including sG, are not stored.

selected_time_steps_for_phase_props = range(0, num_ts, 5)   # Plot every 5 time steps

#%% Load components names from the pickle file
with open('stored_dt_pipe_geometry_other_info.pkl', 'rb') as file:
    _, _, components_names = pickle.load(file)


#%% Pressure profile


# Initialize the plot
plt.figure(figsize=(12, 6))

# Loop through each time step and plot pressure values
# for time_step in [time_steps[-1]]:   # Plot only the last time step
for i, time_step in enumerate(selected_time_steps_for_primary_vars):
    # Filter the data for the pressure profile of the current time step
    pressure_profile = primary_variables_df[time_step][0:num_segments] / bar()

    # Plot the values
    # plt.plot(pressure_profile, list(range(num_segments)), label=time_step)
    plt.plot(pressure_profile, list(range(num_segments)), color=colors[i])   # without legend

# Reverse the y-axis
# plt.gca().invert_yaxis()

# Set the y-axis ticks
plt.gca().yaxis.set_major_locator(MultipleLocator(1))
# plt.gca().xaxis.set_major_locator(MultipleLocator(0.01))

# Set x-axis limits
# plt.xlim(4, 12)

# Add labels and legend
plt.xlabel('Pressure [bar]', fontsize=14)
plt.ylabel('Segment index', fontsize=14)
# plt.title('Pressure profile along the wellbore for different time steps')
plt.title('Pressure profile/profiles along the wellbore', fontsize=14, fontweight='bold')
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
# plt.tight_layout(rect=[0, 0, 0.99, 1])  # Adjust the size of the axes to make space for the legend
# plt.legend(loc='upper right')

plt.tight_layout()
plt.show()


#%% Component/components overall mole fraction profile


for c, comp_name in enumerate(components_names[:-1]):

    # Initialize the plot
    plt.figure(figsize=(12, 6))

    # Loop through each time step and plot overall mole fraction values of component c
    # for time_step in [time_steps[-1]]:   # Plot only the last time step
    for i, time_step in enumerate(selected_time_steps_for_primary_vars):
        # Filter the data for the overall mole fraction profile of component c of the current time step
        component_c_mole_fraction_profile = primary_variables_df[time_step][num_segments * (c + 1):num_segments * (c + 2)]

        # Plot the values
        # plt.plot(CO2_mole_fraction_profile, list(range(num_segments)), label=time_step)
        plt.plot(component_c_mole_fraction_profile, list(range(num_segments)), color=colors[i])   # without legend

    # Reverse the y-axis
    # plt.gca().invert_yaxis()

    # Set the y-axis ticks
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))
    # Set the x-axis ticks
    # plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))

    # Set x-axis limits
    # plt.xlim(4, 12)

    # Add labels and legend
    plt.xlabel(comp_name + ' overall mole fraction [-]', fontsize=14)
    plt.ylabel('Segment index', fontsize=14)
    plt.title(comp_name + ' overall mole fraction profile/profiles along the wellbore', fontsize=14, fontweight='bold')
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
    # plt.tight_layout(rect=[0, 0, 0.99, 1])  # Adjust the size of the axes to make space for the legend
    # plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


#%% Temperature profile


# Temperature profile is plotted if the system is non-isothermal.
if len(primary_variables_df['Initial conditions'])/num_segments > len(components_names):
    # Initialize the plot
    plt.figure(figsize=(12, 6))

    # Loop through each time step and plot temperature values
    # for time_step in [time_steps[-1]]:   # Plot only the last time step
    for i, time_step in enumerate(selected_time_steps_for_primary_vars):
        # Filter the data for the temperature profile of the current time step
        temperature_profile = primary_variables_df[time_step][num_segments * len(components_names):] - 273.15

        # Plot the values
        # plt.plot(temperature_profile, list(range(num_segments)), label=time_step)
        plt.plot(temperature_profile, list(range(num_segments)), color=colors[i])   # without legend

    # Reverse the y-axis
    # plt.gca().invert_yaxis()

    # Set the y-axis ticks
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))
    # plt.gca().xaxis.set_major_locator(MultipleLocator(0.01))

    # Set x-axis limits
    # plt.xlim(4, 12)

    # Add labels and legend
    plt.xlabel('Temperature [\u00B0C]', fontsize=14)
    plt.ylabel('Segment index', fontsize=14)
    # plt.title('Temperature profile along the wellbore for different time steps')
    plt.title('Temperature profile/profiles along the wellbore', fontsize=14, fontweight='bold')
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
    # plt.tight_layout(rect=[0, 0, 0.99, 1])  # Adjust the size of the axes to make space for the legend
    # plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


#%% Gas saturation profile


# Initialize the plot
plt.figure(figsize=(12, 6))

for i, time_step in enumerate(selected_time_steps_for_phase_props):
    sG = phase_props_df["sG"][time_step * num_segments:(time_step + 1) * num_segments]
    plt.plot(sG, list(range(num_segments)), color=colors[i])

# Set the y-axis ticks
plt.gca().yaxis.set_major_locator(MultipleLocator(1))

plt.xlabel('Gas saturation [-]', fontsize=14)
plt.ylabel('Segment index', fontsize=14)

plt.title('Gas saturation profile/profiles along the wellbore', fontsize=14, fontweight='bold')

plt.show()


#%% Profile/profiles of components mole fractions in the gaseous phase


for c, comp_name in enumerate(components_names[:-1]):
    # Initialize the plot
    plt.figure(figsize=(12, 6))

    for i, time_step in enumerate(selected_time_steps_for_phase_props):
        xG_mass = phase_props_df["xG_mass"][time_step * num_segments:(time_step + 1) * num_segments]
        xG_mass_c = np.array([x[c] for x in xG_mass])
        total_moles_in_gas_per_segment = [sum(xG_mass[segment][comp_index] / components_molecular_weights[component_name]
                                          for comp_index, component_name in enumerate(components_names))
                                          for segment in range(num_segments)]
        with np.errstate(invalid='ignore'):  # This suppresses the error for division by zeros in the following line
            xG_mole_c = (xG_mass_c / components_molecular_weights[comp_name]) / total_moles_in_gas_per_segment
        plt.plot(xG_mole_c, list(range(num_segments)), color=colors[i])

    # Set the y-axis ticks
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))

    plt.xlabel(comp_name + ' mole fraction in the gaseous phase [-]', fontsize=14)
    plt.ylabel('Segment index', fontsize=14)

    plt.title(comp_name + ' mole fraction in the gaseous phase profile/profiles along the wellbore', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()


#%% Profile/profiles of components mole fractions in the liquid phase


for c, comp_name in enumerate(components_names[:-1]):
    # Initialize the plot
    plt.figure(figsize=(12, 6))

    for i, time_step in enumerate(selected_time_steps_for_phase_props):
        xL_mass = phase_props_df["xL_mass"][time_step * num_segments:(time_step + 1) * num_segments]
        xL_mass_c = np.array([x[c] for x in xL_mass])
        total_moles_in_liquid_per_segment = [sum(xL_mass[segment][comp_index] / components_molecular_weights[component_name]
                                             for comp_index, component_name in enumerate(components_names))
                                             for segment in range(num_segments)]
        with np.errstate(invalid='ignore'):  # This suppresses the error for division by zeros in the following line
            xL_mole_c = (xL_mass_c / components_molecular_weights[comp_name]) / total_moles_in_liquid_per_segment
        plt.plot(xL_mole_c, list(range(num_segments)), color=colors[i])

    # Set the y-axis ticks
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))

    plt.xlabel(comp_name + ' mole fraction in the liquid phase [-]', fontsize=14)
    plt.ylabel('Segment index', fontsize=14)

    plt.title(comp_name + ' mole fraction in the liquid phase profile/profiles along the wellbore', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()


#%% Gas density profile


# Initialize the plot
plt.figure(figsize=(12, 6))

# Fill the gas density matrix
for i, time_step in enumerate(selected_time_steps_for_phase_props):
    rhoG = phase_props_df["rhoG"][time_step * num_segments:(time_step + 1) * num_segments]
    plt.plot(rhoG, list(range(num_segments)), color=colors[i])

# Set the y-axis ticks
plt.gca().yaxis.set_major_locator(MultipleLocator(1))

plt.xlabel('Gas density [kg/m$^3$]', fontsize=14)
plt.ylabel('Segment index', fontsize=14)

plt.title('Gas density profile/profiles along the wellbore', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()


#%% Liquid density profile


# Initialize the plot
plt.figure(figsize=(12, 6))

# Fill the liquid density matrix
for i, time_step in enumerate(selected_time_steps_for_phase_props):
    rhoL = phase_props_df["rhoL"][time_step * num_segments:(time_step + 1) * num_segments]
    plt.plot(rhoL, list(range(num_segments)), color=colors[i])

# Set the y-axis ticks
plt.gca().yaxis.set_major_locator(MultipleLocator(1))

plt.xlabel('Liquid density [kg/m$^3$]', fontsize=14)
plt.ylabel('Segment index', fontsize=14)

plt.title('Liquid density profile/profiles along the wellbore', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
