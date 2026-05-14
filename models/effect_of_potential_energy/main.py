"""
This example shows the effect of potential energy on the results for a scenario involving a vertical 1-km porous
medium which is initially saturated with water and CO2 is injected into its top grid cell at a constant mass rate.
The bottom cell is considered as an infinitely large grid cell.
"""

import matplotlib.pyplot as plt
import os

from model import Model

simulation_time = 50

""" 1st simulation: Without potential energy """

m = Model()

m.reservoir.grav_acceleration_for_spe = 0.

m.init()
m.set_output()

output_props = m.physics.vars + ['sat_CO2_rich_phase', 'xCO2', 'yH2O']

m.run(simulation_time)

_, output = m.output.output_properties(output_properties=output_props, ts_idx=1)
linear_interval_num_cells = len(m.linear_inverval)   # Number of cells of the linear inverval considered as the reservoir
temp_profile_without_pe = output['temperature'].T[:linear_interval_num_cells]
pressure_profile_without_pe = output['pressure'].T[:linear_interval_num_cells]
H2O_mole_frac_profile_without_pe = output['H2O'].T[:linear_interval_num_cells]


""" 2nd simulation: With potential energy """

m = Model()

# m.reservoir.grav_acceleration_for_spe = 9.80665   # This is specified in the constructor of DartsModel.

m.init()
m.set_output()

m.run(simulation_time)

_, output = m.output.output_properties(output_properties=output_props, ts_idx=1)
temp_profile_with_pe = output['temperature'].T[:linear_interval_num_cells]
pressure_profile_with_pe = output['pressure'].T[:linear_interval_num_cells]
H2O_mole_frac_profile_with_pe = output['H2O'].T[:linear_interval_num_cells]

y = m.reservoir.global_data['depth'][:linear_interval_num_cells]

""" Compare the property profiles """
""" Compare the temperature profiles """
fig, ax = plt.subplots()
ax.plot(temp_profile_without_pe - 273.15, y, color='red', marker='o', label='Without potential energy')
ax.plot(temp_profile_with_pe - 273.15, y, color='blue', marker='*', label='With potential energy')

ax.invert_yaxis()
ax.set_xlabel('Temperature [degree C]')
ax.set_ylabel('Reservoir cell depth [meter]')

ax.set_yticks(y)

ax.legend()
ax.grid()
fig.tight_layout()
fig.savefig(os.path.join(m.output.output_folder, 'figures', 'temperature_profiles.png'))
plt.close(fig)

""" Compare the pressure profiles """
fig, ax = plt.subplots()
ax.plot(pressure_profile_without_pe, y, color='red', marker='o', label='Without potential energy')
ax.plot(pressure_profile_with_pe, y, color='blue', marker='*', label='With potential energy')

ax.invert_yaxis()
ax.set_xlabel('Pressure [bar]')
ax.set_ylabel('Reservoir cell depth [meter]')

ax.set_yticks(y)

ax.legend()
ax.grid()
fig.tight_layout()
fig.savefig(os.path.join(m.output.output_folder, 'figures', 'pressure_profiles.png'))
plt.close(fig)

""" Compare the overall mole fraction profiles """
fig, ax = plt.subplots()
ax.plot(H2O_mole_frac_profile_without_pe, y, color='red', marker='o', label='Without potential energy')
ax.plot(H2O_mole_frac_profile_with_pe, y, color='blue', marker='*', label='With potential energy')

ax.invert_yaxis()
ax.set_xlabel('H$_2$O overall mole fraction [-]')
ax.set_ylabel('Reservoir cell depth [meter]')

ax.set_yticks(y)

ax.legend()
ax.grid()
fig.tight_layout()
fig.savefig(os.path.join(m.output.output_folder, 'figures', 'h2o_overall_mole_fraction_profiles.png'))
plt.close(fig)
