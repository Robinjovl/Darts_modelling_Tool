"""
This example shows the effect of potential energy on the results for a scenario involving a vertical 1-km porous
medium which is initially saturated with water and CO2 is injected into its top grid cell at a constant mass rate.
The bottom cell is considered as an infinitely large grid cell.
"""

import matplotlib.pyplot as plt

from model import Model

simulation_time = 50

""" 1st simulation: Without potential energy """

m = Model()

m.reservoir.grav_acceleration_for_spe = 0.

m.init()
m.set_output()

print_props = m.physics.vars + ['sat_CO2_rich_phase', 'xCO2', 'yH2O']

m.run(simulation_time)

_, output = m.output.output_properties(output_properties=print_props, timestep=1)
linear_interval_num_cells = len(m.linear_inverval)   # Number of cells of the linear inverval considered as the reservoir
temp_profile_without_pe = output['temperature'].T[:linear_interval_num_cells]
pressure_profile_without_pe = output['pressure'].T[:linear_interval_num_cells]
H2O_mole_frac_profile_without_pe = output['H2O'].T[:linear_interval_num_cells]


""" 2nd simulation: With potential energy """

m = Model()

m.reservoir.grav_acceleration_for_spe = 9.80665

m.init()
m.set_output()

m.run(simulation_time)

_, output = m.output.output_properties(output_properties=print_props, timestep=1)
temp_profile_with_pe = output['temperature'].T[:linear_interval_num_cells]
pressure_profile_with_pe = output['pressure'].T[:linear_interval_num_cells]
H2O_mole_frac_profile_with_pe = output['H2O'].T[:linear_interval_num_cells]

y = m.reservoir.global_data['depth'][:linear_interval_num_cells]

""" Compare the temperature profiles """
plt.plot(temp_profile_without_pe - 273.15, y, color='red', marker='o', label='Without potential energy')
plt.plot(temp_profile_with_pe - 273.15, y, color='blue', marker='*', label='With potential energy')

plt.gca().invert_yaxis()
plt.xlabel('Temperature [degree C]')
plt.ylabel('Reservoir cell depth [meter]')

plt.yticks(y)

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

""" Compare the pressure profiles """
plt.plot(pressure_profile_without_pe, y, color='red', marker='o', label='Without potential energy')
plt.plot(pressure_profile_with_pe, y, color='blue', marker='*', label='With potential energy')

plt.gca().invert_yaxis()
plt.xlabel('Pressure [bar]')
plt.ylabel('Reservoir cell depth [meter]')

plt.yticks(y)

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

""" Compare the overall mole fraction profiles """
plt.plot(H2O_mole_frac_profile_without_pe, y, color='red', marker='o', label='Without potential energy')
plt.plot(H2O_mole_frac_profile_with_pe, y, color='blue', marker='*', label='With potential energy')

plt.gca().invert_yaxis()
plt.xlabel('H$_2$O overall mole fraction [-]')
plt.ylabel('Reservoir cell depth [meter]')

plt.yticks(y)

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
