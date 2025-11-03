"""
This example shows the effect of potential energy on the results for a scenario involving a vertical 1-km porous
medium which is initially saturated with water and CO2 is injected into its top grid cell at a constant mass rate.
The bottom cell is considered as an infinitely large grid cell.
"""

import matplotlib.pyplot as plt

from model import Model

""" 1st simulation: Without potential energy """

m = Model()

m.reservoir.grav_acc_for_spe = 0.

m.init()
m.set_output()

print_props = m.physics.vars + ['sat_CO2_rich_phase', 'xCO2', 'yH2O']

m.run(200)

_, output = m.output.output_properties(output_properties=print_props, timestep=1)
temp_profile_without_pe = output['temperature'].T
pressure_profile_without_pe = output['pressure'].T
H2O_mole_frac_profile_without_pe = output['H2O'].T


""" 2nd simulation: With potential energy """

m = Model()

m.reservoir.grav_acc_for_spe = 9.80665

m.init()
m.set_output()

m.run(200)

_, output = m.output.output_properties(output_properties=print_props, timestep=1)
temp_profile_with_pe = output['temperature'].T
pressure_profile_with_pe = output['pressure'].T
H2O_mole_frac_profile_with_pe = output['H2O'].T

y = m.reservoir.global_data['depth']

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
