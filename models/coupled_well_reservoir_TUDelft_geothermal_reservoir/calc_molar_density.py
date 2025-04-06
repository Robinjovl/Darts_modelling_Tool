from dartsflash.libflash import CubicEoS, AQEoS
from dartsflash.components import CompData

from darts.physics.properties.eos_properties import EoSDensity
from darts.physics.properties.density import Garcia2001

components_names = ['CO2', 'C1', 'H2O']
comp_data = CompData(components_names, setprops=True)

ceos = CubicEoS(comp_data, CubicEoS.PR)

density_ev = dict([('gas', EoSDensity(ceos, comp_data.Mw)),
                   ('aqueous', Garcia2001(components_names))])

print(density_ev['gas'].evaluate(15, 298.15, [1e-10, 1 - 2 * 1e-10, 1e-10]))
print(density_ev['aqueous'].evaluate(15, 298.15, [1e-10, 1e-10, 1 - 2 * 1e-10]))
