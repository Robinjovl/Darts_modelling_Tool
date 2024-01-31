import os
from .input_default import *

input_data['case_name'] = 'case_1'

# geometry
input_data['frac_file'] = os.path.join('examples', 'frac_1.txt')

# initial pressure and temperature
# uniform initial pressure and temperature
input_data['initial_uniform'] = True
input_data['uniform_pressure'] = 350.  # bar
input_data['uniform_temperature'] = 348.15  # K

# extrusion - number of layers by Z axis
#input_data['rsv_layers'] = 4

# overburden layers (with fractures)
#input_data['overburden_thickness'] = input_data['height_res'] * 5
#input_data['overburden_layers'] = 1
#input_data['underburden_thickness'] = input_data['height_res'] * 5
#input_data['underburden_layers'] = 1

# second overburden layers (without fractures)
#input_data['overburden_2_thickness'] = input_data['height_res'] * 5
#input_data['overburden_2_layers'] = 1
#input_data['underburden_2_thickness'] = input_data['height_res'] * 5
#input_data['underburden_2_layers'] = 1

