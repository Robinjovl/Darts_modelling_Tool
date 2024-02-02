import os

# add path to import
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)  # 1 level up
parent_dir_2 = os.path.dirname(parent_dir)  # 2 levels up
balmatt_dir = os.path.join(os.path.join(parent_dir_2, 'darts_DFN'), 'examples')
sys.path.insert(0, balmatt_dir)

from .input_default import *

input_data['case_name'] = 'case_2'

# geometry for mesh generation
input_data['margin'] = 1 # [m]
input_data['frac_file'] = os.path.join('examples', 'frac_2.txt')

# cell sizes for mesh generation
input_data['char_len'] = 5  # near fractures (characteristic length for cleaning and mesh generation) [m]
input_data['char_len_boundary'] = 5  # grid size near grid boundaries [m]

# do not use cleaned mesh
input_data['mesh_prefix'] = 'raw_lc'

# well locations
input_data['inj_well_coords'] = [[1, 1, 25]]  # X, Y, Z (only one perforation)
input_data['prod_well_coords'] = [[99, 99, 25]]

# well in the matrix cells or in the fractures
input_data['well_loc_type'] = 'wells_in_frac'

# well controls
input_data['rate_prod'] = None#10  # m3/day
input_data['rate_inj'] = None#5  # m3/day
input_data['delta_temp'] = 10  # inj_temp = initial_temp - delta_temp
input_data['delta_p'] = 5  # inj_bhp = initial_pressure + delta_p