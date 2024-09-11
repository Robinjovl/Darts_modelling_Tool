
def input_data_default():
    input_data = dict()

    ###########################################################################################################
    # DFN framework parameters (for mesh generation)
    input_data['frac_file'] = 'frac.txt'  # fracture tips coordinates X1 Y1 X2 Z2; should contain at least 2 rows (2 fractures)
    #input_data['mesh_prefix'] = 'raw_lc'  #  use mesh with original fracture tips
    input_data['mesh_prefix'] = 'mergefac_0.86_clean_lc'  #  cleaned mesh
    input_data['mesh_clean'] = False  # need gmsh installed and callable from command line in order to mesh

    input_data['margin'] = 100  # [m]
    input_data['box_data'] = None  # [m] mesh bounds (in case of no margin defined)

    # cell sizes
    input_data['char_len'] = 16  # near fractures (characteristic length for cleaning and mesh generation) [m]
    input_data['char_len_boundary'] = 16  # grid size near grid boundaries [m]
    input_data['char_len_well'] = 16  # grid size near wells [m]

    # geometry (both for DFN and model)
    input_data['z_top'] = 0  # [m]
    input_data['height_res'] = 50  # [m]

    # extrusion - number of layers by Z axis
    input_data['rsv_layers'] = 1

    # no overburden layers (fractured) by default
    input_data['overburden_thickness'] = 0
    input_data['overburden_layers'] = 0
    input_data['underburden_thickness'] = 0
    input_data['underburden_layers'] = 0

    # no second overburden layers (without fractures) by default
    input_data['overburden_2_thickness'] = 0
    input_data['overburden_2_layers'] = 0
    input_data['underburden_2_thickness'] = 0
    input_data['underburden_2_layers'] = 0

    # well locations
    input_data['inj_well_coords'] = [[50, 50, 25]]  # X, Y, Z (only one perforation)
    input_data['prod_well_coords'] = [[950, 950, 25]]

    # The properties below do not affect mesh generation stage. So no need to re-generate the mesh if you change them.

    input_data['poro'] = 0.2
    input_data['permx'] = 10  # [mD]
    input_data['permy'] = 10  # [mD]
    input_data['permz'] = 1  # [mD]
    input_data['perm_file'] = None  # if want to read the permeability from netCDF file

    input_data['rock_compressibility'] = 1e-5  # [1/bar]

    # will be passed to UnstructuredDiscretizer
    input_data['frac_aper'] = 1e-3  # (initial) fracture aperture [m]

    # well in the matrix cells or in the fractures
    input_data['well_loc_type'] = 'wells_in_nearest_cell'  # could be in the matrix or in the fracture, depending on the location
    #input_data['well_loc_type'] = 'wells_in_frac'  # put the well into the closest fracture
    #input_data['well_loc_type'] = 'wells_in_mat'  # put the well into the closest matrix cell

    input_data['hcap'] = 2200. # [kJ/m3/K]
    input_data['conduction'] = 181.44  # [kJ/m/day/K]

    # well controls
    input_data['rate_prod'] = None  # m3/day. if None, well will work under BHP control
    input_data['rate_inj'] = None   # m3/day. if None, well will work under BHP control
    input_data['delta_temp'] = 10   # bars. inj_temp = initial_temp - delta_temp
    input_data['delta_p_inj']  = 5  # bars. inj_bhp = initial_pressure + delta_p_inj
    input_data['delta_p_prod'] = 5  # bars. inj_prod = initial_pressure - delta_p_prod

    # principal stress, MPa.
    # Set to None if don't want to recompute fracture apertures by initial stresses
    input_data['Sh_min'] = None #50
    input_data['Sh_max'] = None # 90
    input_data['Sv'] = None # 120
    input_data['SHmax_azimuth'] = None #0  # [°] from X, counter-clockwise
    input_data['sigma_c'] = None #100

    # initial pressure and temperature
    input_data['initial_uniform'] = False

    # uniform
    input_data['uniform_pressure'] = 350.  # bar
    input_data['uniform_temperature'] = 348.15  # K

    # gradient
    input_data['temperature_initial'] = 10  # [C] at the reference depth
    input_data['reference_depth_for_temperature'] = 0  # surface
    input_data['temperature_gradient'] = 0.3  # [C/m]

    input_data['pressure_initial'] = 1  # [bar]
    input_data['reference_depth_for_pressure'] = 0  # m
    input_data['pressure_gradient'] = 0.1  # [bar/m]

    return input_data