from darts.input.input_data import InputData
from darts.physics.geothermal.geothermal import GeothermalIAPWSFluidProps

def input_data_default():
    idata = InputData(type_hydr='thermal', type_mech='none', init_type='gradient')
    idata.geom = dict()
    ###########################################################################################################
    # DFN framework parameters (for mesh generation)
    idata.geom['frac_file'] = 'frac.txt'  # fracture tips coordinates X1 Y1 X2 Z2; should contain at least 2 rows (2 fractures)
    idata.geom['frac_format'] = 'simple'

    #idata.geom['mesh_prefix'] = 'raw_lc'  #  use mesh with original fracture tips
    idata.geom['mesh_prefix'] = 'mergefac_0.86_clean_lc'  #  cleaned mesh
    idata.geom['mesh_clean'] = False  # need gmsh installed and callable from command line in order to mesh

    idata.geom['margin'] = 100  # [m]
    idata.geom['box_data'] = None  # [m] mesh bounds (in case of no margin defined)

    # cell sizes
    idata.geom['char_len'] = 16  # near fractures (characteristic length for cleaning and mesh generation) [m]
    idata.geom['char_len_boundary'] = 16  # grid size near grid boundaries [m]
    idata.geom['char_len_well'] = 16  # grid size near wells [m]

    # geometry (both for DFN and model)
    idata.geom['z_top'] = 0  # [m]
    idata.geom['height_res'] = 50  # [m]

    # extrusion - number of layers by Z axis
    idata.geom['rsv_layers'] = 1

    # no overburden layers (fractured) by default
    idata.geom['overburden_thickness'] = 0
    idata.geom['overburden_layers'] = 0
    idata.geom['underburden_thickness'] = 0
    idata.geom['underburden_layers'] = 0

    # no second overburden layers (without fractures) by default
    idata.geom['overburden_2_thickness'] = 0
    idata.geom['overburden_2_layers'] = 0
    idata.geom['underburden_2_thickness'] = 0
    idata.geom['underburden_2_layers'] = 0

    # well locations
    idata.geom['inj_well_coords'] = [[50, 50, 25]]  # X, Y, Z (only one perforation)
    idata.geom['prod_well_coords'] = [[950, 950, 25]]

    # The properties below do not affect mesh generation stage. So no need to re-generate the mesh if you change them.

    # will be passed to UnstructuredDiscretizer
    idata.geom['frac_aper'] = 1e-3  # (initial) fracture aperture [m]

    # well in the matrix cells or in the fractures
    idata.geom['well_loc_type'] = 'wells_in_nearest_cell'  # could be in the matrix or in the fracture, depending on the location
    #idata.geom['well_loc_type'] = 'wells_in_frac'  # put the well into the closest fracture
    #idata.geom['well_loc_type'] = 'wells_in_mat'  # put the well into the closest matrix cell

    idata.rock.porosity = 0.2
    idata.rock.permx = 10  # [mD]
    idata.rock.permy = 10  # [mD]
    idata.rock.permz = 1  # [mD]
    idata.rock.perm_file = None  # if want to read the permeability from a file

    idata.rock.compressibility = 1e-5  # [1/bars]
    idata.rock.compressibility_ref_p = 1  # [bars]
    idata.rock.compressibility_ref_T = 273.15  # [K]

    idata.rock.heat_capacity = 2200. # [kJ/m3/K]
    idata.rock.conductivity = 181.44  # [kJ/m/day/K]

    idata.fluid = GeothermalIAPWSFluidProps()

    # well controls
    class InputDataWellControls():  # an empty class - to group custom well control input data
        def __init__(self):
            pass
    idata.well_data.controls = InputDataWellControls()
    wctrl = idata.well_data.controls  #short name
    wctrl.prod_rate = None  # m3/day. if None, well will work under BHP control
    wctrl.inj_rate = None   # m3/day. if None, well will work under BHP control
    wctrl.delta_temp = 10   # bars. inj_temp = initial_temp - delta_temp
    wctrl.delta_p_inj  = 5  # bars. inj_bhp = initial_pressure + delta_p_inj
    wctrl.delta_p_prod = 5  # bars. inj_prod = initial_pressure - delta_p_prod
    wctrl.prod_bhp_constraint = 50 # bars
    wctrl.inj_bhp_constraint = 450 # bars

    # principal stress, MPa.
    # Set to None if don't want to recompute fracture apertures by initial stresses
    idata.stress = dict()
    idata.stress['Sh_min'] = None #50
    idata.stress['Sh_max'] = None # 90
    idata.stress['Sv'] = None # 120
    idata.stress['SHmax_azimuth'] = None #0  # [°] from X, counter-clockwise
    idata.stress['sigma_c'] = None #100

    # gradient
    idata.initial.reference_depth_for_pressure = 0  # [m]
    idata.initial.pressure_gradient = 100  # [bar/km]
    idata.initial.pressure_at_ref_depth = 1  # [bars]

    idata.initial.reference_depth_for_temperature = 0  # [m]
    idata.initial.temperature_gradient = 30  # [K/km]
    idata.initial.temperature_at_ref_depth = 273.15 + 10 # [K]

    idata.obl.n_points = 100
    idata.obl.min_p = 50.
    idata.obl.max_p = 500.
    idata.obl.min_e = 1000.
    idata.obl.max_e = 25000.

    return idata