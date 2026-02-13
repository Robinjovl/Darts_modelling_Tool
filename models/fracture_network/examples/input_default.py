import numpy as np

from darts.input.input_data import InputData
from darts.physics.geothermal.geothermal import GeothermalIAPWSFluidProps
from darts.engines import well_control_iface

def input_data_default():
    idata = InputData(type_hydr='thermal', type_mech='none', init_type='gradient')
    idata.geom = dict()
    ###########################################################################################################
    # DFN framework parameters (for mesh generation)
    idata.geom['frac_file'] = 'frac.txt'  # fracture tips coordinates X1 Y1 X2 Z2; should contain at least 2 rows (2 fractures)
    idata.geom['frac_format'] = 'simple'

    idata.geom['frac_tag_start'] = 90000  #  the starting index for physical surfaces for fractures in a mesh, first 6 are for the boundaries
    idata.geom['frac_geom_type'] = 'quad'
    idata.geom['matrix_tags'] = [9991, 9992, 9993, 9994, 9995] # 9991 - rsv, 9992 - overburden, 9993 - underburden, 9994 - overburden2, 9995 - underburden2
    # (they might not be there, but we define them anyway, see also the description in the end of this file
    idata.geom['mesh_type'] = '2.5D'

    idata.geom['mesh_filename'] = None # will be used if the previous item is not 2.5D

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

    # well locations (perforation range) are defined for each case since they depend on the geometry
    #idata.geom['well_coords']['I1'] = [[50, 50, 25, 25]]  # X, Y, Z1, Z2
    #idata.geom['well_coords']['P1']  = [[950, 950, 25, 25]]

    # The properties below do not affect mesh generation stage. So no need to re-generate the mesh if you change them.

    # will be passed to UnstructuredDiscretizer
    idata.geom['frac_aper'] = 1e-3  # (initial) fracture aperture [m]

    # well in the matrix cells or in the fractures
    idata.geom['well_loc_type'] = 'wells_in_nearest_cell'  # could be in the matrix or in the fracture, depending on the location
    #idata.geom['well_loc_type'] = 'wells_in_frac'  # put the well into the closest fracture
    #idata.geom['well_loc_type'] = 'wells_in_mat'  # put the well into the closest matrix cell

    # to mimic an infinite reservoir
    idata.geom['bondary_volume_xy'] = 1e+15  # [m^3]

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
    idata.obl.min_p = 0.
    idata.obl.max_p = 500.
    idata.obl.min_e = 10.
    idata.obl.max_e = 25000.

    # well controls
    class InputDataWellControls():  # an empty class - to group custom well control input data
        def __init__(self):
            pass
    idata.well_data.controls = InputDataWellControls()
    wctrl = idata.well_data.controls  #short name
    wctrl.is_const = True   # the same well controls over all timesteps
    wctrl.prod_rate = None  # m3/day. if None, well will work under BHP control
    wctrl.inj_rate = None   # m3/day. if None, well will work under BHP control
    wctrl.delta_temp = 10   # bars. inj_temp = initial_temp - delta_temp
    wctrl.delta_p_inj  = 5  # bars. inj_bhp = initial_pressure + delta_p_inj
    wctrl.delta_p_prod = 5  # bars. inj_prod = initial_pressure - delta_p_prod
    wctrl.prod_bhp_constraint = 50 # bars
    wctrl.inj_bhp_constraint = 450 # bars
    wctrl.temp_inj_K = 300 # K  #TODO: initial_temperature - wctrl.delta_temp
    #wctrl.mode = 'RATE'  # if use this then set wctrl.prod_rate and wctrl.inj_rate
    wctrl.mode = 'BHP'

    #prod_bhp = P[well_top_perf_idx] - wctrl.delta_p_prod  # rsv block pressure at the top perforation - delta_p
    #inj_bhp = P[well_top_perf_idx] + wctrl.delta_p_prod  # rsv block pressure at the top perforation + delta_p
    #inj_temp = T[well_top_perf_idx] - wctrl.delta_temp

    idata.well_data.well_is_inj = lambda wname: "I" in wname  # determine the well type by its name

    # Time steps
    dt = 60  # Size of the reporting step
    n_time_steps = 12*5   # Number of reporting steps (see above)
    idata.sim.time_steps = np.zeros(n_time_steps) + dt

    return idata

def get_inj_well_coords(idata): # returns a dictionary of injection well coordinates with a well name as a key
    return {k: v for k, v in idata.geom['well_coords'].items() if idata.well_data.well_is_inj(k)}

def get_prod_well_coords(idata): # returns a dictionary of production well coordinates with a well name as a key
    return {k: v for k, v in idata.geom['well_coords'].items() if not idata.well_data.well_is_inj(k)}


'''
# 2.5D mesh from DARTS-gmsh mesh generator
        matrix_tag   surface_tag                             fracture_tag    test_case
        ----------      2     overburden2 top                                     }
        | 9994                    overburden2                                     }
        ----------      2     overburden top       ------------- 90003        }   }
        | 9992                    overburden       | FRACTURE  |              }   }case_1_burden_2
        ----------      2     reservoir top        |-----------| 90001    }   }case_1_burden
        | 9991                    RESERVOIR        | FRACTURE  | 90000    }case_1 }
        ----------      1     reservoir bottom     |-----------| 90002    }   }   }
        | 9993                    underburden      | FRACTURE  |              }   }
        ----------      1     underburden bottom   ------------- 90004        }   }
        | 9995                    underburden2                                    }
        ----------      1     underburden2 bottom                                 }
'''

def add_wells_idata(idata: InputData): # add  wells to idata
    # add wells to idata
    wdata = idata.well_data
    for wname in idata.geom['well_coords'].keys():
        x, y, z = idata.geom['well_coords'][wname][:3]
        wdata.add_well(name=wname, loc_type='xyz', loc_xyz=(x,y,z)) # those x,y,z are not actually used, but let's set it anyway here


def add_well_controls_idata(idata: InputData): # add constant well controls to idata
    wdata = idata.well_data
    wells = wdata.wells  # short name
    wctrl = idata.well_data.controls  #short name

    if wctrl.is_const:
        add_wells_idata(idata)
    else: # if variable well controls are defined, skip this function as wells and controls are added there
        return

    if wctrl.mode == 'RATE':
        for w in wells:
            if idata.well_data.well_is_inj(w): # inj
                wdata.add_inj_rate_control(name=w, rate=wctrl.inj_rate,
                                           rate_type=well_control_iface.VOLUMETRIC_RATE,
                                           bhp_constraint=wctrl.inj_bhp_constraint, temperature=wctrl.temp_inj_K)  # m3/day | bars | K
            else:  # prod
                wdata.add_prd_rate_control(name=w, rate=wctrl.prod_rate,
                                           rate_type=well_control_iface.VOLUMETRIC_RATE,
                                           bhp_constraint=wctrl.prod_bhp_constraint)  # m3/day | bars
    elif wctrl.mode == 'BHP':
        for w in wells:
            if idata.well_data.well_is_inj(w): # inj
                wdata.add_inj_bhp_control(name=w, bhp=wctrl.inj_bhp_constraint, temperature=wctrl.temp_inj_K)  # m3/day | bars | K
            else: # prod
                wdata.add_prd_bhp_control(name=w, bhp=wctrl.prod_bhp_constraint) # m3/day | bars
