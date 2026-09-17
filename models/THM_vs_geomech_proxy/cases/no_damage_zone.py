import numpy as np
from gen_fault_msh_no_damage_zone import DOMAIN_W, DOMAIN_H, LAYER_DY
from cases.base import (
    input_data_base,
    _set_reservoir_bounds,
    _set_wells,
    _set_mesh_tags,
)


def input_data_no_damage_zone(physics_type='single_phase_thermal'):
    idata = input_data_base(thermal='thermal' in physics_type)

    # override permeability (default is 10 mD)
    #idata.rock.permx = idata.rock.permy = idata.rock.permz = 1000.0  # [mD]

    # reservoir geometry (overrides defaults: rsv_top=2000, rsv_bottom=2400, rsv_xy=1000)
    idata.other.rsv_top = 2550.0    # [m]
    idata.other.rsv_bottom = 2600.0  # [m]
    idata.other.rsv_xy = 10000. # [m]

    # well placement (overrides defaults: doublet_shift=500, cell_shift=0)
    idata.other.doublet_shift = 500.0  # [m]

    # doublet rate (default 2000 m3/day); the injector and the producer share it (balanced doublet)
    idata.other.well_rate_m3_day = 8000.0  # [m3/day]
    #idata.other.cell_shift = 500.0     # [m]

    # mesh: Gmsh physical tags: rsv, overburden, underburden
    idata.other.matrix_tags = (99991, 99992, 99993)

    #idata.other.fault = (9991)

    # case_2 and case_3 inherit this, so all three cases share one mesh
    idata.other.mesh_dir = 'no_damage_zone'

    # recompute derived values that depend on the overridden parameters above
    _set_reservoir_bounds(idata)
    _set_wells(idata)
    _set_mesh_tags(idata)
    
    # wells on different sides of the fault (the fault crosses z = 2500 m at x = DOMAIN_W / 2, dipping 60 deg):
    # injector well_offset west and producer well_offset east of it, both at the middle of the domain in y.
    # The fault offsets the reservoir layer (tag 99991): it spans z ~ 2450..2750 m west of the fault and
    # z ~ 2250..2550 m east of it. Both wells request the depth range 2000..3000 m; since perforations are
    # restricted to the reservoir tag, each well is perforated over the whole local reservoir layer.
    x_fault, y_mid = DOMAIN_W / 2., DOMAIN_H / 2.
    well_offset = 500.  # [m] horizontal distance of each well from the fault at mid-depth (wells 1 km apart)
    idata.other.inj_well_coords = [x_fault - well_offset, y_mid, 2000., 3000.]  # [X, Y, Z1, Z2]
    idata.other.prod_well_coords = [x_fault + well_offset, y_mid, 2000., 3000.]  # [X, Y, Z1, Z2]
    idata.other.well_perforation_tags = (99991,)

    idata.other.points_xy = []  # no black reference line for case_1
    idata.other.use_mesh_bounds_in_plot = True
    # 2D slices through the well row at the injector depth: the XZ slice shows both wells and the fault.
    # y_mid lies on a boundary of the LAYER_DY-long mesh layers; the perforations take the layer below it.
    idata.other.plot_slice_origin = [x_fault - well_offset, y_mid - LAYER_DY / 2., 2575.]

    idata.other.set_props_by_tags = True

    rsv_poro = 0.2
    rsv_perm = 10.0     # [mD]

    non_rsv_poro = 0.001
    non_rsv_perm = 0.001  # [mD]

    idata.rock.porosity = np.array([rsv_poro, non_rsv_poro, non_rsv_poro])

    perm = np.array([rsv_perm, non_rsv_perm,  non_rsv_perm])  # isotropic perm per tag
    idata.rock.permx = idata.rock.permy = perm
    idata.rock.permz = perm * 0.1  # vertical perm is 10x lower

    hcap_sand  = 2450.0  # [kJ/m3/K]
    hcap_shale = 2300.0  # [kJ/m3/K]
    idata.rock.heat_capacity = np.array([hcap_sand, hcap_shale, hcap_shale])

    rcond_sand  = 3.0 * 86.4  # [kJ/m/day/K]
    rcond_shale = 2.2 * 86.4  # [kJ/m/day/K]
    idata.rock.thermal_conductivity = np.array([rcond_sand, rcond_shale, rcond_shale])
    
    idata.rock.E = 20.0 * 1e4   # [bars]
    idata.rock.nu = 0.20  # Poisson ratio
                              
    return idata
