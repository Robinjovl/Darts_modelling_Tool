import numpy as np
from cases.base import (
    input_data_base,
    _set_reservoir_bounds,
    _set_wells,
    _set_mesh_tags,
)


def input_data_no_damage_zone():
    idata = input_data_base()

    # override permeability (default is 10 mD)
    #idata.rock.permx = idata.rock.permy = idata.rock.permz = 1000.0  # [mD]

    # reservoir geometry (overrides defaults: rsv_top=2000, rsv_bottom=2400, rsv_xy=1000)
    idata.other.rsv_top = 2550.0    # [m]
    idata.other.rsv_bottom = 2600.0  # [m]
    idata.other.rsv_xy = 10000. # [m]

    # well placement (overrides defaults: doublet_shift=500, cell_shift=0)
    idata.other.doublet_shift = 500.0  # [m]
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
    
    idata.other.prod_well_coords = [idata.other.rsv_xy/2. - 1000., idata.other.rsv_xy/2. + idata.other.doublet_shift,
            idata.other.rsv_top, idata.other.rsv_bottom]  # [X, Y, Z1, Z2]

    idata.other.inj_well_coords = [idata.other.rsv_xy/2. - 1000., idata.other.rsv_xy/2. - idata.other.doublet_shift,
            idata.other.rsv_top, idata.other.rsv_bottom]  # [X, Y, Z1, Z2]

    idata.other.points_xy = []  # no black reference line for case_1
    idata.other.use_mesh_bounds_in_plot = True

    idata.other.set_props_by_tags = True

    rsv_poro = 0.2
    rsv_perm = 100.0    # [mD]

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
    return idata
