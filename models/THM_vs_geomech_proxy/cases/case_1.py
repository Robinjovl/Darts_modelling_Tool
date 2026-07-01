import numpy as np
from cases.base import (
    input_data_base,
    _set_reservoir_bounds,
    _set_wells,
    _set_mesh_tags,
)


def input_data_case_1():
    idata = input_data_base()

    # override permeability (default is 10 mD)
    idata.rock.permx = idata.rock.permy = idata.rock.permz = 1000.0  # [mD]

    # reservoir geometry (overrides defaults: rsv_top=2000, rsv_bottom=2400, rsv_xy=1000)
    idata.other.rsv_top = 1800.0    # [m]
    idata.other.rsv_bottom = 1900.0  # [m]
    idata.other.rsv_xy = 1500.0      # [m]

    # well placement (overrides defaults: doublet_shift=500, cell_shift=0)
    idata.other.doublet_shift = 100.0  # [m]
    idata.other.cell_shift = 500.0     # [m]

    # mesh: three Gmsh physical tags (1=overburden, 2=reservoir, 3=underburden)
    idata.other.matrix_tags = (1, 2, 3)

    # case_2 and case_3 inherit this, so all three cases share one mesh
    idata.other.mesh_dir = 'case_1'

    # recompute derived values that depend on the overridden parameters above
    _set_reservoir_bounds(idata)
    _set_wells(idata)
    _set_mesh_tags(idata)

    idata.other.points_xy = []  # no black reference line for case_1
    idata.other.use_mesh_bounds_in_plot = True
    return idata


def input_data_case_2():
    # same geometry as case_1, but rock properties are assigned per Gmsh tag rather than
    # by coordinate interpolation; tags: 1=overburden, 2=reservoir, 3=underburden
    idata = input_data_case_1()
    idata.other.set_props_by_tags = True

    rsv_poro = 0.2
    rsv_perm = 1000.0    # [mD]
    non_rsv_poro = 0.001
    non_rsv_perm = 0.001  # [mD]

    idata.rock.porosity = np.array([non_rsv_poro, rsv_poro, non_rsv_poro])

    perm = np.array([non_rsv_perm, rsv_perm, non_rsv_perm])  # isotropic perm per tag
    idata.rock.permx = idata.rock.permy = perm
    idata.rock.permz = perm * 0.1  # vertical perm is 10x lower

    return idata


def input_data_case_3():
    # same as case_2 but with heterogeneous thermal properties per tag (sand vs shale)
    idata = input_data_case_2()

    hcap_sand  = 2450.0  # [kJ/m3/K]
    hcap_shale = 2300.0  # [kJ/m3/K]
    idata.rock.heat_capacity = np.array([hcap_shale, hcap_sand, hcap_shale])

    rcond_sand  = 3.0 * 86.4  # [kJ/m/day/K]
    rcond_shale = 2.2 * 86.4  # [kJ/m/day/K]
    idata.rock.thermal_conductivity = np.array([rcond_shale, rcond_sand, rcond_shale])

    return idata
