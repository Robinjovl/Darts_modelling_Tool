import numpy as np
from examples.base import InputDataConfig, build_input_data


def _case_1_config(set_props_by_tags=False):
    return InputDataConfig(
        permeability=1000.0,
        rsv_top=1800.0,
        rsv_bottom=1900.0,
        rsv_xy=1500.0,
        doublet_shift=100.0,
        cell_shift=500.0,
        matrix_tags=(1, 2, 3),
        set_props_by_tags=set_props_by_tags,
        pressure_reference_depth=0.0,
    )


def input_data_case_1():
    idata = build_input_data(_case_1_config())
    idata.other.points_xy = []  # no black reference line for case_1
    idata.other.use_mesh_bounds_in_plot = True
    return idata


def input_data_case_2():
    # heterogeneous poro and perm by mesh tags:
    # same three-layer geometry / mesh as case_1 (matrix tags 1, 2, 3),
    # but rock properties are assigned per tag instead of by interpolation.
    # matrix tags: tag 2 = reservoir, tags 1 & 3 = over/underburden
    idata = input_data_case_1()
    idata.other.set_props_by_tags = True

    # reservoir
    rsv_poro = 0.2
    rsv_perm = 1000.0 # mD     # reservoir (from case_1)

    # over/underburden
    non_rsv_poro = 0.001
    non_rsv_perm = 0.001 # mD

    idata.rock.porosity = np.array([non_rsv_poro, rsv_poro, non_rsv_poro])

    perm = np.array([non_rsv_perm, rsv_perm, non_rsv_perm])  # isotropic perm tensor per tag
    idata.rock.permx = idata.rock.permy = perm
    idata.rock.permz = perm * 0.1

    return idata

def input_data_case_3():
    # same as case_2 but hcap and rcond are heterogeneous, and different non_rsv_perm, non_rsv_poro values
    idata = idata = input_data_case_2()

    hcap_sand = 2450.0 # [kJ/m3/K]
    hcap_shale = 2300.0 # [kJ/m3/K]
    idata.rock.heat_capacity = np.array([hcap_shale, hcap_sand, hcap_shale])

    rcond_sand = 3.0 * 86.4  # [kJ/m/day/K]
    rcond_shale = 2.2 * 86.4  # [kJ/m/day/K]
    idata.rock.thermal_conductivity = np.array([rcond_shale, rcond_sand, rcond_shale])
    return idata
