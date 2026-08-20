import numpy as np
from cases.no_damage_zone import input_data_no_damage_zone
from darts.reservoirs.unstruct_reservoir_mech import (
    get_bulk_modulus,
    get_rock_compressibility,
)


def input_data_no_damage_zone_heter_mech_prop(physics_type='single_phase_thermal'):
    # Inherits `no_damage_zone` (same mesh, flow/thermal props) but with
    # HETEROGENEOUS geomechanical properties per tag: a softer, more
    # compressible sand reservoir sandwiched between stiffer shale burdens.
    #
    # Per-tag array order follows no_damage_zone / matrix_tags
    # (99991, 99992, 99993) = [reservoir(sand), overburden(shale), underburden(shale)].
    idata = input_data_no_damage_zone(physics_type)

    E_sand  = 20.0 * 1e4   # [bars]  (1e4: [GPa] -> [bar])
    E_shale = 10.0 * 1e4  # [bars]
    idata.rock.E = np.array([E_sand, E_shale, E_shale])

    nu_sand  = 0.25  # Poisson ratio
    nu_shale = 0.25
    idata.rock.nu = np.array([nu_sand, nu_shale, nu_shale])

    # recompute derived geomechanical quantities per tag (mirrors _set_rock_mechanics
    # in base.py and input_data_case_4)
    bulk_modulus = get_bulk_modulus(E=idata.rock.E, nu=idata.rock.nu)
    idata.rock.kd = bulk_modulus  # drained bulk modulus (read by set_props_tags / init)
    idata.rock.compressibility = get_rock_compressibility(
        kd=bulk_modulus,
        biot=idata.rock.biot,
        poro0=idata.rock.porosity,
    )
    
    th_exp_sand = 1e-5  # [1/K] linear thermal expansion coefficient
    th_exp_shale = 1e-5  # [1/K] linear thermal expansion coefficient
    idata.rock.th_expn_orig = np.array([th_exp_sand, th_exp_shale, th_exp_shale])
    idata.rock.th_expn = idata.rock.th_expn_orig * bulk_modulus * 3.0  # Cauchy 4.19a/4.21a, linear -> volumetric (4.22)

    return idata
