import numpy as np
from cases.base import (
    input_data_base,
    _set_reservoir_bounds,
    _set_wells,
    _set_mesh_tags,
)
from darts.reservoirs.unstruct_reservoir_mech import (
    get_bulk_modulus,
    get_rock_compressibility,
)


def input_data_case_5(physics_type='single_phase_thermal', mesh_dir='case_5'):
    idata = input_data_base(thermal='thermal' in physics_type)

    # override permeability (default is 10 mD)
    #idata.rock.permx = idata.rock.permy = idata.rock.permz = 1000.0  # [mD]

    # reservoir geometry (overrides defaults: rsv_top=2000, rsv_bottom=2400, rsv_xy=1000)
    idata.other.rsv_top = 2830.0    # [m]
    idata.other.rsv_bottom = 3030.0  # [m]
    idata.other.rsv_xy = 10000. # 4500.0      # [m]
    idata.other.well_rate_m3_day = 8000
    # well placement (overrides defaults: doublet_shift=500, cell_shift=0)
    idata.other.doublet_shift = 500.0  # [m]
    #idata.other.cell_shift = 500.0     # [m]

    # mesh: three Gmsh physical tags (1=overburden, 2=reservoir, 3=underburden, 4=damage zone left,
    # 5=damage zone right 99991=fault)
    idata.other.matrix_tags = (1, 2, 3, 4, 5, 99991)
    # Wells must connect to reservoir rock only, never to a confining layer
    # when their requested endpoint lies exactly on a material interface.
    idata.other.well_perforation_tags = (1,)

    # case_2 and case_3 inherit this, so all three cases share one mesh
    idata.other.mesh_dir = mesh_dir

    # recompute derived values that depend on the overridden parameters above
    _set_reservoir_bounds(idata)
    _set_wells(idata)
    _set_mesh_tags(idata)

    # case_5 owns its well locations, derived from the reservoir geometry above.
    # Carried on idata to both the model and the fault-mesh generator. [X, Y, Z1, Z2].
    xc = idata.other.rsv_xy / 2.0
    idata.other.prod_well_coords = [xc + idata.other.doublet_shift, xc,
                                    idata.other.rsv_top, idata.other.rsv_bottom]
    idata.other.inj_well_coords = [xc - idata.other.doublet_shift, xc,
                                   idata.other.rsv_top, idata.other.rsv_bottom]

    idata.other.points_xy = []  # no black reference line for case_1
    idata.other.use_mesh_bounds_in_plot = True
    # Plot proxy/THM slices over the complete physical mesh domain.  Other
    # cases retain the historical reservoir-focused slice bounds.
    idata.other.proxy_plot_use_mesh_bounds = True

    idata.other.set_props_by_tags = True

    rsv_poro = 0.2
    rsv_perm = 500.0    # [mD]
    damage_zone_perm = 800 # [mD]
    damage_zone_poro = 0.1
    fault_perm = 1 # [mD]
    fault_poro = 0.01
    non_rsv_poro = 0.001
    non_rsv_perm = 0.001  # [mD]

    idata.rock.porosity = np.array([rsv_poro, non_rsv_poro,  non_rsv_poro, damage_zone_poro,damage_zone_poro,fault_poro])
    perm = np.array([rsv_perm,non_rsv_perm,  non_rsv_perm, damage_zone_perm, damage_zone_perm, fault_perm])  # isotropic perm per tag
    idata.rock.permx = idata.rock.permy = perm
    idata.rock.permz = perm * 0.1  # vertical perm is 10x lower

    hcap_sand  = 2450.0  # [kJ/m3/K]
    hcap_shale = 2300.0  # [kJ/m3/K]
    idata.rock.heat_capacity = np.array([hcap_sand, hcap_shale,  hcap_shale, hcap_sand, hcap_sand, hcap_sand])

    rcond_sand  = 3.0 * 86.4  # [kJ/m/day/K]
    rcond_shale = 2.2 * 86.4  # [kJ/m/day/K]
    idata.rock.thermal_conductivity = np.array([rcond_sand, rcond_shale, rcond_shale, rcond_sand,rcond_sand, rcond_sand])
    # Region order: sandstone, shale, shale, damage zone, damage zone, fault core.
    # Values are regional midpoints rounded to two decimal places.
    idata.rock.E = np.array([16.50, 21.00, 21.00, 27.50, 27.50, 5.03]) * 1e4  # [bar] (GPa -> bar)
    idata.rock.nu = np.array([0.18, 0.28, 0.28, 0.25, 0.25, 0.28])
    idata.rock.biot = np.array([0.74, 0.73, 0.73, 0.90, 0.90, 0.95])
    # Preserve original linear thermal expansion coefficients for the proxy [1/K].
    idata.rock.th_expn_orig = np.array([30.00, 25.00, 25.00, 17.50, 17.50, 15.50]) * 1e-6

    # recompute derived geomechanical quantities per tag (mirrors _set_rock_mechanics in base.py)
    bulk_modulus = get_bulk_modulus(E=idata.rock.E, nu=idata.rock.nu)
    idata.rock.kd = bulk_modulus  # drained bulk modulus (read by set_props_tags / init)
    idata.rock.compressibility = get_rock_compressibility(
        kd=bulk_modulus,
        biot=idata.rock.biot,
        poro0=idata.rock.porosity,
    )
    idata.rock.th_expn = idata.rock.th_expn_orig * bulk_modulus * 3.0  # Cauchy 4.19a/4.21a, linear -> volumetric (4.22)
    return idata
