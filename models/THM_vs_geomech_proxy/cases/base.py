import numpy as np
from darts.engines import well_control_iface
from darts.input.input_data import InputData
from darts.reservoirs.unstruct_reservoir_mech import (
    get_bulk_modulus,
    get_isotropic_stiffness,
    get_rock_compressibility,
)


def parse_structured_dims(model_folder):
    dims = model_folder.split("/")[-1].split("_")
    return int(dims[-3]), int(dims[-2]), int(dims[-1])


def input_data_base(thermal=True):
    # True = single_phase_thermal (rate control); False = single_phase (BHP control)
    idata = InputData(
        type_hydr="thermal" if thermal else "isothermal",
        type_mech="thermoporoelasticity" if thermal else "poroelasticity",
        init_type="gradient",
    )

    idata.other.thermal = thermal

    # structured mesh (NX_NY_NZ cases): set True and assign idata.other.nx/ny/nz before calling
    # _set_structured_mesh_coordinates; False skips mesh coordinate setup
    idata.other.add_structured_mesh = False

    # fracture permeability mode: overrides porosity/perm to fracture values and narrows rsv Y bounds
    idata.other.perm_frac = False

    # --- reservoir geometry ---
    idata.other.rsv_top = 2000.0     # [m] reservoir top depth
    idata.other.rsv_bottom = 2400.0  # [m] reservoir bottom depth
    idata.other.rsv_xy = 1000.0      # [m] reservoir half-width; sets rsv_x1/x2/y1/y2 in _set_reservoir_bounds

    # --- well parameters ---
    idata.other.wells_type = "doublet"     # "doublet", "inj", or "prod"
    idata.other.doublet_shift = 500.0      # [m] half-distance between injection and production well
    idata.other.cell_shift = 0.0           # [m] lateral offset to align well with a cell centre
    idata.other.delta_temp_inj = 40.0     # [K] injected fluid temperature delta vs. initial reservoir T
    idata.other.well_rate_m3_day = 2000.0  # [m3/day] volumetric rate (thermal / rate-control mode)
    idata.other.bhp_delta_p = 10.0         # [bar] BHP offset from initial pressure (isothermal / BHP-control mode)

    # --- mesh tags ---
    idata.other.matrix_tags = (99991,)   # single tag for structured-like meshes; override for named cases
    idata.other.set_props_by_tags = False  # if True, rock props assigned per Gmsh physical tag

    _set_rock_data(idata)
    _set_reservoir_bounds(idata)
    _set_rock_mechanics(idata)
    _set_fluid_data(idata)
    _set_initial_conditions(idata)
    _set_wells(idata)
    _set_mesh_tags(idata)

    # a folder name with the mesh file, None if the mesh is generated on the fly
    idata.other.mesh_dir = None

    # reference points [x, y, label] drawn as a black line in plot_vtk (empty = no line)
    idata.other.points_xy = [[250., 250., '(250,250)']]
    idata.other.use_mesh_bounds_in_plot = False

    if idata.other.add_structured_mesh:
        _set_structured_mesh_coordinates(idata)

    _set_obl(idata)
    return idata


def _set_rock_data(idata):
    idata.rock.density = 2650.0            # [kg/m3]
    idata.rock.porosity = 0.1
    idata.rock.permx = idata.rock.permy = idata.rock.permz = 10.0  # [mD]
    idata.rock.thermal_conductivity = 260  # [kJ/m/day/K]
    idata.rock.heat_capacity = 2300        # [kJ/m3/K]
    # rock properties outside the reservoir coordinate bounds
    idata.rock.poro_non_rsv = 0.001
    idata.rock.perm_non_rsv = 0.01        # [mD]

    if idata.other.perm_frac:
        # fracture mode: treat the fracture as fully open (poro=1) with very high perm
        idata.rock.porosity = 1.0
        idata.rock.permx = idata.rock.permy = idata.rock.permz = 1e6
        idata.rock.poro_non_rsv = 0.1
        idata.rock.perm_non_rsv = 1.0  # [mD]


def _set_reservoir_bounds(idata):
    # permeable reservoir vertical boundaries
    idata.other.rsv_x1 = -idata.other.rsv_xy
    idata.other.rsv_x2 =  idata.other.rsv_xy
    idata.other.rsv_y1 = -idata.other.rsv_xy
    idata.other.rsv_y2 =  idata.other.rsv_xy

    if idata.other.perm_frac:
        idata.other.frac_width = 10.0  # [m]
        idata.other.rsv_y1 = -idata.other.frac_width / 2.0
        idata.other.rsv_y2 =  idata.other.frac_width / 2.0


def _set_rock_mechanics(idata):
    idata.rock.biot = 0.7
    idata.rock.E = 12.0 * 1e4  # [GPa]  # 1e4: [GPa] -> [bar]
    idata.rock.nu = 0.25  # Poisson ratio
    idata.rock.E_non_rsv = idata.rock.E  # homogeneous geomechanical properties

    bulk_modulus = get_bulk_modulus(E=idata.rock.E, nu=idata.rock.nu)
    idata.rock.kd = bulk_modulus  # drained bulk modulus (read by set_props_tags / init)
    idata.rock.compressibility = get_rock_compressibility(
        kd=bulk_modulus,
        biot=idata.rock.biot,
        poro0=idata.rock.porosity,
    )
    print("bulk modulus = ", bulk_modulus)
    print("rock compressibility = ", idata.rock.compressibility)
    idata.rock.stiffness = get_isotropic_stiffness(idata.rock.E, idata.rock.nu)

    idata.rock.th_expn = 1e-5  # [1/K] linear thermal expansion coefficient
    idata.rock.th_expn_orig = idata.rock.th_expn  # preserve original for proxy
    idata.rock.th_expn *= bulk_modulus  # Cauchy book formula 4.19a, 4.21a
    idata.rock.th_expn *= 3.0           # linear -> volumetric (Cauchy 4.22)

    idata.rock.th_expn_poro = 0.0  # mechanical term in porosity update


def _set_fluid_data(idata):
    # single-phase water properties
    idata.fluid.Mw = 18.015         # molar weight [g/mol]
    idata.fluid.compressibility = 4.4e-5  # [1/bar]
    idata.fluid.viscosity = 1.0     # [cP]
    idata.fluid.density = 1000.0    # [kg/m3]
    idata.fluid.thermal_conductivity = 0.0  # not used by the engine [kJ/m/day/K]
    idata.fluid.heat_capacity = 75.0        # [kJ/kmol/K] (water: 75.37 kJ/kmol/K)


def _set_initial_conditions(idata):
    # non-zero initial temperature does not converge properly (check t_ref implementation)
    idata.initial.reference_depth_for_temperature = 0.0  # [m]
    idata.initial.temperature_gradient = 0.0             # [K/m]
    idata.initial.temperature_at_ref_depth = 0.0         # [K]
    # pressure_gradient used only for reservoir.p_init; actual initial pressure is from equilibrium
    idata.initial.pressure_gradient = 0.1   # [bar/m]
    idata.initial.pressure_at_ref_depth = 1.0  # [bar]


def _set_wells(idata):
    # place a single well at centre; for doublet, shift both wells symmetrically
    shift = 0.0
    if idata.other.wells_type == "doublet":
        shift = idata.other.doublet_shift  # [m] half-distance between wells

    # single perforation placed at the vertical midpoint of the reservoir
    perf_depth = (idata.other.rsv_top + idata.other.rsv_bottom) * 0.5
    perf_depth_end = idata.other.rsv_bottom - 1.0  # [m]

    # cell_shift aligns the well with a cell centre when the mesh is centred at (0,0)
    if idata.other.wells_type in ("doublet", "prod"):
        idata.other.prod_well_coords = [
            idata.other.cell_shift - shift,
            idata.other.cell_shift,
            perf_depth,
            perf_depth_end,
        ]  # [X, Y, Z1, Z2]
    if idata.other.wells_type in ("doublet", "inj"):
        idata.other.inj_well_coords = [
            idata.other.cell_shift + shift,
            idata.other.cell_shift,
            perf_depth,
            perf_depth_end,
        ]  # [X, Y, Z1, Z2]

    idata.other.delta_temp_inj = idata.other.delta_temp_inj  # [K]
    if idata.other.thermal:  # rate control
        idata.other.delta_p = None
        idata.other.wctrl_type = well_control_iface.MASS_RATE
        idata.other.well_rate = idata.other.well_rate_m3_day * idata.fluid.density  # [m3/day] -> [kg/day]
    else:  # BHP control
        idata.other.delta_p = idata.other.bhp_delta_p  # [bar]
        idata.other.wctrl_type = well_control_iface.BHP
        idata.other.well_rate = None

    idata.other.well_init_depth = perf_depth


def _set_mesh_tags(idata):
    idata.mesh.bnd_tags = {}
    tags = idata.mesh.bnd_tags
    tags["BND_X-"] = 991 
    tags["BND_X+"] = 992
    tags["BND_Y-"] = 993
    tags["BND_Y+"] = 994
    tags["BND_Z-"] = 995   # TOP
    tags["BND_Z+"] = 996   # BOTTOM

    idata.mesh.matrix_tags = list(idata.other.matrix_tags)
    idata.mesh.tags = idata.mesh.bnd_tags.copy()
    for i, mat_tag in enumerate(idata.mesh.matrix_tags, start=1):
        idata.mesh.tags[f"MATRIX_{i}"] = mat_tag


def _set_obl(idata):
    n_points = 400
    idata.obl.zero = 1e-9
    idata.obl.p_step = (1000.0 - 0.0) / (n_points - 1)
    idata.obl.p_origin = 0.0
    idata.obl.t_step = (50.0 - (-50.0)) / (n_points - 1)
    idata.obl.t_origin = -50.0
    idata.obl.z_step = (1.0 - 2.0 * idata.obl.zero) / (n_points - 1)
    idata.obl.z_origin = idata.obl.zero
    idata.obl.epsilon_z = 1e-10


def _xc_from_nx(nx):
    if nx == 1:
        xc_left = np.array([-100])
    elif nx == 7:
        xc_left = np.array([-4000, -2000, -1000, -100])
    elif nx == 17:
        xc_left = np.array([-4000, -2000, -1000, -500, -400, -300, -200, -100, -50])
    elif nx == 97:
        xc_left = np.array(
            [
                -8000,
                -6000,
                -5000,
                -4000,
                -3000,
                -2500,
                -2000,
                -1600,
                -1500,
                -1450,
                -1400,
                -1350,
                -1300,
                -1250,
                -1200,
                -1150,
            ]
            + [-1100, -1050, -1030, -1010, -1000, -990, -980, -950, -900]
            + np.arange(-800, -555, 50).tolist()
            + np.arange(-555, -455, 10).tolist()
            + np.arange(-455, -5, 50).tolist()
        )
    elif nx == 83:
        xc_left = np.array(
            [
                -8000,
                -6000,
                -5000,
                -4000,
                -3000,
                -2500,
                -2000,
                -1600,
                -1500,
                -1450,
                -1400,
                -1350,
                -1300,
                -1250,
                -1200,
                -1150,
            ]
            + [-1100, -1050, -1030, -1010, -1000, -990, -980, -950, -900]
            + np.arange(-800, -100, 100).tolist()
            + np.arange(-100, 0, 10).tolist()
        )
    elif nx == 71:
        xc_left = np.array(
            [
                -8000,
                -6000,
                -5000,
                -4000,
                -3000,
                -2500,
                -2000,
                -1600,
                -1400,
                -1200,
            ]
            + [-1100, -1050, -1030, -1010, -1000, -990, -980, -950, -900]
            + np.arange(-800, -100, 100).tolist()
            + np.arange(-100, 0, 10).tolist()
        )
    elif nx == 41:
        xc_left = np.array(
            [
                -8000,
                -6000,
                -5000,
                -4000,
                -3000,
                -2500,
                -2000,
                -1600,
                -1400,
                -1200,
            ]
            + [-1100, -1000, -900]
            + np.arange(-800, -100, 100).tolist()
            + [-50]
        )
    else:
        raise ValueError(f"not found an option to mesh with nx = {nx}")

    return np.hstack([xc_left, -xc_left[::-1]])


def _set_structured_mesh_coordinates(idata):
    nx, ny, nz = idata.other.nx, idata.other.ny, idata.other.nz
    xc = _xc_from_nx(nx)
    yc = _xc_from_nx(ny)

    if idata.other.perm_frac:
        yc = np.hstack(
            [
                yc[yc < 0],
                np.array([-idata.other.frac_width / 2.0, idata.other.frac_width / 2.0]),
                yc[yc > 0],
            ]
        )

    rsv_top = idata.other.rsv_top
    rsv_bottom = idata.other.rsv_bottom
    if nz == 5:
        zc = np.array([0, 1000, 2000, 2100, 2200, 3000])
    elif nz == 15:
        zc = np.array([0, 1000, 1500, 2000, 2100, 2120, 2140, 2160, 2180, 2200, 2300, 2500, 3000, 4000, 5000, 6000])
    elif nz == 29:
        zc = np.hstack([np.arange(0, rsv_top, 200), np.arange(rsv_top, rsv_bottom, 20), np.arange(rsv_bottom, 5000, 200)])
    elif nz == 37:
        zc = np.hstack([np.arange(0, rsv_top, 150), np.arange(rsv_top, rsv_bottom, 20), np.arange(rsv_bottom, 5000, 150)])
    elif nz == 53:
        zc = np.hstack([np.arange(0, rsv_top, 100), np.arange(rsv_top, rsv_bottom, 20), np.arange(rsv_bottom, 5000, 100)])
    elif nz == 57:
        zc = np.hstack(
            [
                np.arange(0, rsv_top - 100 + 1, 100),
                np.arange(rsv_top - 50, rsv_bottom + 50 + 1, 25),
                rsv_bottom + 100,
                np.arange(rsv_bottom + 200, 5000 + 1, 100),
            ]
        )
    elif nz == 66:
        zc = np.hstack(
            [
                np.arange(0, rsv_top - 100 + 1, 100),
                np.arange(rsv_top - 50, rsv_bottom + 50 + 1, 25),
                rsv_bottom + 100,
                np.arange(rsv_bottom + 200, 5000 + 1, 100),
            ]
        )
    elif nz == 90:
        zc = np.hstack(
            [
                np.arange(0, rsv_top - 500 + 1, 100),
                np.arange(rsv_top - 450, rsv_bottom + 450 + 1, 25),
                rsv_bottom + 500,
                np.arange(rsv_bottom + 600, 5000 + 1, 100),
            ]
        )
    else:
        raise ValueError(f"not found an option to mesh with nz = {nz}")

    idata.other.Xc = xc
    idata.other.Yc = yc
    idata.other.Zc = zc


def input_data_struct_like(model_folder, physics_type, wells_type):
    # unstructured mesh is generated on the fly with rectangular hexahedral cells
    thermal = "thermal" in physics_type
    idata = input_data_base(thermal=thermal)

    # well type comes from the caller (e.g. "doublet", "inj", "prod")
    idata.other.wells_type = wells_type

    # structured mesh dimensions parsed from the folder name (NX_NY_NZ)
    idata.other.nx, idata.other.ny, idata.other.nz = parse_structured_dims(model_folder)
    idata.other.add_structured_mesh = True

    # recompute derived values that depend on the overridden parameters above
    _set_wells(idata)
    _set_initial_conditions(idata)
    _set_structured_mesh_coordinates(idata)

    return idata
