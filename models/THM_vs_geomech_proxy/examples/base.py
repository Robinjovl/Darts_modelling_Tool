from dataclasses import dataclass

import numpy as np
from darts.engines import well_control_iface
from darts.input.input_data import InputData
from darts.reservoirs.unstruct_reservoir_mech import (
    get_bulk_modulus,
    get_isotropic_stiffness,
    get_rock_compressibility,
)


@dataclass
class InputDataConfig:
    type_hydr: str = "thermal"
    type_mech: str = "thermoporoelasticity"
    porosity: float = 0.1
    permeability: float = 10.0
    young_modulus_gpa: float = 12.0
    perm_frac: bool = False
    rsv_top: float = 2000.0
    rsv_bottom: float = 2400.0
    rsv_xy: float = 1000.0
    wells_type: str = "doublet"
    doublet_shift: float = 500.0
    cell_shift: float = 0.0
    thermal: bool = True
    bhp_delta_p: float = 10.0
    well_rate_m3_day: float = 2000.0
    delta_temp_inj: float = 40.0
    matrix_tags: tuple = (99991,)
    pressure_reference_depth: float | None = None
    initial_composition: list | None = None
    dims: tuple | None = None
    add_structured_mesh: bool = False


def parse_structured_dims(model_folder):
    dims = model_folder.split("/")[-1].split("_")
    return int(dims[-3]), int(dims[-2]), int(dims[-1])


def build_input_data(config: InputDataConfig):
    porosity = config.porosity
    permeability = config.permeability

    idata = InputData(
        type_hydr=config.type_hydr,
        type_mech=config.type_mech,
        init_type="gradient",
    )

    if config.dims is not None:
        idata.other.nx, idata.other.ny, idata.other.nz = config.dims

    idata.other.perm_frac = config.perm_frac
    if idata.other.perm_frac:
        porosity = 1.0
        permeability = 1e6

    _set_rock_data(idata, porosity, permeability)
    _set_reservoir_bounds(idata, config)
    _set_rock_mechanics(idata, config.young_modulus_gpa)
    _set_fluid_data(idata)
    _set_initial_conditions(idata, config)
    _set_wells(idata, config)
    _set_mesh_tags(idata, config.matrix_tags)

    # reference points [x, y, label] for the 1D vertical profiles and the black
    # reference line in plot_vtk_pyvista (if empty, no black line is drawn)
    idata.other.points_xy = [[250., 250., '(250,250)']]

    if config.add_structured_mesh:
        _set_structured_mesh_coordinates(idata)

    _set_obl(idata)
    return idata


def _set_rock_data(idata, porosity, permeability):
    idata.rock.density = 2650.0  # [kg/m3]
    idata.rock.porosity = porosity
    idata.rock.permx = idata.rock.permy = idata.rock.permz = permeability  # [mD]

    idata.rock.thermal_conductivity = 260  # [kJ/m/day/K]
    idata.rock.heat_capacity = 2300  # [kJ/m3/K]

    # rock properties for the part of the mesh outside the reservoir boundaries
    idata.rock.poro_non_rsv = 0.001
    idata.rock.perm_non_rsv = 0.01  # this matches proxy and thm (1e-9 matched thm and analytical solution)

    if idata.other.perm_frac:
        idata.rock.poro_non_rsv = 0.1
        idata.rock.perm_non_rsv = 1.0  # [mD]

def _set_reservoir_bounds(idata, config):
    # permeable reservoir vertical boundaries
    idata.other.rsv_top = config.rsv_top  # [m]
    idata.other.rsv_bottom = config.rsv_bottom  # [m]
    # lateral reservoir boundaries
    idata.other.rsv_xy = config.rsv_xy  # [m], laterally limited (rsv width = rsv_xy*2)
    idata.other.rsv_x1 = -idata.other.rsv_xy
    idata.other.rsv_x2 = idata.other.rsv_xy
    idata.other.rsv_y1 = -idata.other.rsv_xy
    idata.other.rsv_y2 = idata.other.rsv_xy

    if idata.other.perm_frac:
        idata.other.frac_width = 10.0  # [m]
        idata.other.rsv_y1 = -idata.other.frac_width / 2.0
        idata.other.rsv_y2 = idata.other.frac_width / 2.0

def _set_rock_mechanics(idata, young_modulus_gpa):
    idata.rock.biot = 0.7  # 0.8 to match dp with geos
    idata.rock.E = 1e4 * young_modulus_gpa  # convert GPa to bars
    idata.rock.nu = 0.25  # poisson ratio
    idata.rock.E_non_rsv = idata.rock.E  # homogeneous geomech prop

    bulk_modulus = get_bulk_modulus(E=idata.rock.E, nu=idata.rock.nu)
    idata.rock.compressibility = get_rock_compressibility(
        kd=bulk_modulus,
        biot=idata.rock.biot,
        poro0=idata.rock.porosity,
    )
    print("bulk modulus = ", bulk_modulus)
    print("rock compressibility = ", idata.rock.compressibility)
    idata.rock.stiffness = get_isotropic_stiffness(idata.rock.E, idata.rock.nu)

    idata.rock.th_expn = 1e-5  # [1/K]
    idata.rock.th_expn_orig = idata.rock.th_expn  # save this for proxy
    idata.rock.th_expn *= bulk_modulus  # Cauchy book formula 4.19a, 4.21a
    idata.rock.th_expn *= 3.0  # Cauchy book formula 4.22; from linear to volumetric

    idata.rock.th_expn_poro = 0.0  # mechanical term in porosity update


def _set_fluid_data(idata):
    # only for a single-phase physics
    idata.fluid.Mw = 18.015  # water molar weight, [g/mol]
    idata.fluid.compressibility = 4.4e-5  # [1/bar]
    idata.fluid.viscosity = 1.0  # [cP]
    idata.fluid.density = 1000.0  # [kg/m^3]
    idata.fluid.thermal_conductivity = 0.0  # not used in the engine, [kJ/m/day/K]
    idata.fluid.heat_capacity = 75.0  # [kJ/kmol/K] (water: 4170 [kJ/m3/K] = 75.37 [kJ/kmol/K])


def _set_initial_conditions(idata, config):
    # initial conditions (p, T gradients)
    # non-zero initial temperature doesn't work properly (doesn't converge, check t_ref implementation)
    idata.initial.reference_depth_for_temperature = 0.0  # [m]
    idata.initial.temperature_gradient = 0.0  # [K/m]
    idata.initial.temperature_at_ref_depth = 0.0  # [K]
    # pressure_gradient is used only in reservoir.get_reservoir_initial_pressure() => reservoir.p_init;
    # the actual initial pressure is computed by equilibrium using the fluid density
    idata.initial.pressure_gradient = 0.1  # [bar/m]
    if config.pressure_reference_depth is not None:
        idata.initial.reference_depth_for_pressure = config.pressure_reference_depth  # [m]
    idata.initial.pressure_at_ref_depth = 1.0  # [bars]
    if config.initial_composition is not None:
        idata.initial.initial_composition = config.initial_composition


def _set_wells(idata, config):
    # vertical well locations
    shift = 0.0  # if a single well - place it to the center
    if config.wells_type == "doublet":
        shift = config.doublet_shift  # half well distance [m]

    eps_perf = 1.0  # [m]
    perf_depth_start = idata.other.rsv_top + eps_perf
    perf_depth_end = idata.other.rsv_bottom - eps_perf
    # as the perf is single, put it to the middle depth of the rsv
    perf_depth_start = (idata.other.rsv_top + idata.other.rsv_bottom) * 0.5

    # cell_shift puts the well into a cell center (when the mesh is centered at (0,0))
    if config.wells_type in ("doublet", "prod"):
        idata.other.prod_well_coords = [
            config.cell_shift - shift,
            config.cell_shift,
            perf_depth_start,
            perf_depth_end,
        ]  # X, Y, Z1, Z2
    if config.wells_type in ("doublet", "inj"):
        idata.other.inj_well_coords = [
            config.cell_shift + shift,
            config.cell_shift,
            perf_depth_start,
            perf_depth_end,
        ]  # X, Y, Z1, Z2

    # well controls
    idata.other.delta_temp_inj = config.delta_temp_inj  # [K], delta for temperature control
    if config.thermal:  # RATE control
        idata.other.delta_p = None
        idata.other.wctrl_type = well_control_iface.MASS_RATE  # mass or molar rate can be chosen here
        idata.other.well_rate = config.well_rate_m3_day * idata.fluid.density  # [m^3/day] -> [kg/day]
    else:  # BHP control
        idata.other.delta_p = config.bhp_delta_p  # [bars]
        idata.other.wctrl_type = well_control_iface.BHP
        idata.other.well_rate = None

    idata.other.well_init_depth = perf_depth_start


def _set_mesh_tags(idata, matrix_tags):
    idata.mesh.bnd_tags = {}
    tags = idata.mesh.bnd_tags
    tags["BND_X-"] = 991
    tags["BND_X+"] = 992
    tags["BND_Y-"] = 993
    tags["BND_Y+"] = 994
    tags["BND_Z-"] = 995
    tags["BND_Z+"] = 996

    idata.mesh.matrix_tags = list(matrix_tags)
    idata.mesh.tags = idata.mesh.bnd_tags.copy()
    for i, mat_tag in enumerate(idata.mesh.matrix_tags, start=1):
        idata.mesh.tags[f"MATRIX_{i}"] = mat_tag


def _set_obl(idata):
    idata.obl.n_points = 400
    idata.obl.zero = 1e-9
    idata.obl.min_p = 0.0
    idata.obl.max_p = 1000.0
    idata.obl.min_t = -50.0
    idata.obl.max_t = 50.0
    idata.obl.min_z = idata.obl.zero
    idata.obl.max_z = 1 - idata.obl.zero
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
