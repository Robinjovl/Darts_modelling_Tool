import numpy as np
import os

from darts.input.input_data import InputData
from darts.models.darts_model import DataTS
from darts.engines import sim_params

class InputDataGeom():  # to group geometry input data
    def __init__(self):
        pass

def get_case_files(case: str):
    prefix = os.path.join('meshes', case[:case.rfind('_')])
    grid_file = os.path.join(prefix, 'grid.grdecl')
    prop_file = os.path.join(prefix, 'reservoir.in')
    sch_file = os.path.join(prefix, 'sch.inc')
    assert os.path.exists(grid_file), 'cannot open' + grid_file
    assert os.path.exists(prop_file), 'cannot open' + prop_file
    assert os.path.exists(sch_file), 'cannot open' + sch_file
    return grid_file, prop_file, sch_file

def input_data_base(idata: InputData, case: str):
    dt = 365.25  # one report timestep length, [days]
    n_time_steps = 20
    idata.sim.time_steps = np.zeros(n_time_steps) + dt

    # time stepping and convergence parameters
    idata.sim.DataTS = DataTS(n_vars=0)
    idata.sim.DataTS.dt_first = 0.01
    idata.sim.DataTS.dt_mult = 2
    idata.sim.DataTS.dt_max = 92
    idata.sim.DataTS.newton_tol = 1e-2
    idata.sim.DataTS.linear_tol = 1e-4
    # use direct linear solver:
    #idata.sim.DataTS.linear_type = sim_params.linear_solver_t.cpu_superlu

    idata.generate_grid = 'generate' in case
    idata.geom = InputDataGeom()
    geom = idata.geom  # a short name
    well_data = idata.well_data  # a short name

    # grid processing parameters
    geom.minpv = 1e-5  # minimal pore volume threshold to set cells inactive, m^3

    # properties processing parameters
    # for the isothermal physics - porosity cutoff value
    # for thermal physics - poro and perm with lower values will be replaced by geom.min_poro:
    #     poro - to keep those cells active even though they have poro=0
    #     perm - to avoid convergence issues
    geom.min_poro = 1e-5

    # allow small flow to avoid pressure jumps
    # since there might pressure change appear due to the temperature change
    geom.min_perm = 1e-5

    # boundary conditions
    geom.bound_volume = 1e10 # lateral boundary volume, m^3

    geom.faultfile = None  # a text file with fault locations and multipliers

    idata.geom.well_index = None  # well index for flow, if None - will be computed by default
    idata.geom.well_indexD = 0.   # well index for thermal conductivity (for closed-loops/U-shaped wells); turned off

    if idata.generate_grid:
        idata.rock.poro = 0.2
        idata.rock.permx = 100  # mD
        idata.rock.permy = 100  # mD
        idata.rock.permz = 10   # mD

    else:  # read from files
        # setup filenames
        gridfile, propfile, schfile = get_case_files(case)
        idata.gridfile = gridfile
        idata.propfile = propfile if os.path.exists(propfile) else gridfile
        idata.schfile = schfile
        # read from a file to idata.well_data.wells[well_name].perforations
        idata.well_data.read_and_add_perforations(idata.schfile)
    idata.grid_out_dir = None  # output path for the generated grid and prop files

    # rock compressibility
    idata.rock.compressibility = 1e-5  # [1/bars]
    idata.rock.compressibility_ref_p = 1 # [bars]
    idata.rock.compressibility_ref_T = 273.15  # [K]

    #########################################################################
    # only for the thermal case (Geothermal physics):
    geom.burden_layers = 4  # the number of additional (generated on-the-fly) overburden/underburden layers
    geom.burden_init_thickness = 10  # first over/under burden layer thickness, [m.]
    idata.rock.burden_prop = 1e-5  # perm and poro value for burden layers

    idata.rock.conduction_shale = 2.2 * 86.4 # Shale conductivity kJ/m/day/K
    idata.rock.conduction_sand = 3 * 86.4 # Sandstone conductivity kJ/m/day/K
    idata.rock.hcap_shale = 2300 # Shale heat capacity kJ/m3/K
    idata.rock.hcap_sand = 2450 # Sandstone heat capacity kJ/m3/K

    # the cells with lower poro will be treated as shale when setting the rock thermal properties
    idata.rock.poro_shale_threshold = 1e-3
    ############################################################################
