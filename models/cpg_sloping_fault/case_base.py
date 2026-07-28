import numpy as np
import os

from darts.input.input_data import InputData, linear_solver_types
from darts.models.darts_model import DataTS
from darts.engines import sim_params

class InputDataGeom():  # to group geometry input data
    def __init__(self):
        pass

def get_case_files(case: str, grid_file: str, prop_file: str, sch_file: str):
    # get full paths for the files, assuming they are in meshes/case (with dropped part after first '_') folder
    # checks file existence and unzips if needed
    prefix = os.path.join('meshes', case[:case.find('_')])
    grid_file_ = os.path.join(prefix, grid_file)
    prop_file_ = os.path.join(prefix, prop_file)
    sch_file_ = os.path.join(prefix, sch_file)
    from darts.tools.keyword_file_tools import compressed_file
    for fname in [grid_file_, prop_file_]:
        compressed_file(fname, verbose=True)
    assert os.path.exists(grid_file_), 'cannot open ' + grid_file_
    assert os.path.exists(prop_file_), 'cannot open ' + prop_file_
    assert os.path.exists(sch_file_), 'cannot open ' + sch_file_
    return grid_file_, prop_file_, sch_file_

def input_data_base(idata: InputData, case: str):
    dt = 365.25  # one report timestep length, [days]
    n_time_steps = 20
    idata.sim.time_steps = np.zeros(n_time_steps) + dt

    # time stepping and convergence parameters
    idata.sim.DataTS = DataTS(n_vars=0)
    idata.sim.DataTS.dt_first = 0.01
    idata.sim.DataTS.dt_mult = 2
    idata.sim.DataTS.dt_max = 92
    idata.sim.DataTS.linear_tol = 1e-4
    # Nonlinear (Newton) tolerance: set on idata (not DataTS); Model_CPG.set_solver()
    # reads it and applies it to the nonlinear-solver spec.
    idata.sim.newton_tolerance = 1e-2
    # use direct linear solver:
    #idata.sim.DataTS.linear_type = sim_params.linear_solver_t.cpu_superlu
    # optional: use PETSc linear solver
    #idata.sim.DataTS.linear_type = linear_solver_types.CPU_PETSC_CPR
    #idata.sim.DataTS.linear_print_level = 0
    # optional: use PARDISO linear solver
    #idata.sim.DataTS.linear_type = linear_solver_types.CPU_PARDISO

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
    else:  # read grid and properties from files
        # setup default filenames
        idata.gridfile = 'grid.grdecl'
        idata.propfile = 'reservoir.in'
        idata.schfile = 'sch.inc'
        idata.gridfile, idata.propfile, idata.schfile = get_case_files(case, idata.gridfile, idata.propfile, idata.schfile)
        idata.propfile = idata.propfile if os.path.exists(idata.propfile) else idata.gridfile
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

    idata.supress_all_output = False
