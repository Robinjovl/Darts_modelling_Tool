from scipy import constants
import numpy as np

from .input_default import  add_wells_idata
from .case_1 import input_data_case_1
from .read_well_rates import read_well_rates

def input_data_case_1_well_rates():
    idata = input_data_case_1()
    idata.geom['case_name'] = 'case_1_well_rates'

    # shortcuts
    wdata = idata.well_data
    wdata.well_rate_fname = r'examples/well_data.xlsx' # a file with time, well history data (rate, temperature)
    wdata.bhp_limit_prod = 50.  # BHP constraint for production wells, bars
    wdata.bhp_limit_inj =  400. # BHP constraint for injection wells, bars
    wdata.bhp_correction = 1000 * constants.g * (idata.geom['z_top'] - 0.) * 1e-5  # hydrostatic depth correction for BHP hist, in case it was measured not at the well top perforation depth
    #3380. is the well top perforation depth; 0. - depth at which BHP was measured; 1000. - water density, kg/m3; 1e-5 - convert Pa to bars
    wdata.coarse_timestep = 1./24/60  # the timestep in the file might be too small, so reinterpolate the data to coarse_timestep, days
    wdata.start_date = None # days, if None - run from the beginning of the dataset
    wdata.end_date = None # days, if None - run until the end of the dataset

    add_wells_idata(idata)
    well_time_coarse = read_well_rates(idata)
    idata.well_data.controls.is_const = False
    idata.sim.time_steps = np.diff(well_time_coarse) # adjust timesteps to well control dataset time range

    return idata
