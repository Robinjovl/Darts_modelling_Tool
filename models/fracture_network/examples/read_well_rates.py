import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

from darts.engines import well_control_iface

def read_well_rates(idata):
    '''
    read xls file and initialize well controls in idata
    well rate and injection temperature columns are used to set well controls, the rest can be used for history matching
    '''

    # shortcuts
    wdata = idata.well_data
    # input data should contain next parameters for this function:
    well_rate_fname = wdata.well_rate_fname
    bhp_limit_prod = wdata.bhp_limit_prod # BHP constraint for production wells, bars
    bhp_limit_inj =  wdata.bhp_limit_inj  # BHP constraint for injection wells, bars
    bhp_correction = wdata.bhp_correction  # hydrostatic depth correction for BHP
    coarse_timestep = wdata.coarse_timestep  # the timestep in the file might be too small, so reinterpolate the data to coarse_timestep, days

    # read the file into a pandas dataset
    well_df = pd.read_excel(well_rate_fname)
    # convert to numpy arrays, filter, and convert units
    well_df = well_df.dropna(subset=['times'])  # drop rows with NaN in 'times' column
    well_time = well_df['times']
    well_rates = np.array(well_df['Flow[m3/h]']) * 24.0  # to m3/days
    well_bhp_hist = np.array(well_df['INJ WHP[bar]']) + bhp_correction
    well_inj_tempr = np.array(well_df['INJ T[C]']) + 273.15  # to K
    well_prd_tempr = np.array(well_df['PROD T[c]']) + 273.15  # to K

    well_time = well_time.dt.to_pydatetime()
    sec2days = 1.0 / 86400.0
    # Reference date (e.g., the first date)
    ref_date = well_time[0]
    well_time_days = np.vectorize(lambda x: (x - ref_date).total_seconds())(
        well_time) * sec2days  # convert to days starting from 0

    # remove backward time shift in the data, if there is
    dt = np.diff(well_time_days)
    i_start_list = np.where(dt < 0)[0]  # find where time is reversed
    i_end_list = []
    for i_start in i_start_list:  # find where time becomes larger again
        i = i_start
        while well_time_days[i] <= well_time_days[i_start]:
            i += 1
        i_end_list.append(i)
    filter = np.ones(well_time_days.shape, dtype=bool)
    for i_start, i_end in zip(i_start_list, i_end_list):
        filter[i_start + 1:i_end] = False  # remove the backward part
    if wdata.start_date is not None:
        filter[well_time_days < wdata.start_date] = False  # keep only after start_date
    if wdata.end_date is not None:
        filter[well_time_days > wdata.end_date] = False
    well_time_days = well_time_days[filter]
    well_rates = well_rates[filter]
    well_bhp_hist = well_bhp_hist[filter]
    well_prd_tempr = well_prd_tempr[filter]
    well_inj_tempr = well_inj_tempr[filter]

    dt = np.diff(well_time_days)
    assert np.all(dt >= 0), "Error in well time data, it should be monotonically increasing"

    well_time_days -= well_time_days.min()  # shift to zero
    well_time_coarse = np.arange(well_time_days.min(), well_time_days.max(),
                                 coarse_timestep)  # daily time steps (coarsened)

    well_rates_cumulative = (well_rates[:-1] * np.diff(well_time_days)).cumsum()  # -1 to match diff size
    well_time_days_1 = well_time_days[:-1]  # -1 to match well_rates_cumulative's size
    well_time_coarse = well_time_coarse[well_time_coarse < well_time_days_1.max()] # cut to avoid interpolation error
    well_rates_cumulative_coarse = interp1d(well_time_days_1, well_rates_cumulative, kind='linear')(well_time_coarse)
    well_rates = np.diff(well_rates_cumulative_coarse) / coarse_timestep  # back to rates but in coarse timesteps now
    wdata.well_rate_hist = well_rates

    wdata.well_bhp_hist = interp1d(well_time_days, well_bhp_hist, kind='linear')(well_time_coarse)
    wdata.well_bht_hist = interp1d(well_time_days, well_prd_tempr, kind='linear')(well_time_coarse)

    # fill input data
    for t, q, inj_tempr, bhp_inj_hist in zip(well_time_coarse, well_rates, well_inj_tempr,
                                             wdata.well_bhp_hist):
        # TODO adjust the well rate to have mass balance using difference in density due to tempr difference
        # rho_mult = calc_rho_mult(prod_bhp, inj_bhp, prod_temp, well_inj_tempr)
        # TODO account for heat loss for prod and warming for inj in the wellbore
        # use bhp_inj_hist instead of bhp_limit_inj
        if q == 0:  # workaround for zero rate since inj well with zero rate doesn't work properly in darts, but maybe it is not needed actually
            wdata.add_prd_rate_control(name='I1', time=t, rate=q, rate_type=well_control_iface.VOLUMETRIC_RATE,
                                       bhp_constraint=bhp_limit_prod)  # m3/day | bars
        else:
            wdata.add_inj_rate_control(name='I1', time=t, rate=q, rate_type=well_control_iface.VOLUMETRIC_RATE,
                                       bhp_constraint=bhp_limit_inj, temperature=inj_tempr)  # m3/day | bars | K
        wdata.add_prd_rate_control(name='P1', time=t, rate=q, rate_type=well_control_iface.VOLUMETRIC_RATE,
                                   bhp_constraint=bhp_limit_prod)  # m3/day | bars

    # this can be used to adjust the timesteps: idata.sim.time_steps = np.diff(well_time_coarse)
    return well_time_coarse
