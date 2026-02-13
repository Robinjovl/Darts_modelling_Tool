# Section of the Python code where we import all dependencies on third party Python modules/libaries or our own
# libraries (exposed C++ code to Python, i.e. darts.engines && darts.physics)
import os
import numpy as np
import pandas as pd
from model import Model
from darts.engines import redirect_darts_output
import shutil
from datetime import datetime
from darts.tools.plot_darts import plot_temp_darts
import pickle
from darts.input.input_data import InputData
from set_case import set_input_data
from make_plots import plot_well_results

def run_simulation(idata : InputData, platform : str ='cpu'):
    print('Running simulation for case', idata.geom['case_name'])

    output_directory = 'sol_' + idata.geom['case_name']

    os.makedirs(output_directory, exist_ok=True)

    redirect_darts_output(os.path.join(output_directory, 'simulation.log'))

    m = Model(idata)

    m.init(verbose = True, platform=platform)
    m.set_output(output_folder = output_directory)

    output_vtk_period = 12  # output each output_vtk_period-th step results to tk

    output_properties_main = m.physics.vars  # only main variables
    output_properties_with_temperature = output_properties_main + ['temperature']

    timesteps, property_array = m.output.output_properties(output_properties=output_properties_with_temperature,
                                                           timestep=0, engine=True)

    # add custom arrays to property_array: fracture index and fracture aperture to be saved to vtk files
    n_fracs = m.reservoir.discretizer.frac_cells_tot
    #frac_index = np.arange(n_fracs).reshape((1, n_fracs))
    frac_aper = m.reservoir.frac_aper if not np.isscalar(m.reservoir.frac_aper) else np.zeros((1, n_fracs)) + m.reservoir.frac_aper
    custom_arrays = {'frac_aperture': frac_aper} #'frac_index': frac_index
    if n_fracs > 0:
        property_array.update(custom_arrays)

    m.output.output_to_vtk(output_data=[timesteps, property_array], ith_step=0, output_directory=output_directory)

    # m.output.save_data_to_h5(kind = 'reservoir')
    ###m.output.output_to_vtk(ith_step=0, output_directory=output_directory)

    sim_time = 0.
    m.print_range(sim_time, part='cells')
    m.print_range(sim_time, part='fracs')

    # Run over all reporting time-steps:
    for ith_step, dt in enumerate(m.idata.sim.time_steps):
        m.set_well_controls_idata(time=sim_time, verbose=True)
        m.run(dt)

        if ith_step % output_vtk_period == 0:
            timesteps, property_array = m.output.output_properties(output_properties=output_properties_with_temperature,
                                                                   timestep=ith_step+1, engine=True)
            m.output.output_to_vtk(output_data=[timesteps, property_array], ith_step=ith_step+1, output_directory=output_directory)

        sim_time += dt
        m.print_range(sim_time, part='cells')
        m.print_range(sim_time, part='fracs')
        m.physics.engine.report()

    m.print_timers()
    m.print_stat()

    # compute and save well time data
    time_data_dict = pd.DataFrame.from_dict(m.physics.engine.time_data_report)
    #time_data_dict = m.output.store_well_time_data(save_output_files=True)

    time_data_dict['Time(years)'] = time_data_dict['time'] / 365.25
    # add history data to the dataframe (drop the last value to make dimension consistent with time_data_report)
    if hasattr(m.idata.well_data, 'well_rate_hist'):
        time_data_dict['rate_hist'] = m.idata.well_data.well_rate_hist
        time_data_dict['BHP_hist'] = m.idata.well_data.well_bhp_hist[:-1]
        time_data_dict['BHT_hist'] = m.idata.well_data.well_bht_hist[:-1] - 273.15 # K to degrees

    # write the files
    df = pd.DataFrame(time_data_dict)
    df.to_pickle(os.path.join(output_directory, "well_time_data.pkl"))
    xls_fname = os.path.join(output_directory, 'well_time_data.xlsx')
    with pd.ExcelWriter(xls_fname) as writer:
        df.to_excel(writer, sheet_name='Sheet1')

    plot_well_results(output_directory)

    return m

if __name__ == "__main__":

    t1 = datetime.now()
    print(t1)

    input_data = set_input_data('case_1')
    run_simulation(input_data)

    t2 = datetime.now()
    print((t2 - t1).total_seconds())
