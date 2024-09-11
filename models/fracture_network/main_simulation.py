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

def run_simulation(input_data):
    print('Running simulation for case', input_data['case_name'])

    output_directory = 'sol_' + input_data['case_name']

    # rename output dir if exists
    if os.path.exists(output_directory):
        ren_fname = output_directory + '_prev'
        if os.path.exists(ren_fname):
            shutil.rmtree(ren_fname)
        os.renames(output_directory, ren_fname)

    os.makedirs(output_directory)

    redirect_darts_output(os.path.join(output_directory, 'out.log'))

    m = Model(input_data)

    m.init(verbose=True, output_folder = output_directory)

    # Specify some other time-related properties (NOTE: all time parameters are in [days])
    size_report_step = 60  # Size of the reporting step 
    num_report_steps = 12*5   # Number of reporting steps (see above)
    output_vtk_period = 12  # output each output_vtk_period-th step results to tk

    m.save_data_to_h5(kind = 'solution')
    m.output_to_vtk(ith_step=0, output_directory=output_directory)

    sim_time = 0.
    m.print_range(sim_time, part='cells')
    m.print_range(sim_time, part='fracs')

    # Run over all reporting time-steps:
    for ith_step in range(num_report_steps):
        m.run(size_report_step)
        m.save_data_to_h5(kind='solution')

        if ith_step % output_vtk_period == 0:
            m.output_to_vtk(ith_step=ith_step+1, output_directory=output_directory)

        sim_time += size_report_step
        m.print_range(sim_time, part='cells')
        m.print_range(sim_time, part='fracs')

    m.print_timers()
    m.print_stat()

    time_data = pd.DataFrame.from_dict(m.physics.engine.time_data)
    time_data['Time (years)'] = time_data['time']/365.

    xls_fname = os.path.join(output_directory, 'time_data.xlsx')
    with pd.ExcelWriter(xls_fname) as writer:
        time_data.to_excel(writer, sheet_name='Sheet1')

    pkl_fname = os.path.join(output_directory, 'time_data.pkl')
    pickle.dump(time_data, open(pkl_fname, 'wb'))

if __name__ == "__main__":

    t1 = datetime.now()
    print(t1)

    input_data = set_input_data('case_1')
    run_simulation(input_data)

    t2 = datetime.now()
    print((t2 - t1).total_seconds())
