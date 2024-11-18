import numpy as np
import pandas as pd

from model import Model
from darts.engines import value_vector, redirect_darts_output

if __name__ == '__main__':
    redirect_darts_output('run.log')
    n = Model()
    # n.params.linear_type = n.params.linear_solver_t.cpu_superlu
    n.init()
    n.output_to_vtk(ith_step=0, output_directory='vtk')

    if True:
        n.run(2000)
        n.print_timers()
        n.print_stat()
        time_data = pd.DataFrame.from_dict(n.physics.engine.time_data)
        time_data.to_pickle("darts_time_data.pkl")
        # n.save_restart_data()
        writer = pd.ExcelWriter('time_data.xlsx')
        time_data.to_excel(writer, sheet_name='Sheet1')
        writer.close()
    else:
        # n.load_restart_data()
        n.load_restart_data('output/solution.h5')
        time_data = pd.read_pickle("darts_time_data.pkl")

    time_data1 = pd.DataFrame.from_dict(n.physics.engine.time_data)
    from darts.tools.plot_darts import *
    writer = pd.ExcelWriter('time_data.xlsx')
    time_data.to_excel(writer, sheet_name='Sheet1')
    writer.close()
    plot_phase_rate_darts('P1', time_data1, 'oil')
    plot_phase_rate_darts('P5', time_data1, 'oil')

    plt.savefig('out.png')

    n.output_to_vtk(ith_step=1, output_directory='vtk')
