import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import Model
from darts.engines import value_vector, redirect_darts_output
from darts.tools.plot_darts import plot_phase_rate_darts
from darts.models.cicd_model import compare_solution_with_reference, get_platform

def run(platform='cpu'):

    redirect_darts_output('rum.log')
    m = Model()
    # m.params.linear_type = m.params.linear_solver_t.cpu_superlu
    m.init(platform=platform)
    m.output_to_vtk(ith_step=0, output_directory='vtk')

    if True:
        m.run(2000)
        m.print_timers()
        m.print_stat()
        time_data = pd.DataFrame.from_dict(m.physics.engine.time_data)
        time_data.to_pickle("darts_time_data.pkl")
        # m.save_restart_data()
        writer = pd.ExcelWriter('time_data.xlsx')
        time_data.to_excel(writer, sheet_name='Sheet1')
        writer.close()
    else:
        # m.load_restart_data()
        m.load_restart_data('output/solutiom.h5')
        time_data = pd.read_pickle("darts_time_data.pkl")

    time_data1 = pd.DataFrame.from_dict(m.physics.engine.time_data)

    writer = pd.ExcelWriter('time_data.xlsx')
    time_data.to_excel(writer, sheet_name='Sheet1')
    writer.close()

    plot_phase_rate_darts('P1', time_data1, 'oil')
    plot_phase_rate_darts('P5', time_data1, 'oil')

    plt.savefig('out.png')

    m.output_to_vtk(ith_step=1, output_directory='vtk')

    # for CI/CD
    failed, sim_time = compare_solution_with_reference(m=m)
    return failed

if __name__ == '__main__':
    exit(run(platform=get_platform()))