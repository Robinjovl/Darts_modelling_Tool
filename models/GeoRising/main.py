from darts.engines import value_vector

from model import Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from darts.models.cicd_model import compare_solution_with_reference, get_platform

def run(platform='cpu'):

    m = Model(iapws_physics=True)

    m.init(platform=platform)
    m.output_to_vtk(ith_step=0, output_directory='vtk')
    m.run(3650)
    m.print_timers()
    m.print_stat()
    m.output_to_vtk(ith_step=1, output_directory='vtk')


    td = pd.DataFrame.from_dict(m.physics.engine.time_data)
    td.to_pickle("darts_time_data.pkl")
    writer = pd.ExcelWriter('time_data.xlsx')
    td.to_excel(writer, sheet_name='Sheet1')
    writer.close()

    string = 'PRD : temperature'
    ax1 = td.plot(x='time', y=[col for col in td.columns if string in col])
    ax1.plot([0, 3650],[348, 348])
    ax1.tick_params(labelsize=14)
    ax1.set_xlabel('Days', fontsize=14)
    ax1.legend(['temp', 'limit'], fontsize=14)
    plt.grid()
    # plt.show()
    plt.savefig('out.png')

    # for CI/CD
    failed, sim_time = compare_solution_with_reference(m=m)
    return failed

if __name__ == '__main__':
    exit(run(platform=get_platform()))