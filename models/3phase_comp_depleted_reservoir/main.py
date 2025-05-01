import numpy as np
import pandas as pd
import sys
from model import Model
from darts.engines import redirect_darts_output

if __name__ == '__main__':
    redirect_darts_output("run.log")

    print('START')

    n = Model()
    n.init()

    output_props = n.physics.property_operators[0].props_name
    n.output_to_vtk(ith_step=0, output_properties=output_props)  # saves initial conditions

    simulation_time = 0.010001 * np.ones(100)
    for i, dt in enumerate(simulation_time):
        n.run(dt)
        n.output_to_vtk(ith_step=i + 1, output_properties=output_props)
    n.print_timers()
    n.print_stat()
    n.save_data_to_h5('solution')

    print('END')
