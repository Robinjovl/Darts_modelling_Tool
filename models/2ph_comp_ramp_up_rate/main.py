"""
This example demonstrates the idata.well_data workflow for ramp-up rate controls.
The injector rate is introduced gradually instead of applying the full target
rate at time zero.
"""

from model import Model
from darts.engines import redirect_darts_output


if __name__ == '__main__':

    redirect_darts_output('run.log')
    n = Model()
    n.init()
    n.set_output()

    n.run(20)
    n.print_timers()
    n.print_stat()

    # compute and save well time data
    # time_data_dict = n.output.store_well_time_data(save_output_files=True)
