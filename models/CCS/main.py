import numpy as np
import pandas as pd
import os
from darts.engines import redirect_darts_output
from model import Model


redirect_darts_output('binary.log')

m = Model(logspace=True)
m.init()
m.set_output()

data_dt = None

timesteps = [1.e-3, 0.92, 2.-1e-3, 8., 55., 300.] + [365] * 9
max_ts = [1e-3, 0.01, 0.02, 0.5, 1., 5.] + [20] * 9

for j, ts in enumerate(timesteps[:2]):

    m.data_ts.dt_mult = 2
    m.data_ts.dt_max = max_ts[j]
    m.data_ts.eta[-1] = 100

    if data_dt is not None:
        data_dt.dt_max = max_ts[j]
        m.run(data_dt, ts)
    else:
        # m.set_sim_params(max_ts=max_ts[j])
        m.run(ts)

""" Define output """
props = ['satV', 'rho_g']
output_props = ['pressure'] + props if not m.physics.thermal else ['pressure', 'temperature'] + props
# output_props += ['y' + comp for comp in m.components[2:]]
lims = {'pressure': [m.p_init, m.p_init+5], 'temperature': [m.t_inj - 30., m.t_init + 10.], 'satV': [1-m.swc, 1.]}
lims.update({'satLCO2': [0., 1.]})
# lims = None
aspect = 'equal'  # 'equal', 'auto' or float
cmap = 'RdBu_r'
logx = True

# Output to xarray and plot with plt
m.output.output_to_plt(sol_filepath=m.output.sol_filepath,  # if not provided, it will plot last timestep from engine
                       output_properties=output_props, lims=lims, plot_zeros=False,
                       aspect_ratio=aspect, logx=logx, cmap=cmap)
m.output.output_to_vtk(output_properties=output_props)

# compute and save well time data in m.output.output_folder
m.output.store_well_time_data(save_output_files=True)

m.print_timers()
m.print_stat()
