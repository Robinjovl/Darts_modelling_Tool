import numpy as np
import pandas as pd
import os
from darts.engines import redirect_darts_output
from model import Model

import matplotlib.pyplot as plt
redirect_darts_output('binary.log')


class DataTS:
    dt_min: float
    omega: float
    eta: float
    dt_max: float
    tol_res: float
    tol_wel: float
    tol_sta: float
    max_it_nl: int
    def __init__(self, nc):
        self.eta = 1e20 * np.ones(nc)  # avoid limitation for changes

        # default values
        self.dt_min = 1e-2
        self.omega = 1
        self.dt_max = 365
        self.tol_res = 1e-2
        self.tol_wel_mult = 1
        self.tol_sta = 1e-2
        self.max_it_nl = 12


filename = 'out'

# define the model
m = Model()

logspace = True
if logspace:
    dr = 1.
    R1 = 1000.
    nz = 24

    poro = np.ones(nz) * 0.001
    perm = np.ones(nz) * 0.001
    poro[4:20] = 0.2
    perm[4:20] = 20
    perm[6:12] = 100

    m.set_reservoir(dr=dr, R1=R1, logspace=logspace, nz=nz, dz=5, poro=poro, perm=perm)

else:
    # lower detfurth
    (nr, nz) = (10, 24)
    dr = 1.
    R1 = nr * dr

    if 0:
        poro = 0.2
        perm = 100

    elif 1:
        poro = np.ones((nr, nz)) * 0.001
        perm = np.ones((nr, nz)) * 0.001
        poro[:, 4:20] = 0.2
        perm[:, 4:20] = 20
        perm[:, 6:12] = 100

    else:
        poro = np.ones((nr, nz)) * 0.075
        perm = np.ones((nr, nz)) * 0.29
        # upper detfurth
        perm[:, :16] = 12.6
        # hardegsen
        poro[:, :10] = 0.09
        perm[:, :10] = 24
        # hardegsen high perm
        poro[:, :8] = 0.2
        perm[:, :8] = 550
        # hardegsen
        poro[:, :6] = 0.09
        perm[:, :6] = 24
        # caprock
        poro[:, :4] = 0.01
        perm[:, :4] = 0.01

    m.set_reservoir(dr=dr, R1=R1, logspace=False, nz=nz, dz=5, poro=poro, perm=perm)

# Specify initial and injection conditions
m.p_init = 100.
m.p_inj = m.p_init + 1.
m.t_init = 350.
m.t_inj = 300.
m.swc = 0.25

zero = 1e-12
m.set_physics(zero, n_points=1001, temperature=None, ph=False, vl_phases=False)
m.inj_stream = [0.001] if m.components[0] == "H2O" else [0.999]

m.set_sim_params(first_ts=1e-7, mult_ts=2, max_ts=1., tol_newton=1e-6, tol_linear=1e-6, it_newton=8,
                 it_linear=50,
                 # newton_type=m.params.newton_global_chop,  # Type of newton method (related to chopping strategy?)
                 # newton_params=value_vector([0.2]),  # Probably chop-criteria(?)
                 )
# m.params.nonlinear_norm_type = m.params.L1
# m.params.linear_type = m.params.cpu_superlu

# init the model
m.init()
# set the output
m.set_output()

""" DEFINE OUTPUT """
props = ['satV', 'rho_g']
output_props = ['pressure'] + props if not m.physics.thermal else ['pressure', 'temperature'] + props
# output_props += ['y' + comp for comp in m.components[2:]]
lims = {'pressure': [m.p_init, m.p_init+5], 'temperature': [m.t_inj - 30., m.t_init + 10.], 'satV': [1-m.swc, 1.]}
lims.update({'satLCO2': [0., 1.]})
# lims = None
aspect = 'equal'  # 'equal', 'auto' or float
cmap = 'RdBu_r'
logx = True

timestep, property_array = m.output.output_properties(output_properties=output_props, timestep=-1)

m.reservoir.output_to_plt(data=property_array, output_props=output_props, lims=lims, plot_zeros=False,
                          aspect_ratio=aspect, logx=logx, cmap=cmap)
# m.output.output_to_vtk(ith_step=0, output_properties=output_props)  # initial conditions
plt.savefig('step0.png', format='png')

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

    timesteps, property_array = m.output.output_properties(output_properties=output_props, timestep=-1)

    m.reservoir.output_to_plt(data=property_array, output_props=output_props, lims=lims, plot_zeros=False,
                              aspect_ratio=aspect, logx=logx, cmap=cmap)

    plt.savefig('step' + str(j+1) + '.png', format='png')

    # compute and save well time data in m.output_folder
    time_data_dict = m.output.store_well_time_data()

m.print_timers()
m.print_stat()
