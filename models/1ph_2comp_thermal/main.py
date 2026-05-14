import numpy as np
import pandas as pd
import os

from model import Model
from darts.engines import value_vector, redirect_darts_output
import matplotlib.pyplot as plt


redirect_darts_output('run.log')

# initialize model
n = Model()
n.init()
n.set_output()

# run model
n.run(1000)
n.print_timers()
n.print_stat()

# compute and save well time data
time_data_dict = n.output.store_well_time_data(save_output_files=True)

# plot solution
Xn = np.array(n.physics.engine.X, copy=False)
nc = n.physics.nc + n.physics.thermal
nb = n.reservoir.mesh.n_res_blocks

plt.figure(num=1, figsize=(12, 8), dpi=100)
str_title = ["pressure", 'water fraction', 'temperature']
for i in range(nc):
    plt.subplot(nc * 100 + 10 + (i + 1))
    plt.plot(Xn[i:nb*nc:nc])
    plt.title(str_title[i])
plt.savefig('out.png')
