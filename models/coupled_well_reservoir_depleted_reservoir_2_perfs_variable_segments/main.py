import numpy as np
import pandas as pd
from model import Model
from darts.engines import redirect_darts_output
import matplotlib.pyplot as plt

redirect_darts_output('run_log.log')
coupled_model = Model()
coupled_model.init()

output_props = coupled_model.physics.property_operators[0].props_name
coupled_model.output_to_vtk(ith_step=0, output_properties=output_props)   # initial conditions

# for i in range(5):
#     coupled_model.run(1)
#     coupled_model.output_to_vtk(ith_step=i+1)   # save a .vtk every day

time_steps = [1/24/60,   # 1 minute
              1/24/30 - 1/24/60,   # 2 minute
              1.001/24/20 - 1/24/30,   # 3 minute
              1/24/12 - 1.001/24/20,   # 5 minute
              1/24/6 - 1/24/12,   # 10 minute
              1/24/3 - 1/24/6,   # 20 minute
              1/24/2 - 1/24/3,   # 30 minute
              1/24/6*5 - 1/24/3,   # 50 minute
              1/24 - 1/24/6*5,   # 1 hour
              2/24 - 1/24,   # 2 hour
              3/24 - 2/24,   # 3 hour
              4.99/24 - 3/24,   # 5 hour
              9.98/24 - 4.99/24,   # 10 hour
              1.0001 - 9.98/24,   # 1 day
              1.9999 - 1.0001,   # 2 day
              3 - 1.9999,   # 3 day
              5.001 - 3,   # 5 day
              10 - 5.001,   # 10 day
              19.99 - 10,   # 20 day
              30.002 - 19.99,   # 30 day
              49.99 - 30.002,   # 50 day
              100.02 - 49.99,   # 100 day
              200.001 - 100.02,   # 200 day
              365 - 200.001]   # 365 day

for i, dt in enumerate(time_steps):
    coupled_model.run(dt)
    coupled_model.output_to_vtk(ith_step=i+1, output_properties=output_props)
