import numpy as np
import pandas as pd
from model import Model
from darts.engines import redirect_darts_output
import matplotlib.pyplot as plt

redirect_darts_output('run_log.log')
coupled_model = Model()
coupled_model.init()

output_props = ["sat_CO2_rich_phase", "sat_aqueous_phase",
                "mole_fraction_CO2__in_CO2_rich_phase", "mole_fraction_CO2__in_aqueous_phase",
                "mole_fraction_CH4__in_CO2_rich_phase", "mole_fraction_CH4__in_aqueous_phase",
                "mole_fraction_H2O__in_CO2_rich_phase", "mole_fraction_H2O__in_aqueous_phase",
                "rho_CO2_rich_phase", "rho_aqueous_phase",
                "miu_CO2_rich_phase", "miu_aqueous_phase"]
coupled_model.output_to_vtk(ith_step=0, output_properties=output_props)   # initial conditions

# for i in range(5):
#     coupled_model.run(1)
#     coupled_model.output_to_vtk(ith_step=i+1)   # save a .vtk every year

time_steps = [1/24/60,   # 1 minute
              1/24/30 - 1/24/60,   # 2 minute
              1/24/20 - 1/24/30,   # 3 minute
              1/24/12 - 1/24/20,   # 5 minute
              1/24/6 - 1/24/12,   # 10 minute
              1/24/3 - 1/24/6,   # 20 minute
              1/24/2 - 1/24/3,   # 30 minute
              1/24/6*5 - 1/24/3,   # 50 minute
              1/24 - 1/24/6*5,   # 1 hour
              2/24 - 1/24,   # 2 hour
              3/24 - 2/24,   # 3 hour
              4.99/24 - 3/24,   # 5 hour
              9.98/24 - 4.99/24,   # 10 hour
              1 - 9.98/24,   # 1 day
              2 - 1,   # 2 day
              3 - 2,   # 3 day
              5 - 3,   # 5 day
              10 - 5,   # 10 day
              19.99 - 10,   # 20 day
              30.002 - 19.99,   # 30 day
              49.99 - 30.002,   # 50 day
              100.02 - 49.99,   # 100 day
              200.001 - 100.02,   # 200 day
              365 - 200.001]   # 365 day
for i, dt in enumerate(time_steps):
    coupled_model.run(dt)
    coupled_model.output_to_vtk(ith_step=i+1, output_properties=output_props)

# coupled_model.run(1)
# # n.reservoir.wells[0].control = n.physics.new_bhp_inj(100, 3*[n.zero])
# # n.run_python(300, restart_dt=1e-3)
# coupled_model.print_timers()
# coupled_model.print_stat()
# time_data = pd.DataFrame.from_dict(coupled_model.physics.engine.time_data)
# time_data.to_pickle("darts_time_data.pkl")
# # n.save_restart_data()
# coupled_model.save_data_to_h5('solution')
# writer = pd.ExcelWriter('time_data.xlsx')
# time_data.to_excel(writer, sheet_name='Sheet1')
# writer.close()
