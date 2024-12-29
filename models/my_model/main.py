import numpy as np
import pandas as pd
from model import Model
from darts.engines import redirect_darts_output
import matplotlib.pyplot as plt

redirect_darts_output('run_log.log')
coupled_model = Model()
coupled_model.init()

coupled_model.run(10)
# n.reservoir.wells[0].control = n.physics.new_bhp_inj(100, 3*[n.zero])
# n.run_python(300, restart_dt=1e-3)
coupled_model.print_timers()
coupled_model.print_stat()
time_data = pd.DataFrame.from_dict(coupled_model.physics.engine.time_data)
time_data.to_pickle("darts_time_data.pkl")
# n.save_restart_data()
coupled_model.save_data_to_h5('solution')
writer = pd.ExcelWriter('time_data.xlsx')
time_data.to_excel(writer, sheet_name='Sheet1')
writer.close()

Xn = np.array(coupled_model.physics.engine.X, copy=False)
nc = coupled_model.physics.nc + coupled_model.physics.thermal
nb = coupled_model.reservoir.mesh.n_res_blocks

plt.figure(num=1, figsize=(12, 8), dpi=100)
for i in range(nc if nc < 3 else 3):
    plt.subplot(330 + (i + 1))
    plt.plot(Xn[i:nb*nc:nc])
plt.savefig('out.png')
