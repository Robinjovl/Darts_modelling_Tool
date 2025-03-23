import pandas as pd
from model import Model
from darts.engines import redirect_darts_output

redirect_darts_output('run_log.log')
coupled_model = Model()
coupled_model.init()

coupled_model.run(1)
time_data = pd.DataFrame.from_dict(coupled_model.physics.engine.time_data)
time_data.to_pickle("darts_time_data.pkl")
coupled_model.save_data_to_h5('solution')

#%% Store output as .vtk files
step = 0
while True:
    try:
        # Export the VTK file
        # if step % 50 == 0:
        coupled_model.output_to_vtk(ith_step=step, output_directory='output')
        step += 1
    except:  # Bare except, since this is not an error
        break  # Gracefully exit the loop without printing anything
