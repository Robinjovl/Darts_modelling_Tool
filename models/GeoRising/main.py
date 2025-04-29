# from darts.engines import value_vector
from model import Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

restart = False

m = Model(iapws_physics=True)
m.init() #(platform='gpu')
m.set_output()

if restart:
    for i in range(2):
        m.run(365/2)
else:
    m.run(365)
m.print_timers()
m.print_stat()

# evaluate properties, save dictionary, load dictionary
output_props = m.physics.vars + m.output.properties
timesteps, property_array = m.output.output_properties(output_properties = output_props)
m.output.save_property_array(timesteps, property_array)
loaded_timesteps, loaded_property_array = m.output.load_property_array(file_directory='output/property_array.h5')

m.output.output_to_vtk(output_properties=output_props) # output all saved time steps to vtk

# compute well time data
time_data_dict = m.output.store_well_time_data()

# save well time data
time_data_df = pd.DataFrame.from_dict(time_data_dict)
time_data_df.to_pickle(os.path.join(m.output_folder, "well_time_data.pkl")) # as a pickle file
writer = pd.ExcelWriter(os.path.join(m.output_folder, "well_time_data.xlsx")) # as an excel file
time_data_df.to_excel(writer, sheet_name='Sheet1', index=False)
writer.close()

# time_data_df.plot(x='time', y=['well_INJ_volumetric_rate_water_at_wh', 'well_PRD_volumetric_rate_water_at_wh'])\
#     .get_figure().savefig(m.output_folder + '/rates.png', dpi=100, bbox_inches='tight')
#
# ax = time_data_df.plot(x='time', y=['well_INJ_BHP', 'well_PRD_BHP'], style=['-b', '-r'], label=['INJ BHP', 'PRD BHP'])
# ax.set_ylabel('BHP [bar]')
# ax2 = ax.twinx()
# time_data_df.plot(x='time', y=['well_INJ_BHT', 'well_PRD_BHT'], ax=ax2, style=['--b', '--r'], label=['INJ BHT', 'PRD BHT'])
# ax2.set_ylabel('BHT [K]')
# lines1, labels1 = ax.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
# plt.tight_layout()
# plt.savefig('BH.png')
# # plt.show()

if restart:
    m_restarted = Model(iapws_physics=True)
    m_restarted.init()
    m_restarted.set_output(output_folder='output/restarted', save_initial=False, all_phase_props=True)

    reservoir_filename = m.sol_filepath
    well_filename = m.well_filepath
    m_restarted.load_restart_data(reservoir_filename, well_filename, timestep = 1) # restart from

    m_restarted.run(365/2, restart_dt = 1e-4)
    # m.print_timers()
    # m.print_stat()

    output_props = m_restarted.physics.vars + m_restarted.output.properties
    m_restarted.output.output_to_vtk(output_properties = output_props)

    time, cell_id, X, var_names = m.output.read_specific_data(m.sol_filepath)
    time_restarted, cell_id, X_restarted, var_names = m_restarted.output.read_specific_data(m_restarted.sol_filepath)

    # check restart position
    assert np.isclose(X[1, :, 0], X_restarted[0, :, 0], rtol=0, atol=0).all(), 'pressure mismatch at restart position'
    assert np.isclose(X[1, :, 1], X_restarted[0, :, 1], rtol=0, atol=0).all(), 'enthalpy mismatch at restart position'

    # plt.figure()
    # for i, name in enumerate(m_restarted.physics.vars):
    #     mae = np.mean(np.abs(X[-1, :, i] - X_restarted[-1, :, i]))
    #     plt.scatter(X[-1, :, i] / np.max(X[-1, :, i]), X_restarted[-1, :, i] / np.max(X_restarted[-1, :, i]).flatten(),
    #                 label=rf'MAE of {name} = {mae:.4e}')
    # plt.legend()
    # plt.show()

    # check final result
    assert np.isclose(X[-1,:,0], X_restarted[-1,:,0], rtol=1e-2, atol=0).all(), f'pressure mismatch at restart position at end of run.'
    assert np.isclose(X[-1,:,1], X_restarted[-1,:,1], rtol=1, atol=0).all(), f'enthalpy mismatch at restart position at end of run.'


