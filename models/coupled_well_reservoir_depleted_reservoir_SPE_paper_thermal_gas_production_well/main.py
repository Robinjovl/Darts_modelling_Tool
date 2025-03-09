"""
Use the SPE paper scenario, but here the reservoir has a high pressure and we want to produce CH4 via the well
Use a large volume as the wellhead of the well to keep the pressure of the wellhead constant
Larger time steps can be used compared to the thermal scenario of the SPE paper
Smaller NR tolerance can be used compared to the thermal scenario of the SPE paper
Uniform temperature in the entire reservoir
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import Model
from darts.engines import redirect_darts_output
from darts.tools.hdf5_tools import load_hdf5_to_dict
from darts.wells.save_results import save_segments_primary_vars_and_phase_props
from darts.wells.visualize_results_heat_maps import visualize_results_heat_maps
from darts.wells.visualize_results_line_graphs import visualize_results_line_graphs

redirect_darts_output('run_log.log')
coupled_model = Model()
coupled_model.init()

if 1:
    output_props = ["sat_CO2/C1_rich_phase", "sat_aqueous_phase",
                    "mole_fraction_CO2__in_CO2/C1_rich_phase", "mole_fraction_CO2__in_aqueous_phase",
                    "mole_fraction_CH4__in_CO2/C1_rich_phase", "mole_fraction_CH4__in_aqueous_phase",
                    "mole_fraction_H2O__in_CO2/C1_rich_phase", "mole_fraction_H2O__in_aqueous_phase",
                    "rho_CO2/C1_rich_phase", "rho_aqueous_phase",
                    "miu_CO2/C1_rich_phase", "miu_aqueous_phase",
                    "enthalpy_CO2/C1_rich_phase", "enthalpy_aqueous_phase"]
    coupled_model.output_to_vtk(ith_step=0, output_properties=output_props)   # initial conditions

    # for i in range(5):
    #     coupled_model.run(1)
    #     coupled_model.output_to_vtk(ith_step=i+1)   # save a .vtk every day

    time_steps = [1/24/60,   # 1 minute
                  1.0001/24/30 - 1/24/60,   # 2 minute
                  1/24/20 - 1.0001/24/30,   # 3 minute
                  1.0001/24/12 - 1/24/20,   # 5 minute
                  1.0001/24/6 - 1.0001/24/12,   # 10 minute
                  1.00001/24/3 - 1.0001/24/6,   # 20 minute
                  1/24/2 - 1.00001/24/3,   # 30 minute
                  1/24/6*5 - 1.00001/24/3,   # 50 minute
                  1/24 - 1.00001/24/6*5,   # 1 hour
                  2/24 - 1.00001/24,   # 2 hour
                  3/24 - 1.00001/24,   # 3 hour
                  4.99/24 - 1.00001/24,   # 5 hour
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

else:
    well_data_file_path = "output/well_data.h5"
    h5_well_data = load_hdf5_to_dict(well_data_file_path)
    save_segments_primary_vars_and_phase_props(h5_well_data, coupled_model)

    primary_vars_and_phase_props_file_address = "output/stored_primary_vars_and_phase_props.pkl"
    visualize_results_heat_maps(primary_vars_and_phase_props_file_address, h5_well_data, coupled_model)
    # visualize_results_line_graphs(primary_vars_and_phase_props_file_address, h5_well_data, coupled_model, 3)
