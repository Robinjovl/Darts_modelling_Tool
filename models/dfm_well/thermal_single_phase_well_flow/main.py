"""
Injection of gaseous CO₂ at a constant mass injection rate with a constant specific enthalpy into a well containing
gaseous CO₂ using a standalone well model to compare its results with those in DWell for a thermal single-phase scenario.

Lessons learned:
    1- Pressure results of DWell and DARTS-well are very similar, but there is a small difference in temperature results
      because in DWell, kinetic energy is considered, and so if it is considered in DARTS-well, the results will
      become identical.

    2- To have identical results with DWell, not only Peaceman model for well index should be considered, but also
       the size of the reservoir block should be accurately specified because the reservoir block size will be used
       to calculate well index in addition to permeability, well radius, etc.

DWell example with which this DARTS-well example is compared is available here:
    https://gitlab.com/open-darts/dwell/-/blob/compare_with_darts_well/examples/single_phase_non_isothermal/for_comparison_with_darts_well.py?ref_type=heads
Comparison of the results are available in the following Excel file:
    https://gitlab.com/open-darts/dwell/-/blob/compare_with_darts_well/examples/single_phase_non_isothermal/dwell_darts_well_comparison.xlsx?ref_type=heads

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from model import Model
from darts.engines import redirect_darts_output
from darts.tools.hdf5_tools import load_hdf5_to_dict
from darts.pipes.save_results import save_segments_primary_vars_and_phase_props
from darts.pipes.viz.plot_heat_map_pcolormesh import plot_heat_map_pcolormesh
from darts.pipes.viz.plot_line_graphs import plot_line_graphs

redirect_darts_output('run_log.log')
coupled_model = Model()
coupled_model.reservoir.grav_acceleration_for_spe = 9.80665
coupled_model.init()
coupled_model.set_output()

if 0:
    output_props = coupled_model.physics.vars + coupled_model.output.properties
    coupled_model.output_to_vtk(ith_step=0, output_properties=output_props)   # saves initial conditions

    time_steps = [10 / 24 / 60,   # 10 minutes
                 ]

    for i, dt in enumerate(time_steps):
        coupled_model.run(dt)
        coupled_model.output_to_vtk(ith_step=i+1, output_properties=output_props)

else:
    well_data_file_path = os.path.join(coupled_model.output.output_folder, "well_data.h5")
    h5_well_data = load_hdf5_to_dict(well_data_file_path)
    save_segments_primary_vars_and_phase_props(h5_well_data, coupled_model)

    primary_vars_and_phase_props_file_address = os.path.join(coupled_model.output.output_folder, "well_primary_vars_and_phase_props.pkl")
    plot_heat_map_pcolormesh(primary_vars_and_phase_props_file_address, h5_well_data, coupled_model)
    plot_line_graphs(primary_vars_and_phase_props_file_address, h5_well_data, coupled_model, 3)
