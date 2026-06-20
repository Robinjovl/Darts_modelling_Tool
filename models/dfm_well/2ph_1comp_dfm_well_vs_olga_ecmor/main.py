"""
Injection of pure gaseous CO₂ at a constant mass injection rate with a constant enthalpy into a well containing
gaseous CO₂ using a standalone well model to compare its results with those in OLGA for a thermal two-phase scenario.

The validation is presented in the ECMOR conference, 2026.

OLGA example with which this DARTS-well example is compared is available here:
    https://gitlab.com/open-darts/TODO
Comparison of the results are available in the following Excel file:
    https://gitlab.com/open-darts/TODO

Corresponding OLGA file is in my_old_laptop/Desktop/march/non-isothermal single-phase model validation with CO2

"""

import os

from darts.engines import redirect_darts_output
from darts.pipes.save_results import save_dfm_well_props
from darts.pipes.viz.plot_well_segment_property_vs_time import (
    ScenarioProfile,
    plot_well_segment_property_vs_time,
)

from model import Model


redirect_darts_output('run.log')
coupled_model = Model()
coupled_model.reservoir.grav_acceleration_for_spe = 9.80665
coupled_model.init()
coupled_model.set_output()

if 1:
    output_props = coupled_model.physics.vars + coupled_model.output.properties
    coupled_model.output.well_output_to_vtp(ith_step=0, output_properties=output_props)  # saves initial well conditions

    time_steps = [
        30 / 24 / 60,
                 ]

    for i, dt in enumerate(time_steps):
        coupled_model.run(dt)
        coupled_model.output.well_output_to_vtp(ith_step=i + 1, output_properties=output_props)

    coupled_model.print_timers()
else:
    save_dfm_well_props('I1', coupled_model)

    # plot_heat_map_pcolormesh('I1', coupled_model, show_plot=False)
    # plot_heat_map_contourf('I1', coupled_model, show_plot=False)

    figures_dir = os.path.join(coupled_model.output_folder, "figures")
    scenarios = [ScenarioProfile(coupled_model.output_folder, "DARTS-well")]
    plot_well_segment_property_vs_time(
        scenarios=scenarios,
        well_name="I1",
        property_key="pressure",
        segment_index=-1,
        output_path=os.path.join(figures_dir, "BHP_time_series.pdf"),
        y_label="BHP [bar]",
        log_x=False,
        show_legend=False,
        show_plot=False,
        marker_style="",
        line_style="--",
        line_color="r",
    )
    plot_well_segment_property_vs_time(
        scenarios=scenarios,
        well_name="I1",
        property_key="temperature",
        segment_index=-1,
        output_path=os.path.join(figures_dir, "BHT_time_series.pdf"),
        property_offset=-273.15,
        y_label="BHT [deg C]",
        log_x=False,
        show_legend=False,
        show_plot=False,
        marker_style="",
        line_style="--",
        line_color="r",
    )
