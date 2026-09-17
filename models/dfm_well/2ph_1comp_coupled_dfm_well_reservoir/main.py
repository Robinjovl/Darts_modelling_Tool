"""
Single-component (CO2), 2-phase (gas and liquid) system
Injection of liquid CO2 into a well and reservoir containing gaseous CO2

When using DFM wells:
    for a 2-phase system, use G as the name of the gaseous phase and L as the name of the liquid phase.
    for a 3-phase system, use G as the name of the gaseous phase, L_a, as the name of one liquid phase,
    and L_b as the name of the other liquid phase.
"""

from darts.engines import redirect_darts_output
from model import Model

from darts.pipes.save_results import save_dfm_well_props
from darts.pipes.viz.plot_heat_map_contourf import plot_heat_map_contourf
from darts.pipes.viz.plot_line_graphs import plot_line_graphs

redirect_darts_output('run.log')
coupled_model = Model()
coupled_model.init()
coupled_model.set_output()

output_props = coupled_model.physics.vars + coupled_model.output.properties + ["temperature"]
coupled_model.output.output_to_vtk(ith_step=0, output_properties=output_props)   # saves initial reservoir conditions
coupled_model.output.well_output_to_vtp(ith_step=0, output_properties=output_props)   # saves initial well conditions

# Two 30 s report steps. The full benchmark instead ramps up to 365 days
# (30 s -> 1 min -> ... -> 165 day report steps) with data_ts.dt_max ratcheted
# alongside, from 5 s at the second report step up to 1 day at the last ones.
report_steps = [
    0.5 / 24 / 60,  # 30 seconds
    0.5 / 24 / 60,  # 1 minute
]

for i, dt in enumerate(report_steps):

    if i == 1:
        coupled_model.data_ts.dt_max = 5 / (24 * 60 * 60)

    coupled_model.run(dt)
    coupled_model.output.output_to_vtk(ith_step=i+1, output_properties=output_props)
    coupled_model.output.well_output_to_vtp(ith_step=i+1, output_properties=output_props)

coupled_model.print_timers()

save_dfm_well_props(
    'I1',
    coupled_model,
    include_overall_composition=True,
    include_phase_velocities=True,
    include_phase_rates=True,
)

plot_heat_map_contourf('I1', coupled_model, y_axis_tick_interval=250, show_plot=False)

# Use line graphs if injection rate is controlled because the wellhead state might change a lot (for numerical reasons)
# at the beginning of simulation and this may create confusion if plot_heat_map_pcolormesh or plot_heat_map_contourf is used.
plot_line_graphs('I1', coupled_model, show_plot=False)
