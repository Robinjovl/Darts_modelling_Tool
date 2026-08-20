"""
Injection of pure gaseous CO₂ at a constant mass rate into a vertical well containing water using a standalone
well model to compare its results with those in OLGA for an isothermal two-phase scenario.
"""

from darts.engines import redirect_darts_output
from model import Model

from darts.pipes.save_results import save_dfm_well_props
from darts.pipes.viz.plot_heat_map_pcolormesh import plot_heat_map_pcolormesh
from darts.pipes.viz.plot_well_property_profiles import plot_well_property_profiles

redirect_darts_output("run.log")
coupled_model = Model()
coupled_model.init()
coupled_model.set_output()

output_props = coupled_model.physics.vars + coupled_model.output.properties
coupled_model.output.well_output_to_vtp(ith_step=0, output_properties=output_props)  # saves initial well conditions

# 5 min, the well-profiles benchmark. The time-series benchmark is 1 hour instead
# (3 report steps of 20 min with data_ts.dt_max ratcheted 5 s -> 10 s -> 15 s).
time_steps = [
    5 / 60 / 24,
]

for i, dt in enumerate(time_steps):
    coupled_model.run(dt)
    coupled_model.output.well_output_to_vtp(ith_step=i + 1, output_properties=output_props)

coupled_model.print_timers()

save_dfm_well_props(
    'I1',
    coupled_model,
    include_overall_composition=True,
    include_phase_velocities=True,
    include_phase_rates=True,
)

plot_heat_map_pcolormesh('I1', coupled_model, show_plot=False)

plot_well_property_profiles(
    'I1',
    coupled_model,
    properties=['pressure', 'vG', 'vL'],
    show_plot=False,
)
