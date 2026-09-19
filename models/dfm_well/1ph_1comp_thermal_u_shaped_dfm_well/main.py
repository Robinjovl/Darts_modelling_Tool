from model import Model

from darts.engines import redirect_darts_output
from darts.pipes.save_results import save_dfm_well_props


redirect_darts_output("run.log")

model = Model()
model.init()
model.set_output()

output_properties = model.physics.vars + model.output.properties
model.output.well_output_to_vtp(ith_step=0, output_properties=output_properties)

# Report-step lengths in seconds. Their cumulative times are
# 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 900, and 1200 seconds.
report_steps_seconds = [1, 1, 3, 5, 10, 10, 30, 60, 180, 300, 300, 300]
assert sum(report_steps_seconds) == 20 * 60

elapsed_seconds = 0
for report_index, report_step_seconds in enumerate(report_steps_seconds, start=1):
    if elapsed_seconds >= 900:
        model.ts_control.dt_max = 6.0 / (24 * 60 * 60)
    elif elapsed_seconds >= 300:
        model.ts_control.dt_max = 4.0 / (24 * 60 * 60)

    model.run(report_step_seconds / (24 * 60 * 60))
    elapsed_seconds += report_step_seconds
    model.output.well_output_to_vtp(ith_step=report_index, output_properties=output_properties)

save_dfm_well_props(
    model.well_name,
    model,
    include_phase_velocities=True,
    include_phase_rates=True,
)
