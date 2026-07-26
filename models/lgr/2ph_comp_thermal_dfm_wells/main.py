from darts.engines import redirect_darts_output

from model import Model


if __name__ == "__main__":
    redirect_darts_output("run.log")

    model = Model(use_amr=True)
    model.init()
    model.set_output(save_initial=False)

    report_steps = [0.001] * 10 + [0.01] * 9 + [0.1] * 9 + [1.0] * 49 + [1.0] * 100
    output_props = model.physics.vars + model.output.properties
    model.output.output_to_vtk(
        ith_step=0,
        output_properties=output_props,
        engine=True,
    )
    model.output.well_output_to_vtp(
        ith_step=0,
        output_properties=output_props,
    )

    for ith_step, dt in enumerate(report_steps):
        if 0.03 < model.physics.engine.t < 0.05:
            model.data_ts.dt_max = 0.05
        elif 1 < model.physics.engine.t < 3:
            model.data_ts.dt_max = 0.1

        model.run(
            dt,
            save_well_data=False,
            save_well_data_after_run=False,
            save_reservoir_data=False,
        )
        model.adapt_lgr(verbose=True)
        model.output.output_to_vtk(
            ith_step=ith_step + 1,
            output_properties=output_props,
            engine=True,
        )
        model.output.well_output_to_vtp(
            ith_step=ith_step + 1,
            output_properties=output_props,
        )

    model.print_timers()
    model.print_stat()
