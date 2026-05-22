import numpy as np

from darts.engines import redirect_darts_output

from model import Model


if __name__ == "__main__":
    redirect_darts_output("run.log")

    model = Model()
    model.init()
    model.set_output()

    report_steps = np.ones(50) * 1
    output_props = model.physics.vars + model.output.properties
    model.output.output_to_vtk(ith_step=0, output_properties=output_props)

    for ith_step, dt in enumerate(report_steps):
        model.run(dt)
        model.output.output_to_vtk(
            ith_step=ith_step + 1,
            output_properties=output_props,
        )

    model.print_timers()
    model.print_stat()
