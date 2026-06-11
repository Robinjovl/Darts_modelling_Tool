from pathlib import Path

from darts.engines import redirect_darts_output
from darts.pipes.save_results import save_dfm_well_props

from model import Model
from plot_paper_comparison import plot_comparison


CASE_DIR = Path(__file__).resolve().parent
PAPER_STEADY_TIME_DAYS = 0.456869e9 / (24 * 60 * 60)


def main():
    redirect_darts_output("run.log")

    model = Model()
    model.init()
    model.set_output()

    output_props = model.physics.vars + model.output.properties
    model.output.well_output_to_vtp(ith_step=0, output_properties=output_props)

    model.run(PAPER_STEADY_TIME_DAYS)
    model.output.well_output_to_vtp(ith_step=1, output_properties=output_props)
    model.print_timers()

    save_dfm_well_props("I1", model)
    plot_comparison(model)


if __name__ == "__main__":
    main()
