from darts.engines import redirect_darts_output
from darts.pipes.save_results import save_dfm_well_props

from model import Model
from comparison_report import print_r2_report
from plot_paper_comparison import plot_comparison


PAPER_STEADY_TIME_DAYS = 0.456869e9 / (24 * 60 * 60)
REFERENCE_PROFILE_FILE = "digitized_t2well_paper_profiles_40_degC_CO2.csv"


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
    plot_comparison(model, REFERENCE_PROFILE_FILE)
    print_r2_report(model, REFERENCE_PROFILE_FILE)


if __name__ == "__main__":
    main()
