from darts.engines import redirect_darts_output
from darts.pipes.save_results import save_dfm_well_props

from model import Model
from comparison_report import print_r2_report
from plot_paper_comparison import plot_comparison


CASE_NAMES = ("40_degC_CO2", "20_degC_air")


def run_case(case_name):
    redirect_darts_output(f"run_{case_name}.log")

    model = Model(case_name)
    model.init()
    model.set_output(output_folder=f"output_{case_name}")

    output_props = model.physics.vars + model.output.properties
    model.output.well_output_to_vtp(ith_step=0, output_properties=output_props)

    model.run(model.paper_steady_time_s / (24 * 60 * 60))
    model.output.well_output_to_vtp(ith_step=1, output_properties=output_props)
    model.print_timers()

    save_dfm_well_props("I1", model)
    plot_comparison(model, model.reference_profile_file, output_label=case_name)
    print_r2_report(model, model.reference_profile_file, case_label=case_name)


def main():
    for case_name in CASE_NAMES:
        print(f"\nRunning {case_name}")
        run_case(case_name)


if __name__ == "__main__":
    main()
