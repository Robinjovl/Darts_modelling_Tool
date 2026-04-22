from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from darts.engines import redirect_darts_output, sim_params

from model import Model, default_corey_regions

DEFAULT_COMPONENTS = ("H2O", "CO2")
DEFAULT_TEMPERATURE = 338.15
DEFAULT_PRODUCER_BHP = 250.0
DEFAULT_INJECTION_RATE = 6.4
DEFAULT_INITIAL_COMPOSITION = {"H2O": 1.0 - 1e-12, "CO2": 1e-12}
DEFAULT_INJECTION_COMPOSITION = {"H2O": 1e-12, "CO2": 1.0 - 1e-12}


@dataclass
class CaseConfig:
    # Physics switches
    hysteresis: bool = True
    components: tuple[str, str] = DEFAULT_COMPONENTS
    temperature: float = DEFAULT_TEMPERATURE
    thermal: bool = False

    # Grid / OBL interpolation
    nx: int = 100
    n_points: int = 1000

    # Initial / boundary conditions
    producer_bhp: float = DEFAULT_PRODUCER_BHP
    initial_composition: dict[str, float] | None = None
    injection_composition: dict[str, float] | None = None
    injection_rate: float = DEFAULT_INJECTION_RATE

    # Time schedule
    total_days: float = 1000.0
    report_step_days: float = 10.0
    stop_injection_after_days: float | None = 400.0
    start_injection_h2o_days: float | None = 800.0
    # Nonlinear / linear solver controls
    first_ts: float = 1e-4
    mult_ts: float = 1.5
    max_ts: float = 1.0
    tol_newton: float = 1e-3
    tol_linear: float = 1e-3
    it_newton: int = 16
    it_linear: int = 20
    dt_eta: float = 0.05

    # Output
    output_folder: str = f"1D_continuous_hys{hysteresis}"
    write_vtk: bool = True
    verbose: bool = True


CONFIG = CaseConfig(
    initial_composition=DEFAULT_INITIAL_COMPOSITION,
    injection_composition=DEFAULT_INJECTION_COMPOSITION,
)
def build_model(config: CaseConfig) -> Model:
    model = Model(hys=config.hysteresis)
    initial_composition = config.initial_composition or {
        config.components[0]: DEFAULT_INITIAL_COMPOSITION[config.components[0]],
        config.components[1]: DEFAULT_INITIAL_COMPOSITION[config.components[1]],
    }
    injection_composition = config.injection_composition or {
        config.components[0]: DEFAULT_INJECTION_COMPOSITION[config.components[0]],
        config.components[1]: DEFAULT_INJECTION_COMPOSITION[config.components[1]],
    }
    model.setup_case(
        nx=config.nx,
        n_points=config.n_points,
        temperature=config.temperature,
        producer_bhp=config.producer_bhp,
        thermal= config.thermal,
        injection_rate=config.injection_rate,
        initial_z_h2o=initial_composition["H2O"],
        injection_stream=injection_composition,
        components=list(config.components),
        corey_regions=default_corey_regions(),
    )
    model.set_sim_params(
        first_ts=config.first_ts,
        mult_ts=config.mult_ts,
        max_ts=config.max_ts,
        tol_newton=config.tol_newton,
        tol_linear=config.tol_linear,
        it_newton=config.it_newton,
        it_linear=config.it_linear,
    )
    model.params.linear_type = sim_params.linear_solver_t.cpu_superlu
    model.data_ts.eta[-1] = config.dt_eta
    model.init()
    model.set_output(output_folder=config.output_folder, all_phase_props=True)
    return model


def update_schedule(model: Model, config: CaseConfig) -> None:
    if (
        config.stop_injection_after_days is not None
        and config.start_injection_h2o_days >= model.physics.engine.t >= config.stop_injection_after_days
    ):
        model.inj_rate[0] = 0.0
        model.inj_rate[1] = 0.0
    elif (
            config.start_injection_h2o_days is not None
            and model.physics.engine.t >= config.start_injection_h2o_days
    ):
        model.inj_rate[0] = 1.728
        model.inj_rate[1] = 0.0



def run_case(config: CaseConfig) -> Model:
    output_dir = Path(config.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    redirect_darts_output(str(output_dir / "binary.log"))

    model = build_model(config)
    print(f"Pore volume = {model.pore_volume()}")

    output_properties = model.vtk_output_properties()
    if config.write_vtk:
        model.output.output_to_vtk(
            ith_step=0,
            output_properties=output_properties,
            engine=True,
        )

    n_steps = int(round(config.total_days / config.report_step_days))
    for step in range(1, n_steps + 1):
        model.run(
            config.report_step_days,
            save_well_data=False,
            verbose=config.verbose,
        )
        update_schedule(model, config)
        if config.write_vtk:
            model.output.output_to_vtk(
                ith_step=step,
                output_properties=output_properties,
                engine=True,
            )

    model.print_timers()
    model.print_stat()
    return model


def main() -> None:
    run_case(CONFIG)


if __name__ == "__main__":
    main()
