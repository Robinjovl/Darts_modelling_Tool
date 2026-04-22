from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from darts.engines import redirect_darts_output

from model import Model, default_corey_regions

DEFAULT_COMPONENTS = ("H2O", "CO2")
DEFAULT_TEMPERATURE = 338.15
DEFAULT_PRODUCER_BHP = 250.0
DEFAULT_INJECTION_RATE = 6.4
DEFAULT_INITIAL_COMPOSITION = {"H2O": 1.0 - 1e-12, "CO2": 1e-12}
DEFAULT_INJECTION_COMPOSITION = {"H2O": 1e-12, "CO2": 1.0 - 1e-12}
DEFAULT_SG_SNAPSHOT_DAYS = (400.0, 800.0, 1000.0)


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
    output_folder: str = "output"
    plot_sg_snapshots: bool = False
    sg_snapshot_days: tuple[float, ...] = DEFAULT_SG_SNAPSHOT_DAYS
    write_vtk: bool = True
    verbose: bool = True


CONFIG = CaseConfig(
    initial_composition=DEFAULT_INITIAL_COMPOSITION,
    injection_composition=DEFAULT_INJECTION_COMPOSITION,
)


def build_model(config: CaseConfig, platform: str = "cpu") -> Model:
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
        thermal=config.thermal,
        injection_rate=config.injection_rate,
        initial_z_h2o=initial_composition["H2O"],
        injection_stream=injection_composition,
        components=list(config.components),
        corey_regions=default_corey_regions(),
        stop_injection_after_days=config.stop_injection_after_days,
        start_injection_h2o_days=config.start_injection_h2o_days,
        water_injection_rate=1.728,
    )
    model.set_sim_params(
        first_ts=config.first_ts,
        mult_ts=config.mult_ts,
        max_ts=config.max_ts,
        runtime=config.total_days,
        tol_newton=config.tol_newton,
        tol_linear=config.tol_linear,
        it_newton=config.it_newton,
        it_linear=config.it_linear,
    )
    model.data_ts.eta[-1] = config.dt_eta
    model.init(platform=platform)
    model.set_output(output_folder=config.output_folder, all_phase_props=True)
    return model


def update_schedule(model: Model, config: CaseConfig) -> None:
    del config
    model.update_injection_schedule()


def save_sg_profile_figure(
    model: Model,
    figure_dir: Path,
    day: float,
) -> None:
    from matplotlib import pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    _, output_props = model.output.output_properties(
        output_properties=["sat_V"],
        engine=True,
    )
    sg = output_props["sat_V"][0]
    x = model.output._get_output_cell_centers()[: model.reservoir.mesh.n_res_blocks, 0]
    sort_idx = x.argsort()

    plt.figure(figsize=(8, 4.5))
    plt.plot(x[sort_idx], sg[sort_idx], linewidth=2.0)
    plt.xlabel("x [m]")
    plt.ylabel("Sg [-]")
    plt.title(f"Sg distribution at {day:.0f} days")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_dir / f"sg_{int(round(day))}d.png", dpi=200)
    plt.close()


def run_case(config: CaseConfig, platform: str = "cpu") -> Model:
    output_dir = Path(config.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figure"
    redirect_darts_output(str(output_dir / "binary.log"))

    model = build_model(config, platform=platform)
    print(f"Pore volume = {model.pore_volume()}")

    output_properties = model.vtk_output_properties()
    if config.write_vtk:
        model.output.output_to_vtk(
            ith_step=0,
            output_properties=output_properties,
            engine=True,
        )

    n_steps = int(round(config.total_days / config.report_step_days))
    remaining_snapshot_days = (
        {float(day) for day in config.sg_snapshot_days}
        if config.plot_sg_snapshots
        else set()
    )
    for step in range(1, n_steps + 1):
        model.run(
            config.report_step_days,
            save_well_data=False,
            verbose=config.verbose,
        )
        update_schedule(model, config)
        current_day = float(model.physics.engine.t)
        for snapshot_day in sorted(remaining_snapshot_days):
            if abs(current_day - snapshot_day) < 1e-9:
                save_sg_profile_figure(model, figure_dir, snapshot_day)
                remaining_snapshot_days.remove(snapshot_day)
                break
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
    platform = "cpu"
    if len(sys.argv) > 1:
        platform = sys.argv[1]
    if platform not in ["cpu", "gpu"]:
        print("unknown platform specified", platform)
        raise SystemExit(1)
    run_case(CONFIG, platform=platform)


if __name__ == "__main__":
    main()
