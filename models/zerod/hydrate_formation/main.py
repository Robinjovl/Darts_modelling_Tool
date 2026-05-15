from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib
import numpy as np
from darts.engines import redirect_darts_output

from model import CASE_CONFIGS, HydrateFormationModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REFERENCE_DIR = Path(__file__).resolve().parent / "reference"


@contextmanager
def tee_stdout(path: Path):
    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, data):
            for file in self.files:
                file.write(data)
                file.flush()

        def flush(self):
            for file in self.files:
                file.flush()

    original_stdout = sys.stdout
    with path.open("w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        try:
            yield
        finally:
            sys.stdout = original_stdout


def _load_csv(path: Path):
    return np.genfromtxt(path, delimiter=",", names=True)


def _series(props: dict, key: str) -> np.ndarray:
    return np.squeeze(np.asarray(props[key], dtype=float))


def _rmse(sim_time, sim_values, ref_time, ref_values) -> float:
    sim_time = np.asarray(sim_time, dtype=float)
    sim_values = np.asarray(sim_values, dtype=float)
    ref_time = np.asarray(ref_time, dtype=float)
    ref_values = np.asarray(ref_values, dtype=float)
    interp = np.interp(ref_time, sim_time, sim_values)
    return float(np.sqrt(np.mean((interp - ref_values) ** 2)))


def _plot_ch4(model, times, props, output_folder: Path):
    pressure_ref = _load_csv(REFERENCE_DIR / "ch4_pressure.csv")
    temperature_ref = _load_csv(REFERENCE_DIR / "ch4_temperature.csv")
    mass_ref = _load_csv(REFERENCE_DIR / "ch4_mass.csv")

    sim_t_hr = np.asarray(times, dtype=float) * 24.0
    boundary_t = np.array([model.boundary_temperature(t) for t in times], dtype=float)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex="col")

    ax[0, 0].plot(sim_t_hr, _series(props, "pressure_MPa"), label="0D", color="tab:blue")
    ax[0, 0].plot(
        pressure_ref["time"],
        pressure_ref["p"],
        label="Reference",
        color="black",
        linestyle="--",
    )
    ax[0, 0].set_ylabel("Pressure [MPa]")
    ax[0, 0].legend()

    ax[0, 1].plot(
        sim_t_hr, _series(props, "temperature_K"), label="0D", color="tab:red"
    )
    ax[0, 1].plot(
        sim_t_hr, boundary_t, label="Boundary", color="tab:orange", linestyle=":"
    )
    ax[0, 1].plot(
        temperature_ref["time"],
        temperature_ref["temperature"],
        label="Reference",
        color="black",
        linestyle="--",
    )
    ax[0, 1].set_ylabel("Temperature [K]")
    ax[0, 1].legend()

    ax[1, 0].plot(sim_t_hr, _series(props, "mass_H2O"), label="H2O", color="tab:blue")
    ax[1, 0].plot(sim_t_hr, _series(props, "mass_CH4"), label="CH4", color="tab:green")
    ax[1, 0].plot(
        sim_t_hr, _series(props, "mass_hydrate"), label="Hydrate", color="tab:red"
    )
    ax[1, 0].plot(
        mass_ref["time"], mass_ref["water"], color="tab:blue", linestyle="--"
    )
    ax[1, 0].plot(mass_ref["time"], mass_ref["gas"], color="tab:green", linestyle="--")
    ax[1, 0].plot(
        mass_ref["time"], mass_ref["hydrate"], color="tab:red", linestyle="--"
    )
    ax[1, 0].set_xlabel("Time [h]")
    ax[1, 0].set_ylabel("Mass [publication units]")
    ax[1, 0].legend()

    ax[1, 1].plot(sim_t_hr, _series(props, "sat_aq"), label="Aq", color="tab:blue")
    ax[1, 1].plot(sim_t_hr, _series(props, "sat_vap"), label="V", color="tab:green")
    ax[1, 1].plot(sim_t_hr, _series(props, "sat_hyd"), label="Hyd", color="tab:red")
    ax[1, 1].set_xlabel("Time [h]")
    ax[1, 1].set_ylabel("Saturation [-]")
    ax[1, 1].legend()

    fig.suptitle(model.case_config.title)
    fig.tight_layout()
    fig.savefig(output_folder / "overview.png", dpi=180)
    plt.close(fig)


def _plot_co2(model, times, props, output_folder: Path):
    temp_ref = _load_csv(REFERENCE_DIR / "co2_temperature.csv")
    sim_t_min = np.asarray(times, dtype=float) * 1440.0
    boundary_t_c = np.array(
        [model.boundary_temperature(t) - 273.15 for t in times], dtype=float
    )

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex="col")

    ax[0, 0].plot(
        sim_t_min, _series(props, "temperature_C"), label="0D", color="tab:red"
    )
    ax[0, 0].plot(
        sim_t_min, boundary_t_c, label="Boundary", color="tab:orange", linestyle=":"
    )
    for key, color in [
        ("tmid", "black"),
        ("tbot", "tab:blue"),
        ("tcap", "tab:green"),
        ("ttop", "tab:purple"),
    ]:
        ax[0, 0].plot(temp_ref["time"], temp_ref[key], label=key, color=color, linestyle="--")
    ax[0, 0].set_ylabel("Temperature [C]")
    ax[0, 0].legend()

    ax[0, 1].plot(
        sim_t_min, _series(props, "pressure_bar"), label="Pressure", color="tab:blue"
    )
    ax[0, 1].set_ylabel("Pressure [bar]")
    ax[0, 1].legend()

    ax[1, 0].plot(sim_t_min, _series(props, "mass_H2O"), label="H2O", color="tab:blue")
    ax[1, 0].plot(sim_t_min, _series(props, "mass_CO2"), label="CO2", color="tab:green")
    ax[1, 0].plot(
        sim_t_min, _series(props, "mass_hydrate"), label="Hydrate", color="tab:red"
    )
    ax[1, 0].set_xlabel("Time [min]")
    ax[1, 0].set_ylabel("Mass [publication units]")
    ax[1, 0].legend()

    ax[1, 1].plot(sim_t_min, _series(props, "sat_aq"), label="Aq", color="tab:blue")
    ax[1, 1].plot(sim_t_min, _series(props, "sat_vap"), label="V", color="tab:green")
    ax[1, 1].plot(sim_t_min, _series(props, "sat_hyd"), label="Hyd", color="tab:red")
    ax[1, 1].set_xlabel("Time [min]")
    ax[1, 1].set_ylabel("Saturation [-]")
    ax[1, 1].legend()

    fig.suptitle(model.case_config.title)
    fig.tight_layout()
    fig.savefig(output_folder / "overview.png", dpi=180)
    plt.close(fig)


def _compare(case: str, times, props) -> dict:
    metrics = {}
    if case == "ch4":
        pressure_ref = _load_csv(REFERENCE_DIR / "ch4_pressure.csv")
        temperature_ref = _load_csv(REFERENCE_DIR / "ch4_temperature.csv")
        mass_ref = _load_csv(REFERENCE_DIR / "ch4_mass.csv")
        sim_time_hr = np.asarray(times, dtype=float) * 24.0
        metrics["pressure_rmse_mpa"] = _rmse(
            sim_time_hr, _series(props, "pressure_MPa"), pressure_ref["time"], pressure_ref["p"]
        )
        metrics["temperature_rmse_k"] = _rmse(
            sim_time_hr,
            _series(props, "temperature_K"),
            temperature_ref["time"],
            temperature_ref["temperature"],
        )
        metrics["hydrate_mass_rmse"] = _rmse(
            sim_time_hr,
            _series(props, "mass_hydrate"),
            mass_ref["time"],
            mass_ref["hydrate"],
        )
    else:
        temp_ref = _load_csv(REFERENCE_DIR / "co2_temperature.csv")
        sim_time_min = np.asarray(times, dtype=float) * 1440.0
        metrics["temperature_rmse_c_tmid"] = _rmse(
            sim_time_min,
            _series(props, "temperature_C"),
            temp_ref["time"],
            temp_ref["tmid"],
        )

    metrics["final_pressure_bar"] = float(_series(props, "pressure_bar")[-1])
    metrics["final_temperature_K"] = float(_series(props, "temperature_K")[-1])
    metrics["final_sat_hyd"] = float(_series(props, "sat_hyd")[-1])
    metrics["final_mass_hydrate"] = float(_series(props, "mass_hydrate")[-1])
    return metrics


def run_case(case: str, output_root: Path, verbose: bool = True):
    cfg = CASE_CONFIGS[case]
    output_folder = output_root / case
    output_folder.mkdir(parents=True, exist_ok=True)
    redirect_darts_output(str(output_folder / "darts.log"))
    with tee_stdout(output_folder / "simulation.log"):
        model = HydrateFormationModel(case=case)
        model.init(output_folder=str(output_folder))
        ok = model.run(days=cfg.runtime_days, method="radau", verbose=verbose)
        if not ok:
            raise RuntimeError(f"{case} run failed to reach final time.")

        model.plot_state_history(
            use_log_p=False, output_path=str(output_folder / "state_history.png")
        )

        times, props = model.extract_property_history()
        if case == "ch4":
            _plot_ch4(model, times, props, output_folder)
        else:
            _plot_co2(model, times, props, output_folder)

        metrics = _compare(case, times, props)
        (output_folder / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"{case}: {json.dumps(metrics, sort_keys=True)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run 0D hydrate formation examples derived from publication 26_hydrate."
    )
    parser.add_argument(
        "--case",
        choices=["ch4", "co2", "all"],
        default="all",
        help="Which reduced publication case to run.",
    )
    parser.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parent / "output"),
        help="Directory where case outputs are written.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-timestep stdout from ZerodModel.run().",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cases = list(CASE_CONFIGS) if args.case == "all" else [args.case]
    for case in cases:
        run_case(case, output_root, verbose=not args.quiet)


if __name__ == "__main__":
    main()
