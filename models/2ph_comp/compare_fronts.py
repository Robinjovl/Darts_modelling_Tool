"""Run and visualize matched SPU/WENO2 front-propagation experiments."""

import argparse
import csv
import json
import math
import time
import traceback
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from darts.engines import redirect_darts_output, sim_params
from model import Model
from scipy.signal import find_peaks, peak_widths

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _json_dump(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, allow_nan=False, indent=2, sort_keys=True) + "\n")


def _timer_value(model: Model, *keys: str) -> float | None:
    timer = model.timer
    try:
        for key in keys:
            timer = timer.node[key]
        return float(timer.get_timer())
    except (KeyError, RuntimeError):
        return None


def _run_summary(
    model: Model,
    scheme: str,
    linear_solver: str,
    runtime: float,
    nx: int,
    report_interval: float,
    wall_seconds: float,
) -> dict:
    engine = model.physics.engine
    stat = engine.stat
    return {
        "scheme": scheme,
        "linear_solver": linear_solver,
        "runtime_days": runtime,
        "completed_time_days": float(engine.t),
        "nx": nx,
        "report_interval_days": report_interval,
        "wall_seconds": wall_seconds,
        "statistics": {
            "timesteps": int(stat.n_timesteps_total),
            "wasted_timesteps": int(stat.n_timesteps_wasted),
            "newton_iterations": int(stat.n_newton_total),
            "wasted_newton_iterations": int(stat.n_newton_wasted),
            "linear_iterations": int(stat.n_linear_total),
            "wasted_linear_iterations": int(stat.n_linear_wasted),
        },
        "timers_seconds": {
            "simulation": _timer_value(model, "simulation"),
            "jacobian_assembly": _timer_value(model, "simulation", "jacobian assembly"),
            "linear_solver_setup": _timer_value(
                model, "simulation", "linear solver setup"
            ),
            "linear_solver_solve": _timer_value(
                model, "simulation", "linear solver solve"
            ),
            "weno_geometry": _timer_value(model, "initialization", "WENO geometry"),
        },
        "weno_fallbacks": {
            "geometry": int(getattr(engine, "weno_geometry_fallback_count", 0)),
            "bounded_state": int(getattr(engine, "weno_bound_fallback_count", 0)),
        },
    }


def simulate(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to append to non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    redirect_darts_output(str(output_dir / "darts_native.log"))
    model = Model(transport_scheme=args.scheme, nx=args.nx, runtime=args.runtime)
    if args.linear_solver == "superlu":
        model.params.linear_type = sim_params.linear_solver_t.cpu_superlu

    started = time.perf_counter()
    try:
        model.init()
        model.set_output(output_folder=str(output_dir), verbose=0)

        n_reports = math.ceil(args.runtime / args.report_interval)
        targets = np.minimum(
            np.arange(1, n_reports + 1, dtype=float) * args.report_interval,
            args.runtime,
        )
        targets = np.unique(targets)

        previous_target = 0.0
        for report_index, target in enumerate(targets, start=1):
            model.run(
                float(target - previous_target),
                save_well_data=False,
                save_well_data_after_run=False,
                save_reservoir_data=True,
                verbose=0,
            )
            previous_target = float(target)
            stat = model.physics.engine.stat
            print(
                f"[{args.scheme}] report {report_index:03d}/{len(targets):03d} "
                f"t={target:g} d, TS={stat.n_timesteps_total}, "
                f"NI={stat.n_newton_total}, LI={stat.n_linear_total}",
                flush=True,
            )

        wall_seconds = time.perf_counter() - started
        summary = _run_summary(
            model,
            args.scheme,
            args.linear_solver,
            args.runtime,
            args.nx,
            args.report_interval,
            wall_seconds,
        )
        _json_dump(output_dir / "run_summary.json", summary)

        if not math.isclose(
            summary["completed_time_days"], args.runtime, rel_tol=0.0, abs_tol=1e-9
        ):
            raise RuntimeError(
                f"Simulation stopped at {summary['completed_time_days']} days instead "
                f"of {args.runtime} days"
            )
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        failure = {
            "scheme": args.scheme,
            "linear_solver": args.linear_solver,
            "runtime_days": args.runtime,
            "nx": args.nx,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _json_dump(output_dir / "run_failure.json", failure)
        raise


def _load_case(case_dir: Path) -> dict:
    with h5py.File(case_dir / "reservoir_solution.h5", "r") as h5_file:
        variable_names = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in h5_file["dynamic/variable_names"][:]
        ]
        data = {
            "time": np.asarray(h5_file["dynamic/time"][:], dtype=float),
            "cell_id": np.asarray(h5_file["dynamic/cell_id"][:], dtype=int),
            "state": np.asarray(h5_file["dynamic/X"][:], dtype=float),
            "variable_names": variable_names,
        }
    data["summary"] = json.loads((case_dir / "run_summary.json").read_text())
    return data


def _gas_saturation(states: np.ndarray) -> np.ndarray:
    evaluator_model = Model(nx=1, runtime=1.0)
    property_container = evaluator_model.physics.property_containers[0]
    saturation = np.empty(states.shape[:2], dtype=float)
    for time_index, state_at_time in enumerate(states):
        for cell_index, state in enumerate(state_at_time):
            saturation[time_index, cell_index] = (
                property_container.compute_saturation_full(state)
            )
        if time_index % 10 == 0 or time_index == states.shape[0] - 1:
            print(
                f"Evaluated gas saturation for snapshot "
                f"{time_index + 1}/{states.shape[0]}",
                flush=True,
            )
    return saturation


def _last_crossing(
    profile: np.ndarray,
    x: np.ndarray,
    baseline: float,
    plateau: float,
    level: float,
) -> float:
    amplitude = plateau - baseline
    if amplitude <= 1e-12:
        return math.nan
    normalized = (profile - baseline) / amplitude
    indices = np.flatnonzero(normalized >= level)
    if indices.size == 0:
        return math.nan
    index = int(indices[-1])
    if index == len(profile) - 1:
        return float(x[index])
    y0 = normalized[index]
    y1 = normalized[index + 1]
    if math.isclose(float(y0), float(y1), abs_tol=1e-14):
        return float(x[index])
    fraction = (level - y0) / (y1 - y0)
    return float(x[index] + fraction * (x[index + 1] - x[index]))


def _front_metrics(
    profile: np.ndarray,
    x: np.ndarray,
    baseline: float,
    plateau: float,
) -> dict:
    x10 = _last_crossing(profile, x, baseline, plateau, 0.10)
    x50 = _last_crossing(profile, x, baseline, plateau, 0.50)
    x90 = _last_crossing(profile, x, baseline, plateau, 0.90)
    width = x10 - x90 if np.isfinite(x10) and np.isfinite(x90) else math.nan
    total_variation = float(np.sum(np.abs(np.diff(profile))))
    endpoint_variation = float(abs(profile[0] - profile[-1]))

    negative_gradient = np.maximum(-np.gradient(profile, x), 0.0)
    peaks, properties = find_peaks(negative_gradient, prominence=1e-5, distance=5)
    leading_x = math.nan
    leading_gradient = math.nan
    leading_fwhm = math.nan
    if peaks.size:
        prominence_limit = max(1e-5, 0.002 * float(np.max(properties["prominences"])))
        qualified_peaks = peaks[properties["prominences"] >= prominence_limit]
        if qualified_peaks.size:
            leading_index = int(qualified_peaks[-1])
            dx = float(np.median(np.diff(x)))
            leading_x = float(x[leading_index])
            leading_gradient = float(negative_gradient[leading_index])
            leading_fwhm = float(
                peak_widths(negative_gradient, [leading_index], rel_height=0.5)[0][0]
                * dx
            )
    return {
        "x10": x10,
        "x50": x50,
        "x90": x90,
        "width_10_90": width,
        "leading_gradient_x": leading_x,
        "leading_gradient_magnitude": leading_gradient,
        "leading_gradient_fwhm": leading_fwhm,
        "total_variation": total_variation,
        "excess_total_variation": total_variation - endpoint_variation,
        "minimum": float(np.min(profile)),
        "maximum": float(np.max(profile)),
        "excess_inventory": float(np.sum(profile - baseline)),
    }


def _first_breakthrough_time(
    time_values: np.ndarray, profiles: np.ndarray, threshold: float
) -> float | None:
    indices = np.flatnonzero(profiles[:, -1] >= threshold)
    return float(time_values[indices[0]]) if indices.size else None


def _write_metrics_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric_history(
    path: Path, time_values: np.ndarray, metrics: dict[str, list[dict]]
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    colors = {"spu": "#1f77b4", "weno2": "#d62728"}
    labels = {"spu": "SPU", "weno2": "WENO2"}
    fields = (
        ("leading_gradient_x", "Leading-gradient position [m]"),
        ("leading_gradient_fwhm", "Leading-gradient FWHM [m]"),
        ("excess_total_variation", "Excess total variation [-]"),
    )
    for axis, (field, ylabel) in zip(axes, fields, strict=True):
        for scheme in ("spu", "weno2"):
            axis.plot(
                time_values,
                [item[field] for item in metrics[scheme]],
                color=colors[scheme],
                label=labels[scheme],
            )
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("Time [days]")
    fig.suptitle("2ph_comp front metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _render_frames(
    output_dir: Path,
    time_values: np.ndarray,
    x: np.ndarray,
    z_spu: np.ndarray,
    z_weno: np.ndarray,
    sat_spu: np.ndarray,
    sat_weno: np.ndarray,
    metrics: dict[str, list[dict]],
) -> None:
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=False)
    colors = {"spu": "#1f77b4", "weno2": "#d62728"}

    z_min = min(float(np.min(z_spu)), float(np.min(z_weno)))
    z_max = max(float(np.max(z_spu)), float(np.max(z_weno)))
    z_margin = max(0.01, 0.04 * (z_max - z_min))
    sat_min = min(float(np.min(sat_spu)), float(np.min(sat_weno)))
    sat_max = max(float(np.max(sat_spu)), float(np.max(sat_weno)))
    sat_margin = max(0.01, 0.04 * (sat_max - sat_min))
    diff_limit = max(float(np.max(np.abs(z_weno - z_spu))), 1e-4)

    for frame_index, time_value in enumerate(time_values):
        fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=100)
        ax_overall, ax_zoom, ax_saturation, ax_difference = axes.flat

        for axis in (ax_overall, ax_zoom):
            axis.plot(x, z_spu[frame_index], color=colors["spu"], label="SPU")
            axis.plot(
                x,
                z_weno[frame_index],
                color=colors["weno2"],
                label="WENO2",
            )
            axis.set_ylim(z_min - z_margin, z_max + z_margin)
            axis.grid(alpha=0.25)
            axis.set_ylabel("Overall CO₂ fraction [-]")

        ax_overall.set_xlim(float(x[0]), float(x[-1]))
        ax_overall.set_title("Full domain")
        ax_overall.legend(loc="upper right")

        front_positions = [
            metrics[scheme][frame_index]["leading_gradient_x"]
            for scheme in ("spu", "weno2")
        ]
        finite_positions = [value for value in front_positions if np.isfinite(value)]
        front_widths = [
            metrics[scheme][frame_index]["leading_gradient_fwhm"]
            for scheme in ("spu", "weno2")
        ]
        finite_widths = [value for value in front_widths if np.isfinite(value)]
        center = float(np.mean(finite_positions)) if finite_positions else float(x[0])
        half_width = max(20.0, 8.0 * max(finite_widths, default=0.0))
        ax_zoom.set_xlim(
            max(float(x[0]), center - half_width),
            min(float(x[-1]), center + half_width),
        )
        ax_zoom.set_title("Moving front detail")

        ax_saturation.plot(x, sat_spu[frame_index], color=colors["spu"], label="SPU")
        ax_saturation.plot(
            x, sat_weno[frame_index], color=colors["weno2"], label="WENO2"
        )
        ax_saturation.set_xlim(float(x[0]), float(x[-1]))
        ax_saturation.set_ylim(sat_min - sat_margin, sat_max + sat_margin)
        ax_saturation.set_xlabel("Distance from injector [m]")
        ax_saturation.set_ylabel("Gas saturation [-]")
        ax_saturation.set_title("Gas-saturation response")
        ax_saturation.grid(alpha=0.25)

        difference = z_weno[frame_index] - z_spu[frame_index]
        ax_difference.axhline(0.0, color="black", linewidth=0.8)
        ax_difference.plot(x, difference, color="#6a3d9a")
        ax_difference.fill_between(x, 0.0, difference, color="#cab2d6", alpha=0.6)
        ax_difference.set_xlim(float(x[0]), float(x[-1]))
        ax_difference.set_ylim(-1.05 * diff_limit, 1.05 * diff_limit)
        ax_difference.set_xlabel("Distance from injector [m]")
        ax_difference.set_ylabel("WENO2 − SPU CO₂ fraction [-]")
        ax_difference.set_title("Scheme difference")
        ax_difference.grid(alpha=0.25)

        spu_metric = metrics["spu"][frame_index]
        weno_metric = metrics["weno2"][frame_index]
        annotation = (
            f"SPU: x_lead={spu_metric['leading_gradient_x']:.2f} m, "
            f"FWHM={spu_metric['leading_gradient_fwhm']:.2f} m\n"
            f"WENO2: x_lead={weno_metric['leading_gradient_x']:.2f} m, "
            f"FWHM={weno_metric['leading_gradient_fwhm']:.2f} m"
        )
        ax_difference.text(
            0.02,
            0.96,
            annotation,
            transform=ax_difference.transAxes,
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

        fig.suptitle(
            f"2ph_comp transport comparison — t = {time_value:g} days",
            fontsize=16,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        frame_path = frame_dir / f"frame_{frame_index:04d}.png"
        fig.savefig(frame_path)
        if frame_index == len(time_values) - 1:
            fig.savefig(output_dir / "final_profile.png")
        plt.close(fig)
        print(f"Rendered {frame_path.name}", flush=True)


def render(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = {
        "spu": _load_case(args.spu_dir.resolve()),
        "weno2": _load_case(args.weno_dir.resolve()),
    }
    if not np.allclose(cases["spu"]["time"], cases["weno2"]["time"]):
        raise ValueError("SPU and WENO2 report times differ")
    if not np.array_equal(cases["spu"]["cell_id"], cases["weno2"]["cell_id"]):
        raise ValueError("SPU and WENO2 reservoir-cell order differs")
    if cases["spu"]["variable_names"] != cases["weno2"]["variable_names"]:
        raise ValueError("SPU and WENO2 state variables differ")

    variable_names = cases["spu"]["variable_names"]
    co2_index = variable_names.index("CO2")
    time_values = cases["spu"]["time"]
    nx = len(cases["spu"]["cell_id"])
    x = np.arange(nx, dtype=float) + 0.5
    z = {scheme: cases[scheme]["state"][:, :, co2_index] for scheme in ("spu", "weno2")}
    saturation = {
        scheme: _gas_saturation(cases[scheme]["state"]) for scheme in ("spu", "weno2")
    }

    baseline = float(np.median(np.concatenate((z["spu"][0], z["weno2"][0]))))
    metrics = {"spu": [], "weno2": []}
    rows = []
    for time_index, time_value in enumerate(time_values):
        plateau = max(
            float(np.max(z["spu"][time_index])),
            float(np.max(z["weno2"][time_index])),
        )
        for scheme in ("spu", "weno2"):
            item = _front_metrics(z[scheme][time_index], x, baseline, plateau)
            metrics[scheme].append(item)
            rows.append({"time_days": float(time_value), "scheme": scheme, **item})

    global_plateau = max(float(np.max(z["spu"])), float(np.max(z["weno2"])))
    breakthrough_threshold = baseline + 0.01 * (global_plateau - baseline)
    final_difference = z["weno2"][-1] - z["spu"][-1]
    final_metrics = {scheme: metrics[scheme][-1] for scheme in ("spu", "weno2")}
    summary = {
        "runtime_days": float(time_values[-1]),
        "nx": nx,
        "snapshots": len(time_values),
        "co2_baseline": baseline,
        "co2_global_plateau": global_plateau,
        "breakthrough_threshold": breakthrough_threshold,
        "breakthrough_time_days": {
            scheme: _first_breakthrough_time(
                time_values, z[scheme], breakthrough_threshold
            )
            for scheme in ("spu", "weno2")
        },
        "final_front_metrics": final_metrics,
        "final_scheme_difference": {
            "mean_absolute": float(np.mean(np.abs(final_difference))),
            "root_mean_square": float(np.sqrt(np.mean(final_difference**2))),
            "maximum_absolute": float(np.max(np.abs(final_difference))),
            "inventory_weno_minus_spu": float(np.sum(final_difference)),
        },
        "maximum_difference_over_all_snapshots": float(
            np.max(np.abs(z["weno2"] - z["spu"]))
        ),
        "run_summaries": {
            scheme: cases[scheme]["summary"] for scheme in ("spu", "weno2")
        },
    }

    _json_dump(output_dir / "analysis_summary.json", summary)
    _write_metrics_csv(output_dir / "front_metrics.csv", rows)
    with h5py.File(output_dir / "comparison_profiles.h5", "w") as h5_file:
        h5_file.create_dataset("time_days", data=time_values)
        h5_file.create_dataset("distance_m", data=x)
        for scheme in ("spu", "weno2"):
            group = h5_file.create_group(scheme)
            group.create_dataset("co2_fraction", data=z[scheme])
            group.create_dataset("gas_saturation", data=saturation[scheme])

    _plot_metric_history(output_dir / "front_metrics.png", time_values, metrics)
    _render_frames(
        output_dir,
        time_values,
        x,
        z["spu"],
        z["weno2"],
        saturation["spu"],
        saturation["weno2"],
        metrics,
    )
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or render the 2ph_comp SPU/WENO2 comparison"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate_parser = subparsers.add_parser("simulate")
    simulate_parser.add_argument("--scheme", choices=("spu", "weno2"), required=True)
    simulate_parser.add_argument(
        "--linear-solver", choices=("default", "superlu"), default="default"
    )
    simulate_parser.add_argument("--runtime", type=float, default=1000.0)
    simulate_parser.add_argument("--nx", type=int, default=1000)
    simulate_parser.add_argument("--report-interval", type=float, default=10.0)
    simulate_parser.add_argument("--output-dir", type=Path, required=True)
    simulate_parser.set_defaults(handler=simulate)

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--spu-dir", type=Path, required=True)
    render_parser.add_argument("--weno-dir", type=Path, required=True)
    render_parser.add_argument("--output-dir", type=Path, required=True)
    render_parser.set_defaults(handler=render)

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.handler(arguments)
