import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np
from darts.engines import redirect_darts_output

from natural_production_model import NaturalProductionConfig, NaturalProductionModel


def parse_float_list(text):
    if isinstance(text, (int, float)):
        return [float(text)]
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_phase_list(text):
    phases = [item.strip().upper() for item in text.split(",") if item.strip()]
    for phase in phases:
        if phase not in ("G", "L"):
            raise ValueError("--pipe-init-phase values must be G or L")
    return phases


def case_name(cfg):
    if cfg.sG_init_target is None:
        state_txt = (
            f"z{cfg.z_co2_init:.0e}"
            if cfg.z_co2_init < 1e-3
            else f"z{cfg.z_co2_init:g}"
        )
    else:
        state_txt = f"sG{cfg.sG_init_target:g}"
    return (
        f"P{cfg.p_init:g}_T{cfg.t_init:g}_{state_txt}_"
        f"WHP{cfg.p_prod_bhp:g}_pipe{cfg.pipe_init_phase.upper()}"
    ).replace(".", "p").replace("-", "m")


def get_series(time_data, key):
    values = time_data.get(key)
    if values is None:
        return np.zeros_like(time_data["time"], dtype=float)
    return np.asarray(values, dtype=float)


def print_pipe_phase_props(model, well_name="P1"):
    pipe = model.wells.get(well_name)
    if pipe is None or not hasattr(pipe, "iter_phases_props"):
        print(f"{well_name} pipe phase properties are not available yet.")
        return

    well = next(
        (well for well in model.reservoir.wells if well.name == well_name),
        None,
    )
    if well is None:
        print(f"{well_name} reservoir well object is not available.")
        return

    n_vars = model.physics.n_vars
    p_idx = model.physics.vars.index("pressure")
    t_idx = (
        model.physics.vars.index("temperature")
        if "temperature" in model.physics.vars
        else None
    )
    start = int(well.well_head_idx)
    stop = start + int(pipe.geometry.num_segments)
    x = np.asarray(model.physics.engine.X, dtype=float).reshape(-1, n_vars)
    pipe_state = x[start:stop]
    pressure = pipe_state[:, p_idx]
    temperature = pipe_state[:, t_idx] if t_idx is not None else None

    xg_mass, xl_mass, sg, rhog, rhol, mug, mul = pipe.iter_phases_props
    sl = 1.0 - np.asarray(sg, dtype=float)
    props = {
        "p": pressure,
        "sG": sg,
        "sL": sl,
        "rhoG": rhog,
        "rhoL": rhol,
        "muG": mug,
        "muL": mul,
    }
    summary = []
    for name, values in props.items():
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            summary.append(f"{name}=nan")
        else:
            summary.append(f"{name}=[{finite.min():.4g},{finite.max():.4g}]")
    print(
        f"T={model.physics.engine.t:.6g} day {well_name} pipe phase props: "
        + ", ".join(summary)
    )
    bottom_idx = int(pipe.geometry.num_segments) - 1
    bottom_items = [
        f"p_top={pressure[0]:.4g} bar",
        f"p_bottom={pressure[bottom_idx]:.4g} bar",
    ]
    if temperature is not None:
        bottom_items.extend(
            [
                f"T_top={temperature[0]:.4g} K",
                f"T_bottom={temperature[bottom_idx]:.4g} K",
            ]
        )
    bottom_items.extend(
        [
            f"sG_bottom={np.asarray(sg, dtype=float)[bottom_idx]:.4g}",
            f"rhoG_bottom={np.asarray(rhog, dtype=float)[bottom_idx]:.4g}",
            f"rhoL_bottom={np.asarray(rhol, dtype=float)[bottom_idx]:.4g}",
        ]
    )
    print(f"{well_name} pipe top/bottom: " + ", ".join(bottom_items))


def summarize_case(cfg, model, time_data, case_dir):
    mass_g = get_series(time_data, "well_P1_mass_rate_G_by_sum_perfs")
    mass_l = get_series(time_data, "well_P1_mass_rate_L_by_sum_perfs")
    total_mass = mass_g + mass_l

    wh_mass_g = get_series(time_data, "well_P1_mass_rate_G_at_wh")
    wh_mass_l = get_series(time_data, "well_P1_mass_rate_L_at_wh")
    wh_total_mass = wh_mass_g + wh_mass_l

    phase_state = model.initial_phase_state()
    threshold = cfg.natural_rate_threshold_kg_day
    natural = bool(np.nanmin(total_mass) < -threshold)

    row = {
        "case": case_name(cfg),
        "output_dir": str(case_dir),
        "p_init_bar": cfg.p_init,
        "t_init_K": cfg.t_init,
        "z_co2_init": cfg.z_co2_init,
        "sG_init_target": (
            np.nan if cfg.sG_init_target is None else cfg.sG_init_target
        ),
        "p_prod_whp_bar": cfg.p_prod_bhp,
        "pipe_init_phase": cfg.pipe_init_phase.upper(),
        "runtime_day": cfg.runtime,
        "natural_threshold_kg_day": threshold,
        "natural_production": int(natural),
        "min_p1_mass_rate_kg_day": float(np.nanmin(total_mass)),
        "final_p1_mass_rate_kg_day": float(total_mass[-1]),
        "mean_p1_mass_rate_kg_day": float(np.nanmean(total_mass)),
        "min_p1_mass_rate_at_wh_kg_day": float(np.nanmin(wh_total_mass)),
        "final_p1_mass_rate_at_wh_kg_day": float(wh_total_mass[-1]),
        "final_p1_mass_rate_G_kg_day": float(mass_g[-1]),
        "final_p1_mass_rate_L_kg_day": float(mass_l[-1]),
        "final_time_day": float(np.asarray(time_data["time"], dtype=float)[-1]),
        "final_WHP_bar": float(get_series(time_data, "well_P1_BHP")[-1]),
    }
    row.update(phase_state)
    return row


def scheduled_step_settings(t, runtime, fixed_report_step=None, fixed_max_ts=None):
    remaining = max(runtime - t, 0.0)
    if fixed_report_step is not None:
        report_dt = min(fixed_report_step, remaining)
        max_ts = fixed_max_ts if fixed_max_ts is not None else fixed_report_step
        return report_dt, min(max_ts, 0.1)

    if t < 0.03:
        report_dt, max_ts = 0.001, 0.001
    elif t < 0.1:
        report_dt, max_ts = 0.01, 0.005
    elif t < 1.0:
        report_dt, max_ts = 0.05, 0.01
    elif t < 5.0:
        report_dt, max_ts = 0.25, 0.0125
    elif t < 10.0:
        report_dt, max_ts = 0.5, 0.025
    elif t < 30.0:
        report_dt, max_ts = 1.0, 0.05
    else:
        report_dt, max_ts = 1.0, 0.1

    return min(report_dt, remaining), max_ts


def run_case(
    cfg,
    output_root,
    save_vtk=True,
    save_well_time_data=True,
    fixed_report_step=None,
    fixed_max_ts=None,
    vtk_stride=1,
):
    case_dir = output_root / case_name(cfg)
    case_dir.mkdir(parents=True, exist_ok=True)
    redirect_darts_output(str(case_dir / "run.log"))

    model = NaturalProductionModel(cfg)
    model.init()
    model.set_output(output_folder=str(case_dir), save_initial=False)

    if save_vtk:
        output_props = model.physics.vars + model.output.properties
        model.output.output_to_vtk(ith_step=0, output_properties=output_props, engine=True)
        model.output.well_output_to_vtp(ith_step=0, output_properties=output_props)

    ith_step = 0
    while model.physics.engine.t < cfg.runtime - 1e-12:
        ith_step += 1
        dt, dt_max = scheduled_step_settings(
            float(model.physics.engine.t),
            cfg.runtime,
            fixed_report_step=fixed_report_step,
            fixed_max_ts=fixed_max_ts,
        )
        model.data_ts.dt_max = dt_max
        model.params.max_ts = dt_max
        model.run(
            dt,
            save_well_data=True,
            save_well_data_after_run=False,
            save_reservoir_data=False,
        )
        print_pipe_phase_props(model, "P1")
        if save_vtk and ith_step % vtk_stride == 0:
            output_props = model.physics.vars + model.output.properties
            model.output.output_to_vtk(
                ith_step=ith_step,
                output_properties=output_props,
                engine=True,
            )
            model.output.well_output_to_vtp(
                ith_step=ith_step,
                output_properties=output_props,
            )

    time_data = model.output.store_well_time_data(save_output_files=save_well_time_data)
    return summarize_case(cfg, model, time_data, case_dir)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Scan no-pump natural production conditions for a DFM producer."
    )
    parser.add_argument("--p-init", default="200", help="Comma-separated reservoir pressures [bar].")
    parser.add_argument("--t-init", default="356.15", help="Comma-separated reservoir temperatures [K].")
    parser.add_argument("--z-co2", default="0.5", help="Comma-separated initial CO2 overall mole fractions.")
    parser.add_argument(
        "--sg-init",
        default="0.3",
        help="Comma-separated target initial gas saturations. Overrides --z-co2 by flash inversion.",
    )
    parser.add_argument("--p-whp", default="1", help="Comma-separated producer WHP targets [bar].")
    parser.add_argument(
        "--p-bhp",
        default=None,
        help="Deprecated alias for --p-whp. In DFM wells this pressure is applied at the pipe head.",
    )
    parser.add_argument("--pipe-init-phase", default="G", help="Comma-separated pipe initial phases: L,G.")
    parser.add_argument("--runtime", type=float, default=182.5, help="Runtime per case [day].")
    parser.add_argument(
        "--report-step",
        type=float,
        default=None,
        help="Use a fixed report/run step [day]. If omitted, a half-year schedule is used.",
    )
    parser.add_argument(
        "--max-ts",
        type=float,
        default=None,
        help="Use a fixed internal max timestep [day], capped at 0.1 day.",
    )
    parser.add_argument("--tol-newton", type=float, default=2e-3, help="Newton residual tolerance.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help="Natural production threshold for negative P1 mass rate [kg/day].",
    )
    parser.add_argument(
        "--output-root",
        default="natural_production_output",
        help="Output directory relative to this script unless absolute.",
    )
    parser.add_argument(
        "--no-vtk",
        action="store_true",
        help="Do not save reservoir VTU and well VTP files.",
    )
    parser.add_argument(
        "--vtk-stride",
        type=int,
        default=1,
        help="Save VTK/VTP every N accepted report steps.",
    )
    parser.add_argument(
        "--no-well-time-data",
        action="store_true",
        help="Do not save well_time_data.pkl/xlsx files.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    script_dir = Path(__file__).resolve().parent
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = script_dir / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    p_values = parse_float_list(args.p_init)
    t_values = parse_float_list(args.t_init)
    if args.sg_init is None:
        z_values = parse_float_list(args.z_co2)
        sg_values = [None]
    else:
        z_values = [parse_float_list(args.z_co2)[0]]
        sg_values = parse_float_list(args.sg_init)
    whp_text = args.p_bhp if args.p_bhp is not None else args.p_whp
    whp_values = parse_float_list(whp_text)
    pipe_phases = parse_phase_list(args.pipe_init_phase)
    rows = []
    for p_init, t_init, z_co2, sg_init, p_whp, pipe_phase in product(
        p_values, t_values, z_values, sg_values, whp_values, pipe_phases
    ):
        cfg = NaturalProductionConfig(
            p_init=p_init,
            t_init=t_init,
            z_co2_init=z_co2,
            sG_init_target=sg_init,
            p_prod_bhp=p_whp,
            pipe_init_phase=pipe_phase,
            runtime=args.runtime,
            max_ts=args.max_ts if args.max_ts is not None else 1e-3,
            tol_newton=args.tol_newton,
            natural_rate_threshold_kg_day=args.threshold,
        )
        rows.append(
            run_case(
                cfg,
                output_root,
                save_vtk=not args.no_vtk,
                save_well_time_data=not args.no_well_time_data,
                fixed_report_step=args.report_step,
                fixed_max_ts=args.max_ts,
                vtk_stride=max(1, args.vtk_stride),
            )
        )

    summary_path = output_root / "natural_production_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} case summary rows to {summary_path}")


if __name__ == "__main__":
    main()
