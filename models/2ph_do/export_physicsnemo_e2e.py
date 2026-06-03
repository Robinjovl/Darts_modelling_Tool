"""
Generate an open-DARTS dataset for the PhysicsNeMo XMGN adapter.

This is an end-to-end plumbing script for the 1D two-phase dead-oil example. It
runs a few short open-DARTS rollouts, saves reservoir HDF5 snapshots, and exports
each case to the schema consumed by PhysicsNeMo's OpenDARTS graph builder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from darts.engines import redirect_darts_output, set_gpu_device
from model import Model

from darts.tools.physicsnemo_export import (
    export_physicsnemo_hdf5,
    validate_physicsnemo_hdf5,
)


def _case_multiplier(case_idx: int, num_cases: int, perturbation: float) -> float:
    """
    Compute a symmetric multiplicative perturbation for one case.

    :param case_idx: Zero-based case index.
    :type case_idx: int
    :param num_cases: Total number of generated cases.
    :type num_cases: int
    :param perturbation: Relative perturbation applied at the sweep endpoints.
    :type perturbation: float
    :return: Multiplicative factor for the selected case.
    :rtype: float
    """

    if num_cases <= 1 or perturbation == 0.0:
        return 1.0
    midpoint = 0.5 * (num_cases - 1)
    normalized = (case_idx - midpoint) / max(midpoint, 1.0)
    return 1.0 + perturbation * normalized


def _smooth_case_profile(nx: int, case_idx: int, seed: int) -> np.ndarray:
    """
    Build a deterministic smooth random profile over the 1D grid.

    :param nx: Number of reservoir cells.
    :type nx: int
    :param case_idx: Zero-based case index used to offset the random seed.
    :type case_idx: int
    :param seed: Base random seed for reproducible profile generation.
    :type seed: int
    :return: Normalized profile with zero mean and unit variance.
    :rtype: np.ndarray
    """

    rng = np.random.default_rng(seed + case_idx)
    x = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    phase_a = rng.uniform(0.0, 2.0 * np.pi)
    phase_b = rng.uniform(0.0, 2.0 * np.pi)
    profile = (
        0.55 * np.sin(2.0 * np.pi * x + phase_a)
        + 0.30 * np.sin(6.0 * np.pi * x + phase_b)
        + 0.15 * rng.standard_normal(nx)
    )
    return (profile - profile.mean()) / (profile.std() + 1e-12)


def _case_arrays(
    nx: int,
    case_idx: int,
    seed: int,
    base_perm: float,
    log_perm_std: float,
    base_poro: float,
    poro_variation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create permeability and porosity arrays for one generated case.

    :param nx: Number of reservoir cells.
    :type nx: int
    :param case_idx: Zero-based case index.
    :type case_idx: int
    :param seed: Base random seed for deterministic heterogeneity.
    :type seed: int
    :param base_perm: Base permeability before lognormal perturbation.
    :type base_perm: float
    :param log_perm_std: Standard deviation of the log-permeability multiplier.
    :type log_perm_std: float
    :param base_poro: Base porosity before spatial perturbation.
    :type base_poro: float
    :param poro_variation: Relative porosity variation applied by the profile.
    :type poro_variation: float
    :return: Permeability and porosity arrays for the case.
    :rtype: tuple[np.ndarray, np.ndarray]
    """

    profile = _smooth_case_profile(nx, case_idx, seed)
    permx = base_perm * np.exp(log_perm_std * profile)
    poro = np.clip(base_poro * (1.0 + poro_variation * profile), 0.05, 0.45)
    return permx.astype(np.float64), poro.astype(np.float64)


def _static_control_features(
    nx: int,
    inj_rate: float,
    prd_bhp: float,
    inj_bhp_limit: float,
    initial_pressure: float,
    initial_water: float,
) -> dict[str, np.ndarray]:
    """
    Create per-cell static arrays for controls and initial conditions.

    :param nx: Number of reservoir cells.
    :type nx: int
    :param inj_rate: Injector molar-rate target.
    :type inj_rate: float
    :param prd_bhp: Producer bottom-hole pressure target.
    :type prd_bhp: float
    :param inj_bhp_limit: Injector bottom-hole pressure constraint.
    :type inj_bhp_limit: float
    :param initial_pressure: Initial reservoir pressure.
    :type initial_pressure: float
    :param initial_water: Initial first-component composition.
    :type initial_water: float
    :return: Mapping from static feature names to per-cell arrays.
    :rtype: dict[str, np.ndarray]
    """

    well_role = np.zeros(nx, dtype=np.float64)
    inj_rate_target = np.zeros(nx, dtype=np.float64)
    prd_bhp_target = np.zeros(nx, dtype=np.float64)
    inj_bhp_limit_feature = np.zeros(nx, dtype=np.float64)

    well_role[0] = 1.0
    well_role[-1] = -1.0
    inj_rate_target[0] = inj_rate
    prd_bhp_target[-1] = prd_bhp
    inj_bhp_limit_feature[0] = inj_bhp_limit

    return {
        "WELL_ROLE": well_role,
        "INJ_RATE_TARGET": inj_rate_target,
        "PRD_BHP_TARGET": prd_bhp_target,
        "INJ_BHP_LIMIT": inj_bhp_limit_feature,
        "INITIAL_PRESSURE": np.full(nx, initial_pressure, dtype=np.float64),
        "INITIAL_WATER": np.full(nx, initial_water, dtype=np.float64),
    }


def _run_case(
    case_name: str,
    output_root: Path,
    export_dir: Path,
    report_steps: int,
    days_per_report_step: float,
    nx: int,
    inj_rate: float,
    prd_bhp: float,
    inj_bhp_limit: float,
    max_ts: float,
    permx: np.ndarray,
    poro: np.ndarray,
    initial_pressure: float,
    initial_water: float,
    platform: str,
    gpu_device: int,
) -> dict:
    """
    Run one open-DARTS case and export its validated PhysicsNeMo HDF5 file.

    :param case_name: Name used for the run folder and exported HDF5 file.
    :type case_name: str
    :param output_root: Directory where raw open-DARTS run folders are written.
    :type output_root: Path
    :param export_dir: Directory where PhysicsNeMo HDF5 files are written.
    :type export_dir: Path
    :param report_steps: Number of report intervals to simulate and save.
    :type report_steps: int
    :param days_per_report_step: Simulation days advanced per report interval.
    :type days_per_report_step: float
    :param nx: Number of reservoir cells.
    :type nx: int
    :param inj_rate: Injector molar-rate target.
    :type inj_rate: float
    :param prd_bhp: Producer bottom-hole pressure target.
    :type prd_bhp: float
    :param inj_bhp_limit: Injector bottom-hole pressure constraint.
    :type inj_bhp_limit: float
    :param max_ts: Maximum open-DARTS timestep in days.
    :type max_ts: float
    :param permx: Per-cell x-direction permeability.
    :type permx: np.ndarray
    :param poro: Per-cell porosity.
    :type poro: np.ndarray
    :param initial_pressure: Initial reservoir pressure.
    :type initial_pressure: float
    :param initial_water: Initial first-component composition.
    :type initial_water: float
    :param platform: open-DARTS execution platform, either ``"cpu"`` or
        ``"gpu"``.
    :type platform: str
    :param gpu_device: Visible GPU device index used when ``platform`` is
        ``"gpu"``.
    :type gpu_device: int
    :return: Validation summary for the exported HDF5 file.
    :rtype: dict
    """

    run_dir = output_root / case_name
    run_dir.mkdir(parents=True, exist_ok=True)
    redirect_darts_output(str(run_dir / "run.log"))

    model = Model(
        nx=nx,
        inj_rate=inj_rate,
        prd_bhp=prd_bhp,
        inj_bhp_limit=inj_bhp_limit,
        max_ts=max_ts,
        permx=permx,
        poro=poro,
        initial_pressure=initial_pressure,
        initial_water=initial_water,
    )
    if platform == "gpu":
        set_gpu_device(gpu_device)
    model.init(platform=platform)

    model.set_output(output_folder=str(run_dir), save_initial=True)

    for _ in range(report_steps):
        model.run(
            days=days_per_report_step,
            save_well_data=False,
            save_well_data_after_run=False,
            save_reservoir_data=True,
            verbose=False,
        )

    export_path = export_dir / f"{case_name}.h5"
    export_physicsnemo_hdf5(
        model=model,
        output_path=export_path,
        case_name=case_name,
        extra_static_cell_data=_static_control_features(
            nx=nx,
            inj_rate=inj_rate,
            prd_bhp=prd_bhp,
            inj_bhp_limit=inj_bhp_limit,
            initial_pressure=initial_pressure,
            initial_water=initial_water,
        ),
    )
    return validate_physicsnemo_hdf5(export_path)


def main() -> int:
    """
    Parse command-line options and generate the requested export cases.

    :return: Process exit code.
    :rtype: int
    """

    parser = argparse.ArgumentParser(
        description="Run 2ph_do and export an OpenDARTS PhysicsNeMo dataset."
    )
    parser.add_argument(
        "--output-root",
        default="../../../open_darts_physicsnemo_e2e/open_darts_runs",
        help="Directory for raw open-DARTS output folders.",
    )
    parser.add_argument(
        "--export-dir",
        default="../../../physicsnemo/examples/reservoir_simulation/dataset/open_darts_exports",
        help="Directory for exported PhysicsNeMo HDF5 cases.",
    )
    parser.add_argument(
        "--case-prefix",
        default="case",
        help="Prefix used in generated case names.",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=100,
        help="Number of cells in the 1D reservoir.",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=3,
        help="Number of short cases to generate. The XMGN preprocessor expects at least 3 cases.",
    )
    parser.add_argument(
        "--report-steps",
        type=int,
        default=5,
        help="Number of one-day report steps saved after the initial state.",
    )
    parser.add_argument(
        "--days-per-report-step",
        type=float,
        default=1.0,
        help="Simulation days advanced before each saved reservoir snapshot.",
    )
    parser.add_argument(
        "--inj-rate",
        type=float,
        default=200.0,
        help="Base injector molar-rate target.",
    )
    parser.add_argument(
        "--prd-bhp",
        type=float,
        default=350.0,
        help="Base producer BHP target.",
    )
    parser.add_argument(
        "--inj-bhp-limit",
        type=float,
        default=450.0,
        help="Injector BHP constraint target.",
    )
    parser.add_argument(
        "--max-ts",
        type=float,
        default=5.0,
        help="Maximum open-DARTS timestep in days.",
    )
    parser.add_argument(
        "--platform",
        choices=("cpu", "gpu"),
        default="cpu",
        help="open-DARTS execution platform used while generating cases.",
    )
    parser.add_argument(
        "--gpu-device",
        type=int,
        default=0,
        help="Visible GPU device index for open-DARTS when --platform=gpu.",
    )
    parser.add_argument(
        "--base-perm",
        type=float,
        default=300.0,
        help="Base permeability in mD.",
    )
    parser.add_argument(
        "--log-perm-std",
        type=float,
        default=0.35,
        help="Standard deviation of the lognormal permeability multiplier.",
    )
    parser.add_argument(
        "--base-poro",
        type=float,
        default=0.2,
        help="Base porosity.",
    )
    parser.add_argument(
        "--poro-variation",
        type=float,
        default=0.12,
        help="Relative porosity variation imposed by the smooth case profile.",
    )
    parser.add_argument(
        "--initial-pressure",
        type=float,
        default=400.0,
        help="Base initial pressure.",
    )
    parser.add_argument(
        "--initial-pressure-variation",
        type=float,
        default=0.05,
        help="Relative +/- initial pressure perturbation across cases.",
    )
    parser.add_argument(
        "--initial-water",
        type=float,
        default=1.0 - 1e-13,
        help="Base initial first-component composition.",
    )
    parser.add_argument(
        "--initial-water-variation",
        type=float,
        default=0.02,
        help="Relative +/- initial composition perturbation across cases.",
    )
    parser.add_argument(
        "--case-perturbation",
        type=float,
        default=0.15,
        help="Relative +/- perturbation applied across cases to rate and producer BHP.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed for deterministic heterogeneous case generation.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    export_dir = Path(args.export_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for case_idx in range(args.num_cases):
        multiplier = _case_multiplier(case_idx, args.num_cases, args.case_perturbation)
        initial_multiplier = _case_multiplier(
            case_idx, args.num_cases, args.initial_pressure_variation
        )
        water_multiplier = _case_multiplier(
            case_idx, args.num_cases, args.initial_water_variation
        )
        permx, poro = _case_arrays(
            nx=args.nx,
            case_idx=case_idx,
            seed=args.seed,
            base_perm=args.base_perm,
            log_perm_std=args.log_perm_std,
            base_poro=args.base_poro,
            poro_variation=args.poro_variation,
        )
        initial_pressure = args.initial_pressure * initial_multiplier
        initial_water = min(max(args.initial_water * water_multiplier, 1e-8), 1.0 - 1e-8)
        summaries.append(
            _run_case(
                case_name=f"{args.case_prefix}_{case_idx + 1:03d}",
                output_root=output_root,
                export_dir=export_dir,
                report_steps=args.report_steps,
                days_per_report_step=args.days_per_report_step,
                nx=args.nx,
                inj_rate=args.inj_rate * multiplier,
                prd_bhp=args.prd_bhp * (2.0 - multiplier),
                inj_bhp_limit=args.inj_bhp_limit,
                max_ts=args.max_ts,
                permx=permx,
                poro=poro,
                initial_pressure=initial_pressure,
                initial_water=initial_water,
                platform=args.platform,
                gpu_device=args.gpu_device,
            )
        )

    for summary in summaries:
        print(
            "{case_name}: {n_timesteps} timesteps, {n_cells} cells, "
            "{n_edges} edges, variables={variable_names}".format(**summary)
        )
    print(f"Exported {len(summaries)} cases to {export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
