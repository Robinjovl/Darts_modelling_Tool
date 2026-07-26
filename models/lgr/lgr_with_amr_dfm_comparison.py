from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from darts.engines import (
    ms_well,
    redirect_darts_output,
    sim_params,
)

from darts.models.cicd_model import CICDModel
from darts.pipes.define_pipe_geometry import PipeGeometry
from darts.pipes.pipe import Pipe
from darts.pipes.set_initial_conditions import LinearAmbientTemperature
from darts.reservoirs.adaptive_lgr import AdaptiveLGRConfig
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.reservoirs.struct_reservoir_with_lgr import LGRPatch

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
DFM_MODEL_PATH = SCRIPT_DIR / "2ph_comp_thermal_dfm_wells" / "model.py"
LGR_COMPARISON_PATH = SCRIPT_DIR / "lgr_comparison.py"
OUTPUT_ROOT = SCRIPT_DIR / "lgr_with_amr_dfm_comparison_output"

RUNTIME_DAYS = 10.0
REPORT_STEPS = tuple([0.001] * 10 + [0.01] * 9 + [0.1] * 9 + [0.5] * 18)
WELL_NAMES = ("I1", "P1")
RES_DEPTH = 2005.0
FINE_STARTUP_MAX_TS = 1e-3
AMR_STARTUP_MAX_TS = 1e-4
MID_MAX_TS = 0.01
LONG_MAX_TS = 0.05


def load_python_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Python module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dfm_model_class() -> type[CICDModel]:
    module = load_python_module("lgr_dfm_thermal_model", DFM_MODEL_PATH)
    return module.Model


BaseDFMModel = load_dfm_model_class()
LGRComparison = load_python_module("lgr_reference_comparison", LGR_COMPARISON_PATH)

PARENT_NX = LGRComparison.PARENT_NX
PARENT_NY = LGRComparison.PARENT_NY
PARENT_NZ = LGRComparison.PARENT_NZ
REFINE = LGRComparison.REFINE
DX_PARENT = LGRComparison.DX_PARENT
DY_PARENT = LGRComparison.DY_PARENT
DZ_PARENT = LGRComparison.DZ_PARENT
WELL_PARENT_CELLS = {
    "I1": (*LGRComparison.LGR_SPECS["inj_lgr"], 1),
    "P1": (*LGRComparison.LGR_SPECS["prod_lgr"], 1),
}


class DFMAMRComparisonModel(BaseDFMModel):
    def __init__(self, grid_kind: str):
        self.grid_kind = grid_kind
        CICDModel.__init__(self)

        self.timer.node["initialization"].start()

        self.zero = 1e-8
        self.use_amr = grid_kind == "amr"
        self.parent_shape = (PARENT_NX, PARENT_NY, PARENT_NZ)
        self.perforation_parent_cells = WELL_PARENT_CELLS.copy()
        self.perforation_fractions = {
            "I1": (0.5, 0.5, 0.5),
            "P1": (0.5, 0.5, 0.5),
        }
        self.amr_config = AdaptiveLGRConfig(
            refine=REFINE,
            buffer_cells=1,
            gradient_threshold=0.25,
            indicator_variables=("CO2"),
            seed_parent_cells=tuple(self.perforation_parent_cells.values()),
            preserve_existing=True,
            max_refined_parent_cells=4,
            patch_name_prefix="amr",
        )
        self.lgrs = self._initial_amr_lgrs() if self.use_amr else []
        self.amr_history = []

        self.set_reservoir()
        self.reservoir.grav_acceleration_for_spe = 9.80665
        self.set_physics()
        self.set_sim_params(
            first_ts=1e-7,
            mult_ts=2,
            max_ts=AMR_STARTUP_MAX_TS if self.use_amr else FINE_STARTUP_MAX_TS,
            runtime=RUNTIME_DAYS,
            tol_newton=1e-3,
            tol_linear=1e-4,
            it_newton=12,
            it_linear=60,
            newton_type=sim_params.newton_local_chop,
            coupled_well_res_norm_method=2,
        )

        self.timer.node["initialization"].stop()

    def _initial_amr_lgrs(self) -> list[LGRPatch]:
        return [
            LGRPatch(
                f"amr_seed_{well_name}",
                (i, i),
                (j, j),
                (k, k),
                self.amr_config.refine,
            )
            for well_name, (i, j, k) in self.perforation_parent_cells.items()
        ]

    def _make_parent_reservoir(self):
        kx, ky, kz = LGRComparison.coarse_permeability()
        parent = StructReservoir(
            self.timer,
            nx=PARENT_NX,
            ny=PARENT_NY,
            nz=PARENT_NZ,
            dx=DX_PARENT,
            dy=DY_PARENT,
            dz=DZ_PARENT,
            permx=kx,
            permy=ky,
            permz=kz,
            poro=0.3,
            depth=RES_DEPTH,
            hcap=2200.0,
            rcond=120.0,
        )
        parent.boundary_volumes["yz_minus"] = 1e20
        parent.boundary_volumes["yz_plus"] = 1e20
        return parent

    def set_reservoir(self):
        if self.grid_kind == "fine":
            self.reservoir = self._make_fine_reservoir()
        else:
            super().set_reservoir()

    def _make_fine_reservoir(self):
        rx, ry, rz = self.amr_config.refine
        kx, ky, kz = LGRComparison.coarse_permeability()

        reservoir = StructReservoir(
            self.timer,
            nx=PARENT_NX * rx,
            ny=PARENT_NY * ry,
            nz=PARENT_NZ * rz,
            dx=DX_PARENT / rx,
            dy=DY_PARENT / ry,
            dz=DZ_PARENT / rz,
            permx=LGRComparison.fine_grid_property(kx),
            permy=LGRComparison.fine_grid_property(ky),
            permz=LGRComparison.fine_grid_property(kz),
            poro=0.3,
            depth=RES_DEPTH,
            hcap=2200.0,
            rcond=120.0,
            cache=False,
        )
        reservoir.boundary_volumes["yz_minus"] = 1e20
        reservoir.boundary_volumes["yz_plus"] = 1e20
        return reservoir

    def _add_dfm_well(
        self,
        well_name: str,
        parent_cell: tuple[int, int, int],
        fractions: tuple[float, float, float],
    ) -> None:
        if self.grid_kind != "fine":
            super()._add_dfm_well(
                well_name=well_name,
                parent_cell=parent_cell,
                fractions=fractions,
            )
            return

        segments_lengths = 50 * np.ones(40)
        segments_lengths = np.append(segments_lengths, 10)
        well_diameter = 0.1524
        well_geometry = PipeGeometry(
            well_name,
            segments_lengths,
            well_diameter,
            inclination_angle=0.0,
        )
        initial_conditions = LinearAmbientTemperature(
            pipe_name=well_name,
            pipe_geom=well_geometry,
            physics=self.physics,
            pipe_head_pressure=1.0,
            pipe_head_temperature=25 + 273.15,
            temp_grad=0.03,
            pipe_head_segment_index=0,
            initial_conditions_dict={
                "phases_names": ["L"],
                "phases_compositions": [[0.01, 0.01, 0.98]],
                "pipe_intervals": [[0.0, well_geometry.pipe_length]],
            },
        )
        self.wells[well_name] = Pipe(
            well_name,
            well_geometry,
            self.physics,
            self.reservoir,
            initial_conditions,
        )

        self.reservoir.add_well(
            well_name,
            ms_well.MS_Type.DFM,
            well_geometry=well_geometry,
        )
        res_cell_idx = self._fine_cell_in_parent(parent_cell, fractions)
        self.reservoir.add_perforation(
            well_name,
            res_cell_idx=res_cell_idx,
            well_seg_idx=well_geometry.num_segments,
            well_diameter=well_geometry.pipe_ID,
            well_indexD=None,
            with_peaceman_for_coupled_well_reservoir=True,
        )

    def _fine_cell_in_parent(
        self,
        parent_cell: tuple[int, int, int],
        fractions: tuple[float, float, float],
    ) -> tuple[int, int, int]:
        return tuple(
            (parent_idx - 1) * refine
            + min(max(int(fraction * refine), 0), refine - 1)
            + 1
            for parent_idx, fraction, refine in zip(
                parent_cell, fractions, self.amr_config.refine, strict=True
            )
        )


def prepare_output_root() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def make_case_output_dir(case_name: str) -> Path:
    candidate = OUTPUT_ROOT / case_name
    if not candidate.exists():
        candidate.mkdir(parents=True)
        return candidate

    for idx in range(1, 1000):
        candidate = OUTPUT_ROOT / f"{case_name}_{idx:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate

    raise RuntimeError(f"Could not create output directory for case '{case_name}'")


def reset_solution_output(case_name: str) -> None:
    for output_kind in ("vtk", "vtp"):
        output_dir = (OUTPUT_ROOT / output_kind / case_name).resolve()
        allowed_root = (OUTPUT_ROOT / output_kind).resolve()
        if output_dir.exists():
            if allowed_root not in output_dir.parents:
                raise RuntimeError(f"Refusing to remove unexpected path: {output_dir}")
            shutil.rmtree(output_dir)


def write_solution_output(
    case_name: str,
    model: DFMAMRComparisonModel,
    ith_step: int,
) -> None:
    output_props = model.physics.vars + model.output.properties
    model.output.output_to_vtk(
        ith_step=ith_step,
        output_directory=str(OUTPUT_ROOT / "vtk" / case_name),
        output_properties=output_props,
        engine=True,
    )
    model.output.well_output_to_vtp(
        ith_step=ith_step,
        output_directory=str(OUTPUT_ROOT / "vtp" / case_name),
        output_properties=output_props,
    )


def update_time_step_controls(model: DFMAMRComparisonModel) -> None:
    current_time = model.physics.engine.t
    if current_time < 0.03:
        model.data_ts.dt_max = (
            AMR_STARTUP_MAX_TS if model.grid_kind == "amr" else FINE_STARTUP_MAX_TS
        )
    elif current_time < 1.0:
        model.data_ts.dt_max = MID_MAX_TS
    else:
        model.data_ts.dt_max = LONG_MAX_TS


def required_output_columns(df: pd.DataFrame) -> list[str]:
    columns = []
    for well_name in WELL_NAMES:
        columns.extend(
            column
            for _, column, _ in available_columns(df, well_name)
            if column in df.columns
        )
    return columns


def assert_finite_output(df: pd.DataFrame, case_name: str) -> None:
    columns = required_output_columns(df)
    if not columns:
        return

    values = df[columns].to_numpy(dtype=float)
    bad = ~np.isfinite(values)
    if not bad.any():
        return

    row_idx, col_idx = np.argwhere(bad)[0]
    time = float(df.iloc[row_idx]["time"])
    column = columns[col_idx]
    raise RuntimeError(
        f"Case '{case_name}' produced non-finite '{column}' at t={time:g} days."
    )


def collect_latest_well_data(
    model: DFMAMRComparisonModel, last_time: float
) -> pd.DataFrame:
    well_data = pd.DataFrame(
        model.output.store_well_time_data(
            phase_molar_rates=True,
            phase_mass_rates=False,
            phase_volumetric_rates=False,
            advective_heat_rates=False,
            save_output_files=False,
        )
    )
    if well_data.empty:
        return well_data
    return well_data[well_data["time"] > last_time + 1e-10].copy()


def run_case(case_name: str, grid_kind: str) -> pd.DataFrame:
    output_dir = make_case_output_dir(case_name)
    redirect_darts_output(str(output_dir / "darts.log"))

    model = DFMAMRComparisonModel(grid_kind=grid_kind)
    model.init(platform="cpu")
    model.set_output(output_folder=str(output_dir), save_initial=False)
    reset_solution_output(case_name)
    write_solution_output(case_name, model, ith_step=0)

    frames: list[pd.DataFrame] = []
    last_time = -np.inf
    for report_idx, report_step in enumerate(REPORT_STEPS, start=1):
        update_time_step_controls(model)
        model.run(
            float(report_step),
            save_well_data=False,
            save_well_data_after_run=True,
            save_reservoir_data=False,
            verbose=False,
        )

        latest_data = collect_latest_well_data(model, last_time)
        if not latest_data.empty:
            assert_finite_output(latest_data, case_name)
            frames.append(latest_data)
            last_time = float(latest_data["time"].max())

        if grid_kind == "amr":
            if model.adapt_lgr(verbose=False) and hasattr(
                model, "_well_output_configured"
            ):
                delattr(model, "_well_output_configured")

        write_solution_output(case_name, model, ith_step=report_idx)
        print(
            f"{case_name}: report {report_idx:03d}/{len(REPORT_STEPS)}, "
            f"time={model.physics.engine.t:.6g} days"
        )

    if not frames:
        raise RuntimeError(f"No well time data was collected for case '{case_name}'")
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset="time")


def available_columns(df: pd.DataFrame, well_name: str) -> list[tuple[str, str, str]]:
    candidates = [
        ("BHP", f"well_{well_name}_BHP", "bar"),
        ("BHT", f"well_{well_name}_BHT", "K"),
    ]
    candidates.extend(
        (
            f"{phase} molar rate",
            f"well_{well_name}_molar_rate_{phase}_by_sum_perfs",
            "kmol/day",
        )
        for phase in ("G", "L")
    )
    return [
        (label, column, unit)
        for label, column, unit in candidates
        if column in df.columns
    ]


def plot_well_timeseries(
    well_name: str,
    fine_data: pd.DataFrame,
    amr_data: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    columns = available_columns(fine_data, well_name)
    fig, axes = plt.subplots(
        len(columns), 1, figsize=(9.5, 2.4 * len(columns)), sharex=True
    )
    if len(columns) == 1:
        axes = [axes]

    for axis, (label, column, unit) in zip(axes, columns, strict=True):
        axis.plot(fine_data["time"], fine_data[column], label="fine", linewidth=1.8)
        axis.plot(
            amr_data["time"],
            amr_data[column],
            label="AMR",
            linewidth=1.4,
            linestyle="--",
        )
        axis.set_ylabel(f"{label}\n[{unit}]")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")

    axes[-1].set_xlabel("time [days]")
    fig.suptitle(f"DFM well {well_name}: fine grid and AMR")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / f"well_{well_name}_timeseries.png", dpi=200)
    plt.close(fig)


def plot_well_difference(
    well_name: str,
    fine_data: pd.DataFrame,
    amr_data: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    columns = available_columns(fine_data, well_name)
    fine_time = fine_data["time"].to_numpy()

    fig, axes = plt.subplots(
        len(columns), 1, figsize=(9.5, 2.4 * len(columns)), sharex=True
    )
    if len(columns) == 1:
        axes = [axes]

    for axis, (label, column, unit) in zip(axes, columns, strict=True):
        amr_interp = np.interp(fine_time, amr_data["time"], amr_data[column])
        diff = amr_interp - fine_data[column].to_numpy()
        axis.plot(fine_time, diff, color="tab:red", linewidth=1.6)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        axis.set_ylabel(f"AMR - fine\n{label} [{unit}]")
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("time [days]")
    fig.suptitle(f"DFM well {well_name}: AMR difference from fine grid")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / f"well_{well_name}_difference.png", dpi=200)
    plt.close(fig)


def print_difference_summary(fine_data: pd.DataFrame, amr_data: pd.DataFrame) -> None:
    fine_time = fine_data["time"].to_numpy()
    print("\nMaximum absolute AMR - fine differences:")
    for well_name in WELL_NAMES:
        for label, column, unit in available_columns(fine_data, well_name):
            amr_interp = np.interp(fine_time, amr_data["time"], amr_data[column])
            diff = amr_interp - fine_data[column].to_numpy()
            print(f"  {well_name:>2s} {label:<14s}: {np.max(np.abs(diff)):.6g} {unit}")


def main() -> None:
    if not np.isclose(sum(REPORT_STEPS), RUNTIME_DAYS):
        raise RuntimeError("Report steps do not sum to the requested runtime")

    prepare_output_root()
    fine_data = run_case("fine", "fine")
    amr_data = run_case("amr", "amr")

    for well_name in WELL_NAMES:
        plot_well_timeseries(well_name, fine_data, amr_data)
        plot_well_difference(well_name, fine_data, amr_data)

    print_difference_summary(fine_data, amr_data)
    print(f"\nPlots written to: {OUTPUT_ROOT}")
    print(f"Reservoir VTK files written to: {OUTPUT_ROOT / 'vtk'}")
    print(f"Well VTP files written to: {OUTPUT_ROOT / 'vtp'}")


if __name__ == "__main__":
    main()
