# ruff: noqa: E402, I001
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import darts

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DARTS_PATH = str(ROOT / "darts")
if LOCAL_DARTS_PATH not in darts.__path__:
    darts.__path__.insert(0, LOCAL_DARTS_PATH)

from darts.engines import (
    ms_well,
    redirect_darts_output,
    sim_params,
)

from darts.models.cicd_model import CICDModel
from darts.pipes.define_pipe_geometry import PipeGeometry
from darts.pipes.pipe import Pipe
from darts.pipes.set_initial_conditions import LinearAmbientTemperature
from darts.reservoirs.struct_reservoir import StructReservoir

DFM_MODEL_PATH = ROOT / "models" / "lgr" / "2ph_comp_thermal_dfm_wells" / "model.py"
OUTPUT_ROOT = ROOT / "models" / "lgr" / "dfm_amr_comparison_output"

RUNTIME_DAYS = 50.0
REPORT_STEPS = tuple([0.001] * 10 + [0.01] * 9 + [0.1] * 9 + [1.0] * 49)
WELL_NAMES = ("I1", "P1")
PHASE_NAMES = ("G", "L")


def load_dfm_model_class() -> type[CICDModel]:
    spec = importlib.util.spec_from_file_location(
        "lgr_dfm_thermal_model", DFM_MODEL_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load DFM model from {DFM_MODEL_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


BaseDFMModel = load_dfm_model_class()


class DFMAMRComparisonModel(BaseDFMModel):
    def __init__(self, grid_kind: str):
        self.grid_kind = grid_kind
        super().__init__(use_amr=(grid_kind == "amr"))
        self.set_sim_params(
            first_ts=1e-7,
            mult_ts=2,
            max_ts=1e-4,
            runtime=RUNTIME_DAYS,
            tol_newton=1e-3,
            tol_linear=1e-4,
            it_newton=12,
            it_linear=60,
            newton_type=sim_params.newton_local_chop,
            coupled_well_res_norm_method=2,
        )

    def set_reservoir(self):
        if self.grid_kind == "fine":
            self.reservoir = self._make_fine_reservoir()
        else:
            super().set_reservoir()

    def _make_fine_reservoir(self):
        rx, ry, rz = self.amr_config.refine
        nx_parent, ny_parent, nz_parent = self.parent_shape

        reservoir = StructReservoir(
            self.timer,
            nx=nx_parent * rx,
            ny=ny_parent * ry,
            nz=nz_parent * rz,
            dx=20.0 / rx,
            dy=20.0 / ry,
            dz=10.0 / rz,
            permx=100.0,
            permy=100.0,
            permz=10.0,
            poro=0.3,
            depth=2005.0,
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


def update_time_step_controls(model: DFMAMRComparisonModel) -> None:
    current_time = model.physics.engine.t
    if 0.03 < current_time < 0.05:
        model.data_ts.dt_max = 0.05
    elif 1.0 < current_time < 3.0:
        model.data_ts.dt_max = 0.1


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
            frames.append(latest_data)
            last_time = float(latest_data["time"].max())

        if grid_kind == "amr":
            if model.adapt_lgr(verbose=False) and hasattr(
                model, "_well_output_configured"
            ):
                delattr(model, "_well_output_configured")

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
        for phase in PHASE_NAMES
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


if __name__ == "__main__":
    main()
