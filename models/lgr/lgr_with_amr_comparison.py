from __future__ import annotations

import os
import shutil

import matplotlib
import numpy as np
import pandas as pd
from darts.engines import redirect_darts_output

from darts.reservoirs.adaptive_lgr import (
    AdaptiveLGRConfig,
    count_lgr_parent_cells,
    plan_adaptive_lgr,
    project_reservoir_state,
    reservoir_state_from_engine,
)
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.reservoirs.struct_reservoir_with_lgr import LGRPatch, StructReservoirWithLGR

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lgr_comparison import (
    DEPTH,
    DX_PARENT,
    DY_PARENT,
    DZ_PARENT,
    LGR_SPECS,
    PARENT_NX,
    PARENT_NY,
    PARENT_NZ,
    RATE_PHASES,
    REFINE,
    ROOT,
    WELL_NAMES,
    FlowComparisonModel,
    coarse_permeability,
    fine_grid_property,
    interpolate_column_to_time,
    structured_well_cell,
)

OUTPUT_ROOT = ROOT / "models" / "lgr" / "lgr_with_amr_comparison_output"

CASE_NAMES = ("lgr_normal", "amr_normal")
PLOT_CASE_NAMES = ("fine", "coarse", *CASE_NAMES)
REPORT_STEPS = np.full(20, 0.5)

AMR_PARENT_CELLS_BY_WELL = {
    "I1": (*LGR_SPECS["inj_lgr"], 1),
    "P1": (*LGR_SPECS["prod_lgr"], 1),
}


class AMRFlowComparisonModel(FlowComparisonModel):
    """
    Compare static well-cell LGR with report-step adaptive LGR.

    The AMR case is refine-only: selected parent cells remain refined after they
    are added. Coarsening is intentionally left for a later implementation stage.
    """

    def __init__(self, grid_kind: str, lgr_mode: str = "normal"):
        self.amr_config = AdaptiveLGRConfig(
            refine=REFINE,
            buffer_cells=1,
            gradient_threshold=0.25,
            indicator_variables=("CO2", "C1"),
            seed_parent_cells=tuple(AMR_PARENT_CELLS_BY_WELL.values()),
            preserve_existing=True,
            max_refined_parent_cells=4,
            patch_name_prefix="amr",
        )
        self.lgrs = initial_amr_lgrs() if grid_kind == "amr_normal" else []
        self.amr_history = []
        self.time_data_history = []
        self._captured_time_data_rows = 0
        super().__init__(grid_kind=grid_kind, lgr_mode=lgr_mode)

    def set_reservoir(self):
        kx, ky, kz = coarse_permeability()
        if self.grid_kind in {"fine", "coarse"}:
            rx, ry, _ = REFINE
            refine_ratio = rx if self.grid_kind == "fine" else 1
            self.reservoir = StructReservoir(
                self.timer,
                nx=PARENT_NX * refine_ratio,
                ny=PARENT_NY * refine_ratio,
                nz=PARENT_NZ,
                dx=DX_PARENT / refine_ratio,
                dy=DY_PARENT / refine_ratio,
                dz=DZ_PARENT,
                permx=fine_grid_property(kx) if self.grid_kind == "fine" else kx,
                permy=fine_grid_property(ky) if self.grid_kind == "fine" else ky,
                permz=fine_grid_property(kz) if self.grid_kind == "fine" else kz,
                poro=0.3,
                depth=DEPTH,
                hcap=2200.0,
                rcond=120.0,
            )
            return

        self.reservoir = self._make_lgr_reservoir(
            self.lgrs if self.grid_kind == "amr_normal" else static_lgrs()
        )

    def set_wells(self):
        for well_name in WELL_NAMES:
            self.reservoir.add_well(well_name)
            if self.grid_kind in {"fine", "coarse"}:
                patch_name = "inj_lgr" if well_name == "I1" else "prod_lgr"
                self.reservoir.add_perforation(
                    well_name,
                    res_cell_idx=structured_well_cell(self.grid_kind, patch_name),
                    well_diameter=0.1524,
                )
            elif self.grid_kind == "amr_normal":
                self.reservoir.add_perforation(
                    well_name,
                    res_cell_idx=self.reservoir.get_cell_index_in_parent_cell(
                        AMR_PARENT_CELLS_BY_WELL[well_name]
                    ),
                    well_diameter=0.1524,
                    well_indexD=None,
                )
            else:
                patch_name = "inj_lgr" if well_name == "I1" else "prod_lgr"
                self.reservoir.add_perforation(
                    well_name,
                    lgr_name=patch_name,
                    lgr_cell_idx=(3, 3, 1),
                    well_diameter=0.1524,
                    well_indexD=None,
                )

    def adapt_lgr(self, verbose: bool = False) -> bool:
        capture_engine_time_data(self)
        if self.grid_kind != "amr_normal":
            return False

        state = reservoir_state_from_engine(self)
        plan = plan_adaptive_lgr(
            self.reservoir,
            state,
            self.physics.vars,
            self.amr_config,
        )
        if not plan.changed:
            return False

        old_reservoir = self.reservoir
        old_time = float(self.physics.engine.t)
        old_vtk_files = dict(getattr(old_reservoir, "vtk_filenames_and_times", {}))

        new_reservoir = self._make_lgr_reservoir(plan.lgrs)
        new_reservoir.init_reservoir(verbose=False)
        new_reservoir.vtk_filenames_and_times = old_vtk_files
        projected_state = project_reservoir_state(
            old_reservoir,
            state,
            new_reservoir,
        )

        self.reservoir = new_reservoir
        self.lgrs = plan.lgrs
        self.set_wells()
        self.has_dfm_well = False
        self.wells = None

        self.reservoir.init_wells()
        self.physics.init_wells(self.reservoir.wells)
        self.set_op_list()
        self.set_boundary_conditions()
        self.set_well_controls()
        self._set_projected_initial_state(projected_state)
        self.reset()
        self._set_engine_reservoir_state(projected_state)
        self.physics.engine.t = old_time
        self._captured_time_data_rows = 0
        self._refresh_output_after_amr()

        self.amr_history.append(
            {
                "time": old_time,
                "n_lgrs": len(self.lgrs),
                "n_selected_parent_cells": len(plan.selected_parent_cells),
                "n_refined_parent_cells": count_lgr_parent_cells(self.lgrs),
                "n_res_blocks": self.reservoir.mesh.n_res_blocks,
            }
        )
        if verbose:
            print(
                "AMR updated LGR layout at "
                f"t={old_time:g} days: {len(self.lgrs)} patches, "
                f"{self.reservoir.mesh.n_res_blocks} reservoir blocks."
            )
        return True

    def _make_lgr_reservoir(self, lgrs: list[LGRPatch]) -> StructReservoirWithLGR:
        kx, ky, kz = coarse_permeability()
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
            depth=DEPTH,
            hcap=2200.0,
            rcond=120.0,
        )
        return StructReservoirWithLGR(
            self.timer,
            parent,
            lgrs,
            lgr_coarse_fine_tran_mode=self.lgr_mode,
        )

    def _set_projected_initial_state(self, projected_state: np.ndarray) -> None:
        input_distribution = {
            var_name: projected_state[:, idx]
            for idx, var_name in enumerate(self.physics.vars)
        }
        self.physics.set_initial_conditions_from_array(
            mesh=self.reservoir.mesh,
            input_distribution=input_distribution,
        )

    def _set_engine_reservoir_state(self, projected_state: np.ndarray) -> None:
        flat_state = np.asarray(projected_state, dtype=float).reshape(-1)
        for name in ("X", "Xn"):
            values = getattr(self.physics.engine, name, None)
            if values is not None:
                np.asarray(values)[: flat_state.size] = flat_state

    def _refresh_output_after_amr(self) -> None:
        if not hasattr(self, "output"):
            return
        self.output.reservoir = self.reservoir
        self.output.op_list = self.op_list
        self.output.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        self.output.wells = self.wells
        self.output.has_dfm_well = self.has_dfm_well


def initial_amr_lgrs() -> list[LGRPatch]:
    return [
        LGRPatch(
            f"amr_seed_{well_name}",
            (i, i),
            (j, j),
            (k, k),
            REFINE,
        )
        for well_name, (i, j, k) in AMR_PARENT_CELLS_BY_WELL.items()
    ]


def static_lgrs() -> list[LGRPatch]:
    return [
        LGRPatch(name, (i, i), (j, j), (1, 1), REFINE)
        for name, (i, j) in LGR_SPECS.items()
    ]


def run_case(case_name: str, grid_kind: str, lgr_mode: str = "normal") -> dict:
    output_dir = OUTPUT_ROOT / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    redirect_darts_output(os.devnull)

    model = AMRFlowComparisonModel(grid_kind=grid_kind, lgr_mode=lgr_mode)
    model.init(platform="cpu")
    model.set_output(output_folder=str(output_dir), save_initial=False)
    reset_reservoir_vtk(case_name)
    write_reservoir_vtk(case_name, model, ith_step=0)

    for ith_step, dt in enumerate(REPORT_STEPS, start=1):
        model.run(
            float(dt),
            save_well_data=False,
            save_well_data_after_run=False,
            save_reservoir_data=False,
            verbose=False,
        )
        model.adapt_lgr(verbose=grid_kind == "amr_normal")
        write_reservoir_vtk(case_name, model, ith_step=ith_step)

    capture_engine_time_data(model)
    return {
        "model": model,
        "well_time_data": engine_history_to_well_df(model),
    }


def capture_engine_time_data(model: AMRFlowComparisonModel) -> None:
    time_data = model.physics.engine.time_data
    if "time" not in time_data:
        return

    n_rows = len(time_data["time"])
    start = getattr(model, "_captured_time_data_rows", 0)
    if start >= n_rows:
        return

    keys = list(time_data.keys())
    for row_idx in range(start, n_rows):
        model.time_data_history.append(
            {
                key: time_data[key][row_idx]
                for key in keys
                if row_idx < len(time_data[key])
            }
        )
    model._captured_time_data_rows = n_rows


def engine_history_to_well_df(model: AMRFlowComparisonModel) -> pd.DataFrame:
    engine_df = pd.DataFrame(model.time_data_history)
    data = {"time": engine_df["time"].to_numpy(dtype=float)}
    for well_name in WELL_NAMES:
        data[f"well_{well_name}_BHP"] = engine_df[f"{well_name} : BHP (bar)"].to_numpy(
            dtype=float
        )
        for phase in RATE_PHASES:
            data[f"well_{well_name}_{phase}_phase_rate"] = engine_df[
                f"{well_name} : {phase} rate (m3/day)"
            ].to_numpy(dtype=float)
    return pd.DataFrame(data)


def reset_reservoir_vtk(case_name: str) -> None:
    vtk_dir = OUTPUT_ROOT / "vtk" / case_name
    if vtk_dir.exists():
        shutil.rmtree(vtk_dir)


def write_reservoir_vtk(
    case_name: str,
    model: AMRFlowComparisonModel,
    ith_step: int,
) -> None:
    vtk_dir = OUTPUT_ROOT / "vtk" / case_name
    output_props = model.physics.vars + model.output.properties
    model.output.output_to_vtk(
        ith_step=ith_step,
        output_directory=str(vtk_dir),
        output_properties=output_props,
        engine=True,
    )


def write_well_timeseries_comparison(cases: dict[str, dict]) -> None:
    fine_well_df = cases["fine"]["well_time_data"]
    fine_time = fine_well_df["time"].to_numpy(dtype=float)
    long_rows = []

    for well_name, metric_name, column in comparison_columns():
        fine_values = fine_well_df[column].to_numpy(dtype=float)
        row_data = {
            "time": fine_time,
            "well": np.full_like(fine_time, well_name, dtype=object),
            "metric": np.full_like(fine_time, metric_name, dtype=object),
            "fine": fine_values,
        }
        for case_name in ("coarse", *CASE_NAMES):
            case_well_df = cases[case_name]["well_time_data"]
            case_values = interpolate_column_to_time(case_well_df, column, fine_time)
            row_data[case_name] = case_values

        long_rows.append(pd.DataFrame(row_data))

    plot_well_timeseries_comparison(pd.concat(long_rows, ignore_index=True))


def comparison_columns() -> list[tuple[str, str, str]]:
    columns = []
    for well_name in WELL_NAMES:
        columns.append((well_name, "BHP", f"well_{well_name}_BHP"))
        for phase in RATE_PHASES:
            metric_name = f"{phase}_phase_rate"
            column = f"well_{well_name}_{phase}_phase_rate"
            columns.append((well_name, metric_name, column))
    return columns


def plot_well_timeseries_comparison(timeseries_df: pd.DataFrame) -> None:
    plot_dir = OUTPUT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    case_labels = {
        "fine": "fine grid",
        "coarse": "coarse grid",
        "lgr_normal": "well-cell LGR normal",
        "amr_normal": "adaptive LGR normal",
    }
    colors = {
        "fine": "black",
        "coarse": "tab:orange",
        "lgr_normal": "tab:blue",
        "amr_normal": "tab:green",
    }
    linestyles = {
        "fine": "-",
        "coarse": (0, (3, 1, 1, 1)),
        "lgr_normal": "--",
        "amr_normal": "-.",
    }

    for well_name in WELL_NAMES:
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.0), sharex=True)
        plot_specs = [
            ("BHP", "BHP [bar]"),
            ("gas_phase_rate", "gas rate [m3/day]"),
            ("aqueous_phase_rate", "aqueous rate [m3/day]"),
        ]
        for axis, (metric, ylabel) in zip(axes, plot_specs, strict=True):
            df = timeseries_df[
                (timeseries_df["well"] == well_name)
                & (timeseries_df["metric"] == metric)
            ]
            for case_name in PLOT_CASE_NAMES:
                axis.plot(
                    df["time"],
                    df[case_name],
                    label=case_labels[case_name],
                    color=colors[case_name],
                    linestyle=linestyles[case_name],
                    linewidth=1.8,
                )
            axis.set_ylabel(ylabel)
            axis.grid(True, linewidth=0.4, alpha=0.5)
        axes[-1].set_xlabel("time [days]")
        axes[0].legend(loc="best", fontsize=8)
        fig.suptitle(f"{well_name}: AMR BHP and phase rates")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{well_name}_amr_bhp_phase_rates.png", dpi=180)
        plt.close(fig)


def cleanup_non_plot_outputs() -> None:
    for path in OUTPUT_ROOT.iterdir():
        if path.name in {"plots", "vtk"}:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    vtk_root = OUTPUT_ROOT / "vtk"
    if vtk_root.exists():
        shutil.rmtree(vtk_root)

    cases = {
        "fine": run_case("fine", "fine"),
        "coarse": run_case("coarse", "coarse"),
        "lgr_normal": run_case("lgr_normal", "lgr_normal"),
        "amr_normal": run_case("amr_normal", "amr_normal"),
    }
    write_well_timeseries_comparison(cases)
    cleanup_non_plot_outputs()

    print(f"Plots written to {OUTPUT_ROOT / 'plots'}")
    print(f"Reservoir VTK files written to {OUTPUT_ROOT / 'vtk'}")


if __name__ == "__main__":
    main()
