from __future__ import annotations

import os
import shutil

import matplotlib
import numpy as np
import pandas as pd
from darts.engines import redirect_darts_output

from darts.reservoirs.struct_reservoir import StructReservoir
from darts.reservoirs.struct_reservoir_with_lgr import LGRPatch, StructReservoirWithLGR

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flow_based_upscaling_reference_comparison import (
    DEPTH,
    DX_PARENT,
    DY_PARENT,
    DZ_PARENT,
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
)

OUTPUT_ROOT = ROOT / "models" / "lgr" / "flow_based_corridor_comparison_output"

INJ_PARENT_CELL = (3, 3)
PROD_PARENT_CELL = (7, 3)
CORRIDOR_PARENT_CELLS = tuple((i, INJ_PARENT_CELL[1]) for i in range(3, 8))
CORRIDOR_PATCH_BY_WELL = {
    "I1": f"corridor_i{INJ_PARENT_CELL[0]}_j{INJ_PARENT_CELL[1]}",
    "P1": f"corridor_i{PROD_PARENT_CELL[0]}_j{PROD_PARENT_CELL[1]}",
}
CASE_NAMES = ("lgr_normal", "lgr_steady")
PLOT_CASE_NAMES = ("fine", "coarse", *CASE_NAMES)
REPORT_STEPS = np.full(20, 0.5)


class CorridorFlowComparisonModel(FlowComparisonModel):
    """
    Compare LGR transmissibility modes with a refined injector-producer corridor.
    """

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
        lgrs = [
            LGRPatch(
                name=f"corridor_i{i}_j{j}",
                i_range=(i, i),
                j_range=(j, j),
                k_range=(1, 1),
                refine=REFINE,
            )
            for i, j in CORRIDOR_PARENT_CELLS
        ]
        self.reservoir = StructReservoirWithLGR(
            self.timer,
            parent,
            lgrs,
            lgr_coarse_fine_transmissibility_mode=self.lgr_mode,
        )

    def set_wells(self):
        for well_name in WELL_NAMES:
            self.reservoir.add_well(well_name)
            if self.grid_kind in {"fine", "coarse"}:
                self.reservoir.add_perforation(
                    well_name,
                    res_cell_idx=structured_corridor_well_cell(
                        self.grid_kind, well_name
                    ),
                    well_diameter=0.1524,
                )
            else:
                self.reservoir.add_perforation(
                    well_name,
                    lgr_name=CORRIDOR_PATCH_BY_WELL[well_name],
                    lgr_cell_idx=(3, 3, 1),
                    well_diameter=0.1524,
                    well_indexD=None,
                )


def structured_corridor_well_cell(
    grid_kind: str, well_name: str
) -> tuple[int, int, int]:
    if well_name == "I1":
        parent_i, parent_j = INJ_PARENT_CELL
    elif well_name == "P1":
        parent_i, parent_j = PROD_PARENT_CELL
    else:
        raise KeyError(f"Unknown well name {well_name!r}.")

    if grid_kind == "coarse":
        return parent_i, parent_j, 1

    rx, ry, _ = REFINE
    fine_i = (parent_i - 1) * rx + rx // 2 + 1
    fine_j = (parent_j - 1) * ry + ry // 2 + 1
    return fine_i, fine_j, 1


def run_case(case_name: str, grid_kind: str, lgr_mode: str = "flow_based") -> dict:
    output_dir = OUTPUT_ROOT / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    redirect_darts_output(os.devnull)

    model = CorridorFlowComparisonModel(grid_kind=grid_kind, lgr_mode=lgr_mode)
    model.init(platform="cpu")
    model.set_output(output_folder=str(output_dir))
    reset_reservoir_vtk(case_name)
    write_reservoir_vtk(case_name, model, ith_step=0)

    for ith_step, dt in enumerate(REPORT_STEPS, start=1):
        model.run(
            float(dt),
            save_well_data=False,
            save_well_data_after_run=True,
            save_reservoir_data=False,
            verbose=False,
        )
        write_reservoir_vtk(case_name, model, ith_step=ith_step)

    well_time_data = model.output.store_well_time_data(
        phase_molar_rates=True,
        phase_mass_rates=False,
        phase_volumetric_rates=False,
        component_molar_rates=False,
        component_mass_rates=False,
        save_output_files=False,
    )
    return {
        "model": model,
        "well_time_data": pd.DataFrame(well_time_data),
    }


def reset_reservoir_vtk(case_name: str) -> None:
    vtk_dir = OUTPUT_ROOT / "vtk" / case_name
    if vtk_dir.exists():
        shutil.rmtree(vtk_dir)


def write_reservoir_vtk(
    case_name: str,
    model: CorridorFlowComparisonModel,
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
            metric_name = f"{phase}_phase_molar_rate_by_sum_perfs"
            column = f"well_{well_name}_molar_rate_{phase}_by_sum_perfs"
            columns.append((well_name, metric_name, column))
    return columns


def plot_well_timeseries_comparison(timeseries_df: pd.DataFrame) -> None:
    plot_dir = OUTPUT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    case_labels = {
        "fine": "fine grid",
        "coarse": "coarse grid",
        "lgr_normal": "corridor LGR normal",
        "lgr_steady": "corridor LGR steady flow-based",
    }
    colors = {
        "fine": "black",
        "coarse": "tab:orange",
        "lgr_normal": "tab:blue",
        "lgr_steady": "tab:green",
    }
    linestyles = {
        "fine": "-",
        "coarse": (0, (3, 1, 1, 1)),
        "lgr_normal": "--",
        "lgr_steady": "-.",
    }

    for well_name in WELL_NAMES:
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.0), sharex=True)
        plot_specs = [
            ("BHP", "BHP [bar]"),
            ("gas_phase_molar_rate_by_sum_perfs", "gas molar rate"),
            ("aqueous_phase_molar_rate_by_sum_perfs", "aqueous molar rate"),
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
        fig.suptitle(f"{well_name}: refined-corridor BHP and phase rates")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{well_name}_corridor_bhp_phase_rates.png", dpi=180)
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
        "lgr_normal": run_case("lgr_normal", "lgr_steady", lgr_mode="normal"),
        "lgr_steady": run_case("lgr_steady", "lgr_steady"),
    }
    write_well_timeseries_comparison(cases)
    cleanup_non_plot_outputs()

    print(f"Plots written to {OUTPUT_ROOT / 'plots'}")
    print(f"Reservoir VTK files written to {OUTPUT_ROOT / 'vtk'}")


if __name__ == "__main__":
    main()
