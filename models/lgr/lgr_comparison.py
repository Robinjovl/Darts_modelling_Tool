from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from darts.engines import redirect_darts_output, sim_params, well_control_iface

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from darts.models.darts_model import DartsModel
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic
from darts.physics.properties.flash import ConstantK
from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.reservoirs.flow_based_tran_for_lgr import scale_by_raw_distribution
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.reservoirs.struct_reservoir_with_lgr import LGRPatch, StructReservoirWithLGR

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "models" / "lgr" / "lgr_comparison_output"

PARENT_NX = 9
PARENT_NY = 5
PARENT_NZ = 1
REFINE = (5, 5, 1)
DX_PARENT = 20.0
DY_PARENT = 20.0
DZ_PARENT = 10.0
DEPTH = 1000.0

LGR_SPECS = {
    "inj_lgr": (3, 3),
    "prod_lgr": (7, 3),
}
SIDE_NAMES = {
    (0, -1): "left",
    (0, 1): "right",
    (1, -1): "up",
    (1, 1): "down",
}
CASE_NAMES = ("lgr_normal", "lgr_steady", "lgr_kairan")
PLOT_CASE_NAMES = ("fine", "coarse", *CASE_NAMES)
WELL_NAMES = ("I1", "P1")
RATE_PHASES = ("gas", "aqueous")
REPORT_STEPS = np.full(20, 0.5)


def coarse_permeability() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a deliberately heterogeneous parent-grid permeability field."""
    kx = np.array(
        [
            [80, 80, 90, 140, 220],
            [90, 70, 80, 130, 210],
            [110, 80, 70, 120, 190],
            [130, 100, 90, 140, 180],
            [180, 170, 160, 150, 170],
            [260, 230, 210, 190, 160],
            [340, 300, 260, 220, 180],
            [420, 360, 300, 240, 190],
            [500, 420, 340, 260, 200],
        ],
        dtype=float,
    )
    ky = np.array(
        [
            [60, 120, 220, 120, 70],
            [70, 140, 260, 140, 80],
            [80, 160, 320, 160, 90],
            [90, 180, 360, 180, 100],
            [100, 210, 410, 210, 110],
            [110, 190, 330, 190, 120],
            [120, 170, 270, 170, 130],
            [130, 150, 220, 150, 140],
            [140, 140, 180, 140, 150],
        ],
        dtype=float,
    )
    kz = 0.1 * kx
    return kx[:, :, None], ky[:, :, None], kz[:, :, None]


def fine_grid_property(parent_property: np.ndarray) -> np.ndarray:
    rx, ry, _ = REFINE
    return np.repeat(np.repeat(parent_property, rx, axis=0), ry, axis=1)


class KairanTransientLGRReservoir(StructReservoirWithLGR):
    """Comparison-only reservoir using Kairan's transient apparent side trans."""

    def __init__(self, *args, output_root: Path, **kwargs):
        self.kairan_output_root = Path(output_root)
        self.kairan_side_trans_cache: dict[str, dict[str, float]] = {}
        super().__init__(*args, **kwargs)

    def _apply_flow_based_coarse_fine_transmissibility(
        self,
        coarse_fine_connections,
        tran,
        tran_thermal,
    ) -> None:
        if not coarse_fine_connections:
            return

        patch_by_name = {patch.name: patch for patch in self.lgrs}
        grouped = {}
        for conn in coarse_fine_connections:
            grouped.setdefault((conn.patch_name, conn.axis, conn.side), []).append(conn)

        diagnostics = []
        for (patch_name, axis, side), group in sorted(grouped.items()):
            side_name = SIDE_NAMES.get((axis, side))
            if side_name is None:
                continue

            patch = patch_by_name[patch_name]
            side_trans = self._kairan_side_transmissibilities(patch)
            raw_tran = np.asarray([conn.raw_tran for conn in group], dtype=float)
            target_total = side_trans[side_name] * len(group)
            scaled_tran = scale_by_raw_distribution(raw_tran, target_total)
            hydraulic_scale = target_total / float(np.sum(raw_tran))

            for conn, tm in zip(group, scaled_tran, strict=True):
                tran[conn.index] = float(tm)
                tran_thermal[conn.index] = float(
                    conn.raw_tran_thermal * hydraulic_scale
                )

            diagnostics.append(
                {
                    "patch": patch_name,
                    "axis": axis,
                    "side": side,
                    "n_connections": len(group),
                    "raw_tran_total": float(np.sum(raw_tran)),
                    "scaled_tran_total": float(np.sum(scaled_tran)),
                    "kairan_avg_link_trans": float(side_trans[side_name]),
                }
            )

        self.lgr_coarse_fine_transmissibility_diagnostics = diagnostics

    def _kairan_side_transmissibilities(self, patch: LGRPatch) -> dict[str, float]:
        if patch.name not in self.kairan_side_trans_cache:
            self.kairan_side_trans_cache[patch.name] = run_kairan_local_upscaling(
                patch,
                self._lgr_parent_arrays,
                self.kairan_output_root,
            )
        return self.kairan_side_trans_cache[patch.name]


class FlowComparisonModel(DartsModel):
    def __init__(self, grid_kind: str, lgr_mode: str = "flow_based"):
        super().__init__()
        self.grid_kind = grid_kind
        self.lgr_mode = lgr_mode
        self.zero = 1e-8

        self.timer.node["initialization"].start()
        self.set_reservoir()
        self.set_physics()
        self.set_sim_params(
            first_ts=1e-3,
            mult_ts=2.0,
            max_ts=0.5,
            runtime=10.0,
            tol_newton=1e-3,
            tol_linear=1e-4,
            it_newton=12,
            it_linear=80,
            newton_type=sim_params.newton_local_chop,
        )
        self.timer.node["initialization"].stop()

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
            LGRPatch(name, (i, i), (j, j), (1, 1), REFINE)
            for name, (i, j) in LGR_SPECS.items()
        ]
        reservoir_cls = (
            KairanTransientLGRReservoir
            if self.grid_kind == "lgr_kairan"
            else StructReservoirWithLGR
        )
        kwargs = {}
        if reservoir_cls is KairanTransientLGRReservoir:
            kwargs["output_root"] = OUTPUT_ROOT / "kairan_upscaling"
        self.reservoir = reservoir_cls(
            self.timer,
            parent,
            lgrs,
            lgr_coarse_fine_tran_mode=self.lgr_mode,
            **kwargs,
        )

    def set_wells(self):
        for well_name, patch_name in (("I1", "inj_lgr"), ("P1", "prod_lgr")):
            self.reservoir.add_well(well_name)
            if self.grid_kind in {"fine", "coarse"}:
                self.reservoir.add_perforation(
                    well_name,
                    res_cell_idx=structured_well_cell(self.grid_kind, patch_name),
                    well_diameter=0.1524,
                )
            else:
                self.reservoir.add_perforation(
                    well_name,
                    lgr_name=patch_name,
                    lgr_cell_idx=(3, 3, 1),
                    well_diameter=0.1524,
                    well_indexD=None,
                )

    def set_physics(self):
        epsilon = 1e-9
        components = ["CO2", "C1", "H2O"]
        phases = ["gas", "aqueous"]
        molecular_weights = [44.01, 16.04, 18.015]

        property_container = PropertyContainer(
            phases_name=phases,
            components_name=components,
            Mw=molecular_weights,
            eps_z=epsilon,
            temperature=1.0,
        )
        property_container.flash_ev = ConstantK(
            len(components), [4.0, 2.0, 0.1], self.zero
        )
        property_container.density_ev = {
            "gas": DensityBasic(compr=1e-3, dens0=200.0),
            "aqueous": DensityBasic(compr=1e-5, dens0=600.0),
        }
        property_container.viscosity_ev = {
            "gas": ConstFunc(0.05),
            "aqueous": ConstFunc(0.5),
        }
        property_container.rel_perm_ev = {
            "gas": PhaseRelPerm("gas"),
            "aqueous": PhaseRelPerm("oil"),
        }

        self.physics = Compositional(
            components,
            phases,
            self.timer,
            state_spec=Compositional.StateSpecification.P,
            n_points=200,
            min_p=1.0,
            max_p=300.0,
            min_z=0.0,
            max_z=1.0,
            epsilon_z=epsilon,
            extrapolation_flag=True,
        )
        self.physics.add_property_region(property_container)

    def set_initial_conditions(self):
        input_distribution = {
            self.physics.vars[0]: 100.0,
            self.physics.vars[1]: 0.1,
            self.physics.vars[2]: 0.2,
        }
        self.physics.set_initial_conditions_from_array(
            mesh=self.reservoir.mesh,
            input_distribution=input_distribution,
        )

    def set_well_controls(self):
        injection_composition = [1.0 - 2.0 * self.zero, self.zero]
        for idx, well in enumerate(self.reservoir.wells):
            if idx == 0:
                self.physics.set_well_controls(
                    wctrl=well.control,
                    control_type=well_control_iface.BHP,
                    is_inj=True,
                    target=140.0,
                    inj_composition=injection_composition,
                )
            else:
                self.physics.set_well_controls(
                    wctrl=well.control,
                    control_type=well_control_iface.BHP,
                    is_inj=False,
                    target=80.0,
                )


def structured_well_cell(grid_kind: str, patch_name: str) -> tuple[int, int, int]:
    parent_i, parent_j = LGR_SPECS[patch_name]
    if grid_kind == "coarse":
        return parent_i, parent_j, 1

    rx, ry, _ = REFINE
    fine_i = (parent_i - 1) * rx + rx // 2 + 1
    fine_j = (parent_j - 1) * ry + ry // 2 + 1
    return fine_i, fine_j, 1


def run_kairan_local_upscaling(
    patch: LGRPatch,
    parent_arrays: dict[str, np.ndarray],
    output_root: Path,
) -> dict[str, float]:
    tran_fc = load_kairan_tran_fc()
    kx_patch, ky_patch, kz_patch = extract_parent_patch(parent_arrays, patch)
    output_dir = output_root / patch.name
    output_dir.mkdir(parents=True, exist_ok=True)

    model = tran_fc.FlowUpscalingModel(
        kx_patch,
        ky_patch,
        kz_patch,
        refine=REFINE[:2],
        dx_parent=DX_PARENT,
        dy_parent=DY_PARENT,
        dz_parent=DZ_PARENT,
        start_z=DEPTH,
        poro=0.3,
    )
    analyzer = tran_fc.PatchEffectiveTransAnalyzer(
        model=model,
        nx=model.nx,
        ny=model.ny,
        nz=model.nz,
        patch_size=REFINE[0],
        patch_center_1b=model.patch_center_1b,
        n_nb_cols=REFINE[0],
    )
    redirect_darts_output(os.devnull)
    model.init(platform="cpu")
    model.set_output(output_folder=str(output_dir))
    for _ in range(20):
        model.run(
            365.0,
            save_well_data=False,
            save_well_data_after_run=False,
            save_reservoir_data=False,
            verbose=False,
        )

    summary = analyzer.all_faces(k0=0, mobility_mode="interface_avg")["summary_df"]
    return {
        str(row["side"]): float(row["T_eff_avglink"]) for _, row in summary.iterrows()
    }


def load_kairan_tran_fc():
    path = ROOT / "LGR_kairan" / "Egg model" / "tran_fc.py"
    spec = importlib.util.spec_from_file_location("kairan_tran_fc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_parent_patch(
    parent_arrays: dict[str, np.ndarray],
    patch: LGRPatch,
    patch_size: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half = patch_size // 2
    i_center = patch.i_range[0] - 1
    j_center = patch.j_range[0] - 1
    i_min = i_center - half
    i_max = i_center + half
    j_min = j_center - half
    j_max = j_center + half
    if i_min < 0 or i_max >= PARENT_NX or j_min < 0 or j_max >= PARENT_NY:
        raise ValueError(f"Patch {patch.name!r} is too close to the model boundary.")

    def extract(name: str) -> np.ndarray:
        values = np.asarray(parent_arrays[name], dtype=float)
        out = np.empty((patch_size, patch_size), dtype=float)
        for jj, parent_j in enumerate(range(j_min, j_max + 1)):
            for ii, parent_i in enumerate(range(i_min, i_max + 1)):
                out[ii, jj] = values[parent_i + parent_j * PARENT_NX]
        return out

    return extract("permx"), extract("permy"), extract("permz")


def run_case(case_name: str, grid_kind: str, lgr_mode: str = "flow_based") -> dict:
    output_dir = OUTPUT_ROOT / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    redirect_darts_output(os.devnull)

    model = FlowComparisonModel(grid_kind=grid_kind, lgr_mode=lgr_mode)
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
    model: FlowComparisonModel,
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
        "lgr_normal": "LGR normal",
        "lgr_steady": "LGR steady flow-based",
        "lgr_kairan": "LGR Kairan transient",
    }
    colors = {
        "fine": "black",
        "coarse": "tab:orange",
        "lgr_normal": "tab:blue",
        "lgr_steady": "tab:green",
        "lgr_kairan": "tab:red",
    }
    linestyles = {
        "fine": "-",
        "coarse": (0, (3, 1, 1, 1)),
        "lgr_normal": "--",
        "lgr_steady": "-.",
        "lgr_kairan": ":",
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
        fig.suptitle(f"{well_name}: BHP and phase rates")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{well_name}_bhp_phase_rates.png", dpi=180)
        plt.close(fig)


def interpolate_column_to_time(
    well_df: pd.DataFrame, column: str, target_time: np.ndarray
) -> np.ndarray:
    return np.interp(
        target_time,
        well_df["time"].to_numpy(dtype=float),
        well_df[column].to_numpy(dtype=float),
    )


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
        "lgr_kairan": run_case("lgr_kairan", "lgr_kairan"),
    }
    write_well_timeseries_comparison(cases)
    cleanup_non_plot_outputs()

    print(f"Plots written to {OUTPUT_ROOT / 'plots'}")
    print(f"Reservoir VTK files written to {OUTPUT_ROOT / 'vtk'}")


if __name__ == "__main__":
    main()
