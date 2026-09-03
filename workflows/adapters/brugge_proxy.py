"""Adapter for the CI coarse Brugge proxy (``models/Uniform_Brugge``).

Cheap knobs: transmissibility multipliers (per interface or scalar), well-index multipliers (per
well or scalar), per-cell porosity, per-well BHP targets. The gmsh mesh is generated once per study
into the input snapshot and reused by every member (``regenerate_mesh=False``). Members write no
VTK or spreadsheet files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from workflows.adapter import InputSnapshot, ModelAdapter, add_generated_input
from workflows.spec import ObservationSpec

MESH_NAME = "Brugge_model.msh"
QUANTITIES = {
    "oil_rate": "well_{w}_volumetric_rate_oil_at_wh",
    "wat_rate": "well_{w}_volumetric_rate_wat_at_wh",
    "gas_rate": "well_{w}_volumetric_rate_gas_at_wh",
    "bhp": "well_{w}_BHP",
    "bht": "well_{w}_BHT",
}


def _import_model(model_dir: Path):
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from model import Model

    return Model


class Adapter(ModelAdapter):
    input_patterns = ("Brugge_struct/*",)
    rebuild_scope = {
        "tran_multiplier": "cheap",
        "wi_multiplier": "cheap",
        "poro": "cheap",
        "bhp": "cheap",
        "relperm": "physics",
        "logperm": "geometry",
    }
    member_seconds = (8.0, 12.0)

    def prepare_snapshot(self, snapshot: InputSnapshot) -> InputSnapshot:
        mesh = Path(snapshot.root) / MESH_NAME
        if MESH_NAME not in snapshot.files:
            model_cls = _import_model(self.model_dir)
            model_cls(
                input_dir=snapshot.root, mesh_file=str(mesh), regenerate_mesh=True
            )
            snapshot = add_generated_input(snapshot, mesh)
        return snapshot

    def build(self, paths: dict, realization: dict):
        model_cls = _import_model(self.model_dir)
        inputs = Path(paths["inputs"])
        model = model_cls(
            input_dir=str(inputs),
            mesh_file=str(inputs / MESH_NAME),
            regenerate_mesh=False,
        )
        model.init()
        model.set_output(
            output_folder=str(Path(paths["workdir"]) / "output"), verbose=False
        )
        return model

    def apply(self, model, realization: dict) -> None:
        from darts.engines import value_vector, well_control_iface

        mesh = model.reservoir.mesh
        touched = False
        if "tran_multiplier" in realization:
            tran, tran_d = value_vector([]), value_vector([])
            mesh.get_res_tran(tran, tran_d)
            factor = np.asarray(realization["tran_multiplier"], dtype=float)
            new_tran = np.array(tran) * factor
            new_tran_d = np.array(tran_d) * factor
            mesh.set_res_tran(value_vector(new_tran), value_vector(new_tran_d))
            touched = True
        if "wi_multiplier" in realization:
            wi = value_vector([])
            mesh.get_wells_tran(wi)
            mesh.set_wells_tran(
                value_vector(
                    np.array(wi) * np.asarray(realization["wi_multiplier"], dtype=float)
                )
            )
            touched = True
        if "poro" in realization:
            np.array(mesh.poro, copy=False)[: mesh.n_res_blocks] = np.asarray(
                realization["poro"], dtype=float
            )
            touched = True
        if "bhp" in realization:
            for well in model.reservoir.wells:
                if well.name in realization["bhp"]:
                    is_inj = well.name.startswith("I")
                    kwargs = (
                        {"inj_composition": model.inj_composition} if is_inj else {}
                    )
                    model.physics.set_well_controls(
                        wctrl=well.control,
                        control_type=well_control_iface.BHP,
                        is_inj=is_inj,
                        target=float(realization["bhp"][well.name]),
                        **kwargs,
                    )
        if touched:
            model.reset()

    def run(self, model, report_times) -> None:
        current = 0.0
        for target in report_times:
            step = float(target) - current
            if step <= 0:
                raise ValueError(
                    "report times must be strictly increasing and positive"
                )
            model.run(step, save_reservoir_data=False, verbose=0)
            current = float(target)

    def observe(self, model, observation: ObservationSpec | None) -> dict:
        data = model.output.store_well_time_data(save_output_files=False)
        times = np.asarray(data["time"], dtype=float)
        if observation is None:
            return {"time": times.tolist(), "values": {}}
        report = np.asarray(observation.report_times, dtype=float)
        rows = []
        for t in report:
            hit = np.flatnonzero(np.isclose(times, t, rtol=0.0, atol=1e-9))
            if hit.size == 0:
                raise RuntimeError(
                    f"no simulation row at report time {t}; run() must stop there"
                )
            rows.append(int(hit[-1]))
        values = {}
        for well in observation.wells:
            for quantity in observation.quantities:
                key = QUANTITIES[quantity].format(w=well)
                values[f"{well}:{quantity}"] = np.asarray(data[key], dtype=float)[
                    rows
                ].tolist()
        return {"time": report.tolist(), "values": values}
