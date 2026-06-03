"""
Export open-DARTS simulation data for PhysicsNeMo reservoir surrogates.

The exporter writes a compact HDF5 file that the PhysicsNeMo XMeshGraphNet
prototype can convert into PyTorch Geometric graphs. It is intentionally
additive: call it after a model has been initialized, and preferably after
reservoir HDF5 output has been saved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

SCHEMA_VERSION = "0.1"


def export_physicsnemo_hdf5(
    model: Any,
    output_path: str | Path | None = None,
    reservoir_h5_path: str | Path | None = None,
    case_name: str | None = None,
    extra_static_cell_data: dict[str, Any] | None = None,
    include_properties: bool = True,
) -> Path:
    """
    Export an initialized open-DARTS model to a PhysicsNeMo-ready HDF5 file.

    The export includes static reservoir features, graph connectivity, well
    completion metadata, and dynamic state arrays copied from the reservoir
    output file or read from the current engine state.

    :param model: Initialized ``DartsModel``-like object with
        ``model.reservoir.mesh`` available.
    :type model: Any
    :param output_path: Destination HDF5 path. Defaults to
        ``<output_folder>/physicsnemo_export.h5``.
    :type output_path: str | Path | None
    :param reservoir_h5_path: Existing open-DARTS reservoir solution HDF5 file.
        Defaults to ``model.output.sol_filepath`` when available.
    :type reservoir_h5_path: str | Path | None
    :param case_name: Optional case name stored as file metadata.
    :type case_name: str | None
    :param extra_static_cell_data: Additional per-cell arrays to store under
        ``/static``.
    :type extra_static_cell_data: dict[str, Any] | None
    :param include_properties: Copy the existing ``/properties`` group from the
        source reservoir solution file when it is present.
    :type include_properties: bool
    :return: Path to the exported HDF5 file.
    :rtype: Path
    :raises ValueError: If the model has no reservoir mesh or the export path
        matches the source reservoir HDF5 path.
    """

    reservoir = getattr(model, "reservoir", None)
    if reservoir is None or getattr(reservoir, "mesh", None) is None:
        raise ValueError(
            "model.reservoir.mesh is required; call model.init(...) first."
        )

    output_path = _default_output_path(model, output_path)
    reservoir_h5_path = _default_reservoir_h5_path(model, reservoir_h5_path)
    if (
        reservoir_h5_path is not None
        and output_path.resolve() == reservoir_h5_path.resolve()
    ):
        raise ValueError(
            "output_path must be different from the source reservoir_h5_path; "
            "the exporter writes a new PhysicsNeMo HDF5 file."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = reservoir.mesh
    n_res_blocks = int(mesh.n_res_blocks)

    with h5py.File(output_path, "w") as out:
        out.attrs["schema"] = "open_darts_physicsnemo"
        out.attrs["schema_version"] = SCHEMA_VERSION
        out.attrs["source"] = "open-DARTS"
        out.attrs["case_name"] = case_name or output_path.stem

        _write_static_group(out, model, n_res_blocks, extra_static_cell_data)
        _write_edges_group(out, mesh, n_res_blocks)
        _write_wells_group(out, reservoir)
        _write_dynamic_group(out, model, reservoir_h5_path, n_res_blocks)

        if (
            include_properties
            and reservoir_h5_path is not None
            and reservoir_h5_path.is_file()
        ):
            with h5py.File(reservoir_h5_path, "r") as src:
                if "properties" in src:
                    src.copy("properties", out)

    return output_path


def validate_physicsnemo_hdf5(path: str | Path) -> dict[str, Any]:
    """
    Validate an exported open-DARTS/PhysicsNeMo HDF5 file.

    The validator checks the structural contract consumed by the PhysicsNeMo
    OpenDARTS graph builder, including required datasets and shape consistency
    between static cell data, graph edges, times, and dynamic state arrays.

    :param path: Exported HDF5 file to validate.
    :type path: str | Path
    :return: Summary with file path, case name, dimensions, edge count, and
        dynamic variable names.
    :rtype: dict[str, Any]
    :raises FileNotFoundError: If the export file does not exist.
    :raises ValueError: If required datasets are missing or internally
        inconsistent.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Export file not found: {path}")

    with h5py.File(path, "r") as h5:
        _require_datasets(
            h5,
            [
                "static/PORV",
                "static/cell_centers",
                "static/edges/cell_m",
                "static/edges/cell_p",
                "dynamic/time",
                "dynamic/X",
                "dynamic/variable_names",
            ],
        )

        x = h5["dynamic/X"]
        if x.ndim != 3:
            raise ValueError(
                f"/dynamic/X must be 3D [time, cell, variable], got {x.shape}."
            )
        n_timesteps, n_cells, n_vars = x.shape

        time = h5["dynamic/time"]
        if time.shape[0] != n_timesteps:
            raise ValueError(
                f"/dynamic/time length {time.shape[0]} does not match X timesteps {n_timesteps}."
            )

        variable_names = h5["dynamic/variable_names"]
        if variable_names.shape[0] != n_vars:
            raise ValueError(
                f"/dynamic/variable_names length {variable_names.shape[0]} "
                f"does not match X variables {n_vars}."
            )

        centers = h5["static/cell_centers"]
        if centers.shape != (n_cells, 3):
            raise ValueError(
                f"/static/cell_centers must have shape ({n_cells}, 3), got {centers.shape}."
            )

        porv = h5["static/PORV"]
        if porv.shape[0] != n_cells:
            raise ValueError(
                f"/static/PORV length {porv.shape[0]} does not match X cells {n_cells}."
            )

        cell_m = np.asarray(h5["static/edges/cell_m"])
        cell_p = np.asarray(h5["static/edges/cell_p"])
        if cell_m.shape != cell_p.shape:
            raise ValueError(
                f"edge cell_m shape {cell_m.shape} does not match cell_p shape {cell_p.shape}."
            )
        if cell_m.size and (cell_m.min() < 0 or cell_p.min() < 0):
            raise ValueError("Edge indices must be non-negative.")
        if cell_m.size and (cell_m.max() >= n_cells or cell_p.max() >= n_cells):
            raise ValueError("Edge indices exceed number of reservoir cells.")

        for edge_name in ("tran", "tranD", "grav_coef", "edge_type"):
            edge_path = f"static/edges/{edge_name}"
            if edge_path in h5 and h5[edge_path].shape[0] != cell_m.shape[0]:
                raise ValueError(
                    f"/{edge_path} length {h5[edge_path].shape[0]} does not match edge count {cell_m.shape[0]}."
                )

        return {
            "path": str(path),
            "case_name": h5.attrs.get("case_name", path.stem),
            "n_timesteps": int(n_timesteps),
            "n_cells": int(n_cells),
            "n_variables": int(n_vars),
            "n_edges": int(cell_m.shape[0]),
            "variable_names": _decode_h5_strings(variable_names[:]),
        }


def _require_datasets(h5: h5py.File, paths: list[str]) -> None:
    missing = [path for path in paths if path not in h5]
    if missing:
        raise ValueError(f"Missing required HDF5 dataset(s): {', '.join(missing)}")


def _default_output_path(model: Any, output_path: str | Path | None) -> Path:
    if output_path is not None:
        return Path(output_path)
    output_folder = Path(getattr(model, "output_folder", "."))
    return output_folder / "physicsnemo_export.h5"


def _decode_h5_strings(values: Any) -> list[str]:
    decoded = []
    for value in values:
        decoded.append(
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
        )
    return decoded


def _default_reservoir_h5_path(
    model: Any, reservoir_h5_path: str | Path | None
) -> Path | None:
    if reservoir_h5_path is not None:
        path = Path(reservoir_h5_path)
        return path if str(path) else None
    output = getattr(model, "output", None)
    if output is not None and hasattr(output, "sol_filepath"):
        path = Path(output.sol_filepath)
        return path if str(path) else None
    if hasattr(model, "sol_filepath"):
        path = Path(model.sol_filepath)
        return path if str(path) else None
    return None


def _write_static_group(
    out: h5py.File,
    model: Any,
    n_res_blocks: int,
    extra_static_cell_data: dict[str, Any] | None,
) -> None:
    reservoir = model.reservoir
    mesh = reservoir.mesh
    static = out.require_group("static")

    centers = _resolve_cell_centers(model, n_res_blocks)
    _write_dataset(static, "cell_centers", centers)
    _write_dataset(static, "X", centers[:, 0])
    _write_dataset(static, "Y", centers[:, 1])
    _write_dataset(static, "Z", centers[:, 2])

    poro = _cell_vector(getattr(mesh, "poro", None), n_res_blocks, default=0.0)
    volume = _cell_vector(getattr(mesh, "volume", None), n_res_blocks, default=1.0)
    depth = _cell_vector(getattr(mesh, "depth", None), n_res_blocks, default=0.0)

    _write_dataset(static, "PORO", poro)
    _write_dataset(static, "VOLUME", volume)
    _write_dataset(static, "PORV", poro * volume)
    _write_dataset(static, "DEPTH", depth)

    for name in ("PERMX", "PERMY", "PERMZ"):
        values = _resolve_reservoir_property(reservoir, name.lower(), n_res_blocks)
        if values is not None:
            _write_dataset(static, name, values)

    for mesh_name, dataset_name in (
        ("op_num", "OP_NUM"),
        ("rock_cond", "RCOND"),
        ("heat_capacity", "HCAP"),
    ):
        values = _cell_vector(
            getattr(mesh, mesh_name, None), n_res_blocks, default=None
        )
        if values is not None:
            _write_dataset(static, dataset_name, values)

    if extra_static_cell_data:
        for name, values in extra_static_cell_data.items():
            _write_dataset(
                static,
                str(name).upper(),
                _cell_vector(values, n_res_blocks, default=np.nan),
            )


def _write_edges_group(out: h5py.File, mesh: Any, n_res_blocks: int) -> None:
    edges = out.require_group("static").require_group("edges")

    block_m = np.asarray(mesh.block_m, dtype=np.int64)
    block_p = np.asarray(mesh.block_p, dtype=np.int64)
    mask = (block_m < n_res_blocks) & (block_p < n_res_blocks)

    _write_dataset(edges, "cell_m", block_m[mask].astype(np.int32, copy=False))
    _write_dataset(edges, "cell_p", block_p[mask].astype(np.int32, copy=False))

    for mesh_name, dataset_name in (
        ("tran", "tran"),
        ("tranD", "tranD"),
        ("grav_coef", "grav_coef"),
    ):
        values = np.asarray(getattr(mesh, mesh_name, []), dtype=np.float64)
        if values.size == block_m.size:
            _write_dataset(edges, dataset_name, values[mask])

    _write_dataset(edges, "edge_type", np.zeros(int(mask.sum()), dtype=np.int32))


def _write_wells_group(out: h5py.File, reservoir: Any) -> None:
    wells = getattr(reservoir, "wells", []) or []
    group = out.require_group("wells")

    names: list[str] = []
    completion_cell: list[int] = []
    completion_well_id: list[int] = []
    completion_well_block: list[int] = []
    completion_wi: list[float] = []
    completion_wid: list[float] = []

    for well_id, well in enumerate(wells):
        names.append(str(getattr(well, "name", f"well_{well_id}")))
        for perf in getattr(well, "perforations", []) or []:
            if len(perf) < 2:
                continue
            completion_well_block.append(int(perf[0]))
            completion_cell.append(int(perf[1]))
            completion_well_id.append(well_id)
            completion_wi.append(float(perf[2]) if len(perf) > 2 else np.nan)
            completion_wid.append(float(perf[3]) if len(perf) > 3 else np.nan)

    string_dtype = h5py.string_dtype(encoding="utf-8")
    _write_dataset(group, "names", np.asarray(names, dtype=object), dtype=string_dtype)
    _write_dataset(
        group, "completion_cell", np.asarray(completion_cell, dtype=np.int32)
    )
    _write_dataset(
        group, "completion_well_id", np.asarray(completion_well_id, dtype=np.int32)
    )
    _write_dataset(
        group,
        "completion_well_block",
        np.asarray(completion_well_block, dtype=np.int32),
    )
    _write_dataset(group, "completion_wi", np.asarray(completion_wi, dtype=np.float64))
    _write_dataset(
        group, "completion_wid", np.asarray(completion_wid, dtype=np.float64)
    )


def _write_dynamic_group(
    out: h5py.File,
    model: Any,
    reservoir_h5_path: Path | None,
    n_res_blocks: int,
) -> None:
    if reservoir_h5_path is not None and reservoir_h5_path.is_file():
        with h5py.File(reservoir_h5_path, "r") as src:
            if "dynamic" not in src:
                raise ValueError(f"{reservoir_h5_path} has no /dynamic group.")
            src.copy("dynamic", out)
        return

    physics = getattr(model, "physics", None)
    if physics is None or getattr(physics, "engine", None) is None:
        raise ValueError(
            "No reservoir HDF5 file was found and model.physics.engine is unavailable."
        )

    dynamic = out.require_group("dynamic")
    time = np.asarray([float(getattr(physics.engine, "t", 0.0))], dtype=np.float64)
    state = np.asarray(physics.engine.X, dtype=np.float64)
    n_vars = int(physics.n_vars)
    x = state[: n_res_blocks * n_vars].reshape(1, n_res_blocks, n_vars)
    names = getattr(physics, "vars", [f"var_{i}" for i in range(n_vars)])

    _write_dataset(dynamic, "time", time)
    _write_dataset(dynamic, "CFL_max", np.asarray([physics.engine.CFL_max]))
    _write_dataset(dynamic, "cell_id", np.arange(n_res_blocks, dtype=np.int32))
    _write_dataset(dynamic, "X", x)
    _write_dataset(
        dynamic,
        "variable_names",
        np.asarray([str(name) for name in names], dtype=object),
        dtype=h5py.string_dtype(encoding="utf-8"),
    )


def _resolve_cell_centers(model: Any, n_res_blocks: int) -> np.ndarray:
    output = getattr(model, "output", None)
    if output is not None and hasattr(output, "_get_output_cell_centers"):
        centers = np.asarray(output._get_output_cell_centers(), dtype=np.float64)
    else:
        centers = None
        reservoir = model.reservoir
        if hasattr(reservoir, "discretizer"):
            for name in ("centroids_all_cells", "centroid_all_cells"):
                if hasattr(reservoir.discretizer, name):
                    centers = np.asarray(getattr(reservoir.discretizer, name))
                    break
        if centers is None and hasattr(reservoir, "centroids_all_cells"):
            centers = np.asarray(reservoir.centroids_all_cells)

    if centers is None:
        depth = _cell_vector(model.reservoir.mesh.depth, n_res_blocks, default=0.0)
        centers = np.column_stack(
            [
                np.arange(n_res_blocks, dtype=np.float64),
                np.zeros(n_res_blocks, dtype=np.float64),
                depth,
            ]
        )

    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(
            f"Expected cell centers with shape (n_cells, 3), got {centers.shape}."
        )
    return centers[:n_res_blocks]


def _resolve_reservoir_property(
    reservoir: Any, name: str, n_res_blocks: int
) -> np.ndarray | None:
    for source in (
        getattr(reservoir, name, None),
        getattr(reservoir, "global_data", {}).get(name, None)
        if hasattr(reservoir, "global_data")
        else None,
        getattr(reservoir, "global_data", {}).get(name.upper(), None)
        if hasattr(reservoir, "global_data")
        else None,
    ):
        values = _cell_vector(
            source,
            n_res_blocks,
            default=None,
            global_to_local=_global_to_local(reservoir),
        )
        if values is not None:
            return values
    return None


def _global_to_local(reservoir: Any) -> np.ndarray | None:
    if hasattr(reservoir, "global_data") and "global_to_local" in reservoir.global_data:
        return np.asarray(reservoir.global_data["global_to_local"])
    if hasattr(reservoir, "discretizer") and hasattr(
        reservoir.discretizer, "global_to_local"
    ):
        return np.asarray(reservoir.discretizer.global_to_local)
    return None


def _cell_vector(
    values: Any,
    n_cells: int,
    default: float | None,
    global_to_local: np.ndarray | None = None,
) -> np.ndarray | None:
    if values is None:
        if default is None:
            return None
        return np.full(n_cells, default, dtype=np.float64)

    arr = np.asarray(values)
    if arr.dtype == object and arr.size == 0:
        if default is None:
            return None
        return np.full(n_cells, default, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n_cells, float(arr), dtype=np.float64)

    flat = np.asarray(arr, dtype=np.float64).reshape(-1, order="F")
    if flat.size == n_cells:
        return flat

    if global_to_local is not None and flat.size >= global_to_local.size:
        out = np.full(n_cells, np.nan if default is None else default, dtype=np.float64)
        for global_idx, local_idx in enumerate(global_to_local.astype(np.int64)):
            if 0 <= local_idx < n_cells:
                out[local_idx] = flat[global_idx]
        return out

    if flat.size > n_cells:
        return flat[:n_cells]

    if default is None:
        return None
    out = np.full(n_cells, default, dtype=np.float64)
    out[: flat.size] = flat
    return out


def _write_dataset(
    group: h5py.Group,
    name: str,
    data: Any,
    dtype: Any | None = None,
) -> None:
    if name in group:
        del group[name]
    if dtype is None:
        group.create_dataset(name, data=data)
    else:
        group.create_dataset(name, data=data, dtype=dtype)


def _main() -> int:
    """
    Run the command-line validation helper.

    :return: Process exit code.
    :rtype: int
    """

    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Utilities for open-DARTS PhysicsNeMo export files."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate an exported PhysicsNeMo HDF5 file."
    )
    validate_parser.add_argument("path", help="Path to exported HDF5 file.")

    args = parser.parse_args()
    if args.command == "validate":
        summary = validate_physicsnemo_hdf5(args.path)
        print(json.dumps(summary, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
