"""Bake SPE11b FluidFlower geometry to flat per-cell JSON arrays.

Runs once to materialise the heterogeneous reservoir grid (porosity,
permeabilities, depth, heat-capacity, conductivity, op_num) for the
schema-first JSON model in :file:`models/SPE11b_json/SPE11b.json`.

The bake-out reuses the existing :class:`FluidFlowerStruct` reservoir builder
from :mod:`models.SPE11b.fluidflower_str_b` so the result is bit-for-bit
identical to the Python baseline.  We also compute the (i, k) cell indices for
the two injection well centers so the JSON config can name them directly.

Usage::

    python bake_arrays.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from darts.engines import timer_node


def _make_timer() -> timer_node:
    """Build a timer that already carries the ``initialization`` sub-node.

    :class:`ReservoirBase.__init__` reads ``timer.node['initialization']`` on
    construction, so the bake script must seed that entry; the live model
    normally does it inside :meth:`DartsModel.__init__`.
    """
    timer = timer_node()
    timer.start()
    timer.node["initialization"] = timer_node()
    return timer

REPO_ROOT = Path(__file__).resolve().parents[2]
SPE11B_DIR = REPO_ROOT / "models" / "SPE11b"
OUT_DIR = Path(__file__).resolve().parent / "arrays"

# Resolve the existing SPE11b Python model on the import path so we can reuse
# its FluidFlowerStruct (and its custom geometry/shape modules) directly.
sys.path.insert(0, str(SPE11B_DIR))

from fluidflower_str_b import FluidFlowerStruct  # noqa: E402
from model_b import layer_props, layers_to_regions  # noqa: E402

# Reporting grid resolution from the SPE11b baseline (main.py).
NX = 840 // 10
NY = 1
NZ = 120 // 10

WELL_CENTERS = {
    "I1": [2700.0, 0.0, 300.0],
    "I2": [5100.0, 0.0, 700.0],
}


def _to_list(arr) -> list:
    """Convert a numpy array (any dtype, any shape) to a flat Python list for JSON.

    For 3D arrays we flatten in Fortran order to match the KJI loop
    ``global_idx = k*nx*ny + j*nx + i`` used by :class:`FluidFlowerStruct`,
    keeping the per-cell ordering consistent with the rest of the
    :class:`StructReservoir` global_data arrays.
    """
    a = np.asarray(arr)
    if a.ndim > 1:
        a = a.ravel(order="F")
    if np.issubdtype(a.dtype, np.integer):
        return [int(x) for x in a.tolist()]
    return [float(x) for x in a.tolist()]


def main() -> None:
    """Build the reservoir, dump arrays + initial-conditions table, print well indices."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Minimal model_specs subset that FluidFlowerStruct actually reads.
    model_specs = {
        "nx": NX,
        "nz": NZ,
        "RHS": False,  # standard mass-rate path (matches the JSON minimal example)
    }

    timer = _make_timer()
    reservoir = FluidFlowerStruct(
        timer=timer,
        layer_properties=layer_props,
        layers_to_regions=layers_to_regions,
        model_specs=model_specs,
        well_centers=WELL_CENTERS,
    )

    # Per-cell arrays — FluidFlowerStruct stores them in global_data after
    # super().__init__() and they are flat KJI-ordered numpy arrays of length
    # nx*ny*nz.
    gd = reservoir.global_data
    nx, ny, nz = reservoir.nx, reservoir.ny, reservoir.nz
    n_total = nx * ny * nz

    # Sanity: every per-cell array must have the expected element count.
    for key in ("poro", "permx", "permy", "permz", "depth", "hcap", "rcond", "op_num"):
        arr = np.asarray(gd[key])
        if arr.size != n_total:
            raise RuntimeError(
                f"global_data['{key}'] has size {arr.size}, expected {n_total}"
            )

    arrays = {
        "poro": gd["poro"],
        "permx": gd["permx"],
        "permy": gd["permy"],
        "permz": gd["permz"],
        "depth": gd["depth"],
        "hcap": gd["hcap"],
        "rcond": gd["rcond"],
        "op_num": gd["op_num"],
    }

    for name, arr in arrays.items():
        path = OUT_DIR / f"{name}.json"
        flat = _to_list(arr)
        with path.open("w", encoding="utf-8") as f:
            json.dump(flat, f)
        print(
            f"wrote {path.relative_to(REPO_ROOT)}  "
            f"(n={len(flat)}, dtype={np.asarray(arr).dtype})"
        )

    # Initial-conditions depth table (two-point hydrostatic + geothermal).
    # Matches model_b.set_initial_conditions: pres_in = 212 bar, hydrostatic
    # gradient 0.09775 bar/m applied to depth.
    #
    # Temperature follows the SPE11b geothermal: T(z_spe) = 70°C - 0.025·z_spe
    # in *SPE11b* coordinates where z_spe = 0 at the bottom and z_spe = 1200 m
    # at the top.  Inside the DARTS reservoir, ``depth`` is the conventional
    # "distance below the top" axis, so ``z_spe = 1200 - depth`` and
    # T(depth) = 273.15 + 70 - 0.025·(1200 - depth) = 313.15 + 0.025·depth.
    depth = np.asarray(gd["depth"], dtype=float)
    depth_top = float(depth.min())  # shallowest cell
    depth_bot = float(depth.max())  # deepest cell
    pres_top = 212.0
    pres_bot = pres_top + depth_bot * 0.09775
    temp_top = 313.15 + 0.025 * depth_top
    temp_bot = 313.15 + 0.025 * depth_bot
    h2o = 1.0 - 1e-10  # zero = 1e-10 in the Python baseline

    initial_conditions = {
        "input_depth": [depth_top, depth_bot],
        "input_distribution": {
            "pressure": [pres_top, pres_bot],
            "H2O": [h2o, h2o],
            "temperature": [temp_top, temp_bot],
        },
    }
    ic_path = OUT_DIR / "initial_conditions.json"
    with ic_path.open("w", encoding="utf-8") as f:
        json.dump(initial_conditions, f, indent=2)
    print(f"wrote {ic_path.relative_to(REPO_ROOT)}")

    # Cell sizes in the structured grid — extracted so the JSON config can use
    # scalar dx/dy/dz instead of arrays.  FluidFlowerStruct sets dx, dy, dz on
    # the discretizer's global_data dict.
    dx_val = np.asarray(gd["dx"]).ravel()[0]
    dy_val = np.asarray(gd["dy"]).ravel()[0]
    dz_val = np.asarray(gd["dz"]).ravel()[0]
    print(f"dx, dy, dz = {float(dx_val)}, {float(dy_val)}, {float(dz_val)}")
    print(f"nx, ny, nz = {nx}, {ny}, {nz}")

    # Resolve the two well-center coordinates to (i, k) cell indices for the
    # JSON wells block.  Matches the convention in FluidFlowerStruct.set_wells:
    # i = int(x // dx), k = int(z // dz), j = 1 (the only y-layer).
    dx = 8400.0 / nx
    dz = 1200.0 / nz
    for name, center in WELL_CENTERS.items():
        x, _, z = center
        i = min(max(int(x // dx), 0), nx - 1)
        k = min(max(int(z // dz), 0), nz - 1)
        # JSON config uses 1-based indices.
        print(f"well {name}: ijk = [{i + 1}, 1, {k + 1}]  (x={x}, z={z})")

    # Boundary-volume multiplier used by the SPE11b Python baseline:
    # ``5e9 * (1200 / nz)`` applied to yz_minus and yz_plus.
    print(f"boundary_volumes.yz_minus/plus = {5e9 * (1200.0 / nz)}")


if __name__ == "__main__":
    os.chdir(SPE11B_DIR)  # so relative `shapes`/`fluidflower` imports resolve
    main()
