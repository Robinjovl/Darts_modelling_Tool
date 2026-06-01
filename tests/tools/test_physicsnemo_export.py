from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np


def _load_export_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "darts" / "tools" / "physicsnemo_export.py"
    spec = importlib.util.spec_from_file_location("physicsnemo_export", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Mesh:
    n_res_blocks = 3
    block_m = [0, 1, 1, 2, 0, 3]
    block_p = [1, 0, 2, 1, 3, 0]
    tran = [10.0, 10.0, 20.0, 20.0, 99.0, 99.0]
    tranD = [1.0, 1.0, 2.0, 2.0, 9.0, 9.0]
    grav_coef = [0.1, -0.1, 0.2, -0.2, 9.0, 9.0]
    poro = [0.2, 0.25, 0.3]
    volume = [100.0, 110.0, 120.0]
    depth = [1000.0, 1001.0, 1002.0]
    op_num = [0, 0, 1]
    rock_cond = [1.0, 1.0, 1.0]
    heat_capacity = [2.0, 2.0, 2.0]


class _Reservoir:
    mesh = _Mesh()
    permx = [100.0, 200.0, 300.0]
    permy = [80.0, 90.0, 100.0]
    permz = [10.0, 20.0, 30.0]
    wells = []


class _Output:
    sol_filepath = ""

    def _get_output_cell_centers(self):
        return np.array([[0.0, 0.0, 1000.0], [1.0, 0.0, 1001.0], [2.0, 0.0, 1002.0]])


class _Engine:
    t = 5.0
    CFL_max = 0.1
    X = [100.0, 0.1, 110.0, 0.2, 120.0, 0.3]


class _Physics:
    engine = _Engine()
    n_vars = 2
    vars = ["pressure", "sat"]


class _Model:
    reservoir = _Reservoir()
    output = _Output()
    physics = _Physics()
    output_folder = "."


def test_export_physicsnemo_hdf5_writes_valid_schema(tmp_path):
    exporter = _load_export_module()
    output_path = tmp_path / "case_001.h5"

    exporter.export_physicsnemo_hdf5(
        _Model(),
        output_path=output_path,
        reservoir_h5_path=None,
        case_name="case_001",
    )
    summary = exporter.validate_physicsnemo_hdf5(output_path)

    assert summary["case_name"] == "case_001"
    assert summary["n_cells"] == 3
    assert summary["n_edges"] == 4
    assert summary["variable_names"] == ["pressure", "sat"]

    with h5py.File(output_path, "r") as h5:
        np.testing.assert_allclose(h5["static/PORV"][:], [20.0, 27.5, 36.0])
        assert h5["dynamic/X"].shape == (1, 3, 2)
        assert h5["static/edges/cell_m"][:].tolist() == [0, 1, 1, 2]
