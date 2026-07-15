from pathlib import Path

import numpy as np

from darts.engines import sim_params, timer_node
from darts.reservoirs.cpg_reservoir import CPG_Reservoir
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.reservoirs.unstruct_reservoir import UnstructReservoir
from darts.tools.gen_cpg_grid import gen_cpg_grid


def initialization_timer():
    timer = timer_node()
    timer.node["initialization"] = timer_node()
    return timer


def test_structured_weno_static_data_and_expanded_graph():
    reservoir = StructReservoir(
        initialization_timer(),
        nx=3,
        ny=3,
        nz=3,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        permx=100.0,
        permy=100.0,
        permz=100.0,
        poro=0.2,
    )
    reservoir.init_reservoir()

    params = sim_params()
    params.transport_scheme = sim_params.weno2
    reservoir.prepare_weno(params)

    mesh = reservoir.mesh
    assert mesh.weno_enabled
    assert not mesh.weno_finalized
    central_cell = 13
    candidate_offsets = np.asarray(mesh.weno_cell_candidate_offset)
    assert candidate_offsets[central_cell + 1] - candidate_offsets[central_cell] == 8

    reservoir.init_wells()
    assert mesh.weno_finalized
    assert np.all(np.asarray(mesh.weno_cell_status) == 1)
    spu_nonzeros = 2 * len(reservoir.cell_m) + mesh.n_blocks
    assert mesh.weno_jacobian_row_offset[-1] > spu_nonzeros


def test_unstructured_tetrahedra_weno_mapping():
    mesh_file = (
        Path(__file__).parents[1]
        / "discretizer"
        / "tests"
        / "compare_discretizers"
        / "meshes"
        / "unit_tetra.msh"
    )
    reservoir = UnstructReservoir(
        initialization_timer(),
        mesh_file=str(mesh_file),
        permx=100.0,
        permy=100.0,
        permz=100.0,
        poro=0.2,
        frac_aper=0.0,
    )
    reservoir.physical_tags["matrix"] = [99991]
    reservoir.physical_tags["boundary"] = [991, 992, 993, 994, 995, 996]
    reservoir.init_reservoir()

    params = sim_params()
    params.transport_scheme = sim_params.weno2
    reservoir.prepare_weno(params)
    assert reservoir.mesh.weno_enabled
    assert np.all(np.asarray(reservoir.mesh.weno_cell_status) == 1)

    reservoir.init_wells()
    assert reservoir.mesh.weno_finalized


def test_corner_point_weno_geometry(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    arrays = gen_cpg_grid(
        nx=3,
        ny=3,
        nz=3,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        permx=100.0,
        permy=100.0,
        permz=100.0,
        poro=0.2,
    )
    reservoir = CPG_Reservoir(initialization_timer(), arrays=arrays)
    reservoir.init_reservoir()

    params = sim_params()
    params.transport_scheme = sim_params.weno2
    reservoir.prepare_weno(params)
    assert reservoir.mesh.weno_enabled
    assert np.all(np.asarray(reservoir.mesh.weno_cell_status) == 1)

    reservoir.init_wells()
    assert reservoir.mesh.weno_finalized
