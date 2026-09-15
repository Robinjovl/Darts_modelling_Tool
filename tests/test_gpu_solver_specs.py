"""GPU solver specs on the CPU platform (host-assembled Jacobian).

The ``GPUSolverSpec`` family builds through the open-source registry (``gpu_*``
names, wrapped in ``linsolv_host_adapter``) so a CPU engine can run a GPU
solver and switch between CPU and GPU solvers mid-run. These tests guard:

1. the spec <-> registry contract on every build (names, ``available()``, the
   ``NotImplementedError`` of ``build()`` on a build without the solver);
2. on a CUDA build, an end-to-end run of the small ``models/2ph_do`` flow model
   on platform 'cpu' with the GPU GMRES + block-ILU(0) solver, a mid-run switch
   to the CPU default and back (no Jacobian reallocation), and the direct
   solvers when they are compiled in.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "models" / "2ph_do"

GPU_SPECS = {
    "gpu_gmres_ilu0": "GPUGMRESILU0SolverSpec",
    "gpu_cudss": "CuDSSSolverSpec",
    "gpu_cusolver": "GPUCuSolverSpec",
    "gpu_gmres_cpr_amgx": "AMGXCPRSolverSpec",
    "gpu_bicgstab_cpr_amgx": "GPUBiCGStabCPRSolverSpec",
}


def _jac_ptr(engine):
    return np.asarray(engine.jac_vals).__array_interface__["data"][0]


@pytest.fixture()
def solvers():
    solvers = pytest.importorskip("darts.linear_solvers")
    if not getattr(solvers, "_have_compiled_solvers", False):
        pytest.skip("open-source solver registry not available in this build")
    return solvers


def test_gpu_specs_name_registry_solvers(solvers):
    from darts.linear_solvers import specs

    registered = set(solvers.registered_solvers())
    for name, cls_name in GPU_SPECS.items():
        cls = getattr(specs, cls_name)
        assert cls.registry_name == name
        assert cls.linear_type_name  # platform 'gpu' still uses the enum
        assert cls.available() == (name in registered)
        spec = cls()
        config = spec._make_config()
        assert isinstance(config, solvers.GPUSolverConfig)
        assert config.device_num == 0
        if not cls.available():
            with pytest.raises(NotImplementedError):
                spec.build(2)
        else:
            assert spec.build(2) is not None


def test_gpu_spec_config_fields(solvers):
    from darts.linear_solvers import GPUGMRESILU0SolverSpec

    spec = GPUGMRESILU0SolverSpec(
        tolerance=1e-7,
        max_iterations=120,
        restart=30,
        ilu_single_precision=True,
        device_num=1,
    )
    config = spec._make_config()
    assert config.tolerance == 1e-7
    assert config.max_iterations == 120
    assert config.restart == 30
    assert config.ilu_single_precision is True
    assert config.device_num == 1
    assert spec.to_dict()["restart"] == 30


def test_adaptive_accepts_available_gpu_candidates_only(solvers):
    from darts.linear_solvers import (
        AdaptiveSolverSpec,
        CuDSSSolverSpec,
        GMRESSolverSpec,
        GPUGMRESILU0SolverSpec,
    )

    for cls in (GPUGMRESILU0SolverSpec, CuDSSSolverSpec):
        if cls.available():
            AdaptiveSolverSpec(candidates=[GMRESSolverSpec(), cls()])
        else:
            with pytest.raises(TypeError):
                AdaptiveSolverSpec(candidates=[GMRESSolverSpec(), cls()])


@pytest.fixture()
def model(solvers):
    cwd = os.getcwd()
    os.chdir(MODEL_DIR)
    sys.path.insert(0, str(MODEL_DIR))
    try:
        from model import Model

        from darts.engines import redirect_darts_output

        redirect_darts_output("test_gpu_solver_specs.log")
        yield Model()
    finally:
        sys.path.remove(str(MODEL_DIR))
        os.chdir(cwd)


def _run_and_check(m, days):
    m.run(days, verbose=False)
    assert m.nonlinear_solver.stats.n_timesteps_total > 0
    assert m.nonlinear_solver.stats.n_newton_total > 0


def test_gpu_solver_on_cpu_platform_and_switching(model):
    from darts.linear_solvers import (
        CPRSolverSpec,
        GMRESSolverSpec,
        GPUGMRESILU0SolverSpec,
        LinearSolver,
    )

    if not GPUGMRESILU0SolverSpec.available():
        pytest.skip("no CUDA build: gpu_gmres_ilu0 is not registered")
    m = model
    assert getattr(m, "platform", "cpu") == "cpu"
    m.linear_solver = LinearSolver(model=m)
    m.linear_solver.spec = GPUGMRESILU0SolverSpec(
        tolerance=1e-6, max_iterations=200, restart=40
    )
    m.init()
    m.set_output(output_folder="test_gpu_solver_specs_out")
    _run_and_check(m, 10)
    ptr0 = _jac_ptr(m.physics.engine)
    assert ptr0 != 0
    li_gpu = m.nonlinear_solver.stats.n_linear_total
    assert li_gpu > 0

    # GPU -> CPU default: rebuilt and re-injected on the same Jacobian.
    assert (
        m.linear_solver.update_solver(spec=GMRESSolverSpec(prec=CPRSolverSpec()))
        == "rebuilt"
    )
    assert _jac_ptr(m.physics.engine) == ptr0
    _run_and_check(m, 10)

    # CPU -> GPU again, and a hot field update on the GPU spec.
    assert (
        m.linear_solver.update_solver(spec=GPUGMRESILU0SolverSpec(tolerance=1e-6))
        == "rebuilt"
    )
    assert _jac_ptr(m.physics.engine) == ptr0
    _run_and_check(m, 10)
    m.linear_solver.update_solver(tolerance=1e-7)
    assert m.params.tolerance_linear == 1e-7
    _run_and_check(m, 5)


@pytest.mark.parametrize("cls_name", ["CuDSSSolverSpec", "GPUCuSolverSpec"])
def test_gpu_direct_solvers_on_cpu_platform(model, cls_name):
    from darts.linear_solvers import LinearSolver, specs

    cls = getattr(specs, cls_name)
    if not cls.available():
        pytest.skip(f"{cls_name} is not registered in this build")
    m = model
    m.linear_solver = LinearSolver(model=m)
    m.linear_solver.spec = cls()
    m.init()
    m.set_output(output_folder="test_gpu_solver_specs_out")
    _run_and_check(m, 10)
    # a direct solve reports one linear iteration per Newton iteration
    assert (
        m.nonlinear_solver.stats.n_linear_total
        == m.nonlinear_solver.stats.n_newton_total
    )
