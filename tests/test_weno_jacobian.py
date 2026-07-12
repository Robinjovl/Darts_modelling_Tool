import importlib.util
from pathlib import Path

import numpy as np


def load_2ph_model():
    model_path = Path(__file__).parents[1] / "models" / "2ph_comp" / "model.py"
    spec = importlib.util.spec_from_file_location("weno_2ph_comp_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


def jacobian_column(engine, block_column: int, variable: int) -> np.ndarray:
    n_vars = engine.get_n_vars()
    rows = np.asarray(engine.jac_rows)
    columns = np.asarray(engine.jac_cols)
    values = np.asarray(engine.jac_vals)
    result = np.zeros(len(engine.RHS))
    for block_row in range(rows.size - 1):
        row_columns = columns[rows[block_row] : rows[block_row + 1]]
        matches = np.flatnonzero(row_columns == block_column)
        if matches.size:
            slot = rows[block_row] + matches[0]
            block = values[
                slot * n_vars * n_vars : (slot + 1) * n_vars * n_vars
            ].reshape(n_vars, n_vars)
            result[block_row * n_vars : (block_row + 1) * n_vars] = block[:, variable]
    return result


def test_weno_jacobian_matches_finite_difference():
    model = load_2ph_model()(transport_scheme="weno2", nx=12, runtime=0.001)
    model.init()

    engine = model.physics.engine
    n_vars = engine.get_n_vars()
    n_reservoir = model.reservoir.mesh.n_res_blocks
    state = np.asarray(engine.X)
    dt = 0.001
    source_cell = 5
    variable = 1
    epsilon = 1.0e-7
    source_index = source_cell * n_vars + variable
    for pressure_slope, remote_row in ((0.7, source_cell - 2), (-0.7, source_cell + 2)):
        for cell in range(n_reservoir):
            state[cell * n_vars] = 50.3 + pressure_slope * cell
            state[cell * n_vars + 1] = 0.08 + 0.0037 * cell
            state[cell * n_vars + 2] = 0.24 - 0.0019 * cell

        engine.assemble_linear_system(dt)
        analytic = jacobian_column(engine, source_cell, variable)
        original = state[source_index]
        state[source_index] = original + epsilon
        engine.assemble_linear_system(dt)
        residual_plus = np.array(engine.RHS, copy=True)
        state[source_index] = original - epsilon
        engine.assemble_linear_system(dt)
        residual_minus = np.array(engine.RHS, copy=True)
        state[source_index] = original

        finite_difference = (residual_plus - residual_minus) / (2.0 * epsilon)
        np.testing.assert_allclose(
            analytic,
            finite_difference,
            rtol=2.0e-4,
            atol=2.0e-6,
        )

        # The selected row is not a physical neighbour.  A nonzero entry proves
        # that the test exercised an expanded WENO dependency rather than only SPU.
        remote_slice = analytic[remote_row * n_vars : (remote_row + 1) * n_vars]
        assert np.linalg.norm(remote_slice, ord=np.inf) > 1.0e-8
