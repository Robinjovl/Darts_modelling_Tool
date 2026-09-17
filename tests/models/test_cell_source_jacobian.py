"""FD consistency tests for the :class:`CellSource` diagonal-block Jacobian (M3).

:class:`~darts.models.conditions.CellSource` accepts an optional ``d_rates``
(constant array or ``f(t, states)`` callable) and then writes an analytic
diagonal-block Jacobian: ``jac_diag(b) -= d_rates[k] * dt`` next to its residual
contribution ``rhs[b * n_vars + c] -= q_c * dt``. Those two writes must be
consistent -- the block has to be the derivative of the residual the SAME item
wrote -- because a Jacobian-writing ``CellSource`` is what carries the
Python-side pseudo-wells of ``models/2ph_constant_k`` (a wrong block there does
not fail loudly, it just degrades or silently redirects Newton).

Everything runs against pure-numpy stubs, in the style of ``test_conditions.py``:
a hand-built 4-block CSR (3x3 dense blocks) wrapped in a real ``BlockCSRView``,
a stub model, and an ``AssemblyContext`` over plain arrays -- no reservoir, no
assembly.

Covered:

* a smooth, variable-coupling rate law with a ``d_rates`` CALLABLE: every column
  of every diagonal block matches a central-difference derivative of the
  residual the item scattered, and nothing outside the addressed diagonal blocks
  is touched;
* the composition-splitting pseudo-well law of ``models/2ph_constant_k``
  (``q_c = -mass_rate * z_c / Mw_c`` with ``z_last = 1 - sum(z)``), whose
  ``d_rates`` is a CONSTANT array: same FD check, plus the expected sparsity and
  the residual sign convention;
* a negative control: the same FD comparison against a deliberately wrong
  ``d_rates`` must fail, so the check above cannot pass vacuously.
"""

import numpy as np
import pytest

# BlockCSRView/CellSource live next to code importing darts.engines; skip the
# module on a checkout without the compiled extension.
pytest.importorskip("darts.engines")

from darts.models.conditions import (  # noqa: E402
    AssemblyContext,
    BlockCSRView,
    CellSource,
)

N_VARS = 3
BLOCK_SIZE = N_VARS * N_VARS
N_BLOCKS = 4
N_CSR_POSITIONS = 10


class FakeCSREngine:
    """4-block CSR with 3x3 dense blocks and pattern::

    row 0: (0,0) (0,1)
    row 1: (1,0) (1,1) (1,2)
    row 2:       (2,1) (2,2) (2,3)
    row 3:             (3,2) (3,3)
    """

    def __init__(self):
        self.jac_rows = np.array([0, 2, 5, 8, 10], dtype=np.int64)
        self.jac_cols = np.array([0, 1, 0, 1, 2, 1, 2, 3, 2, 3], dtype=np.int64)
        self.jac_diags = np.array([0, 3, 6, 9], dtype=np.int64)
        self.jac_vals = np.zeros(N_CSR_POSITIONS * BLOCK_SIZE)
        self.opt_history_matching = False


class _StubMesh:
    def __init__(self, n_blocks):
        self.n_blocks = n_blocks
        self.n_res_blocks = n_blocks


class _StubReservoir:
    def __init__(self, n_blocks):
        self.mesh = _StubMesh(n_blocks)
        self.wells = []


class _StubPhysics:
    def __init__(self, engine):
        self.n_vars = N_VARS
        self.engine = engine


class _StubModel:
    def __init__(self, engine):
        self.physics = _StubPhysics(engine)
        self.reservoir = _StubReservoir(N_BLOCKS)
        self.platform = "cpu"


def _apply(item, model, engine, X, dt, t=1.0):
    """Apply ``item`` at state ``X`` on a zeroed Jacobian; return (rhs, jac_vals)."""
    engine.jac_vals[:] = 0.0
    ctx = AssemblyContext(
        rhs=np.zeros(N_BLOCKS * N_VARS),
        jac=BlockCSRView(engine, N_VARS),
        X=np.asarray(X, dtype=float),
        Xn=np.zeros(N_BLOCKS * N_VARS),
        dt=dt,
        t=t,
        iteration=0,
        n_vars=N_VARS,
        n_res_blocks=N_BLOCKS,
    )
    item.apply(ctx)
    return ctx.rhs.copy(), engine.jac_vals.copy()


def _diagonal_block(jac_vals, engine, block):
    start = int(engine.jac_diags[block]) * BLOCK_SIZE
    return jac_vals[start : start + BLOCK_SIZE].reshape(N_VARS, N_VARS)


def _fd_jacobian_columns(item, model, engine, X0, dt, cell, h=1e-6):
    """Central-difference d(residual rows of ``cell``)/d(state of ``cell``)."""
    rows = slice(cell * N_VARS, (cell + 1) * N_VARS)
    columns = np.zeros((N_VARS, N_VARS))
    for v in range(N_VARS):
        x_plus = np.array(X0, dtype=float)
        x_minus = np.array(X0, dtype=float)
        x_plus[cell * N_VARS + v] += h
        x_minus[cell * N_VARS + v] -= h
        rhs_plus, _ = _apply(item, model, engine, x_plus, dt)
        rhs_minus, _ = _apply(item, model, engine, x_minus, dt)
        columns[:, v] = (rhs_plus[rows] - rhs_minus[rows]) / (2.0 * h)
    return columns


class _CoupledRateSource(CellSource):
    """Smooth, variable-coupling rate law with an analytic ``d_rates`` callable.

    ``q_c = alpha_c * x0**2 + beta_c * sin(x1) + gamma_c * x0 * x2``, so every
    column of the derivative block is populated and state-dependent.
    """

    def __init__(self, cells, alpha, beta, gamma):
        self.alpha = np.asarray(alpha, dtype=float)
        self.beta = np.asarray(beta, dtype=float)
        self.gamma = np.asarray(gamma, dtype=float)
        super().__init__(
            cells=cells, rates=self.state_rates, d_rates=self.state_d_rates
        )

    def state_rates(self, t, states):
        x0 = states[:, 0:1]
        x1 = states[:, 1:2]
        x2 = states[:, 2:3]
        return self.alpha * x0**2 + self.beta * np.sin(x1) + self.gamma * x0 * x2

    def state_d_rates(self, t, states):
        x0 = states[:, 0:1]
        x1 = states[:, 1:2]
        x2 = states[:, 2:3]
        d_rates = np.zeros((len(self.cells), N_VARS, N_VARS))
        d_rates[:, :, 0] = 2.0 * self.alpha * x0 + self.gamma * x2
        d_rates[:, :, 1] = self.beta * np.cos(x1)
        d_rates[:, :, 2] = self.gamma * x0
        return d_rates


class _CompositionSplitSource(CellSource):
    """The rate-controlled pseudo-well law of ``models/2ph_constant_k``.

    A fixed MASS rate split over the components by the cell's own composition:
    ``q_c = -mass_rate * z_c / Mw_c`` with ``z_c = x_{c+1}`` and
    ``z_last = 1 - sum(x_1..x_{n_vars-1})``. The derivative is constant, which is
    exactly why it is worth checking against the residual it accompanies.
    """

    def __init__(self, cells, mass_rate, Mw):
        self.mass_rate = float(mass_rate)
        self.Mw = np.asarray(Mw, dtype=float)
        n_vars = self.Mw.size
        d_rates = np.zeros((len(cells), n_vars, n_vars))
        for c in range(n_vars - 1):
            d_rates[:, c, c + 1] = -self.mass_rate / self.Mw[c]
            d_rates[:, n_vars - 1, c + 1] = self.mass_rate / self.Mw[n_vars - 1]
        super().__init__(cells=cells, rates=self.composition_rates, d_rates=d_rates)

    def composition_rates(self, t, states):
        z = np.empty_like(states)
        z[:, :-1] = states[:, 1:]
        z[:, -1] = 1.0 - states[:, 1:].sum(axis=1)
        return -(self.mass_rate * z / self.Mw)


def _state_vector():
    """A state with distinct, O(1) entries in every block."""
    # (pressure, z1, z2) per block
    blocks = [
        (120.0, 0.30, 0.25),
        (135.0, 0.40, 0.15),
        (150.0, 0.20, 0.35),
        (170.0, 0.10, 0.45),
    ]
    return np.array(blocks, dtype=float).ravel()


def test_callable_d_rates_matches_finite_difference_of_the_residual():
    engine = FakeCSREngine()
    model = _StubModel(engine)
    cells = [1, 2]
    item = _CoupledRateSource(
        cells,
        alpha=[[0.7, -0.4, 0.2], [0.1, 0.5, -0.3]],
        beta=[[1.3, 0.8, -0.6], [-0.9, 0.4, 1.1]],
        gamma=[[0.05, -0.02, 0.03], [-0.04, 0.06, 0.01]],
    )
    assert item.provides_jacobian is True
    item.bind(model)

    dt = 0.37
    X0 = _state_vector()
    _, jac_vals = _apply(item, model, engine, X0, dt)

    for cell in cells:
        analytic = _diagonal_block(jac_vals, engine, cell)
        finite_difference = _fd_jacobian_columns(item, model, engine, X0, dt, cell)
        assert analytic == pytest.approx(finite_difference, rel=1e-6, abs=1e-9)


def test_jacobian_contribution_stays_in_the_addressed_diagonal_blocks():
    engine = FakeCSREngine()
    model = _StubModel(engine)
    item = _CoupledRateSource(
        [1], alpha=[[0.7, -0.4, 0.2]], beta=[[1.3, 0.8, -0.6]], gamma=[[0.05, 0.0, 0.0]]
    )
    item.bind(model)
    _, jac_vals = _apply(item, model, engine, _state_vector(), dt=0.37)

    touched = np.ones(N_CSR_POSITIONS, dtype=bool)
    touched[int(engine.jac_diags[1])] = False
    assert np.all(jac_vals.reshape(N_CSR_POSITIONS, BLOCK_SIZE)[touched] == 0.0)


def test_composition_split_source_jacobian_matches_finite_difference():
    engine = FakeCSREngine()
    model = _StubModel(engine)
    cells = [0, 3]
    Mw = [44.01, 16.04, 58.12]
    item = _CompositionSplitSource(cells, mass_rate=10.0, Mw=Mw)
    item.bind(model)

    dt = 0.75
    X0 = _state_vector()
    rhs, jac_vals = _apply(item, model, engine, X0, dt)

    for cell in cells:
        analytic = _diagonal_block(jac_vals, engine, cell)
        finite_difference = _fd_jacobian_columns(item, model, engine, X0, dt, cell)
        assert analytic == pytest.approx(finite_difference, rel=1e-6, abs=1e-9)

        # only d q_c / d z_c (entry (c, c+1)) and the last equation's
        # d q_last / d z_v (row n_vars - 1) are non-zero
        expected_nonzero = np.zeros((N_VARS, N_VARS), dtype=bool)
        for c in range(N_VARS - 1):
            expected_nonzero[c, c + 1] = True
            expected_nonzero[N_VARS - 1, c + 1] = True
        assert np.array_equal(analytic != 0.0, expected_nonzero)

    # residual sign convention: rhs -= q * dt, and the pseudo-well PRODUCES for a
    # positive mass rate (q = -mass_rate * z / Mw is out of the block)
    z = np.array([X0[1], X0[2], 1.0 - X0[1] - X0[2]])
    expected_rhs = 10.0 * z / np.asarray(Mw) * dt
    assert rhs[0:N_VARS] == pytest.approx(expected_rhs)


def test_finite_difference_check_rejects_a_wrong_d_rates():
    """The FD comparison above must not be able to pass vacuously."""
    engine = FakeCSREngine()
    model = _StubModel(engine)
    item = _CompositionSplitSource([2], mass_rate=10.0, Mw=[44.01, 16.04, 58.12])
    item.d_rates = 2.0 * np.asarray(item.d_rates)  # deliberately wrong
    item.bind(model)

    dt = 0.75
    X0 = _state_vector()
    _, jac_vals = _apply(item, model, engine, X0, dt)
    analytic = _diagonal_block(jac_vals, engine, 2)
    finite_difference = _fd_jacobian_columns(item, model, engine, X0, dt, 2)
    assert analytic != pytest.approx(finite_difference, rel=1e-6, abs=1e-9)


# ---------------------------------------------------------------------------
# Duplicate cells (review finding R3): a cell may legitimately appear twice in
# `cells`, and BOTH scatters must accumulate. Advanced-index subtraction on the
# residual dropped the first contribution (the last write won) while the
# np.add.at Jacobian scatter kept both -- so the advertised analytic Jacobian
# was not the derivative of the assembled residual.
# ---------------------------------------------------------------------------


def test_duplicate_cells_accumulate_in_the_residual():
    """The review's reproduction: cells [0, 0] with constant rates must yield
    the ADDITIVE residual, not the last row alone."""
    engine = FakeCSREngine()
    model = _StubModel(engine)
    item = CellSource(cells=[0, 0], rates=[[1.0, 2.0, 5.0], [3.0, 4.0, 7.0]])
    item.bind(model)

    rhs, _ = _apply(item, model, engine, _state_vector(), dt=1.0)
    assert rhs[0:N_VARS] == pytest.approx([-4.0, -6.0, -12.0])
    assert np.all(rhs[N_VARS:] == 0.0)


def test_duplicate_cells_with_constant_d_rates_are_consistent():
    """Constant `d_rates` on a duplicated cell: the accumulated diagonal block
    must be the derivative of the accumulated residual (FD across the
    duplicate), and the residual must carry BOTH contributions."""
    engine = FakeCSREngine()
    model = _StubModel(engine)
    Mw = [44.01, 16.04, 58.12]
    single = _CompositionSplitSource([2], mass_rate=10.0, Mw=Mw)
    single.bind(model)
    item = _CompositionSplitSource([2, 2], mass_rate=10.0, Mw=Mw)
    item.bind(model)

    dt = 0.75
    X0 = _state_vector()
    rhs_single, _ = _apply(single, model, engine, X0, dt)
    rhs, jac_vals = _apply(item, model, engine, X0, dt)

    # duplicating the cell exactly doubles the residual contribution
    assert rhs == pytest.approx(2.0 * rhs_single)

    analytic = _diagonal_block(jac_vals, engine, 2)
    finite_difference = _fd_jacobian_columns(item, model, engine, X0, dt, 2)
    assert analytic == pytest.approx(finite_difference, rel=1e-6, abs=1e-9)


def test_duplicate_cells_with_a_state_dependent_callable_are_consistent():
    """A rate CALLABLE with an analytic `d_rates` callable on a duplicated
    cell: residual accumulates both rows, and the accumulated Jacobian matches
    a central finite difference of the accumulated residual."""
    engine = FakeCSREngine()
    model = _StubModel(engine)
    item = _CoupledRateSource(
        [1, 1],
        alpha=[[0.7, -0.4, 0.2], [0.1, 0.5, -0.3]],
        beta=[[1.3, 0.8, -0.6], [-0.9, 0.4, 1.1]],
        gamma=[[0.05, -0.02, 0.03], [-0.04, 0.06, 0.01]],
    )
    item.bind(model)

    dt = 0.37
    X0 = _state_vector()
    rhs, jac_vals = _apply(item, model, engine, X0, dt)

    # the residual carries the SUM of the two rate rows of the same cell
    states = X0[N_VARS : 2 * N_VARS][None, :].repeat(2, axis=0)
    both = item.state_rates(0.0, states)
    assert rhs[N_VARS : 2 * N_VARS] == pytest.approx(-dt * both.sum(axis=0))

    analytic = _diagonal_block(jac_vals, engine, 1)
    finite_difference = _fd_jacobian_columns(item, model, engine, X0, dt, 1)
    # rel 5e-6: the doubled trig terms carry twice the FD truncation error
    assert analytic == pytest.approx(finite_difference, rel=5e-6, abs=1e-9)
