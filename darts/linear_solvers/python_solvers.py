"""Python-resident linear solvers for the open-DARTS unified solver framework.

PETSc (via :mod:`petsc4py`) and Pardiso (via :mod:`pypardiso` / Intel MKL) run
inside the Python process and cannot be injected into the C++ engine the way the
engine-resident solvers (``mgr``, ``superlu``, HYPRE, the GPU solvers) are. They
are wrapped here as stateful :class:`PythonLinearSolver` objects so they still
fit the unified :class:`~darts.linear_solvers.specs.LinearSolverSpec` framework.

The key idea (see ``SOLVER_REFACTORING_PLAN.md`` section 13) is the
*setup-once / solve-many* split: the sparsity pattern is fixed for a whole run
(``MATRIX_TYPE_CSR_FIXED_STRUCTURE``), so the expensive structural work --
Pardiso's symbolic analysis and the block->scalar sparsity expansion -- happens
once in :meth:`PythonLinearSolver.setup`, and only the matrix *values* are
refreshed per Newton iteration in :meth:`PythonLinearSolver.solve`.

The :mod:`petsc4py` / :mod:`pypardiso` imports are deferred into ``setup`` /
``solve`` so this module imports cleanly when those optional packages are
absent.
"""

from __future__ import annotations

import numpy as np


def _extract_block_csr(engine):
    """Return the engine Jacobian as a zero-copy block-CSR view.

    :returns: ``(rows, cols, vals, rhs, sol, block_size)`` where ``rows`` /
        ``cols`` / ``vals`` are numpy views onto the ``block_csr_matrix`` host
        arrays, ``rhs`` / ``sol`` are views onto the engine RHS / dX vectors,
        and ``block_size`` is the number of equations per cell.
    """
    rows = np.asarray(engine.jac_rows)
    cols = np.asarray(engine.jac_cols)
    vals = np.asarray(engine.jac_vals)
    rhs = np.array(engine.RHS, copy=False)
    sol = np.array(engine.dX, copy=False)

    n_blocks_nnz = cols.size
    assert n_blocks_nnz > 0, (
        "Jacobian is not exposed to Python -- a Python-resident solver "
        "(PETSc / Pardiso) needs the engine Jacobian; this happens e.g. when a "
        "direct C++ solver is selected instead."
    )
    block_size = int(round((vals.size / n_blocks_nnz) ** 0.5))
    return rows, cols, vals, rhs, sol, block_size


class _BlockToScalarExpander:
    """Block-CSR -> scalar CSR expansion with a precomputed values gather.

    The structure (scalar ``row_ptr`` / ``col_ind``) and the block->scalar value
    permutation are computed once from the fixed block sparsity pattern. Each
    :meth:`refresh` call only gathers the current block values into the scalar
    layout, so the per-Newton-iteration cost is one ``np.take`` -- no
    allocation, no symbolic work.

    This replaces the legacy ``scipy.sparse.bsr_matrix(...).tocsr()`` round-trip
    inside the Newton loop.
    """

    def __init__(self, rows, cols, block_size):
        n_block_rows = rows.size - 1
        b = block_size
        b2 = b * b
        nnz_blocks = cols.size

        self.scalar_rows = np.empty(n_block_rows * b + 1, dtype=np.int32)
        self.scalar_cols = np.empty(nnz_blocks * b2, dtype=np.int32)
        self._gather_idx = np.empty(nnz_blocks * b2, dtype=np.int64)
        self.scalar_vals = np.empty(nnz_blocks * b2, dtype=np.float64)
        self.n = n_block_rows * b
        self.block_size = b

        # Vectorised construction (the previous interpreted triple loop over
        # nnz_blocks * b^2 took minutes-to-hours at million-cell scale).
        rows64 = rows.astype(np.int64)
        cols64 = cols.astype(np.int64)
        d = np.diff(rows64)  # blocks per block-row
        # Blocks per scalar row: each block-row contributes b scalar rows of
        # d[ib] blocks each.
        reps = np.repeat(d, b)
        self.scalar_rows[0] = 0
        self.scalar_rows[1:] = np.cumsum(reps * b)
        # Block index for every (scalar row, k) pair: grouped arange over the
        # row's block range, repeated for each of the b scalar sub-rows.
        csum = np.cumsum(reps)
        total = int(csum[-1]) if csum.size else 0
        grouped = np.arange(total, dtype=np.int64) - np.repeat(csum - reps, reps)
        bi_seq = np.repeat(np.repeat(rows64[:-1], b), reps) + grouped
        r_seq = np.repeat(np.arange(n_block_rows * b, dtype=np.int64) % b, reps)
        # Expand each (block, r) into the b scalar columns of the block.
        bi_full = np.repeat(bi_seq, b)
        r_full = np.repeat(r_seq, b)
        c_full = np.tile(np.arange(b, dtype=np.int64), bi_seq.size)
        self.scalar_cols[:] = (cols64[bi_full] * b + c_full).astype(np.int32)
        self._gather_idx[:] = bi_full * b2 + r_full * b + c_full

    def refresh(self, block_vals):
        """Gather the current block values into ``scalar_vals`` (in place)."""
        np.take(block_vals.ravel(), self._gather_idx, out=self.scalar_vals)
        return self.scalar_vals


# Mirror of ``opendarts::linear_solvers::solve_result`` in linear_solver.hpp --
# the residual-regression rule must be identical in every backend, native or
# Python-resident, or the same solve would be classified differently depending
# on which one ran it. Keep the value in sync with the C++ constant.
RESIDUAL_GROWTH_RTOL = 1.0e-12


def _residual_did_not_regress(final_residual: float, initial_residual: float) -> bool:
    """True when the final residual is no worse than the initial one, allowing
    ``RESIDUAL_GROWTH_RTOL`` of round-off growth. Starting from an exactly zero
    residual the system is already solved, so any positive final residual is
    growth -- the relative slack has nothing to scale."""
    if not initial_residual > 0.0:
        return final_residual <= 0.0
    return final_residual <= initial_residual * (1.0 + RESIDUAL_GROWTH_RTOL)


class PythonLinearSolver:
    """Base class for a Python-resident linear solver.

    Stateful by design: :meth:`setup` is called once (the sparsity pattern is
    fixed for the whole run) and :meth:`solve` every Newton iteration. Use
    :meth:`solve_system` as the entry point -- it performs the lazy one-time
    setup and then the solve.
    """

    def __init__(
        self,
        tolerance: float = 1e-5,
        max_iterations: int = 50,
        print_level: int = 0,
    ):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.print_level = print_level
        self._is_set_up = False

    # -- interface ---------------------------------------------------------
    def setup(self, rows, cols, block_size):
        """Build the persistent backend state for a fixed block-CSR pattern."""
        raise NotImplementedError

    def solve(self, rows, cols, vals, block_size, rhs, sol):
        """Refresh the matrix values and solve ``A x = rhs`` into ``sol``.

        Return status contract (mirrors the C++ ``linsolv_iface::solve``):
        ``(rc, n_iters, residual)``; ``rc`` is ``0`` on success -- including a
        plain max-iterations / rtol miss, which
        the outer inexact-Newton loop rides via its residual gate (bos parity,
        cf. ``linsolv_gmres.cpp``); nonzero on a HARD failure (a non-finite
        solution from breakdown / PC failure / NaN), so the Newton loop cuts
        the timestep instead of applying a garbage update.
        """
        raise NotImplementedError

    # -- entry point -------------------------------------------------------
    def solve_system(self, engine):
        """Solve the engine's current Newton linear system.

        :returns: ``0`` on success, nonzero on a hard linear-solver failure
            (see :meth:`solve`). The engine mirrors this into
            ``NonlinearSolver.status.linear_solver_rc`` so the Newton loop / adaptive
            fallback can react.
        """
        rows, cols, vals, rhs, sol, block_size = _extract_block_csr(engine)
        if not self._is_set_up:
            self.setup(rows, cols, block_size)
            self._is_set_up = True
        return self.solve(rows, cols, vals, block_size, rhs, sol)


class PETScSolver(PythonLinearSolver):
    """PETSc (petsc4py) Krylov solver wrapped for the unified framework.

    The CPR / fixed-stress preconditioners split *within* each cell-block
    (pressure vs transport / displacement), which PETSc's PCFIELDSPLIT requires
    a scalar (AIJ) matrix for -- BAIJ is rejected for sub-block splits. The
    block-CSR Jacobian is therefore expanded to scalar CSR, but the structural
    expansion happens once in :meth:`setup` and per-Newton-iteration only a
    values gather (``np.take``) is needed -- the legacy
    ``scipy.sparse.bsr_matrix(...).tocsr()`` round-trip is eliminated.

    :param variant: ``"cpr"`` (CPR for flow) or ``"fs"`` (fixed-stress
        fieldsplit for poromechanics).
    """

    def __init__(self, variant: str = "cpr", **kwargs):
        super().__init__(**kwargs)
        if variant not in ("cpr", "fs"):
            raise ValueError(f"PETScSolver: unknown variant {variant!r}")
        self.variant = variant
        self._PETSc = None
        self._expander = None

    def _petsc_args(self) -> str:
        """Assemble the PETSc command-line option string for this variant.

        PETSc's composite / fieldsplit preconditioner configuration is driven
        by the option database -- petsc4py exposes no programmatic
        ``addCompositePC``. The fieldsplit *index sets* are still attached
        programmatically (they depend on runtime indices).
        """
        args = ""
        if self.print_level >= 2:
            args += "-ksp_monitor_short "
        if self.print_level >= 5:
            args += "-omp_view "
        args += "-ksp_pc_side right "
        args += f"-ksp_max_it {self.max_iterations} "
        args += f"-ksp_rtol {self.tolerance} "
        if self.variant == "cpr":
            # CPR: composite PC -- fieldsplit (AMG on pressure, jacobi on
            # transport) then ILU on the full system.
            args += (
                "-pc_type composite -pc_composite_type multiplicative "
                "-pc_composite_pcs fieldsplit,ilu "
                "-sub_0_pc_fieldsplit_type schur "
                "-sub_0_pc_fieldsplit_schur_fact_type upper "
                "-sub_0_pc_fieldsplit_schur_precondition selfp "
                "-sub_0_fieldsplit_transport_ksp_type preonly "
                "-sub_0_fieldsplit_transport_pc_type jacobi "
                "-sub_0_fieldsplit_pressure_ksp_type preonly "
                "-sub_0_fieldsplit_pressure_pc_type gamg "
            )
        else:  # fs
            # Fixed-stress: block LDU fieldsplit, AMG on each field.
            args += (
                "-pc_type fieldsplit -pc_fieldsplit_type schur "
                "-pc_fieldsplit_schur_fact_type upper "
                "-pc_fieldsplit_schur_precondition selfp "
                "-fieldsplit_displacement_ksp_type preonly "
                "-fieldsplit_displacement_pc_type gamg "
                "-fieldsplit_pressure_ksp_type preonly "
                "-fieldsplit_pressure_pc_type gamg "
            )
        return args

    def setup(self, rows, cols, block_size):
        """Initialise petsc4py once and precompute the block->scalar gather."""
        import petsc4py

        petsc4py.init(self._petsc_args())
        from petsc4py import PETSc

        self._PETSc = PETSc
        self._expander = _BlockToScalarExpander(rows, cols, block_size)
        # Persistent PETSc objects, created on the first solve and reused for
        # the whole run (previously the AIJ matrix, KSP, fieldsplit index sets
        # and the composite PC -- including the GAMG hierarchy setup -- were
        # rebuilt from scratch on every Newton iteration).
        self._mat = None
        self._ksp = None

    def _field_index_sets(self, block_size):
        """Build the fieldsplit index sets (pressure / transport|displacement).

        Variable 0 of each cell is pressure; the remaining ``block_size - 1``
        are transport (CPR) or displacement (FS).
        """
        PETSc = self._PETSc
        n = self._expander.n
        pressure_idx = np.arange(0, n, block_size, dtype="int32")
        other_idx = np.concatenate(
            [np.arange(v, n, block_size, dtype="int32") for v in range(1, block_size)]
        )
        is_pressure = PETSc.IS().createGeneral(pressure_idx)
        is_other = PETSc.IS().createGeneral(np.sort(other_idx))
        if self.variant == "fs" and block_size - 1 > 1:
            # Displacement is a vector field.
            is_other.setBlockSize(block_size - 1)
        return is_pressure, is_other

    def _build_aij(self, block_size):
        """Wrap the current scalar-expanded values as a PETSc AIJ matrix."""
        PETSc = self._PETSc
        exp = self._expander
        n = exp.n
        return PETSc.Mat().createAIJ(
            size=(n, n),
            csr=(exp.scalar_rows, exp.scalar_cols, exp.scalar_vals),
        )

    def _ensure_ksp(self, block_size):
        """Build the persistent AIJ matrix + KSP + fieldsplit wiring once.

        Composite / fieldsplit *types* come from the option database (see
        _petsc_args); the fieldsplit *index sets* are attached
        programmatically because they depend on the runtime layout.
        """
        PETSc = self._PETSc
        mat = self._build_aij(block_size)
        mat.assemble()

        ksp = PETSc.KSP().create()
        ksp.setFromOptions()
        ksp.setOperators(mat, mat)

        is_pressure, is_other = self._field_index_sets(block_size)
        pc = ksp.getPC()

        if self.variant == "cpr":
            # Composite PC: setUp() first to materialise the inner PCs, then
            # attach the fieldsplit IS to stage 0 (the fieldsplit Schur PC).
            pc.setUp()
            stage = pc.getCompositePC(0)
            stage.setFieldSplitIS(("transport", is_other), ("pressure", is_pressure))
            stage.setOperators(mat, mat)
            stage.setUp()
            pc.getCompositePC(1).setUp()
        else:  # fs
            # The PC *is* the fieldsplit -- attach the IS before setUp(),
            # otherwise PETSc errors with "must have at least two fields".
            pc.setFieldSplitIS(("displacement", is_other), ("pressure", is_pressure))

        ksp.setUp()
        self._mat = mat
        self._ksp = ksp

    def solve(self, rows, cols, vals, block_size, rhs, sol):
        PETSc = self._PETSc
        self._expander.refresh(vals)

        if self._ksp is None:
            # First solve: scalar_vals already hold the refreshed values.
            self._ensure_ksp(block_size)
        else:
            # Subsequent Newton iterations: in-place value update on the fixed
            # structure. The state bump from assemble() makes the KSP re-run
            # PCSetUp on the next solve (fresh GAMG hierarchy on the new
            # values), while the AIJ matrix, index sets, KSP and composite PC
            # objects are all reused.
            exp = self._expander
            self._mat.setValuesCSR(exp.scalar_rows, exp.scalar_cols, exp.scalar_vals)
            self._mat.assemble()

        ksp = self._ksp
        petsc_rhs = PETSc.Vec().createWithArray(rhs, rhs.size)
        petsc_sol = PETSc.Vec().createWithArray(sol, sol.size)
        if self.print_level >= 4:
            ksp.view()
        # Residual this solve starts from, captured BEFORE ksp.solve overwrites
        # petsc_sol with the solution. It is the reference for the residual-
        # safety rule of the unified convention below. PETSc's default zero
        # initial guess makes it ||b||; a nonzero guess needs ||b - A x0||.
        if ksp.getInitialGuessNonzero():
            _r0_vec = petsc_rhs.duplicate()
            self._mat.mult(petsc_sol, _r0_vec)
            _r0_vec.aypx(-1.0, petsc_rhs)  # r0 = b - A x0
            r0 = float(_r0_vec.norm())
            _r0_vec.destroy()
        else:
            r0 = float(np.linalg.norm(rhs))
        ksp.solve(petsc_rhs, petsc_sol)
        n_iters = int(ksp.getIterationNumber())
        residual = float(ksp.getResidualNorm())
        if self.print_level >= 1:
            print(
                f"PETSc {self.variant}: solved, its={n_iters}, rnorm={residual:.3e}",
                flush=True,
            )
        # mat / ksp are persistent (destroyed with the solver); only the
        # per-solve vector wrappers are released.
        petsc_rhs.destroy()
        petsc_sol.destroy()

        # Unified return convention (mirrors the C++ linear_solver::solve()):
        #   2 -- hard failure: non-finite iterate (breakdown / PC failure /
        #        NaN-or-Inf) or any PETSc divergence other than the iteration
        #        cap. Without this a diverged NaN solve would drive
        #        newton_residual to NaN, whose comparisons are all false, and
        #        be silently accepted as converged.
        #   3 -- iteration cap hit on a finite iterate ('not converged,
        #        usable'); NonlinearSolverSpec.on_linear_nonconvergence decides.
        #   0 -- converged.
        from petsc4py import PETSc

        reason = int(ksp.getConvergedReason())
        max_it = int(getattr(PETSc.KSP.ConvergedReason, "DIVERGED_MAX_IT", -3))
        # Residual-safety rule of the unified convention: an exhausted budget
        # only yields a USABLE iterate when the solution AND the residual are
        # finite and the residual did not regress past the one this solve
        # started from (r0, captured before the solve).
        usable = (
            np.isfinite(sol).all()
            and np.isfinite(residual)
            and _residual_did_not_regress(residual, r0)
        )
        if not usable:
            rc = 2
        elif reason < 0:
            rc = 3 if reason == max_it else 2
        else:
            rc = 0
        return rc, n_iters, residual


class PardisoSolver(PythonLinearSolver):
    """Pardiso (pypardiso / Intel MKL) sparse direct solver.

    The block-CSR Jacobian is expanded to scalar CSR via the shared
    :class:`_BlockToScalarExpander` (structure built once, per-iteration
    values-only gather). The :class:`pypardiso.PyPardisoSolver` is persistent
    and reuses its symbolic analysis across iterations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pardiso = None
        self._expander = None

    def setup(self, rows, cols, block_size):
        """Expand the block sparsity to scalar CSR and create the Pardiso solver."""
        self._expander = _BlockToScalarExpander(rows, cols, block_size)

        import pypardiso

        self._pardiso = pypardiso.PyPardisoSolver()

    def solve(self, rows, cols, vals, block_size, rhs, sol):
        from scipy.sparse import csr_matrix

        exp = self._expander
        exp.refresh(vals)
        mat = csr_matrix(
            (exp.scalar_vals, exp.scalar_cols, exp.scalar_rows),
            shape=(exp.n, exp.n),
            copy=False,
        )
        # PyPardisoSolver reuses its symbolic analysis when the pattern is
        # unchanged; only the numerical factorisation repeats. It raises on a
        # singular/failed factorisation, so a returned solution is well-defined;
        # still guard against a non-finite result (return nonzero so the Newton
        # loop cuts the timestep) for parity with the C++ direct-solver checks.
        self._pardiso.factorize(mat)
        sol[:] = self._pardiso.solve(mat, rhs)
        # Direct solve: report a single "iteration" and no iterative residual,
        # matching the (rc, n_iters, residual) contract of the nonlinear driver.
        return (0 if np.isfinite(sol).all() else 2), 1, 0.0
