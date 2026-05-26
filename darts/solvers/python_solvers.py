"""Python-resident linear solvers for the open-DARTS unified solver framework.

PETSc (via :mod:`petsc4py`) and Pardiso (via :mod:`pypardiso` / Intel MKL) run
inside the Python process and cannot be injected into the C++ engine the way the
engine-resident solvers (``mgr``, ``superlu``, HYPRE, the GPU solvers) are. They
are wrapped here as stateful :class:`PythonLinearSolver` objects so they still
fit the unified :class:`~darts.solvers.specs.LinearSolverSpec` framework.

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

        self.scalar_rows[0] = 0
        pos = 0
        for ib in range(n_block_rows):
            blk_start, blk_end = rows[ib], rows[ib + 1]
            for r in range(b):
                for bi in range(blk_start, blk_end):
                    jb = cols[bi]
                    for c in range(b):
                        self.scalar_cols[pos] = jb * b + c
                        self._gather_idx[pos] = bi * b2 + r * b + c
                        pos += 1
                self.scalar_rows[ib * b + r + 1] = pos

    def refresh(self, block_vals):
        """Gather the current block values into ``scalar_vals`` (in place)."""
        np.take(block_vals.ravel(), self._gather_idx, out=self.scalar_vals)
        return self.scalar_vals


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
        """Refresh the matrix values and solve ``A x = rhs`` into ``sol``."""
        raise NotImplementedError

    # -- entry point -------------------------------------------------------
    def solve_system(self, engine):
        """Solve the engine's current Newton linear system."""
        rows, cols, vals, rhs, sol, block_size = _extract_block_csr(engine)
        if not self._is_set_up:
            self.setup(rows, cols, block_size)
            self._is_set_up = True
        self.solve(rows, cols, vals, block_size, rhs, sol)


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

    def solve(self, rows, cols, vals, block_size, rhs, sol):
        PETSc = self._PETSc
        self._expander.refresh(vals)

        mat = self._build_aij(block_size)
        mat.assemble()
        petsc_rhs = PETSc.Vec().createWithArray(rhs, rhs.size)
        petsc_sol = PETSc.Vec().createWithArray(sol, sol.size)

        # Build the KSP. Composite / fieldsplit *types* come from the option
        # database (see _petsc_args); the fieldsplit *index sets* are attached
        # programmatically here because they depend on the runtime layout.
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
        if self.print_level >= 4:
            ksp.view()
        ksp.solve(petsc_rhs, petsc_sol)
        if self.print_level >= 1:
            print(
                f"PETSc {self.variant}: solved, "
                f"its={ksp.getIterationNumber()}, "
                f"rnorm={ksp.getResidualNorm():.3e}",
                flush=True,
            )
        mat.destroy()
        petsc_rhs.destroy()
        petsc_sol.destroy()
        ksp.destroy()


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
        # unchanged; only the numerical factorisation repeats.
        self._pardiso.factorize(mat)
        sol[:] = self._pardiso.solve(mat, rhs)
