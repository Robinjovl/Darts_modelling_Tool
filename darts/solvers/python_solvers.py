"""Python-resident linear solvers for the open-DARTS unified solver framework.

PETSc (via :mod:`petsc4py`) and Pardiso (via :mod:`pypardiso` / Intel MKL) run
inside the Python process and cannot be injected into the C++ engine the way the
engine-resident solvers (``mgr``, ``superlu``, HYPRE, the GPU solvers) are. They
are wrapped here as stateful :class:`PythonLinearSolver` objects so they still
fit the unified :class:`~darts.solvers.specs.LinearSolverSpec` framework.

The key idea (see ``SOLVER_REFACTORING_PLAN.md`` section 13) is the
*setup-once / solve-many* split: the sparsity pattern is fixed for a whole run
(``MATRIX_TYPE_CSR_FIXED_STRUCTURE``), so the expensive structural work --
PETSc's KSP/PC construction, Pardiso's symbolic analysis, the block->scalar
sparsity expansion -- happens once in :meth:`PythonLinearSolver.setup`, and only
the matrix *values* are refreshed per Newton iteration in
:meth:`PythonLinearSolver.solve`.

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
    # vals holds n_blocks_nnz blocks of block_size x block_size scalars.
    block_size = int(round((vals.size / n_blocks_nnz) ** 0.5))
    return rows, cols, vals, rhs, sol, block_size


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
        """Solve the engine's current Newton linear system.

        Extracts the block-CSR Jacobian, performs the one-time :meth:`setup` on
        the first call, then :meth:`solve`. ``sol`` (engine ``dX``) is written
        in place.
        """
        rows, cols, vals, rhs, sol, block_size = _extract_block_csr(engine)
        if not self._is_set_up:
            self.setup(rows, cols, block_size)
            self._is_set_up = True
        self.solve(rows, cols, vals, block_size, rhs, sol)


class PETScSolver(PythonLinearSolver):
    """PETSc (petsc4py) Krylov solver wrapped for the unified framework.

    The engine Jacobian is block-CSR, which is exactly PETSc's ``BAIJ`` layout,
    so the system matrix is built as a ``BAIJ`` matrix with ``bsize`` equal to
    the cell block size -- no block->scalar (``.tocsr()``) expansion. The KSP
    and its preconditioner are constructed once in :meth:`setup`; each
    :meth:`solve` re-uploads the block values and solves.

    :param variant: ``"cpr"`` (CPR for flow) or ``"fs"`` (fixed-stress
        fieldsplit for poromechanics).
    """

    def __init__(self, variant: str = "cpr", **kwargs):
        super().__init__(**kwargs)
        if variant not in ("cpr", "fs"):
            raise ValueError(f"PETScSolver: unknown variant {variant!r}")
        self.variant = variant
        self._PETSc = None
        self._ksp = None
        self._block_size = None
        self._n_scalar = None

    def _petsc_args(self) -> str:
        """Assemble the PETSc command-line option string for this variant."""
        args = ""
        if self.print_level >= 2:
            args += "-ksp_monitor_short "
        if self.print_level >= 5:
            args += "-omp_view "
        args += "-ksp_pc_side right "
        args += f"-ksp_max_it {self.max_iterations} "
        args += f"-ksp_rtol {self.tolerance} "
        if self.variant == "cpr":
            # CPR: composite PC -- fieldsplit (AMG on pressure) then ILU.
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
        """Initialise petsc4py and record the fixed pattern dimensions.

        The KSP / PC are built lazily on the first :meth:`solve`, where the
        BAIJ matrix (needed to attach the fieldsplit index sets) is available.
        """
        import petsc4py

        petsc4py.init(self._petsc_args())
        from petsc4py import PETSc

        self._PETSc = PETSc
        self._block_size = block_size
        self._n_scalar = (rows.size - 1) * block_size

    def _build_baij(self, rows, cols, vals, block_size):
        """Wrap the block-CSR Jacobian as a PETSc BAIJ matrix (no expansion)."""
        PETSc = self._PETSc
        n = self._n_scalar
        # BAIJ block values, block-row-major -- the engine's native layout.
        block_vals = vals.reshape(cols.size, block_size, block_size)
        mat = PETSc.Mat().createBAIJ(
            size=(n, n),
            bsize=block_size,
            csr=(rows.astype("int32"), cols.astype("int32"), block_vals),
        )
        mat.assemble()
        return mat

    def _field_index_sets(self, block_size):
        """Build the fieldsplit index sets (pressure / transport|displacement).

        Variable 0 of each cell is pressure; the remaining ``block_size - 1``
        are transport (CPR) or displacement (FS).
        """
        PETSc = self._PETSc
        n = self._n_scalar
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

    def solve(self, rows, cols, vals, block_size, rhs, sol):
        PETSc = self._PETSc

        mat = self._build_baij(rows, cols, vals, block_size)
        petsc_rhs = PETSc.Vec().createWithArray(rhs, rhs.size)
        petsc_sol = PETSc.Vec().createWithArray(sol, sol.size)

        if self._ksp is None:
            # One-time KSP / PC construction (the structural, expensive part).
            ksp = PETSc.KSP().create()
            ksp.setFromOptions()
            ksp.setOperators(mat, mat)
            is_pressure, is_other = self._field_index_sets(block_size)
            pc = ksp.getPC()
            pc.setUp()
            if self.variant == "cpr":
                stage = pc.getCompositePC(0)
                stage.setFieldSplitIS(
                    ("transport", is_other), ("pressure", is_pressure)
                )
                stage.setOperators(mat, mat)
                stage.setUp()
                pc.getCompositePC(1).setUp()
            else:  # fs
                pc.setFromOptions()
                pc.setFieldSplitIS(
                    ("displacement", is_other), ("pressure", is_pressure)
                )
            self._ksp = ksp
        else:
            # Subsequent iterations: same pattern, refreshed block values.
            self._ksp.setOperators(mat, mat)

        self._ksp.setUp()
        if self.print_level >= 4:
            self._ksp.view()
        self._ksp.solve(petsc_rhs, petsc_sol)
        if self.print_level >= 1:
            true_res = np.linalg.norm(mat * petsc_sol - petsc_rhs)
            print(f"PETSc: solved, true residual = {true_res:.3e}")
        mat.destroy()
        petsc_rhs.destroy()
        petsc_sol.destroy()


class PardisoSolver(PythonLinearSolver):
    """Pardiso (pypardiso / Intel MKL) sparse direct solver.

    Pardiso is a scalar solver, so the block-CSR Jacobian must be expanded to
    scalar CSR. The expansion *structure* (scalar ``row_ptr`` / ``col_ind``) and
    a block->scalar value **gather index** are computed once in :meth:`setup`;
    each :meth:`solve` only gathers the current block values into the scalar
    value array, so no structural work or allocation happens in the hot path.
    The :class:`pypardiso.PyPardisoSolver` is persistent and reuses its symbolic
    analysis across iterations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pardiso = None
        self._scalar_rows = None
        self._scalar_cols = None
        self._gather_idx = None
        self._scalar_vals = None
        self._n_scalar = None

    def setup(self, rows, cols, block_size):
        """Expand the block sparsity to scalar CSR and precompute the gather."""
        n_block_rows = rows.size - 1
        b = block_size
        b2 = b * b
        n_scalar = n_block_rows * b
        nnz_blocks = cols.size

        scalar_rows = np.empty(n_scalar + 1, dtype=np.int32)
        scalar_cols = np.empty(nnz_blocks * b2, dtype=np.int32)
        gather_idx = np.empty(nnz_blocks * b2, dtype=np.int64)

        scalar_rows[0] = 0
        pos = 0
        for ib in range(n_block_rows):
            blk_start, blk_end = rows[ib], rows[ib + 1]
            for r in range(b):
                for bi in range(blk_start, blk_end):
                    jb = cols[bi]
                    for c in range(b):
                        scalar_cols[pos] = jb * b + c
                        # block value (bi, r, c) in the flat engine layout
                        gather_idx[pos] = bi * b2 + r * b + c
                        pos += 1
                scalar_rows[ib * b + r + 1] = pos

        self._scalar_rows = scalar_rows
        self._scalar_cols = scalar_cols
        self._gather_idx = gather_idx
        self._scalar_vals = np.empty(nnz_blocks * b2, dtype=np.float64)
        self._n_scalar = n_scalar

        import pypardiso

        self._pardiso = pypardiso.PyPardisoSolver()

    def solve(self, rows, cols, vals, block_size, rhs, sol):
        from scipy.sparse import csr_matrix

        # Values-only refresh: gather block values into the scalar layout.
        np.take(vals.ravel(), self._gather_idx, out=self._scalar_vals)
        mat = csr_matrix(
            (self._scalar_vals, self._scalar_cols, self._scalar_rows),
            shape=(self._n_scalar, self._n_scalar),
            copy=False,
        )
        # PyPardisoSolver reuses its symbolic analysis when the pattern is
        # unchanged; only the numerical factorisation repeats.
        self._pardiso.factorize(mat)
        sol[:] = self._pardiso.solve(mat, rhs)
