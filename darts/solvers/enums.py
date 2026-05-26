"""HYPRE MGR integer-code enumerations.

These ``IntEnum`` classes mirror the C++ enums in
``solvers/include/MGRStrategy.hpp``. The values MUST match HYPRE's official
numbering. Being ``IntEnum``, they can be used anywhere a plain ``int`` is
expected -- e.g. the integer fields of :class:`darts.solvers.specs.MGRLevelSpec`.
"""

from enum import IntEnum


class FRelaxation(IntEnum):
    """MGR F-relaxation type (``MGRLevelSpec.frelax_type``)."""

    NONE = -1
    WEIGHTED_JACOBI = 0
    SINGLE_VCYCLE_SMOOTHER = 1
    AMG_VCYCLE = 2
    HYBRID_GAUSS_SEIDEL_FORWARD = 3
    HYBRID_GAUSS_SEIDEL_BACKWARD = 4
    HYBRID_CHAOTIC_GAUSS_SEIDEL = 5
    HYBRID_SYMMETRIC_GAUSS_SEIDEL = 6
    JACOBI = 7
    L1_HYBRID_SYMMETRIC_GAUSS_SEIDEL = 8
    GAUSSIAN_ELIMINATION = 9
    L1_GAUSS_SEIDEL_FORWARD = 13
    L1_GAUSS_SEIDEL_BACKWARD = 14
    FCF_JACOBI = 17
    L1_JACOBI = 18
    SPARSE_DIRECT_SOLVER = 29
    ILU = 32
    GAUSSIAN_ELIMINATION_W_PIVOTING = 99
    DIRECT_INVERSE = 199


class Interpolation(IntEnum):
    """MGR interpolation type (``MGRLevelSpec.interp_type``)."""

    INJECTION = 0
    L1_JACOBI = 1
    JACOBI = 2
    CLASSICAL_MODIFIED = 3
    APPROXIMATE_INVERSE = 4
    BLOCK_JACOBI = 12
    BLOCK_ROW_SUM = 13
    BLOCK_ROW_SUM_ABS = 14


class Restriction(IntEnum):
    """MGR restriction type (``MGRLevelSpec.restrict_type``)."""

    INJECTION = 0
    UNSCALED = 1
    JACOBI = 2
    APPROXIMATE_INVERSE = 3
    PAIR_DISTANCE_1 = 4
    PAIR_DISTANCE_2 = 5
    BLOCK_JACOBI = 12
    CPR_LIKE = 13
    BLOCK_COL_LUMPED = 14
    PARTIAL_COL_LUMPED = 15


class CoarseGrid(IntEnum):
    """MGR coarse-grid method (``MGRLevelSpec.coarse_method``)."""

    GALERKIN = 0
    NON_GALERKIN_BLOCK_DIAG = 1
    NON_GALERKIN_CPR_DIAG = 2
    NON_GALERKIN_CPR_BLOCK_DIAG = 3
    NON_GALERKIN_SPARSE_APPROX_INV = 4
    NON_GALERKIN_A_CC = 5


class GlobalSmoother(IntEnum):
    """MGR global smoother type (``MGRLevelSpec.smoother_type``)."""

    NONE = -1
    BLOCK_JACOBI = 0
    BLOCK_GAUSS_SEIDEL = 1
    JACOBI = 2
    GAUSS_SEIDEL_SEQUENTIAL = 3
    GAUSS_SEIDEL_PARALLEL = 4
    HYBRID_GAUSS_SEIDEL_FORWARD = 5
    HYBRID_GAUSS_SEIDEL_BACKWARD = 6
    EUCLID_ILU = 8
    HYPRE_ILU = 16
    L1_JACOBI = 18


class VariableRole(IntEnum):
    """Physical role of a degree of freedom.

    Used for ``MGRSolverSpec.reservoir_variable_roles`` /
    ``well_variable_roles``. Mirrors the ``VariableRole`` enum in
    ``solvers/include/CompositionalFlowStrategy.hpp``.
    """

    PRESSURE = 0
    COMPOSITION = 1
    SATURATION = 2
    TEMPERATURE = 3
    VOLUME_CONSTRAINT = 4
    WELL_PRESSURE = 100
    WELL_SECONDARY = 101
    FACILITY = 200
    ROCK_MECHANICS = 300
    DISPLACEMENT = 301
    STRESS = 302
    OTHER = 999
