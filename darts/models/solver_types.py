"""Linear solver type enum (relocated from ``darts.input.input_data``).

Kept as a separate module so :mod:`darts.models.darts_model` and
:mod:`darts.pipes.viz.plot_live` can import it without pulling in the
legacy ``InputData`` god-struct.
"""

from enum import Enum


class linear_solver_types(Enum):
    """PETSC and direct linear solver identifiers.

    Negative values distinguish PETSc-backed solvers from the C++ solvers.
    """

    CPU_PETSC_CPR = -1  # CPR for flow
    CPU_PETSC_FS = -2  # fixed stress for poromechanics
    CPU_PARDISO = -10  # direct parallel solver


__all__ = ["linear_solver_types"]
