"""Shared exception types for geochemical-equilibrium (flash) backends.

A single base class lets the model's Newton loop catch a non-convergence from *any*
flash backend (PHREEQC, Reaktoro, ...) and convert it into a timestep cut, without
coupling the backends to each other or masking unrelated bugs.
"""


class FlashError(RuntimeError):
    """Base class for flash (chemical/VLE equilibrium) non-convergence errors.

    Backends raise a subclass when equilibrium cannot be computed for a given state.
    Catch ``FlashError`` to treat the timestep as non-converged and cut ``dt``.
    """
