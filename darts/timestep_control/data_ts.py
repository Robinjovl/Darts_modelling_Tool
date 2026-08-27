import numpy as np


class DataTS:
    """Timestep-control parameters.

    Holds ONLY the timestep controls (``dt_first``/``dt_min``/``dt_mult``/
    ``dt_max``/``eta``) plus the total ``runtime``. Neither solver's settings
    are mirrored here — each lives at its own single source of truth:
    ``DartsModel.nonlinear_solver.spec`` (a
    :class:`darts.nonlinear_solvers.NonlinearSolverSpec`, !327) and
    ``DartsModel.linear_solver.spec`` (a
    :class:`darts.linear_solvers.LinearSolverSpec`, !280 — the transitional
    ``linear_*`` attributes this structure carried are now removed).

    This is the timestepping analogue of the validated, serializable nonlinear
    and linear solver specs: :meth:`validate` and :meth:`to_dict` mirror
    ``NonlinearSolverSpec``'s, so the effective timestepping configuration is
    serializable and inspectable (see :meth:`DartsModel.print_config`).
    """

    _FIELDS = (
        "eta",
        "dt_first",
        "dt_min",
        "dt_mult",
        "dt_max",
        "runtime",
    )

    def __init__(self, n_vars=0):
        # timestep control (owned by this structure)
        self.eta = (
            1e20 * np.ones(n_vars)
        )  # controls the timestep by the variable change from the previous newton iteration
        # dX = Xn - X. Eta has a size of number of DOFs per cell. Set to a large value by default, so doesn't affect the timestep choice
        self.dt_first = 1.0  # initial timestep [days]
        # minimal allowed timestep [days] = the divergence-abort floor. NOTE: a
        # known discrepancy remains -- set_sim_params(min_ts=) defaults to 1e-15,
        # so a model configured through set_sim_params (without an explicit min_ts)
        # floors lower than one that constructs DataTS directly. This value is left
        # at 1e-12 deliberately (behaviour-preserving); the *effective* value is now
        # surfaced by DartsModel.print_config(). Unifying the two paths is a follow-up
        # that requires re-baselining the models that ride the abort floor.
        self.dt_min = 1e-12
        self.dt_mult = 2.0  # timestep multiplier, affects the next timestep choice
        self.dt_max = 10.0  # maximal allowed timestep [days]
        self.runtime = (
            1000.0  # total runtime [days]; read by run() when days is not given
        )

    def validate(self):
        """Raise ``ValueError`` on an obviously-invalid timestepping configuration
        (mirror of ``NonlinearSolverSpec.validate()``); called at ``init()``."""
        if self.dt_first <= 0:
            raise ValueError(f"data_ts.dt_first must be > 0, got {self.dt_first}")
        if self.dt_min <= 0:
            raise ValueError(f"data_ts.dt_min must be > 0, got {self.dt_min}")
        if self.dt_max < self.dt_min:
            raise ValueError(
                f"data_ts.dt_max ({self.dt_max}) must be >= dt_min ({self.dt_min})"
            )
        if self.dt_mult < 1.0:
            raise ValueError(f"data_ts.dt_mult must be >= 1, got {self.dt_mult}")
        if self.runtime <= 0:
            raise ValueError(f"data_ts.runtime must be > 0, got {self.runtime}")

    def to_dict(self):
        """Serializable snapshot (mirror of ``NonlinearSolverSpec.to_dict()``)."""
        return {
            "eta": np.asarray(self.eta).tolist(),
            "dt_first": self.dt_first,
            "dt_min": self.dt_min,
            "dt_mult": self.dt_mult,
            "dt_max": self.dt_max,
            "runtime": self.runtime,
        }

    def print(self):
        print("Timestepping parameters:")
        for k in self._FIELDS:
            print("\t", k, "=", getattr(self, k))


def apply_legacy_ts_kwargs(
    model,
    first_ts: float = None,
    mult_ts: float = None,
    min_ts: float = 1e-15,
    max_ts: float = None,
    runtime: float = 1000,
):
    """One-deprecation-cycle helper for ``LinearSolver.set_sim_params()``'s
    timestep-only portion: build a fresh :class:`DataTS`, assign the
    (possibly ``None``) legacy ``*_ts`` kwargs onto it, install it as
    ``model.data_ts``. Returns the new ``DataTS``."""
    ts = DataTS(model.physics.n_vars)
    ts.dt_first = first_ts if first_ts is not None else ts.dt_first
    ts.dt_min = min_ts if min_ts is not None else ts.dt_min
    ts.dt_max = max_ts if max_ts is not None else ts.dt_max
    ts.dt_mult = mult_ts if mult_ts is not None else ts.dt_mult
    ts.runtime = runtime
    model.data_ts = ts
    return model.data_ts
