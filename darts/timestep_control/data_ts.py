import warnings

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

    def set_sim_params(
        self,
        model,
        first_ts: float = None,
        mult_ts: float = None,
        min_ts: float = 1e-15,
        max_ts: float = None,
        runtime: float = 1000,
        **legacy,
    ):
        """
        Deprecated: set the timestep controls on this structure (installing it as
        ``model.data_ts``) and, for one deprecation cycle, migrate removed
        nonlinear/linear solver keyword arguments onto ``model``'s solvers.

        The nonlinear solver parameters are NOT set here anymore — specify them
        on ``model.nonlinear_solver`` (a :class:`darts.nonlinear_solvers.NewtonSolver`),
        typically in a ``set_solver()`` override::

            self.nonlinear_solver = NewtonSolver(tolerance=1e-4, max_iterations=15,
                                                 chop=ChopSpec(mode='local', factor=0.2))

        For one deprecation cycle the removed nonlinear keyword arguments
        (``tol_newton``, ``it_newton``, ``newton_type``, ``newton_params``,
        ``line_search``, ``coupled_well_res_norm_method``) are still accepted:
        they emit a :class:`DeprecationWarning` and are mapped onto
        ``model.nonlinear_solver.spec``. Any other unexpected keyword still raises
        :class:`TypeError`.

        :param first_ts: First timestep
        :type first_ts: float
        :param mult_ts: Timestep multiplier
        :type mult_ts: float
        :param max_ts: Maximum timestep
        :type max_ts: float
        :param runtime: Total runtime in days, default is 1000
        :type runtime: float
        Linear-solver parameters are NOT set here either: the linear solver is
        configured through ``model.linear_solver.spec`` / ``.spec.tolerance`` /
        ``.max_iterations`` in :meth:`DartsModel.set_solver`. For one deprecation
        cycle the removed linear keyword arguments (``tol_linear``, ``it_linear``)
        are accepted on the same terms as the nonlinear ones above.

        .. deprecated::
            Set timestep controls on ``model.data_ts``, the nonlinear solver on
            ``model.nonlinear_solver`` and the linear solver on
            ``model.linear_solver.spec`` instead.
        """
        warnings.warn(
            "set_sim_params() is deprecated; set timestep controls on "
            "DartsModel.data_ts and specify DartsModel.nonlinear_solver in set_solver()",
            DeprecationWarning,
            stacklevel=3,  # blame the set_solver() override, not the LinearSolver delegator
        )
        # Solver settings are NOT set here -- they live on model.nonlinear_solver /
        # model.linear_solver. One-cycle migration of legacy solver kwargs: apply
        # them now if the solvers already exist, otherwise DEFER to set_solver(),
        # which materializes the defaults. Deferring (rather than materializing
        # here) keeps set_sim_params() from re-entering the overridable
        # set_solver() hook -- models call set_sim_params() from their set_solver()
        # override, so calling back would recurse infinitely.
        if legacy:
            if getattr(model, "nonlinear_solver", None) is not None:
                model.linear_solver._migrate_legacy_solver_kwargs(legacy)
            else:
                model._pending_legacy_solver_kwargs = legacy

        # Time stepping parameters. if None, the current value is kept.
        self.dt_first = first_ts if first_ts is not None else self.dt_first
        self.dt_min = min_ts if min_ts is not None else self.dt_min
        self.dt_max = max_ts if max_ts is not None else self.dt_max
        self.dt_mult = mult_ts if mult_ts is not None else self.dt_mult
        self.runtime = runtime
        model.data_ts = self
        return self
