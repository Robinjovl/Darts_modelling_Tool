"""Deprecated configuration shims of :class:`~darts.models.darts_model.DartsModel`.

One-deprecation-cycle compatibility layer, extracted from ``darts_model.py``
(review feedback: keep the base model readable). Everything in
:class:`LegacyConfigShims` is scheduled for removal -- when the deprecation
cycle ends, delete this module and drop the mixin from ``DartsModel``'s bases.

Scope: ``set_sim_params()`` / ``set_sim_params_data_ts()`` (timestep-only,
warn on use), the one-cycle mapping of removed nonlinear/linear keyword
arguments onto the solver specs (``_migrate_legacy_solver_kwargs``), and the
retired ``copy_data_ts_to_sim_params()`` no-op.
"""

import warnings


class LegacyConfigShims:
    """Mixin holding the deprecated ``set_sim_params`` family (see module docstring)."""

    def set_sim_params_data_ts(self, data_ts):
        """Deprecated: assign ``nonlinear_solver`` and set timestep controls on
        ``data_ts`` instead."""
        warnings.warn(
            "set_sim_params_data_ts() is deprecated; specify DartsModel.nonlinear_solver "
            "in set_solver() and set timestep controls on DartsModel.data_ts instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from darts.models.darts_model import DataTS  # lazy: avoids a circular import

        self.set_solver()
        self.data_ts = DataTS(self.physics.n_vars)
        # copy attributes except eta
        for k in DataTS._FIELDS:
            if k == "eta":
                continue
            if hasattr(data_ts, k):
                setattr(self.data_ts, k, getattr(data_ts, k))

    def set_sim_params(
        self,
        first_ts: float = None,
        mult_ts: float = None,
        min_ts=1e-15,
        max_ts: float = None,
        runtime: float = 1000,
        **legacy,
    ):
        """
        Function to set the timestep and linear solver parameters.

        The nonlinear solver parameters are NOT set here anymore — specify them
        on ``self.nonlinear_solver`` (a :class:`darts.nonlinear_solvers.NewtonSolver`),
        typically in a ``set_solver()`` override::

            self.nonlinear_solver = NewtonSolver(tolerance=1e-4, max_iterations=15,
                                                 chop=ChopSpec(mode='local', factor=0.2))

        For one deprecation cycle the removed nonlinear keyword arguments
        (``tol_newton``, ``it_newton``, ``newton_type``, ``newton_params``,
        ``line_search``, ``coupled_well_res_norm_method``) are still accepted:
        they emit a :class:`DeprecationWarning` and are mapped onto
        ``self.nonlinear_solver.spec``. Any other unexpected keyword still raises
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
        configured through ``self.linear_solver`` (a
        :class:`darts.linear_solvers.LinearSolver`, tuned via
        ``self.linear_solver.spec.tolerance`` / ``.max_iterations``) in
        :meth:`set_solver`. For one deprecation cycle the removed linear keyword
        arguments (``tol_linear``, ``it_linear``) are accepted on the same terms
        as the nonlinear ones above.

        .. deprecated::
            Set timestep controls on ``self.data_ts``, the nonlinear solver on
            ``self.nonlinear_solver`` and the linear solver on
            ``self.linear_solver`` instead.
        """
        warnings.warn(
            "set_sim_params() is deprecated; set timestep controls on "
            "DartsModel.data_ts and specify DartsModel.nonlinear_solver in set_solver()",
            DeprecationWarning,
            stacklevel=2,
        )
        # Solver settings are NOT set here -- they live on self.nonlinear_solver /
        # self.linear_solver. One-cycle migration of legacy solver kwargs: apply them
        # now if the solvers already exist, otherwise DEFER to set_solver(), which
        # materializes the defaults. Deferring (rather than materializing here) keeps
        # set_sim_params() from re-entering the overridable set_solver() hook --
        # models call set_sim_params() from their set_solver() override, so calling
        # back would recurse infinitely.
        if legacy:
            if getattr(self, "nonlinear_solver", None) is not None:
                self._migrate_legacy_solver_kwargs(legacy)
            else:
                self._pending_legacy_solver_kwargs = legacy

        from darts.models.darts_model import DataTS  # lazy: avoids a circular import

        # fresh timestep-control structure
        self.data_ts = DataTS(self.physics.n_vars)
        ts = self.data_ts

        # Time stepping parameters. if None, default value will be used
        ts.dt_first = first_ts if first_ts is not None else ts.dt_first
        ts.dt_min = min_ts if min_ts is not None else ts.dt_min
        ts.dt_max = max_ts if max_ts is not None else ts.dt_max
        ts.dt_mult = mult_ts if mult_ts is not None else ts.dt_mult

        # NOTE: neither solver's parameters are accepted here -- this method
        # configures time-stepping only. Nonlinear settings live on
        # self.nonlinear_solver.spec (!327), linear settings on
        # self.linear_solver.spec (!280):
        #     super().set_solver()                       # platform default solvers
        #     self.nonlinear_solver.spec.tolerance = 1e-3
        #     self.linear_solver.spec.tolerance = 1e-6
        # Legacy kwargs of either family are mapped for one deprecation cycle by
        # _migrate_legacy_solver_kwargs() above.

        # single-sourced on data_ts.runtime via the runtime property
        self.runtime = runtime

    def _migrate_legacy_solver_kwargs(self, legacy: dict):
        """One-deprecation-cycle shim: map removed ``set_sim_params`` solver
        keyword arguments -- nonlinear (!327) and linear (!280) alike -- onto
        ``self.nonlinear_solver.spec`` / ``self.linear_solver.spec`` and warn.
        Unknown keys raise TypeError so genuine typos still fail loudly."""
        from darts.nonlinear_solvers.newton import _ENUM_TO_CHOP_MODE

        spec = self.nonlinear_solver.spec
        handled = []
        # --- linear family (MR280): the spec is the single owner ---
        if "tol_linear" in legacy or "it_linear" in legacy:
            lin = getattr(self.linear_solver, "spec", None)
            if lin is None:
                raise RuntimeError(
                    "set_sim_params(tol_linear=/it_linear=) cannot be migrated: "
                    "self.linear_solver holds no LinearSolverSpec (raw handle)."
                )
            if "tol_linear" in legacy:
                lin.tolerance = legacy.pop("tol_linear")
                handled.append("tol_linear -> linear_solver.spec.tolerance")
            if "it_linear" in legacy:
                lin.max_iterations = legacy.pop("it_linear")
                handled.append("it_linear -> linear_solver.spec.max_iterations")
        if "line_search" in legacy:
            # !327 removed the line-search solver entirely; accept and ignore.
            legacy.pop("line_search")
            handled.append("line_search -> removed (no longer supported)")
        if "tol_newton" in legacy:
            spec.tolerance = legacy.pop("tol_newton")
            handled.append("tol_newton -> nonlinear_solver.spec.tolerance")
        if "it_newton" in legacy:
            spec.max_iterations = legacy.pop("it_newton")
            handled.append("it_newton -> nonlinear_solver.spec.max_iterations")
        if "coupled_well_res_norm_method" in legacy:
            spec.coupled_well_res_norm_method = legacy.pop(
                "coupled_well_res_norm_method"
            )
            handled.append(
                "coupled_well_res_norm_method -> "
                "nonlinear_solver.spec.coupled_well_res_norm_method"
            )
        if "newton_type" in legacy:
            nt = legacy.pop("newton_type")
            # Accept the legacy int (0/1/2), the compiled sim_params.newton_solver_t
            # enum (int-CONVERTIBLE but not an int instance -- isinstance(nt, int)
            # is False for a pybind enum), the mode string, or None.
            if nt is not None and not isinstance(nt, str):
                try:
                    nt = int(nt)
                except (TypeError, ValueError):
                    pass
            spec.chop.mode = (
                _ENUM_TO_CHOP_MODE.get(nt, nt) if isinstance(nt, int) else nt
            )
            spec.chop.__post_init__()  # validate the mapped mode
            handled.append("newton_type -> nonlinear_solver.spec.chop.mode")
        if "newton_params" in legacy:
            np_val = legacy.pop("newton_params")
            spec.chop.factor = np_val[0] if isinstance(np_val, list | tuple) else np_val
            handled.append("newton_params[0] -> nonlinear_solver.spec.chop.factor")
        if legacy:
            raise TypeError(
                f"set_sim_params() got unexpected keyword argument(s) {sorted(legacy)}"
            )
        warnings.warn(
            "set_sim_params() solver keyword arguments are removed; mapped for "
            "this release only (" + "; ".join(handled) + "). Migrate to "
            "self.nonlinear_solver = NewtonSolver(...) / self.linear_solver = "
            "<LinearSolverSpec> in set_solver().",
            DeprecationWarning,
            stacklevel=3,
        )

    def copy_data_ts_to_sim_params(self):
        """No-op retained for compatibility: nothing is mirrored into ``sim_params``
        any more.

        * timestep controls live on ``data_ts`` and are read directly by :meth:`run`;
        * nonlinear settings are synced into the engine by the nonlinear solver
          (``NonlinearSolverSpec.sync_to_engine``, !327);
        * linear settings are mirrored by :meth:`_sync_solver_to_sim_params` from
          ``self.linear_solver.spec`` before ``engine.init()`` (!280).

        The corresponding C++ ``sim_params`` fields (``first_ts``/``max_ts``/
        ``mult_ts``/``tolerance_newton``/``max_i_newton``) no longer exist.
        """
