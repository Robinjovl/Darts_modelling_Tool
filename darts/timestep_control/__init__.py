"""Timestep control for DARTS.

This package is the single source of timestep-control parameters, the
timestepping analogue of :mod:`darts.linear_solvers` (!280) and
:mod:`darts.nonlinear_solvers` (!327):

- :class:`TimestepControl` — a plain settings holder for ``dt_first``/``dt_min``/
  ``dt_mult``/``dt_max``/``eta`` plus the total ``runtime``. Unlike the two
  solvers it has no build/bind step: it's consumed directly by
  ``DartsModel.run()``'s timestep-size adaptation loop, with no C++-side
  mirroring.

``DartsModel.ts_control`` is a plain member, constructed with ``n_vars=0`` in
``DartsModel.__init__`` (before ``physics`` exists) and resized to the
model's actual ``physics.n_vars`` by ``DartsModel._apply_nonlinear()`` during
``init()``. Tune it via ``self.ts_control.dt_first = ...`` etc., or replace it
outright with ``self.ts_control = TimestepControl(...)`` in ``set_solver()``.

``TimestepControl.set_sim_params()`` is the deprecated ``first_ts=``/``mult_ts=``/...
legacy entry point (one deprecation cycle), reachable as
``model.linear_solver.set_sim_params(...)`` (a thin backward-compatible
delegator kept there since ~40 example models call it) or directly as
``model.ts_control.set_sim_params(model, ...)``; new code should assign
``model.ts_control`` fields directly instead.
"""

from darts.timestep_control.timestep_control import TimestepControl

__all__ = [
    "TimestepControl",
]
