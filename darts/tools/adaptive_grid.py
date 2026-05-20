"""
Convenience helpers for constructing adaptive OBL interpolators using only a
per-axis step (axes_step), without specifying axes_min / axes_max.

Background: as of the multi-index-keyed adaptive interpolator (Phase 1+2), the
prescribed window (axes_min, axes_max) is no longer load-bearing — supporting
points are cached on demand wherever the solver lands in state space. The
remaining requirements from the C++ constructor are:

  * axes_points  — affects the legacy integer-key encoding and the advisory
                   "in-bounds" window (used for pickle cache compatibility);
  * axes_min     — interpreted as the grid origin offset;
  * axes_max     — derived from axes_min + (n_points - 1) * axes_step.

This module provides factory functions that take only `axes_step` (and optional
`axes_origin`) and produce a working adaptive interpolator. `axes_points` is
set to a moderate default (the integer key range used for pickle export) and
the resulting bounds are advisory: queries past them work without warnings or
performance loss.

Example:
    from darts.tools.adaptive_grid import make_multilinear_adaptive_cpu

    itor = make_multilinear_adaptive_cpu(
        evaluator,
        axes_step=[0.01, 0.01, 1.0, 1.0],   # per-axis cell size
        n_ops=24,
    )
    itor.init()
"""

from collections.abc import Sequence

import numpy as np

from darts.engines import index_vector, value_vector

# Default advisory grid extent in cells per axis. This drives:
#   - the size of the legacy mixed-radix integer key used for pickle export;
#   - the "in-bounds" predicate used by the point_data getter.
# It does NOT bound the actual cache — cells outside this window are still
# evaluated and held in memory; they are merely skipped during pickle export.
DEFAULT_ADVISORY_AXIS_POINTS = 1024


def _resolve_interpolator(
    kind: str, platform: str, n_dims: int, n_ops: int, precision: str = "d"
):
    """Return the templatized interpolator class for the requested (kind, n_dims, n_ops).

    Mirrors the discovery logic in darts.physics.base.physics_base.PhysicsBase
    but without depending on a PhysicsBase instance.
    """
    import importlib

    module = importlib.import_module("darts.interpolators")

    # 32-bit index first for compactness, then 64-bit. The legacy 128-bit variant
    # ("ll" tag) was removed once adaptive storage migrated to signed multi-index keys
    # (see git history; __uint128_t was only needed for the old dense mixed-radix enumeration).
    for index_tag in ("i", "l"):
        name = f"{kind}_adaptive_{platform}_interpolator_{index_tag}_{precision}_{n_dims}_{n_ops}"
        cls = getattr(module, name, None)
        if cls is not None:
            return cls, name
    raise RuntimeError(
        f"No {kind}_adaptive_{platform}_interpolator template exposed for "
        f"n_dims={n_dims}, n_ops={n_ops}, precision={precision}. Rebuild with that pair."
    )


def _build_grid_args(
    axes_step: Sequence[float],
    n_ops: int,
    axes_origin: Sequence[float] | None = None,
    n_axes_points: Sequence[int] | None = None,
):
    """Derive the (axes_points, axes_min, axes_max) tuple required by the C++ constructor.

    `axes_min` becomes `axes_origin` (default 0 per axis).
    `axes_max` is derived from `axes_min + (n_points - 1) * axes_step`.
    """
    n_dims = len(axes_step)
    step = np.asarray(axes_step, dtype=np.float64)
    if axes_origin is None:
        origin = np.zeros(n_dims, dtype=np.float64)
    else:
        origin = np.asarray(axes_origin, dtype=np.float64)
        if origin.shape != (n_dims,):
            raise ValueError(
                f"axes_origin must have {n_dims} entries, got {origin.shape}"
            )
    if n_axes_points is None:
        n_points = [DEFAULT_ADVISORY_AXIS_POINTS] * n_dims
    else:
        n_points = list(n_axes_points)
        if len(n_points) != n_dims:
            raise ValueError(
                f"n_axes_points must have {n_dims} entries, got {len(n_points)}"
            )
    axes_max = origin + (np.asarray(n_points, dtype=np.float64) - 1.0) * step
    return n_points, origin.tolist(), axes_max.tolist()


def make_multilinear_adaptive_cpu(
    evaluator,
    axes_step: Sequence[float],
    n_ops: int,
    axes_origin: Sequence[float] | None = None,
    n_axes_points: Sequence[int] | None = None,
    precision: str = "d",
):
    """Construct an adaptive multilinear CPU interpolator from `axes_step` only.

    :param evaluator: operator_set_evaluator_iface that returns operator values per state.
    :param axes_step: per-axis cell size (also defines `n_dims`).
    :param n_ops: number of operators to be interpolated.
    :param axes_origin: optional per-axis origin (default: zeros).
    :param n_axes_points: optional advisory number of supporting points per axis
        (default: DEFAULT_ADVISORY_AXIS_POINTS). Drives the in-bounds window used
        for the legacy pickle-cache export only.
    :param precision: "d" (double, default) or "s" (single).
    :returns: an initialized adaptive multilinear interpolator. `init()` is NOT called —
        the caller must call `.init()` after configuring timers/cache.
    """
    n_dims = len(axes_step)
    cls, _name = _resolve_interpolator("multilinear", "cpu", n_dims, n_ops, precision)
    n_points, origin, axes_max = _build_grid_args(
        axes_step, n_ops, axes_origin, n_axes_points
    )
    return cls(
        evaluator,
        index_vector(n_points),
        value_vector(origin),
        value_vector(axes_max),
    )


def make_linear_adaptive_cpu(
    evaluator,
    axes_step: Sequence[float],
    n_ops: int,
    axes_origin: Sequence[float] | None = None,
    n_axes_points: Sequence[int] | None = None,
    use_barycentric: bool = False,
    precision: str = "d",
):
    """Construct an adaptive linear (simplex / barycentric) CPU interpolator from `axes_step` only."""
    n_dims = len(axes_step)
    cls, _name = _resolve_interpolator("linear", "cpu", n_dims, n_ops, precision)
    n_points, origin, axes_max = _build_grid_args(
        axes_step, n_ops, axes_origin, n_axes_points
    )
    return cls(
        evaluator,
        index_vector(n_points),
        value_vector(origin),
        value_vector(axes_max),
        use_barycentric,
    )


def make_multilinear_adaptive_gpu(
    evaluator,
    axes_step: Sequence[float],
    n_ops: int,
    axes_origin: Sequence[float] | None = None,
    n_axes_points: Sequence[int] | None = None,
    precision: str = "d",
):
    """Construct an adaptive multilinear GPU interpolator from `axes_step` only."""
    n_dims = len(axes_step)
    cls, _name = _resolve_interpolator("multilinear", "gpu", n_dims, n_ops, precision)
    n_points, origin, axes_max = _build_grid_args(
        axes_step, n_ops, axes_origin, n_axes_points
    )
    return cls(
        evaluator,
        index_vector(n_points),
        value_vector(origin),
        value_vector(axes_max),
    )
