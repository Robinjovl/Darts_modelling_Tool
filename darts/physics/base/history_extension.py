"""
Provide reusable support for extending OBL with history state.

History variables are extra OBL interpolation axes that are *not* Newton unknowns: the
physics advances them outside of Newton, after a converged timestep (e.g. the maximum gas
saturation ``sg_max`` that Killough relative-permeability hysteresis needs). The engine
keeps them in a separate ``Xhistory`` buffer and builds the interpolator input as
``Xop = [X | Xhistory]``.

Ordering contract (load-bearing, do not change casually): the descriptors are held as an
ordered *list*, not a dict, because they define not only the axis labels but also the order
in which the history axes are appended to the OBL state. That order must stay consistent
across three places -- this module, the engine's ``Xhistory`` buffer, and the interpolator
state ``[X | Xhistory]``. An empty list disables history-aware behaviour entirely; a
non-empty list extends the OBL interpolation state without touching the Newton system.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from darts.engines import conn_mesh, engine_base, ms_well, value_vector


@dataclass
class HistoryField:
    """
    Describe one OBL history variable declaratively.

    History variables enter the OBL interpolator state but are NOT Newton unknowns -- the
    physics advances them outside of Newton (e.g. max gas saturation updated after each
    converged timestep for Killough relative-permeability hysteresis).

    :param label: Axis label used for interpolator state ordering (e.g. ``"sg_max"``).
    :param axes_step: Cell size for this history axis.
    :param axes_origin: Grid origin for this history axis.
    :param default: Reservoir initial value and fallback value at wells and boundaries.
    """

    label: str
    axes_step: float = 1.0
    axes_origin: float = 0.0
    default: float = 0.0

    def __post_init__(self) -> None:
        """
        Normalize and validate the descriptor after dataclass initialization.

        :raises ValueError: If the label is empty, if ``axes_step`` is not finite and
            strictly positive, or if ``axes_origin`` / ``default`` are not finite.
        """
        self.axes_step = float(self.axes_step)
        self.axes_origin = float(self.axes_origin)
        self.default = float(self.default)
        if not self.label:
            raise ValueError("HistoryField.label must be non-empty")
        if not (0.0 < self.axes_step < float("inf")):
            raise ValueError(
                f"HistoryField {self.label!r} axes_step={self.axes_step!r} "
                "must be finite and strictly positive"
            )
        if not (-float("inf") < self.axes_origin < float("inf")):
            raise ValueError(
                f"HistoryField {self.label!r} axes_origin={self.axes_origin!r} "
                "must be finite"
            )
        if not (-float("inf") < self.default < float("inf")):
            raise ValueError(
                f"HistoryField {self.label!r} default={self.default!r} must be finite"
            )


class HistoryStateSupport:
    """
    Manage OBL history descriptors and their runtime engine buffers.

    The descriptors are stored in OBL axis order; see the module docstring for the ordering
    contract this order participates in. An empty descriptor list makes every operation on
    this object an inert no-op, so callers need no ``if history is configured`` guards.

    The class deliberately does not own a physics lifecycle. ``PhysicsBase`` composes it and
    invokes the explicit integration methods while creating an engine, interpolators,
    boundary state, and wells.

    :param fields: Ordered descriptors for auxiliary OBL state axes.
    """

    def __init__(self, fields: Iterable[HistoryField] | None = None) -> None:
        """
        Store ordered history descriptors.

        :param fields: Ordered descriptors for auxiliary OBL state axes.
        """
        self.fields = fields

    @property
    def fields(self) -> list[HistoryField]:
        """
        Return configured history descriptors in OBL storage order.

        :returns: Ordered history descriptors.
        """
        return self._fields

    @fields.setter
    def fields(self, fields: Iterable[HistoryField] | None) -> None:
        """
        Reconfigure the history descriptors.

        :param fields: Ordered descriptors for auxiliary OBL state axes, or None to disable.
        """
        self._fields: list[HistoryField] = list(fields or [])

    @property
    def n_fields(self) -> int:
        """
        Return the number of configured history variables.

        :returns: Number of auxiliary OBL state axes.
        """
        return len(self.fields)

    @property
    def labels(self) -> list[str]:
        """
        Return history labels in their storage order.

        :returns: Ordered history-axis labels.
        """
        return [field.label for field in self.fields]

    @property
    def defaults(self) -> list[float]:
        """
        Return the configured default values in their storage order.

        :returns: Ordered history-axis defaults.
        """
        return [field.default for field in self.fields]

    def default(self, label: str) -> float:
        """
        Return the configured default value for a history field.

        :param label: Label of the requested history field.
        :returns: Default field value.
        :raises KeyError: If ``label`` is not configured.
        """
        return self.fields[self._field_index(label)].default

    def extend_axes(
        self, axes_origin: list[float], axes_step: list[float]
    ) -> tuple[list[float], list[float]]:
        """
        Append history-axis metadata to primary OBL axes.

        :param axes_origin: Primary OBL axis origins.
        :param axes_step: Primary OBL axis cell sizes.
        :returns: Extended ``(axes_origin, axes_step)`` lists, in that order.
        """
        return (
            list(axes_origin) + [field.axes_origin for field in self.fields],
            list(axes_step) + [field.axes_step for field in self.fields],
        )

    def configure_engine(self, engine: engine_base) -> None:
        """
        Configure the engine's runtime history-buffer width before initialization.

        Must be called before ``engine.init()``, which allocates ``Xop`` / ``Xhistory``
        from this width. A width of 0 disables those engine code paths entirely.

        :param engine: DARTS engine instance.
        :type engine: :class:`engine_base`
        """
        if hasattr(engine, "n_history_runtime"):
            engine.n_history_runtime = self.n_fields

    def configure_property_container(self, property_container: Any) -> None:
        """
        Propagate history metadata to one property container.

        The container needs the count to locate primary variables correctly (e.g.
        temperature at position ``[nc]`` rather than ``[-1]`` once ``sg_max`` is appended),
        and the ordered labels to expose ``{label: value}`` to history-aware evaluators.

        :param property_container: Property container receiving OBL state metadata.
        :type property_container: :class:`PropertyContainer`
        """
        if hasattr(property_container, "n_history"):
            property_container.n_history = self.n_fields
        if hasattr(property_container, "history_labels"):
            property_container.history_labels = self.labels

    def get_engine_history_array(
        self, engine: engine_base, label: str, n_blocks: int | None = None
    ) -> np.ndarray:
        """
        Return a per-cell copy of one history axis from ``engine.Xhistory``.

        ``Xhistory`` is a flat ``[(n_blocks + n_bounds) * n_history]`` buffer in cell-major
        order (all ``n_history`` values for cell 0, then cell 1, ...). The returned array is
        a copy and is safe to mutate.

        :param engine: DARTS engine instance that owns ``Xhistory``.
        :type engine: :class:`engine_base`
        :param label: Label of the requested history field.
        :param n_blocks: Number of cells to read; defaults to all history-buffer cells.
        :returns: Requested history values as a one-dimensional array.
        :raises RuntimeError: If no history fields are configured.
        :raises KeyError: If ``label`` is not configured.
        """
        self._require_fields()
        values = np.asarray(engine.Xhistory, copy=False)
        if n_blocks is None:
            n_blocks = values.size // self.n_fields
        field_index = self._field_index(label)
        return values.reshape(-1, self.n_fields)[:n_blocks, field_index].copy()

    def get_interpolator_state(
        self, engine: engine_base, n_vars: int, n_blocks: int | None = None
    ) -> np.ndarray:
        """
        Return the flattened cell-major OBL state ``[X | Xhistory]``.

        The layout is interleaved so callers that stride by ``n_vars + n_history`` pick out
        one state variable per cell.

        :param engine: DARTS engine instance that owns primary and history state.
        :type engine: :class:`engine_base`
        :param n_vars: Number of primary Newton variables per cell.
        :param n_blocks: Number of cells to read; defaults to all primary-state cells.
        :returns: Primary and history state interleaved per cell.
        """
        primary = np.asarray(engine.X, copy=False)
        if n_blocks is None:
            n_blocks = primary.size // n_vars
        primary = primary.reshape(-1, n_vars)[:n_blocks]
        if not self.fields:
            return primary.flatten()
        history = np.asarray(engine.Xhistory, copy=False).reshape(-1, self.n_fields)[
            :n_blocks
        ]
        return np.concatenate([primary, history], axis=1).flatten()

    def set_engine_history_array(
        self,
        engine: engine_base,
        label: str,
        values: float | np.ndarray | list[float],
        n_blocks: int | None = None,
    ) -> None:
        """
        Overwrite one history axis in ``engine.Xhistory``.

        The selected column is updated through a reshaped writable NumPy view, so the write
        lands in the engine-owned buffer.

        :param engine: DARTS engine instance that owns ``Xhistory``.
        :type engine: :class:`engine_base`
        :param label: Label of the history field to write.
        :param values: Scalar for every cell or one value per target cell.
        :param n_blocks: Number of cells to write; defaults to all history-buffer cells.
        :raises RuntimeError: If no history fields are configured.
        :raises KeyError: If ``label`` is not configured.
        """
        self._require_fields()
        flat = np.asarray(engine.Xhistory, copy=False)
        if n_blocks is None:
            n_blocks = flat.size // self.n_fields
        if np.isscalar(values):
            field_values = np.full(n_blocks, float(values))
        else:
            field_values = np.asarray(values, dtype=float).reshape(-1)
        field_index = self._field_index(label)
        flat.reshape(-1, self.n_fields)[:n_blocks, field_index] = field_values

    def populate_boundary_defaults(self, mesh: conn_mesh) -> None:
        """
        Fill boundary-cell history values with configured defaults.

        The engine's ``build_Xop`` reads boundary-cell history values from
        ``mesh.Xhistory_bounds`` and falls back to zero when the buffer is empty. Writing the
        configured defaults here makes engines with boundary cells (MPFA / mechanical paths)
        start from the intended value instead. No-op when no fields are configured or when
        the mesh has no boundary cells.

        :param mesh: Connection mesh that may expose boundary cells.
        :type mesh: :class:`conn_mesh`
        """
        if not self.fields:
            return
        n_bounds = int(getattr(mesh, "n_bounds", 0))
        if n_bounds <= 0:
            return
        # cell-major layout: [(h_0, ..., h_{n_history-1}) for cell 0, cell 1, ...]
        defaults = np.array(self.defaults, dtype=float)
        mesh.Xhistory_bounds = value_vector(np.tile(defaults, n_bounds).tolist())

    def populate_well_defaults(self, wells: list[ms_well]) -> None:
        """
        Fill well and well-control history values with configured defaults.

        :param wells: Initialized DARTS well objects.
        :type wells: list[:class:`ms_well`]
        """
        if not self.fields:
            return
        defaults = value_vector(self.defaults)
        for well in wells:
            well.Xhistory_well_default = defaults
            if hasattr(well, "control"):
                well.control.Xhistory_well_default = defaults
            if hasattr(well, "constraint"):
                well.constraint.Xhistory_well_default = defaults

    def _field_index(self, label: str) -> int:
        """
        Return the storage index of a configured history label.

        :param label: Label of the requested history field.
        :returns: Zero-based storage index.
        :raises KeyError: If ``label`` is not configured.
        """
        for index, field in enumerate(self.fields):
            if field.label == label:
                return index
        raise KeyError(label)

    def _require_fields(self) -> None:
        """
        Reject engine-buffer operations without configured history fields.

        :raises RuntimeError: If no history fields are configured.
        """
        if not self.fields:
            raise RuntimeError("Physics has no history fields configured")
