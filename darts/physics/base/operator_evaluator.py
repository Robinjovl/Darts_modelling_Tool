import warnings
from itertools import product

import numpy as np

from darts.interpolators import operator_set_evaluator_iface, value_vector
from darts.physics.base.property_container import PropertyContainer


def supports_flash_reuse(property_container) -> bool:
    """
    Return whether a property container can share tabulated flash results.

    Reuse requires the container's effective ``evaluate`` to be the base-class
    composition of ``evaluate_flash`` + ``evaluate_properties``.
    Containers that override ``evaluate`` monolithically are no longer supported
    Both ``OperatorsBase.__init__`` and ``PhysicsBase.set_operators`` raise ``ValueError``
    for such a container instead of silently evaluating without flash reuse.

    :param property_container: Property container instance to inspect
    :type property_container: PropertyContainer
    :return: True if flash results can be tabulated and shared between operator sets
    :rtype: bool
    """
    return getattr(type(property_container), "evaluate", None) is (
        PropertyContainer.evaluate
    )


def assert_flash_snapshot_consistent(property_container) -> None:
    """
    Raise if a property container's flash-store row methods
    (``get_flash_snapshot``, ``set_flash_results``, ``flash_row_width``)
    aren't all defined together by the same class.

    :class:`FlashOperators` tabulates flash results as fixed-size float rows in a
    C++ point store (see :meth:`FlashOperators.attach_point_store`), reading
    and writing them through exactly these three methods. ``PropertyContainer``
    (base) defines all three together for the ``(nu, x, temperature, pressure)``
    layout; a subclass overriding the flash-snapshot format (e.g. chemistry's
    geochemical-equilibrium outputs) must override all three together, so the row
    ``get_flash_snapshot`` packs and the row ``set_flash_results`` unpacks always
    agree. Overriding only some of them would silently break the flash store (wrong
    field packed in the wrong slot) rather than announce the mismatch, so this is
    checked once, eagerly, when a region's operators are built
    (:meth:`~darts.physics.base.physics.PhysicsBase.set_operators`) instead of
    being discovered later as corrupted data or a silent cache never hitting.

    :param property_container: Property container instance to validate
    :type property_container: PropertyContainer
    :raises ValueError: If the three methods are not all defined by the same class
    """
    cls = type(property_container)

    def _owner(name):
        for klass in cls.__mro__:
            if name in klass.__dict__:
                return klass
        return None

    names = ("get_flash_snapshot", "set_flash_results", "flash_row_width")
    owners = {name: _owner(name) for name in names}
    if len(set(owners.values())) != 1:
        raise ValueError(
            f"{cls.__name__}: get_flash_snapshot/set_flash_results/"
            f"flash_row_width must be overridden together (whichever class "
            f"changes the flash-snapshot format must define all three) -- got "
            f"{ {k: v.__name__ if v else None for k, v in owners.items()} }"
        )


class OperatorsBase(operator_set_evaluator_iface):
    def __init__(
        self,
        property_container: PropertyContainer,
        thermal: bool,
        extrapolation_flag: bool = True,
        dz: float = None,
        flash_operators: 'FlashOperators' = None,
    ):
        """
        Constructor of OperatorsBase base class

        :param property_container: Property container of type PropertyContainer
        :param thermal: Switch to indicate if energy conservation equation is there
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[last component] < 0 in case nc >= 3)
        :param dz: Composition OBL cell size(s) used to step onto neighbouring grid nodes
                    during boundary extrapolation. Scalar (uniform spacing) or a per-axis
                    vector of length nc-1 (non-uniform cell size across composition axes).
        :param flash_operators: Shared :class:`FlashOperators` instance for reuse of tabulated flash results;
                    when None, or reuse is disabled for containers that override ``evaluate`` monolithically, a private instance is created.
                    Usually wraps this same ``property_container``.
                    May wrap a different one when regions share one FlashOperators
                    (see :meth:`~darts.physics.base.physics.PhysicsBase.add_property_region`'s ``flash_region``)
                    In that case ``evaluate_property_container`` copies the flash results onto this container before use.
        :type flash_operators: FlashOperators, optional
        """
        super().__init__()

        self.property = property_container

        self.thermal = thermal

        self.nc = property_container.nc
        self.ne = self.nc + self.thermal
        self.nph = property_container.nph

        # Some containers (e.g. chemistry's output-only OutputPropertyContainer) have
        # no solid/fluid phase split of their own; default to "no solid components",
        # matching PropertyContainer's own defaults (nc_sol=0, np_sol=0).
        self.ns = getattr(property_container, 'ns', 0)
        self.nc_fl = getattr(property_container, 'nc_fl', self.nc)
        self.np_fl = getattr(property_container, 'np_fl', self.nph)

        self.n_ops: int = None

        self.eps_z = (
            property_container.eps_z if hasattr(property_container, 'eps_z') else 1e-13
        )

        self.extrapolation_flag = extrapolation_flag
        # dz: composition-axis OBL cell size(s) used to step onto neighbouring grid
        # nodes during boundary extrapolation. Accepts a scalar / length-1 value
        # (uniform cell size across all composition axes — the legacy case) or a
        # per-axis vector of length nc-1 (non-uniform OBL cell size across
        # composition axes). Stored as a 1-D float array; a length-1 array is
        # broadcast to every composition axis.
        self.dz = np.atleast_1d(np.asarray(dz, dtype=float)) if dz is not None else None
        assert self.nc <= 2 or not extrapolation_flag or self.dz is not None, (
            "Please provide dz for extrapolation"
        )
        if self.dz is not None:
            # Validate shape/values, not just length: a stray 2-D array, NaN/inf, or
            # non-positive step would otherwise corrupt the per-axis stepping silently.
            assert self.dz.ndim == 1, (
                f"dz must be a scalar or 1-D vector, got ndim={self.dz.ndim}"
            )
            assert np.all(np.isfinite(self.dz)), "dz entries must be finite"
            if extrapolation_flag and self.nc > 2:
                assert self.dz.size in (1, self.nc - 1), (
                    f"dz must be scalar or length nc-1={self.nc - 1}, got {self.dz.size}"
                )
                assert np.all(self.dz > 0), "dz entries must be strictly positive"

        # Flash-reuse wiring: all operator sets of a region share one FlashOperators
        # instance that tabulates the flash results per supporting point.
        # It usually wraps this same property_container
        # When it wraps a different one (see PhysicsBase.add_property_region's flash_region),
        # evaluate_property_container() copies the results across.
        if flash_operators is not None:
            self.flash = flash_operators
        elif isinstance(self, FlashOperators):
            self.flash = self
        elif supports_flash_reuse(property_container):
            # Standalone use (no shared instance passed): private tabulation
            self.flash = FlashOperators(
                property_container,
                thermal,
                extrapolation_flag=extrapolation_flag,
                dz=dz,
            )
        else:
            raise ValueError(
                f"{type(property_container).__name__} overrides evaluate() monolithically. "
                f"PropertyContainer subclasses must implement evaluate_flash()/evaluate_properties() "
                f"so flash results can be tabulated and shared across operator sets. "
                f"Monolithic evaluate() overrides are no longer supported. "
                f"(see darts.physics.base.property_container.PropertyContainer)."
            )

        # C++ point-store wiring; see attach_point_store(). None (the default)
        # means there is no cache: extrapolate() evaluates supporting points
        # directly, and FlashOperators.ensure_flash() recomputes every flash.
        self._point_itor = None
        self._point_n_slots = 0
        self._point_axes_origin = None
        self._point_axes_step_inv = None

    def attach_point_store(self, itor, n_slots=None, axes_origin=None, axes_step=None):
        """
        Wire this operator set to its own interpolator's supporting-point store,
        so tabulated rows are read/written directly through
        ``itor.try_get_point``/``itor.set_point``. Called by
        :meth:`~darts.physics.base.physics.PhysicsBase.set_interpolators` once the
        operator set's interpolator has been created (and its on-disk cache, if
        any, loaded).

        The store is the same ``point_data_store`` the interpolator materializes
        supporting points into, keyed on the integer multi-index this
        interpolator's OBL axes give a state (see :meth:`_point_key`). Two users
        share this wiring:

        - :meth:`extrapolate` reads/writes the operator values at extrapolation
          supporting points instead of re-evaluating them on every boundary
          extrapolation. Rows inserted this way are reused by later interpolation
          (which skips ``evaluate()`` for them) and vice versa: points the
          interpolator already materialized are extrapolation cache hits.
        - :class:`FlashOperators` tabulates its flash-snapshot rows through the
          identical mechanism (see :meth:`FlashOperators.ensure_flash`); for the
          base container layout the snapshot row ``(nu, x, T, P)`` is a strict
          prefix-superset of its operator values ``(nu, x, T)``, so both users
          agree on the stored rows.

        Rows tabulated either way are picked up by the incremental disk
        persistence (``OblCacheCodec`` via ``PhysicsBase.write_cache``) that
        already covers the interpolator, with no separate save/load path.

        Leave unattached (or pass ``itor=None``) to disable caching entirely.
        An ``itor`` that does not expose ``try_get_point``/``set_point`` is
        treated as None, so static-mode interpolators (single-point store access
        is adaptive-only) and prebuilt extensions predating those bindings
        degrade to uncached evaluation without the caller having to check.
        Also unattached: regions with no compiled OBL template for this
        (n_dims, n_ops), and evaluators wrapped in ``ParallelEvaluator``
        (worker-process evaluators are fresh instances that are never attached).

        :param itor: This operator set's own adaptive interpolator, or None.
        :param n_slots: Values per stored point: the interpolator's compiled N_OPS.
                    May exceed ``self.n_ops`` (compiled-template fallback with
                    spare operator slots, or an interpolator built wider than this
                    operator set, e.g. property interpolators built with the
                    physics-wide n_ops).
        :param axes_origin: Per-axis OBL grid origin over this interpolator's
                    axes set (primary + history for the reservoir/property/well
                    interpolators; primary-only for the flash store).
        :param axes_step: Per-axis OBL grid cell size, same length as ``axes_origin``.
        """
        if itor is None or not (
            hasattr(itor, 'try_get_point') and hasattr(itor, 'set_point')
        ):
            self._point_itor = None
            return
        self._point_itor = itor
        self._point_n_slots = n_slots
        self._point_axes_origin = np.asarray(axes_origin, dtype=np.float64)
        self._point_axes_step_inv = 1.0 / np.asarray(axes_step, dtype=np.float64)

    def _point_key(self, state_np):
        """
        Return the point-store multi-index key for a supporting point: the signed
        per-axis grid index over the first ``len(axes_origin)`` state coordinates
        (the axes passed to :meth:`attach_point_store`), saturated to int32.
        Axes beyond those (e.g. history axes for the flash store's primary-only
        wiring) are excluded from the key.

        :param state_np: State at the supporting point
        :type state_np: np.ndarray
        :return: int32 multi-index key of length len(axes_origin)
        :rtype: np.ndarray
        """
        scaled = (
            state_np[: self._point_axes_origin.size] - self._point_axes_origin
        ) * self._point_axes_step_inv
        idx = np.clip(np.rint(scaled), -2147483648, 2147483647)
        return idx.astype(np.int32)

    def _evaluate_supporting_point(self, ref_state):
        """
        Operator values at an extrapolation supporting point (a real OBL grid
        node), as a length-``n_ops`` array. Read from the attached interpolator
        point store when possible (see :meth:`attach_point_store`); on a miss (or
        with no store attached) the point is evaluated through :meth:`evaluate`
        and, when a store is attached, tabulated so later interpolation and
        extrapolation reuse it.

        :param ref_state: Full-length state at the supporting point
        :type ref_state: np.ndarray
        :return: Operator values at the supporting point, length ``self.n_ops``
        :rtype: np.ndarray
        """
        if self._point_itor is None:
            ref_vals = value_vector(np.zeros(self.n_ops))
            self.evaluate(value_vector(ref_state), ref_vals)
            return ref_vals.to_numpy()

        key = self._point_key(ref_state)
        cached = self._point_itor.try_get_point(key)
        if cached is not None:
            return np.asarray(cached)[: self.n_ops]

        # Evaluate into a zero-initialized row of the interpolator's full N_OPS
        # width, reproducing exactly the row C++ materialization would store.
        ref_vals = value_vector(np.zeros(self._point_n_slots))
        self.evaluate(value_vector(ref_state), ref_vals)
        # evaluate() itself may have tabulated this key with a more complete row
        # (FlashOperators.ensure_flash stores the full flash snapshot, including
        # slots its operator values leave zero) -- never overwrite an existing row.
        if self._point_itor.try_get_point(key) is None:
            self._point_itor.set_point(key, ref_vals.to_numpy())
        return ref_vals.to_numpy()[: self.n_ops]

    def insert_point_rows(self, rows):
        """
        Insert externally computed supporting-point rows into the attached point
        store, skipping keys already present (never overwrite). Rows narrower
        than the store's ``n_slots`` are zero-padded on the right: a row
        evaluated into an ``n_ops``-wide vector then padded matches the
        ``n_slots``-wide row C++ materialization would store, whose spare
        trailing slots stay zero.

        Used by :class:`~darts.physics.base.parallel_evaluator.ParallelEvaluator`
        to merge rows tabulated by worker-local stores during batch evaluation
        back into the parent interpolator's store, so later interpolation,
        extrapolation and disk persistence reuse them. No-op when no store is
        attached.

        :param rows: Iterable of ``(key, row)`` pairs: ``key`` an int sequence of
                    the store's axes length, ``row`` a float array of length
                    <= ``n_slots``
        """
        if self._point_itor is None:
            return
        for key, row in rows:
            k = np.asarray(key, dtype=np.int32)
            if self._point_itor.try_get_point(k) is None:
                row = np.asarray(row, dtype=np.float64)
                full = np.zeros(self._point_n_slots)
                n = min(row.size, self._point_n_slots)
                full[:n] = row[:n]
                self._point_itor.set_point(k, full)

    def ensure_flash_results(self, state_np):
        """
        Make ``self.property`` hold the flash results for this supporting point,
        reusing tabulated results through this operator set's :class:`FlashOperators`.

        :param state_np: State at the supporting point [pres, comp_0, ..., comp_N-1, (temp), (history)]
        :type state_np: np.ndarray
        """
        self.flash.ensure_flash(state_np)
        if self.flash.property is not self.property:
            # This region shares another region's FlashOperators
            # (see PhysicsBase.add_property_region's flash_region)
            # Copy the flash results it just computed/restored onto this region's
            # own container via the same row contract the flash point store uses,
            # since the callers read the results from self.property.
            row = np.zeros(self.flash.property.flash_row_width())
            self.flash.property.get_flash_snapshot(row)
            self.property.set_flash_results(row)

    def evaluate_property_container(self, state_np):
        """
        Evaluate the shared property container at a supporting point.

        Reuses tabulated flash results through this region's :class:`FlashOperators`.

        :param state_np: State at the supporting point [pres, comp_0, ..., comp_N-1, (temp), (history)]
        :type state_np: np.ndarray
        """
        self.ensure_flash_results(state_np)
        self.property.evaluate_properties(state_np)

    def evaluate_batch(self, states, n_points, values, n_ops):
        """
        Default serial batch evaluation: loops calling evaluate() per point.
        Override in a subclass or wrapper (e.g. ParallelEvaluator) for parallel dispatch.

        :param states: Flat array of coordinates [n_points * n_dims]
        :param n_points: Number of points to evaluate
        :param values: Flat output array [n_points * n_ops], pre-allocated
        :param n_ops: Number of operators per point
        :return: 0 if successful
        """
        states_np = np.asarray(states)
        values_np = np.asarray(values)
        n_dims = len(states_np) // n_points
        for i in range(n_points):
            sv = value_vector(states_np[i * n_dims : (i + 1) * n_dims].copy())
            vv = value_vector(np.zeros(n_ops))
            self.evaluate(sv, vv)
            values_np[i * n_ops : (i + 1) * n_ops] = np.asarray(vv)
        return 0

    def apply_extrapolation(self, state, values):
        """
        Method that determines whether or not extrapolation should be applied to current state (z[-1] < 0).
        If so, it will call extrapolate() and return True, such that evaluate() skips further evaluation of operators

        :param state: Vector with state [P, z, (T/H)]
        :param values: Vector with operator values
        :return: Whether or not extrapolation has been applied to this state
        """
        # Find composition, if last composition is negative, apply extrapolation
        zc = np.append(state[1 : self.nc], 1 - np.sum(state[1 : self.nc]))

        if len(zc) > 2 and zc[-1] < 0.99 * self.eps_z and self.extrapolation_flag:
            # TODO: Fix second condition, this is problematic for small eps_z values (~1e-14)
            self.extrapolate(state, values)
            return True
        else:
            return False

    def extrapolate(self, state, values):
        """
        When composition lies outside the simplex (∑ z_i ≠ 1 or some z_i < 0), perform exact hyperplane extrapolation:
        Fit each operator value via val = a·z + c through exactly d+1 valid reference points ,
        then evaluate at the out‑of‑bounds composition. Pressure (and temperature) remain constant.
        State layout: [ p, z₁, …, z_d, (T), (history) ] -- reference states copy the
        full incoming state and only step the compositions, so pressure, temperature
        and any history-axis coordinates carry over unchanged.

        Supporting-point operator values are read from / tabulated into the attached
        interpolator point store when one is wired (see :meth:`attach_point_store`),
        so repeated boundary extrapolations and ordinary interpolation share work.
        """
        # Unpack composition from state
        vec = state.to_numpy()
        z = vec[1 : self.nc].copy()

        zero_comps = [i for i in range(self.nc - 1) if z[i] <= 2 * self.eps_z]
        nonzero_comps = [1 if z[i] > 2 * self.eps_z else 0 for i in range(self.nc - 1)]
        nonzero_comp_idxs = [
            i for i, is_nonzero in enumerate(nonzero_comps) if is_nonzero
        ]
        dims = np.sum(nonzero_comps)

        # Per-axis composition cell size (length nc-1). A scalar / length-1 dz is
        # broadcast to every composition axis, so uniform OBL grids reproduce the
        # original behaviour exactly; a length nc-1 dz gives each composition axis
        # its own step, i.e. non-uniform OBL cell size across composition axes.
        dz_axis = (
            self.dz
            if self.dz.size == (self.nc - 1)
            else np.full(self.nc - 1, self.dz[0])
        )

        # --- Candidate supporting points --------------------------------------
        # Step −offset_i · dz_axis[i] along each active composition axis using
        # INDEPENDENT per-axis offsets (0..n_steps_max), excluding the all-zero
        # offset (the incoming point). Independent offsets are required on a
        # non-uniform grid: a valid *physical* support can need a mixed move such
        # as (1·dz0, 2·dz1) that a single shared multiplier never produces. Each
        # candidate lands on a real OBL grid node.
        #
        # n_steps_max gives reach 2 per axis, enough to recover `dims + 1` physical
        # supports for boundary nodes (which sit within one cell of the surface).
        n_steps_max = max(2, dims + 1 - (2**dims - 1))
        candidates = []  # (dist2_index, offsets, zp, admissible)
        for offsets in product(range(n_steps_max + 1), repeat=dims):
            if not any(offsets):
                continue
            zp = z.copy()
            for i, axis in enumerate(nonzero_comp_idxs):
                if offsets[i]:
                    zp[axis] -= offsets[i] * dz_axis[axis]
            # Distance in grid-index units (the offset magnitude), so the selection
            # geometry is independent of the per-axis cell sizes and not biased
            # toward coarser axes.
            dist2 = float(sum(o * o for o in offsets))
            last_z = 1.0 - np.sum(zp)
            # Admissible = the support is physical: it must lie inside the simplex
            # on BOTH the normalization constraint (last_z = 1 − Σz ≥ 0) AND the
            # axis-aligned constraints (every explicit composition ≥ 0). Checking
            # last_z alone would accept a support with a negative component and
            # flash it as physical (the paper requires *physical* supporting
            # points). The small tolerance absorbs round-off on exactly-on-axis
            # grid nodes. This per-node half-space test needs no explicit
            # hypercube/normalization-surface intersection.
            admissible = (last_z >= 0.0) and bool(np.all(zp >= -1e-12))
            candidates.append((dist2, offsets, zp, admissible))

        # --- Rank-revealing selection of d+1 physical, independent supports ----
        # Walk the admissible candidates nearest-first and greedily keep a point
        # only if it adds a NEW affine direction (raises the rank). This guarantees
        # the selected d+1 supports are affinely independent (so B is non-singular)
        # AND physical — closing the gap where a distance-only "furthest+closest"
        # rule can pick collinear or non-physical points. Independence is tracked by
        # modified Gram–Schmidt on the offset vectors; because zp − z = −offset·dz
        # (a fixed positive per-axis scaling), offset-space rank equals zp-space
        # rank, so this is exact and scale-free.
        admissible_sorted = sorted((c for c in candidates if c[3]), key=lambda c: c[0])
        n_supporting_points = dims + 1
        selected = []
        anchor = None
        basis = []  # orthonormal directions already spanned (offset/index units)
        for c in admissible_sorted:
            ov = np.asarray(c[1], dtype=float)
            if anchor is None:
                anchor = ov
                selected.append(c)
                continue
            v = ov - anchor
            for b in basis:
                v = v - np.dot(v, b) * b
            nv = np.linalg.norm(v)
            if nv > 1e-9:
                basis.append(v / nv)
                selected.append(c)
            if len(selected) == n_supporting_points:
                break

        # Fallback: the admissible set genuinely spans fewer than `dims` directions
        # (the physical neighbourhood of this node is lower-dimensional than dims).
        # Top up nearest-first from the remaining candidates so the fit can still
        # proceed; evaluating a non-physical top-up node recurses into another
        # extrapolation rather than flashing it.
        if len(selected) < n_supporting_points:
            chosen = {c[1] for c in selected}
            for c in sorted(
                (c for c in candidates if c[1] not in chosen), key=lambda c: c[0]
            ):
                selected.append(c)
                if len(selected) == n_supporting_points:
                    break

        supporting_points = [c[2] for c in selected]

        # Gather valid reference points, through the interpolator point store
        zps_list = []
        vals_list = []
        for zp in supporting_points:
            ref_state = vec.copy()
            ref_state[1 : self.nc] = zp
            zps_list.append(zp)
            vals_list.append(self._evaluate_supporting_point(ref_state))

        # Use the first d+1 valid points to define hyperplane implicitly via val = a·z + c
        zps = np.stack(zps_list[: dims + 1])  # shape (dims+1, dims)
        vals = np.stack(vals_list[: dims + 1])  # shape (dims+1, n_ops)

        # Build and solve B · X = vals, where B = [zps | 1]
        B = np.hstack((zps, np.ones((dims + 1, 1))))  # shape (dims+1, dims+1)
        B = np.delete(B, zero_comps, axis=1)
        try:
            X = np.linalg.solve(B, vals)  # shape (dims+1, n_ops)
        except np.linalg.LinAlgError:
            # A singular system means support SELECTION produced affinely-dependent
            # points — a selection problem, not an expected numerical condition.
            # Don't abort the run, but surface it loudly (per-process warning, which
            # works in serial and in multiprocessing workers alike) rather than
            # silently least-squares'ing it away, so it can be investigated.
            warnings.warn(
                f"OBL extrapolation: singular supporting set (dims={int(dims)}) — "
                "falling back to least-squares. This indicates degenerate support "
                "selection and should be investigated.",
                RuntimeWarning,
                stacklevel=2,
            )
            X = np.linalg.lstsq(B, vals, rcond=None)[0]

        # Separate coefficients
        a = X[:-1, :]  # shape (dims, n_ops)
        c = X[-1, :]  # shape (n_ops,)

        # Extrapolate values
        z_nonzero = z[nonzero_comp_idxs]
        ext = a.T.dot(z_nonzero) + c

        # Write back into values array
        out = np.array(values, copy=False)
        out[: self.n_ops] = ext
        return out


class FlashOperators(OperatorsBase):
    """
    Operator set that takes care of the flash / thermodynamics at OBL supporting points.

    One instance is shared by all operator sets of a property region (see
    :meth:`~darts.physics.base.physics.PhysicsBase.set_operators`): the first operator
    set to evaluate at a supporting point computes the flash and tabulates the result
    in the C++ flash point store (see :meth:`attach_point_store`); every other operator
    set evaluating at the same point restores the tabulated result via :meth:`ensure_flash`
    instead of recomputing it. The tabulation is keyed on the primary state coordinates of
    the supporting point (OBL history axes are excluded — the flash does not depend on
    them), so reuse is independent of the order in which the interpolators discover their
    supporting points. All interpolators of a region share identical axes origin/step,
    hence coinciding supporting points produce bit-identical coordinates and exact keys
    match. When no flash store is attached (itor_mode='static', or a standalone instance
    never wired to a Physics interpolator — try_get_point/set_point exist only on adaptive
    interpolators), there is no cache at all: :meth:`ensure_flash` runs the flash directly
    on every call.

    It also implements the evaluator interface itself, exposing phase fractions ``nu``,
    phase compositions ``x`` and temperature as operator values, so it can back an
    interpolator for output or diagnostics.
    """

    def __init__(
        self,
        property_container: PropertyContainer,
        thermal: bool,
        extrapolation_flag: bool = True,
        dz: float = None,
    ):
        """
        Constructor of FlashOperators class

        :param property_container: Property container of type PropertyContainer
        :param thermal: Switch to indicate if energy conservation equation is there
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[last component] < 0 in case nc >= 3)
        :param dz: Composition OBL cell size(s) used to step onto neighbouring grid nodes
                    during boundary extrapolation. Scalar (uniform spacing) or a per-axis
                    vector of length nc-1 (non-uniform cell size across composition axes).
        """
        super().__init__(
            property_container, thermal, extrapolation_flag=extrapolation_flag, dz=dz
        )

        # Operator layout (meaningful when the container defines a fluid-phase layout;
        # e.g. the chemistry containers do not, and use this class for reuse only)
        if self.np_fl > 0 and self.nc_fl > 0:
            self.NU_OP = 0  # phase mole fraction operator - np_fl
            self.X_OP = (
                self.NU_OP + self.np_fl
            )  # phase composition operator - np_fl * nc_fl
            self.TEMP_OP = (
                self.X_OP + self.np_fl * self.nc_fl
            )  # temperature operator - 1
            self.n_ops = self.TEMP_OP + 1
            self.op_names = [
                (self.NU_OP, "NU"),
                (self.X_OP, "X"),
                (self.TEMP_OP, "TEMP"),
            ]
        else:
            self.n_ops = 0

        # Hit = a flash-store lookup found a tabulated result; miss = evaluate_flash had
        # to run (flash-store miss, or no flash store attached at all -- see ensure_flash).
        self.cache_hits = 0
        self.cache_misses = 0

    def attach_point_store(self, itor, n_slots=None, axes_origin=None, axes_step=None):
        """
        Wire this FlashOperators to its dedicated interpolator's supporting-point
        store, so :meth:`ensure_flash` reads/writes flash-snapshot rows directly
        (same mechanism as :meth:`OperatorsBase.attach_point_store`, which see).
        The stored rows are the container's flash snapshots -- for the base layout
        ``(nu, x, T, P)``, a strict prefix-superset of this operator set's values
        ``(nu, x, T)``, so :meth:`extrapolate`'s cache hits stay consistent with
        :meth:`evaluate`. The store is keyed on the primary OBL axes only
        (history axes are excluded -- the flash does not depend on them), so pass
        primary-only ``axes_origin``/``axes_step``.

        Refuses to attach when the container declares no flash row
        (``flash_row_width() <= 0``); :meth:`ensure_flash` then runs the flash
        directly on every call. (The container's flash-snapshot methods are
        validated once, eagerly, by :func:`assert_flash_snapshot_consistent` in
        :meth:`~darts.physics.base.physics.PhysicsBase.set_operators`, so by the
        time this is called they're already known to be consistent.)

        :param itor: This region's dedicated FlashOperators interpolator, or None.
        :param n_slots: Values per cached point: at least
                    ``property_container.flash_row_width()``; may be larger
                    when ``create_interpolator`` fell back to a compiled template
                    with spare operator slots.
        :param axes_origin: Per-axis OBL grid origin, length ``self.ne``.
        :param axes_step: Per-axis OBL grid cell size, length ``self.ne``.
        """
        if self.property.flash_row_width() <= 0:
            itor = None
        super().attach_point_store(
            itor, n_slots=n_slots, axes_origin=axes_origin, axes_step=axes_step
        )

    def _ensure_flash_stored(self, state_np):
        """
        ``ensure_flash`` via the C++ flash point store (see
        :meth:`attach_point_store`). Packing/unpacking the fixed-width float row is
        delegated to the property container's own ``get_flash_snapshot``/
        ``set_flash_results`` (validated by :func:`assert_flash_snapshot_consistent`
        in :meth:`~darts.physics.base.physics.PhysicsBase.set_operators`), so this
        method works for any container layout, not just the base NU/X/TEMP/PRES one.
        """
        key = self._point_key(state_np)
        cached = self._point_itor.try_get_point(key)
        if cached is not None:
            self.property.set_flash_results(cached)
            self.cache_hits += 1
            return

        self.property.evaluate_flash(state_np)
        row = np.zeros(self._point_n_slots)
        self.property.get_flash_snapshot(row)
        self._point_itor.set_point(key, row)
        self.cache_misses += 1

    def ensure_flash(self, state_np):
        """
        Make the property container hold the flash results for this supporting point.

        On a miss, the flash is computed via ``PropertyContainer.evaluate_flash`` and
        the result tabulated; on a hit, the tabulated snapshot is restored via
        ``PropertyContainer.set_flash_results``. Restoring (rather than skipping when
        the container "already holds" the state) makes the hit path correct regardless
        of what ran in between (other supporting points, ``compute_total_enthalpy``, ...).

        Tabulation goes through the C++ flash point store when one is attached (see
        :meth:`attach_point_store`); otherwise there is no cache and the flash is
        recomputed on every call (itor_mode='static', or a standalone instance never
        wired to a Physics interpolator).

        :param state_np: State at the supporting point [pres, comp_0, ..., comp_N-1, (temp), (history)]
        :type state_np: np.ndarray
        """
        if self._point_itor is not None:
            self._ensure_flash_stored(state_np)
            return

        self.property.evaluate_flash(state_np)
        self.cache_misses += 1

    def evaluate(self, state, values):
        """
        Evaluate the flash operators: phase mole fractions nu, phase compositions x
        and temperature at the given state.

        :param state: Vector of state variables [pres, comp_0, ..., comp_N-1, (temp)]
        :type state: darts.interpolators.value_vector
        :param values: Vector for storage of operator values
        :type values: darts.interpolators.value_vector
        :return: 0 if successful
        :rtype: int
        """
        # Check if extrapolation needs to be applied
        if self.apply_extrapolation(state, values):
            return 0

        state_np = state.to_numpy()
        values_np = values.to_numpy()
        values_np[:] = 0

        self.ensure_flash(state_np)

        values_np[self.NU_OP : self.NU_OP + self.np_fl] = self.property.nu
        values_np[self.X_OP : self.X_OP + self.np_fl * self.nc_fl] = (
            self.property.x.ravel()
        )
        values_np[self.TEMP_OP] = self.property.temperature

        return 0


class WellCtrlOperators(OperatorsBase):
    """
    Set of operators for well controls of EPM and DFM wells.

    Operator layout:
    NP EPM molar-rate, NP EPM mass-rate, NP EPM volumetric-rate,
    NP EPM advective-heat-rate, pressure, temperature,
    NP DFM molar-rate, NP DFM mass-rate, NP DFM volumetric-rate,
    NP DFM advective-heat-rate.
    """

    def __init__(
        self,
        property_container: PropertyContainer,
        thermal: bool,
        extrapolation_flag: bool = True,
        dz: float = None,
        flash_operators: FlashOperators = None,
    ):
        """
        Constructor of WellCtrlOperators class

        :param property_container: Property container of type PropertyContainer
        :param thermal: Switch to indicate if energy conservation equation is there
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[last component] < 0 in case nc >= 3)
        :param dz: Composition OBL cell size(s) used to step onto neighbouring grid nodes
                    during boundary extrapolation. Scalar (uniform spacing) or a per-axis
                    vector of length nc-1 (non-uniform cell size across composition axes).
        :param flash_operators: Shared :class:`FlashOperators` of this property region
        :type flash_operators: FlashOperators, optional
        """
        super().__init__(
            property_container,
            thermal,
            extrapolation_flag=extrapolation_flag,
            dz=dz,
            flash_operators=flash_operators,
        )

        self.n_rate_ctrl_types = 4  # molar, mass, volumetric, and advective heat rates
        self.n_state_ctrl_ops = 2  # pressure and temperature
        self.epm_rate_ctrl_ops_offset = 0
        self.state_ctrl_ops_offset = self.n_rate_ctrl_types * self.nph
        self.dfm_rate_ctrl_ops_offset = (
            self.state_ctrl_ops_offset + self.n_state_ctrl_ops
        )
        self.n_ops = self.n_state_ctrl_ops + 2 * self.n_rate_ctrl_types * self.nph

    def _fill_rate_ctrl_ops(self, values, offset, rate_factor):
        # Molar rate ctrl operator
        idx = offset
        values[idx + self.property.ph] = (
            self.property.dens_m[self.property.ph] * rate_factor
        )

        # Mass rate ctrl operator
        idx += self.nph
        values[idx + self.property.ph] = (
            self.property.dens[self.property.ph] * rate_factor
        )

        # Volumetric rate ctrl operator
        idx += self.nph
        values[idx + self.property.ph] = rate_factor

        # Advective heat rate ctrl operator
        idx += self.nph
        if self.thermal:
            values[idx + self.property.ph] = (
                self.property.enthalpy[self.property.ph]
                * self.property.dens_m[self.property.ph]
                * rate_factor
            )

    def evaluate(self, state, values):
        # Check if extrapolation needs to be applied
        if self.apply_extrapolation(state, values):
            return 0

        state_np = state.to_numpy()
        values_np = values.to_numpy()
        values_np[:] = 0

        self.evaluate_property_container(state_np)
        if self.thermal:
            self.property.evaluate_thermal(state_np)

        epm_rate_factor = (
            self.property.kr[self.property.ph] / self.property.mu[self.property.ph]
        )
        self._fill_rate_ctrl_ops(
            values_np, self.epm_rate_ctrl_ops_offset, epm_rate_factor
        )

        # Store pressure (P) and temperature (T) of the current state for a generic state specification.
        # This is needed when pressure or temperature is not part of the state variables
        # (e.g., volume instead of pressure, or enthalpy instead of temperature).
        idx = self.state_ctrl_ops_offset
        values_np[idx + 0] = state[0]
        values_np[idx + 1] = self.property.temperature

        dfm_rate_factor = self.property.sat[self.property.ph]
        self._fill_rate_ctrl_ops(
            values_np, self.dfm_rate_ctrl_ops_offset, dfm_rate_factor
        )

        return 0


class ThermalVarOperator(OperatorsBase):
    """
    ThermalVarOperator gives the thermal variable for generic state specification
    """

    def __init__(
        self,
        property_container: PropertyContainer,
        thermal: bool,
        is_pt: bool = True,
        extrapolation_flag: bool = True,
        dz: float = None,
        flash_operators: FlashOperators = None,
    ):
        """
        Constructor of ThermalVarOperator class

        :param property_container: Property container of type PropertyContainer
        :param thermal: Switch to indicate if energy conservation equation is there
        :param is_pt: Switch to indicate if state specification is P, PT, or PH
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[last component] < 0 in case nc >= 3)
        :param dz: Composition OBL cell size(s) used to step onto neighbouring grid nodes
                    during boundary extrapolation. Scalar (uniform spacing) or a per-axis
                    vector of length nc-1 (non-uniform cell size across composition axes).
        :param flash_operators: Shared :class:`FlashOperators` of this property region
        :type flash_operators: FlashOperators, optional
        """
        super().__init__(
            property_container,
            thermal,
            extrapolation_flag,
            dz,
            flash_operators=flash_operators,
        )

        self.n_ops = 1
        self.is_pt = is_pt

    def evaluate(self, state_pt, values):
        # Check if extrapolation needs to be applied
        if self.apply_extrapolation(state_pt, values):
            return 0

        values_np = values.to_numpy()
        values_np[:] = 0

        if not self.thermal:
            values_np[0] = self.property.temperature
        elif self.is_pt:
            values_np[0] = state_pt[-1]
        else:
            values_np[0] = self.property.compute_total_enthalpy(state_pt=state_pt)

        return 0


class PropertyOperators(OperatorsBase):
    """
    This class contains a set of operators for evaluation of output properties.
    A set of interpolators is created in the :class:`Physics` object to rapidly obtain properties after simulation.
    """

    def __init__(
        self,
        property_container: PropertyContainer,
        thermal: bool,
        props: dict = None,
        extrapolation_flag: bool = True,
        dz: float = None,
        flash_operators: FlashOperators = None,
    ):
        """
        This is the constructor for PropertyOperator.
        The properties to be obtained from the PropertyOperators are passed to PropertyContainer as a dictionary.

        :param property_container: PropertyContainer object to evaluate properties at given state
        :param thermal: Bool for thermal
        :param props: Optional dictionary of properties, default is taken from PropertyContainer
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[last component] < 0 in case nc >= 3)
        :param dz: Composition OBL cell size(s) used to step onto neighbouring grid nodes
                    during boundary extrapolation. Scalar (uniform spacing) or a per-axis
                    vector of length nc-1 (non-uniform cell size across composition axes).
        :param flash_operators: Shared :class:`FlashOperators` of this property region
        :type flash_operators: FlashOperators, optional
        """
        super().__init__(
            property_container,
            thermal,
            extrapolation_flag,
            dz,
            flash_operators=flash_operators,
        )

        self.props = property_container.output_props if props is None else props
        self.props_name = [key for key in self.props.keys()]
        self.props_idx = {prop: j for j, prop in enumerate(self.props_name)}
        self.n_ops = len(self.props_name)

    def evaluate(self, state: value_vector, values: value_vector):
        """
        This function evaluates the properties at given `state` (P,z) or (P,T,z) from the :class:`PropertyContainer` object.
        The user-specified properties are stored in the `values` object.

        :param state: Vector of state variables [pres, comp_0, ..., comp_N-1, (temp)]
        :type state: darts.interpolators.value_vector
        :param values: Vector for storage of operator values
        :type values: darts.interpolators.value_vector
        """
        # Check if extrapolation needs to be applied
        if self.apply_extrapolation(state, values):
            return 0

        state_np = state.to_numpy()
        values_np = values.to_numpy()
        self.evaluate_property_container(state_np)
        if self.thermal:
            self.property.evaluate_thermal(state_np)

        for i, prop in enumerate(self.props_name):
            output = self.props[prop]()
            values_np[i] = output if not np.isnan(output) else 0.0

        return 0


class ReservoirOperators(OperatorsBase):
    def __init__(
        self,
        property_container: PropertyContainer,
        thermal: bool,
        extrapolation_flag: bool = True,
        dz: float = None,
        flash_operators: FlashOperators = None,
    ):
        """
        Constructor of ReservoirOperators class

        :param property_container: Property container of type PropertyContainer
        :param thermal: Switch to indicate if energy conservation equation is there
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[last component] < 0 in case nc >= 3)
        :param dz: Composition interval along OBL composition axes to obtain consistent points for extrapolation
                    (must be equal along all composition axes in current setup)
        :param flash_operators: Shared :class:`FlashOperators` of this property region
        :type flash_operators: FlashOperators, optional
        """
        super().__init__(
            property_container,
            thermal,
            extrapolation_flag=extrapolation_flag,
            dz=dz,
            flash_operators=flash_operators,
        )  # Initialize base-class

        # Operator order
        self.ACC_OP = 0  # accumulation operator - ne
        self.FLUX_OP = self.ACC_OP + self.ne  # flux operator - ne * nph
        self.DENS_OP = self.FLUX_OP + self.ne * self.nph  # density operator
        self.UPSAT_OP = self.DENS_OP + self.nph  # saturation operator
        self.GRAD_OP = self.UPSAT_OP + self.nph  # gradient operator - ne * nph
        self.KIN_OP = self.GRAD_OP + self.ne * self.nph  # kinetic operator - ne
        self.GRAV_OP = self.KIN_OP + self.ne  # gravity operator - nph
        self.PC_OP = self.GRAV_OP + self.nph  # capillary operator - nph
        self.MULT_OP = self.PC_OP + self.nph  # permeability multiplier operator - 1
        self.LAMBDA_OP = self.MULT_OP + 1  # mobility operator - nph
        self.SAT_OP = self.LAMBDA_OP + self.nph  # saturation operator - nph
        self.ENTH_OP = self.SAT_OP + self.nph  # enthalpy operator - nph
        self.TEMP_OP = self.ENTH_OP + self.nph  # temperature operator - 1
        self.PRES_OP = self.TEMP_OP + 1  # pressure operator - 1
        self.n_ops = self.PRES_OP + 1

        # Operator names
        self.op_names = [
            (self.ACC_OP, "ACC"),
            (self.FLUX_OP, "FLUX"),
            (self.DENS_OP, "DENS"),
            (self.UPSAT_OP, "UPSAT"),
            (self.GRAD_OP, "GRAD"),
            (self.KIN_OP, "KIN"),
            (self.GRAV_OP, "GRAV"),
            (self.PC_OP, "PC"),
            (self.MULT_OP, "MULT"),
            (self.LAMBDA_OP, "LAMBDA"),
            (self.SAT_OP, "SAT"),
            (self.ENTH_OP, "ENTH"),
            (self.TEMP_OP, "TEMP"),
            (self.PRES_OP, "PRES"),
        ]

    def print_operators(self, state, values):
        """Method for printing operators, grouped"""
        print("================================================")
        print("STATE", state)
        print("ALPHA (accumulation)", values[self.ACC_OP : self.FLUX_OP])
        for j in range(self.nph):
            idx0, idx1 = self.FLUX_OP + j * self.ne, self.FLUX_OP + (j + 1) * self.ne
            print(f"BETA (flux) {j}", values[idx0:idx1])
        print("GAMMA (diffusion)", values[self.UPSAT_OP : self.GRAD_OP])
        for j in range(self.nph):
            idx0, idx1 = self.GRAD_OP + j * self.ne, self.GRAD_OP + (j + 1) * self.ne
            print(f"CHI (diffusion) {j}", values[idx0:idx1])
        print("DELTA (reaction)", values[self.KIN_OP : self.GRAV_OP])
        print("GRAVITY", values[self.GRAV_OP : self.PC_OP])
        print("CAPILLARITY", values[self.PC_OP : self.MULT_OP])
        print("PERM_MULT", values[self.MULT_OP])
        print("LAMBDA", values[self.LAMBDA_OP : self.SAT_OP])
        print("SAT", values[self.SAT_OP : self.ENTH_OP])
        print("ENTHALPY", values[self.ENTH_OP : self.ENTH_OP + self.nph])
        print("TEMPERATURE, PRESSURE", values[self.TEMP_OP], values[self.PRES_OP])
        return

    def evaluate(self, state, values):
        """
        Evaluate the non-thermal reservoir operators for the super engine

        :param state: state variables [pres, comp_0, ..., comp_N-1, temp]
        :type state: value_vector in open-darts, pylvarray.Array in GEOS
        :param values: values of the operators (used for storing the operator values)
        :type values: value_vector in open-darts, pylvarray.Array in GEOS
        :return: updated value for operators, stored in values
        """
        # Check if extrapolation needs to be applied
        if self.apply_extrapolation(state, values):
            return 0

        # Composition vector and pressure from state:
        state_np = state.to_numpy()
        values_np = values.to_numpy()
        values_np[:] = 0

        # Evaluate properties at current state, reusing tabulated flash results
        self.evaluate_property_container(state_np)
        self.compr = self.property.rock_compr_ev.evaluate(state_np[0])

        density_tot = np.sum(
            self.property.sat[: self.np_fl] * self.property.dens_m[: self.np_fl]
        )
        zc = np.append(state_np[1 : self.nc], 1 - np.sum(state_np[1 : self.nc]))
        self.phi_s = np.sum(zc[self.nc_fl :])
        self.phi_f = 1.0 - self.phi_s

        """ CONSTRUCT OPERATORS HERE """

        """ Alpha operator represents accumulation term """
        # fluid mass accumulation: c_r [1/bar] z_c* [-] rho_m^T [kmol/m3]
        values_np[self.ACC_OP : self.ACC_OP + self.nc_fl] = (
            self.compr * density_tot * zc[: self.nc_fl]
        )

        """ and alpha for mineral components """
        # solid mass accumulation: c_r phi^T z_s* [-] rho_ms [kmol/m3]
        values_np[self.ACC_OP + self.nc_fl : self.ACC_OP + self.nc_fl + self.ns] = (
            self.compr
            * self.property.dens_m[self.np_fl : self.np_fl + self.ns]
            * zc[self.nc_fl : self.nc_fl + self.ns]
        )

        """ Beta operator """
        for j in self.property.ph:
            # fluid convective mass flux: x_cj [-] rho_mj [kmol/m3] (kmol/m3)
            values_np[
                self.FLUX_OP + j * self.ne : self.FLUX_OP + j * self.ne + self.nc_fl
            ] = self.property.x[j][: self.nc_fl] * self.property.dens_m[j]

        """ Molar density operator """
        # molar density: rho_mj [kmol/m3]
        values_np[self.DENS_OP + self.property.ph] = self.property.dens_m[
            self.property.ph
        ]

        """ Gamma operator for diffusion (for heat conduction and molecular diffusion) """
        # fluid diffusive flux sat: c_r [1/bar] phi_f s_j (1/bar)
        values_np[self.UPSAT_OP + self.property.ph] = (
            self.compr * self.phi_f * self.property.sat[self.property.ph]
        )
        # solid diffusive flux sat: c_r [1/bar] z_s* (1/bar)
        values_np[self.UPSAT_OP + self.np_fl : self.UPSAT_OP + self.np_fl + self.ns] = (
            self.compr * zc[self.nc_fl : self.nc_fl + self.ns]
        )

        """ Chi operator for diffusion """
        for j in self.property.ph:
            D = self.property.diffusion_ev[self.property.phases_name[j]].evaluate()
            # fluid diffusive flux: D_cj [m2/day] x_cj [-] (m2/day)
            values_np[
                self.GRAD_OP + j * self.ne : self.GRAD_OP + j * self.ne + self.nc_fl
            ] = D[: self.nc_fl] * self.property.x[j][: self.nc_fl]

        """ Delta operator for reaction """
        # fluid/solid mass source: n_c [kmol/m3/day] (kmol/m3/day)
        values_np[self.KIN_OP : self.KIN_OP + self.nc] = self.property.mass_source

        """ Gravity and capillarity operators """
        # E3-> gravity
        values_np[self.GRAV_OP + self.property.ph] = self.property.dens[
            self.property.ph
        ]

        # E4-> capillarity
        values_np[self.PC_OP : self.PC_OP + self.property.np_fl] = self.property.pc

        """ Permeability multiplier k/kmax """
        # E5_> permeability multiplier due to permporo relationship
        values_np[self.MULT_OP] = self.property.permporo_mult_ev.evaluate(self.phi_f)

        """ Lambda operator (phase mobility) """
        # phase mobility: k_rj [-] / mu_j [cP ∝ bar.day] (1/(bar.day))
        values_np[self.LAMBDA_OP + self.property.ph] = (
            self.property.kr[self.property.ph] / self.property.mu[self.property.ph]
        )

        """ Saturation operator for phase volumetric calculations in the wellbore """
        # phase saturation: s_j [-]
        values_np[self.SAT_OP + self.property.ph] = self.property.sat[self.property.ph]

        """ Pressure operator """
        # Pressure operator (for generic state specification where no pressure in the state, for instance V,T)
        values_np[self.PRES_OP] = state_np[0]

        if self.thermal:
            self.evaluate_thermal(state_np, values_np)

        # self.print_operators(state, values)

        return 0

    def evaluate_thermal(self, state, values):
        """
        Evaluate the thermal reservoir operators for the super engine

        :param state: state variables [pres, comp_0, ..., comp_N-1, temp]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        pressure = state[0]

        # Evaluate thermal properties at current state
        self.property.evaluate_thermal(state)

        """ Alpha operator represents accumulation term """
        # fluid enthalpy: phi_f[-] s_j [-] rho_mj [kmol/m3] H_j [kJ/kmol] (kJ/m3)
        values[self.ACC_OP + self.nc] += (
            self.compr
            * self.phi_f
            * np.sum(
                self.property.sat[self.property.ph]
                * self.property.dens_m[self.property.ph]
                * self.property.enthalpy[self.property.ph]
            )
        )  # fluid enthalpy (kJ/m3)
        # solid enthalpy: phi_s[-] s_j [-] rho_mj [kmol/m3] H_j [kJ/kmol] (kJ/m3)
        values[self.ACC_OP + self.nc] += (
            self.compr
            * self.phi_s
            * np.sum(
                self.property.sat[self.np_fl : self.np_fl + self.ns]
                * self.property.dens_m[self.np_fl : self.np_fl + self.ns]
                * self.property.enthalpy[self.np_fl : self.np_fl + self.ns]
            )
        )
        # Enthalpy to internal energy conversion
        values[self.ACC_OP + self.nc] -= self.compr * 100 * pressure

        """ Beta operator """
        # fluid convective energy flux: H_j [kJ/kmol] rho_mj [kmol/m3] (kJ/m3)
        values[self.FLUX_OP + self.property.ph * self.ne + self.nc] = (
            self.property.enthalpy[self.property.ph]
            * self.property.dens_m[self.property.ph]
        )

        """ Chi operator for temperature in conduction """
        # fluid/solid conductive flux: kappa_j [kJ/m.K.day] T [K] (kJ/m.day)
        values[self.GRAD_OP + self.property.ph * self.ne + self.nc] = (
            self.property.temperature * self.property.cond[self.property.ph]
        )

        """ Delta operator for reaction """
        # energy source: Q [kJ/m3/day] (kJ/m3/day)
        values[self.KIN_OP + self.nc] = self.property.energy_source

        """ Phase enthalpy operator """
        for j in range(self.nph):
            values[self.ENTH_OP + j] = self.property.enthalpy[j]

        """ Additional energy operators """
        # Temperature operator
        values[self.TEMP_OP] = self.property.temperature

        return 0


class WellOperators(ReservoirOperators):
    def evaluate(self, state, values):
        """
        Evaluate the non-thermal well operators for the super engine

        :param state: state variables [pres, comp_0, ..., comp_N-1, temp]
        :type state: value_vector in open-darts, pylvarray.Array in GEOS
        :param values: values of the operators (used for storing the operator values)
        :type state: value_vector in open-darts, pylvarray.Array in GEOS
        :return: updated value for operators, stored in values
        """
        # Check if extrapolation needs to be applied
        if self.apply_extrapolation(state, values):
            return 0

        # Composition vector and pressure from state:
        state_np = state.to_numpy()
        values_np = values.to_numpy()
        values_np[:] = 0

        # Evaluate properties at current state, reusing tabulated flash results
        self.evaluate_property_container(state_np)

        density_tot = np.sum(
            self.property.sat[: self.np_fl] * self.property.dens_m[: self.np_fl]
        )
        zc = np.append(state_np[1 : self.nc], 1 - np.sum(state_np[1 : self.nc]))
        self.phi_s = np.sum(zc[self.nc_fl :])
        self.phi_f = 1.0 - self.phi_s

        """ CONSTRUCT OPERATORS HERE """

        """ Alpha operator represents accumulation term """
        # fluid mass accumulation: z_c* [-] rho_m^T [kmol/m3]
        values_np[self.ACC_OP : self.ACC_OP + self.nc_fl] = (
            density_tot * zc[: self.nc_fl]
        )

        """ and alpha for mineral components """
        # solid mass accumulation: z_s* [-] rho_ms [kmol/m3]
        values_np[self.ACC_OP + self.nc_fl : self.ACC_OP + self.nc_fl + self.ns] = (
            self.property.dens_m[self.np_fl : self.np_fl + self.ns]
            * zc[self.nc_fl : self.nc_fl + self.ns]
        )

        """ Beta operator """
        for j in self.property.ph:
            # fluid convective mass flux: x_cj [-] rho_mj [kmol/m3] (kmol/m3)
            values_np[
                self.FLUX_OP + j * self.ne : self.FLUX_OP + j * self.ne + self.nc_fl
            ] = self.property.x[j][: self.nc_fl] * self.property.dens_m[j]

        """ Molar density operator """

        """ Gamma operator for diffusion (for heat conduction and molecular diffusion) """

        """ Chi operator for diffusion """

        """ Delta operator for reaction """
        # fluid/solid mass source: n_c [kmol/m3/day] (kmol/m3/day)
        values_np[self.KIN_OP : self.KIN_OP + self.nc] = self.property.mass_source

        """ Gravity and capillarity operators """
        # E3-> gravity
        values_np[self.GRAV_OP + self.property.ph] = self.property.dens[
            self.property.ph
        ]

        # E4-> capillarity

        """ Permeability multiplier k/kmax """
        # E5_> permeability multiplier due to permporo relationship
        values_np[self.MULT_OP] = 1.0

        """ Lambda operator (phase mobility) """
        # phase mobility: k_rj [-] / mu_j [cP ∝ bar.day] (1/(bar.day))
        values_np[self.LAMBDA_OP + self.property.ph] = (
            self.property.kr[self.property.ph] / self.property.mu[self.property.ph]
        )

        """ Saturation operator for phase volumetric calculations in the wellbore """
        # phase saturation: s_j [-]
        values_np[self.SAT_OP + self.property.ph] = self.property.sat[self.property.ph]

        """ Pressure operator """
        # Pressure operator (for generic state specification where no pressure in the state, for instance V,T)
        values_np[self.PRES_OP] = state_np[0]

        if self.thermal:
            self.evaluate_thermal(state_np, values_np)

        # self.print_operators(state, values)

        return 0

    def evaluate_thermal(self, state, values):
        """
        Evaluate the thermal well operators for the super engine

        :param state: state variables [pres, comp_0, ..., comp_N-1, temp]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        pressure = state[0]

        # Evaluate thermal properties at current state
        self.property.evaluate_thermal(state)

        """ Alpha operator represents accumulation term """
        # fluid enthalpy: s_j [-] rho_mj [kmol/m3] H_j [kJ/kmol] (kJ/m3)
        values[self.ACC_OP + self.nc] += self.phi_f * np.sum(
            self.property.sat[self.property.ph]
            * self.property.dens_m[self.property.ph]
            * self.property.enthalpy[self.property.ph]
        )  # fluid enthalpy (kJ/m3)
        # solid enthalpy: s_j [-] rho_mj [kmol/m3] H_j [kJ/kmol] (kJ/m3)
        values[self.ACC_OP + self.nc] += self.phi_s * np.sum(
            self.property.sat[self.np_fl : self.np_fl + self.ns]
            * self.property.dens_m[self.np_fl : self.np_fl + self.ns]
            * self.property.enthalpy[self.np_fl : self.np_fl + self.ns]
        )

        # Enthalpy to internal energy conversion
        values[self.ACC_OP + self.nc] -= 100 * pressure

        """ Beta operator """
        # fluid convective energy flux: H_j [kJ/kmol] rho_mj [kmol/m3] (kJ/m3)
        values[self.FLUX_OP + self.property.ph * self.ne + self.nc] = (
            self.property.enthalpy[self.property.ph]
            * self.property.dens_m[self.property.ph]
        )

        """ Chi operator for temperature in conduction """
        # fluid/solid conductive flux: kappa_j [kJ/m.K.day] T [K] (kJ/m.day)
        values[self.GRAD_OP + self.property.ph * self.ne + self.nc] = (
            self.property.temperature * self.property.cond[self.property.ph]
        )

        """ Delta operator for reaction """
        # energy source: V [m3] dt [day] c_r phi^T Q [kJ/m3.days] (kJ/m3)
        values[self.KIN_OP + self.nc] = self.property.energy_source

        """ Phase enthalpy operator """
        for j in range(self.nph):
            values[self.ENTH_OP + j] = self.property.enthalpy[j]

        """ Additional energy operators """
        # Temperature operator
        values[self.TEMP_OP] = self.property.temperature

        return 0


class GeomechanicsReservoirOperators(ReservoirOperators):
    def __init__(
        self,
        property_container: PropertyContainer,
        thermal: bool,
        extrapolation_flag: bool = True,
        dz: float = None,
        flash_operators: FlashOperators = None,
    ):
        """
        Constructor of GeomechanicsReservoirOperators class

        :param property_container: Property container of type PropertyContainer
        :param thermal: Switch to indicate if energy conservation equation is there
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[last component] < 0 in case nc >= 3)
        :param dz: Composition interval along OBL composition axes to obtain consistent points for extrapolation
                    (must be equal along all composition axes in current setup)
        :param flash_operators: Shared :class:`FlashOperators` of this property region
        :type flash_operators: FlashOperators, optional
        """
        super().__init__(
            property_container,
            thermal,
            extrapolation_flag,
            dz,
            flash_operators=flash_operators,
        )  # Initialize base-class

        self.ROCK_DENS_OP = self.PRES_OP + 1  # used only in mechanical engine
        self.n_ops = self.ROCK_DENS_OP + 1

    def evaluate(self, state, values):
        """
        Evaluate the geomechanical reservoir operators for the super engine

        :param state: state variables [pres, comp_0, ..., comp_N-1, temp]
        :type state: value_vector in open-darts, pylvarray.Array in GEOS
        :param values: values of the operators (used for storing the operator values)
        :type values: value_vector in open-darts, pylvarray.Array in GEOS
        :return: updated value for operators, stored in values
        """
        # Reservoir operators
        super().evaluate(state, values)

        # Rock density operator
        self.n_ops = self.ROCK_DENS_OP + 1
        # TODO: function of matrix pressure = I1 / 3 = (s_xx + s_yy + s_zz) / 3
        values.to_numpy()[self.ROCK_DENS_OP] = self.property.rock_density_ev.evaluate()

        return 0

    def print_operators(self, state, values):
        """Method for printing operators, grouped"""
        self.print_operators(state, values)
        print("ROCK DENSITY", values[self.ROCK_DENS_OP])
        return


class SinglePhaseGeomechanicsOperators(OperatorsBase):
    def evaluate(self, state, values):
        """
        Evaluate the single-phase geomechanical operators for the super engine

        :param state: state variables [pres, comp_0, ..., comp_N-1, temp]
        :type state: value_vector in open-darts, pylvarray.Array in GEOS
        :param values: values of the operators (used for storing the operator values)
        :type values: value_vector in open-darts, pylvarray.Array in GEOS
        :return: updated value for operators, stored in values
        """
        # Check if extrapolation needs to be applied
        if self.apply_extrapolation(state, values):
            return 0

        state_np = state.to_numpy()
        values_np = values.to_numpy()
        self.evaluate_property_container(state_np)
        values_np[0] = self.property.dens[0]
        values_np[1] = self.property.dens[0] / self.property.mu[0]

        return 0
