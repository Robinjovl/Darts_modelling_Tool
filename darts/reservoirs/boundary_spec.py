"""Declarative boundary-condition spec for the mech / MPFA discretizer family.

The Robin form of a boundary condition, ``a * u + b * f = r``, used to be
spelled out four times in the code base: in the ``bound_cond`` vocabulary dicts
(:mod:`darts.reservoirs.unstruct_reservoir_mech`), in the ``a``/``b`` arrays
packed for the C++ ``mech_discretizer`` (``dis::BoundaryCondition`` /
``dis::THMBoundaryCondition``), in the ``6 x 1`` ``[an, bn, at, bt, a, b]``
matrices packed for the ``pm_discretizer``, and once more in the pure-Python
MPFA discretizer.  This module holds ONE typed description of a boundary and a
compiler per backend, so a new backend or a new boundary type is written once.

Layout
------

* :class:`FaceBC` -- the ``(a, b)`` TYPE of one Robin channel.  ``r`` is not
  part of it: types are discretization input (they shape the gradient
  reconstruction and are fixed for the run) while values are per-timestep data.
* :class:`ScalarBC` / :class:`MechBC` -- one channel WITH its value, i.e. the
  legacy ``{'a', 'b', 'r'}`` and ``{'an', 'bn', 'rn', 'at', 'bt', 'rt'}``
  dicts; the vocabulary constructors (:func:`no_flow`, :func:`aquifer`,
  :func:`roller`, :func:`stuck`, :func:`load`, :func:`free`, :func:`robin`,
  :func:`dirichlet`, :func:`neumann`, ...) return these and mirror
  ``bound_cond`` exactly.
* :class:`BoundaryFacetSpec` -- the four TYPE channels of one boundary tag
  (flow / thermal / mech_normal / mech_tangential); :class:`BoundaryTypeSpec`
  maps ``tag -> BoundaryFacetSpec``.
* :class:`BoundaryValues` / :class:`BoundaryValueSpec` -- the matching ``r``
  side, constant or ``callable(t)``.
* :class:`BoundarySpec` -- the pair, with
  :meth:`BoundarySpec.from_legacy_dict` ingesting the EXISTING
  ``boundary_conditions`` dicts verbatim and splitting them into the two.

Compilers (pure functions of the spec, no model needed):

* :func:`compile_mech_discretizer` -- ``THMBoundaryCondition`` +
  ``BoundaryCondition`` objects for the standalone C++ discretizer,
* :func:`compile_pm_discretizer` -- the ``6 x 1`` matrices for
  ``pm_discretizer`` (rejects the thermal facet: thermoporoelasticity is not
  supported on that path).

The pure-Python MPFA copy in ``darts.reservoirs.mesh.unstruct_discretizer``
(``a``/``b`` inlined in ``calc_mpfa_connections_all_cells``) is NOT wired
through this module yet -- it is the remaining fourth encoding.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from darts.models.conditions import ConditionItem

__all__ = [
    'FaceBC',
    'ScalarBC',
    'MechBC',
    'BoundaryFacetSpec',
    'BoundaryValues',
    'BoundaryTypeSpec',
    'BoundaryValueSpec',
    'BoundarySpec',
    'BoundaryValueBC',
    'robin',
    'dirichlet',
    'neumann',
    'no_flow',
    'aquifer',
    'flow_rate',
    'roller',
    'free',
    'stuck',
    'stuck_roller',
    'load',
    'stuck_t_load_n',
    'mech_discretizer_arrays',
    'compile_mech_discretizer',
    'pm_discretizer_row',
    'compile_pm_discretizer',
]

# defaults of the legacy packing for boundary elements not covered by any tag
# (``ap = np.ones(n_bounds)``, everything else ``np.zeros(n_bounds)``)
DEFAULT_FLOW_FACE_AB = (1.0, 0.0)
DEFAULT_OTHER_FACE_AB = (0.0, 0.0)


def _value(v, t: float = 0.0):
    """Resolve a constant-or-``callable(t)`` boundary value."""
    return v(t) if callable(v) else v


@dataclass(frozen=True)
class FaceBC:
    """TYPE of one Robin channel of one facet: ``a * u + b * f = r``.

    ``a = 1, b = 0`` is Dirichlet (prescribed pressure / temperature /
    displacement), ``a = 0, b = 1`` is Neumann (prescribed flux / load).
    The right-hand side ``r`` deliberately lives in :class:`BoundaryValues`.
    """

    a: float
    b: float

    @property
    def is_dirichlet(self) -> bool:
        return self.a != 0.0 and self.b == 0.0

    @property
    def is_neumann(self) -> bool:
        return self.a == 0.0 and self.b != 0.0


@dataclass(frozen=True)
class ScalarBC:
    """A scalar channel (flow or thermal) with its value: legacy ``{a, b, r}``."""

    face: FaceBC
    r: object = 0.0

    @property
    def a(self) -> float:
        return self.face.a

    @property
    def b(self) -> float:
        return self.face.b

    def to_legacy(self) -> dict:
        return {'a': self.face.a, 'b': self.face.b, 'r': self.r}

    @classmethod
    def from_legacy(cls, d: Mapping) -> ScalarBC:
        return cls(FaceBC(d['a'], d['b']), d.get('r', 0.0))


@dataclass(frozen=True)
class MechBC:
    """The mechanical channel with its values: legacy ``{an, bn, rn, at, bt, rt}``.

    ``normal``/``rn`` act along the facet normal, ``tangential``/``rt`` in the
    facet plane (``rt`` is a 3-vector).
    """

    normal: FaceBC
    tangential: FaceBC
    rn: object = 0.0
    rt: object = (0.0, 0.0, 0.0)

    def to_legacy(self) -> dict:
        return {
            'an': self.normal.a,
            'bn': self.normal.b,
            'rn': self.rn,
            'at': self.tangential.a,
            'bt': self.tangential.b,
            'rt': np.array(self.rt),
        }

    @classmethod
    def from_legacy(cls, d: Mapping) -> MechBC:
        return cls(
            normal=FaceBC(d['an'], d['bn']),
            tangential=FaceBC(d['at'], d['bt']),
            rn=d.get('rn', 0.0),
            rt=d.get('rt', (0.0, 0.0, 0.0)),
        )


# --------------------------------------------------------------------------
# vocabulary -- one-to-one with ``bound_cond`` in unstruct_reservoir_mech.py
# --------------------------------------------------------------------------


def robin(a: float, b: float, r=0.0) -> ScalarBC:
    """General scalar Robin channel ``a * u + b * f = r``."""
    return ScalarBC(FaceBC(float(a), float(b)), r)


def dirichlet(r=0.0) -> ScalarBC:
    """Prescribed value ``u = r``."""
    return ScalarBC(FaceBC(1.0, 0.0), r)


def neumann(r=0.0) -> ScalarBC:
    """Prescribed flux ``f = r``."""
    return ScalarBC(FaceBC(0.0, 1.0), r)


def no_flow() -> ScalarBC:
    """``bound_cond.NO_FLOW`` -- zero flux."""
    return ScalarBC(FaceBC(0.0, 1.0), 0.0)


def aquifer(p) -> ScalarBC:
    """``bound_cond.AQUIFER`` -- prescribed pressure / temperature."""
    return ScalarBC(FaceBC(1.0, 0.0), p)


def flow_rate(q) -> ScalarBC:
    """``bound_cond.FLOW`` -- prescribed flow rate."""
    return ScalarBC(FaceBC(0.0, 1.0), q)


def roller() -> MechBC:
    """``bound_cond.ROLLER`` -- ``un = 0``, ``st = 0``."""
    return MechBC(FaceBC(1.0, 0.0), FaceBC(0.0, 1.0), 0.0, np.array([0, 0, 0]))


def free() -> MechBC:
    """``bound_cond.FREE`` -- ``sn = 0``, ``st = 0``."""
    return MechBC(FaceBC(0.0, 1.0), FaceBC(0.0, 1.0), 0.0, np.array([0, 0, 0]))


def stuck(un, ut) -> MechBC:
    """``bound_cond.STUCK`` -- ``un = Un``, ``ut = Ut``."""
    return MechBC(FaceBC(1.0, 0.0), FaceBC(1.0, 0.0), un, np.array(ut))


def stuck_roller(un) -> MechBC:
    """``bound_cond.STUCK_ROLLER`` -- ``un = Un``, ``st = 0``."""
    return MechBC(FaceBC(1.0, 0.0), FaceBC(0.0, 1.0), un, np.array([0.0, 0.0, 0.0]))


def load(Fn, Ft) -> MechBC:
    """``bound_cond.LOAD`` -- ``sn = Fn``, ``st = Ft``."""
    return MechBC(FaceBC(0.0, 1.0), FaceBC(0.0, 1.0), Fn, np.array(Ft))


def stuck_t_load_n(Fn, ut) -> MechBC:
    """``bound_cond.STUCK_T_LOAD_N`` -- ``sn = Fn``, ``ut = Ut``."""
    return MechBC(FaceBC(0.0, 1.0), FaceBC(1.0, 0.0), Fn, np.array(ut))


# --------------------------------------------------------------------------
# per-tag specs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryFacetSpec:
    """TYPE (``a``, ``b``) of every channel of one boundary tag."""

    flow: FaceBC
    mech_normal: FaceBC
    mech_tangential: FaceBC
    thermal: FaceBC | None = None


@dataclass(frozen=True)
class BoundaryValues:
    """VALUE (``r``) of every channel of one boundary tag, constant or ``f(t)``."""

    flow: object = 0.0
    mech_normal: object = 0.0
    mech_tangential: object = (0.0, 0.0, 0.0)
    thermal: object = None

    def evaluate(self, t: float = 0.0) -> BoundaryValues:
        """Resolve every callable channel at time ``t``."""
        return BoundaryValues(
            flow=_value(self.flow, t),
            mech_normal=_value(self.mech_normal, t),
            mech_tangential=_value(self.mech_tangential, t),
            thermal=None if self.thermal is None else _value(self.thermal, t),
        )


class _TagMapping(Mapping):
    """Immutable ``tag -> entry`` mapping shared by the type/value specs."""

    __slots__ = ('_entries',)

    def __init__(self, entries: Mapping):
        self._entries = dict(entries)

    def __getitem__(self, tag):
        return self._entries[tag]

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)

    def __repr__(self):
        return f'{type(self).__name__}({self._entries!r})'


class BoundaryTypeSpec(_TagMapping):
    """``{tag: BoundaryFacetSpec}`` -- the discretization-time half of a spec."""

    @property
    def thermal(self) -> bool:
        """True when any tag carries a thermal facet."""
        return any(f.thermal is not None for f in self._entries.values())


class BoundaryValueSpec(_TagMapping):
    """``{tag: BoundaryValues}`` -- the per-timestep half of a spec."""


@dataclass(frozen=True)
class BoundarySpec:
    """A boundary description split into its TYPE and VALUE halves."""

    types: BoundaryTypeSpec
    values: BoundaryValueSpec

    @property
    def thermal(self) -> bool:
        return self.types.thermal

    @property
    def tags(self):
        return list(self.types)

    @classmethod
    def from_legacy_dict(
        cls, boundary_conditions: Mapping, thermal: bool | None = None
    ) -> BoundarySpec:
        """Ingest the existing ``{tag: {'flow': ..., 'mech': ..., 'temp': ...}}``
        dicts verbatim and split them into a type spec and a value spec.

        Extra keys (e.g. the ``'cells'`` list added by
        ``set_boundary_conditions_pm_discretizer``) are ignored.

        :param thermal: ``None`` (default) takes the thermal facet from the
            presence of a ``'temp'`` entry; ``False`` drops it even when
            present (models hand the same dicts to a poroelastic and to a
            thermoporoelastic run); ``True`` requires it.
        """
        types, values = {}, {}
        for tag, entry in boundary_conditions.items():
            for channel in ('flow', 'mech'):
                if channel not in entry:
                    raise KeyError(
                        f"boundary tag {tag!r} has no '{channel}' condition; "
                        "the mech/MPFA boundary spec needs both 'flow' and "
                        "'mech' (and optionally 'temp')."
                    )
            if thermal and 'temp' not in entry:
                raise KeyError(
                    f"boundary tag {tag!r} has no 'temp' condition, which a "
                    'thermoporoelastic run requires.'
                )
            flow = ScalarBC.from_legacy(entry['flow'])
            mech = MechBC.from_legacy(entry['mech'])
            use_temp = 'temp' in entry if thermal is None else bool(thermal)
            temp = ScalarBC.from_legacy(entry['temp']) if use_temp else None
            types[tag] = BoundaryFacetSpec(
                flow=flow.face,
                mech_normal=mech.normal,
                mech_tangential=mech.tangential,
                thermal=None if temp is None else temp.face,
            )
            values[tag] = BoundaryValues(
                flow=flow.r,
                mech_normal=mech.rn,
                mech_tangential=mech.rt,
                thermal=None if temp is None else temp.r,
            )
        return cls(BoundaryTypeSpec(types), BoundaryValueSpec(values))

    def to_legacy_dict(self) -> dict:
        """Rebuild the legacy ``boundary_conditions`` dicts (canonical keys only)."""
        out = {}
        for tag, facets in self.types.items():
            vals = self.values[tag]
            entry = {
                'flow': ScalarBC(facets.flow, vals.flow).to_legacy(),
                'mech': MechBC(
                    facets.mech_normal,
                    facets.mech_tangential,
                    vals.mech_normal,
                    vals.mech_tangential,
                ).to_legacy(),
            }
            if facets.thermal is not None:
                entry['temp'] = ScalarBC(facets.thermal, vals.thermal).to_legacy()
            out[tag] = entry
        return out


# --------------------------------------------------------------------------
# backend compilers
# --------------------------------------------------------------------------


def mech_discretizer_arrays(
    types: BoundaryTypeSpec, tag_ids: Mapping, n_bounds: int
) -> dict:
    """Scatter the per-tag ``(a, b)`` types onto per-boundary-element arrays.

    :param tag_ids: ``{tag: integer array of boundary-element indices}``.
    :param n_bounds: number of boundary elements.
    :returns: ``{'flow_a', 'flow_b', 'mech_normal_a', 'mech_normal_b',
        'mech_tangen_a', 'mech_tangen_b'}`` plus ``'thermal_a'``/``'thermal_b'``
        when the spec has a thermal facet.  Elements not covered by any tag keep
        the legacy defaults (flow ``a = 1``, everything else ``0``).
    """
    thermal = types.thermal
    out = {
        'flow_a': np.full(n_bounds, DEFAULT_FLOW_FACE_AB[0]),
        'flow_b': np.full(n_bounds, DEFAULT_FLOW_FACE_AB[1]),
        'mech_normal_a': np.zeros(n_bounds),
        'mech_normal_b': np.zeros(n_bounds),
        'mech_tangen_a': np.zeros(n_bounds),
        'mech_tangen_b': np.zeros(n_bounds),
    }
    if thermal:
        out['thermal_a'] = np.zeros(n_bounds)
        out['thermal_b'] = np.zeros(n_bounds)
    for tag, ids in tag_ids.items():
        facets = types[tag]
        ids = np.asarray(ids, dtype=np.int64)
        out['flow_a'][ids] = facets.flow.a
        out['flow_b'][ids] = facets.flow.b
        out['mech_normal_a'][ids] = facets.mech_normal.a
        out['mech_normal_b'][ids] = facets.mech_normal.b
        out['mech_tangen_a'][ids] = facets.mech_tangential.a
        out['mech_tangen_b'][ids] = facets.mech_tangential.b
        if thermal:
            if facets.thermal is None:
                raise ValueError(
                    f'boundary tag {tag!r} has no thermal facet while other '
                    'tags do; a thermoporoelastic run needs one per tag.'
                )
            out['thermal_a'][ids] = facets.thermal.a
            out['thermal_b'][ids] = facets.thermal.b
    return out


def compile_mech_discretizer(types: BoundaryTypeSpec, tag_ids: Mapping, n_bounds: int):
    """Compile the spec into the C++ ``mech_discretizer`` boundary objects.

    :returns: ``(thm_bc, flow_bc, heat_bc)`` where ``thm_bc`` is the
        ``THMBoundaryCondition`` consumed by
        ``reconstruct_displacement_gradients_per_cell`` and ``flow_bc`` /
        ``heat_bc`` are the plain ``BoundaryCondition`` objects consumed by
        ``reconstruct_pressure(_temperature)_gradients_per_cell``.  ``heat_bc``
        is ``None`` for a non-thermal spec.
    """
    from darts.discretizer import BoundaryCondition, THMBoundaryCondition, value_vector

    arrays = mech_discretizer_arrays(types, tag_ids, n_bounds)

    thm_bc = THMBoundaryCondition()
    thm_bc.flow.a = value_vector(arrays['flow_a'])
    thm_bc.flow.b = value_vector(arrays['flow_b'])
    thm_bc.mech_normal.a = value_vector(arrays['mech_normal_a'])
    thm_bc.mech_normal.b = value_vector(arrays['mech_normal_b'])
    thm_bc.mech_tangen.a = value_vector(arrays['mech_tangen_a'])
    thm_bc.mech_tangen.b = value_vector(arrays['mech_tangen_b'])

    # the base discretizer's reconstruct_pressure_gradients_per_cell() does not
    # know THMBoundaryCondition, so the scalar channels are handed over as
    # plain BoundaryCondition objects too
    flow_bc = BoundaryCondition()
    flow_bc.a = value_vector(arrays['flow_a'])
    flow_bc.b = value_vector(arrays['flow_b'])

    heat_bc = None
    if types.thermal:
        thm_bc.thermal.a = value_vector(arrays['thermal_a'])
        thm_bc.thermal.b = value_vector(arrays['thermal_b'])
        heat_bc = BoundaryCondition()
        heat_bc.a = value_vector(arrays['thermal_a'])
        heat_bc.b = value_vector(arrays['thermal_b'])

    return thm_bc, flow_bc, heat_bc


def pm_discretizer_row(facets: BoundaryFacetSpec) -> list:
    """The ``pm_discretizer`` ``6 x 1`` packing ``[an, bn, at, bt, a, b]``."""
    if facets.thermal is not None:
        raise AssertionError('thermoporoelasticity is not supported in pm_discretizer')
    return [
        facets.mech_normal.a,
        facets.mech_normal.b,
        facets.mech_tangential.a,
        facets.mech_tangential.b,
        facets.flow.a,
        facets.flow.b,
    ]


def compile_pm_discretizer(types: BoundaryTypeSpec, prop_ids) -> list:
    """Compile the spec into one ``pm_discretizer`` ``6 x 1`` matrix per face.

    :param prop_ids: boundary tag of every boundary face, in face order.
    :returns: list of ``darts.engines.matrix`` objects to append to ``pm.bc``.
    """
    from darts.engines import matrix

    if types.thermal:
        raise AssertionError('thermoporoelasticity is not supported in pm_discretizer')
    rows = {tag: pm_discretizer_row(facets) for tag, facets in types.items()}
    return [matrix(rows[tag], 6, 1) for tag in prop_ids]


# --------------------------------------------------------------------------
# value plumbing
# --------------------------------------------------------------------------


class BoundaryValueBC(ConditionItem):
    """The VALUE half of a boundary spec, owned and driven by a mech reservoir.

    Where the compilers above own the ``(a, b)`` TYPE half (fixed at
    discretization time), this item owns the ``r`` half: the ``bc_rhs`` fill
    that ``init_bc_rhs`` used to inline, the zero-copy view writes into
    ``mesh.bc`` / ``mesh.bc_prev`` that ``update_trans`` does, the ``n -> n+1``
    rotation that ``update`` does, and the ``bc`` vs ``pz_bounds`` Dirichlet
    coherence that the (commented-out) assert in ``mech_operators.cpp`` used to
    check.

    It is a :class:`~darts.models.conditions.ConditionItem` by type, but a
    VALUE-ARRAY item: it writes ``conn_mesh`` arrays that C++ consumes BEFORE
    equation scaling, never ``ctx.rhs`` / ``ctx.jac``, so :meth:`apply` is a
    no-op.  ``ConditionSet.compile`` currently rejects every item on a
    mechanics model (guard iii, "rows are rescaled inside assembly"), which is
    correct for residual items and wrong for this one -- hence the reservoir
    owns and drives it directly from ``init_bc_rhs`` / ``update_trans`` /
    ``update`` instead of being registered on ``model.conditions``.
    :attr:`writes_mesh_arrays` marks the exemption that guard should honour
    when the item moves onto ``model.conditions`` in M3.
    """

    #: marks a value-array item: writes mesh arrays, never the scaled residual
    writes_mesh_arrays = True
    provides_jacobian = False
    adjoint_transparent = True

    def __init__(self, reservoir, strict_dirichlet: bool = False):
        self.reservoir = reservoir
        self.spec = BoundarySpec(BoundaryTypeSpec({}), BoundaryValueSpec({}))
        #: raise instead of warning when a Dirichlet bc and pz_bounds disagree
        self.strict_dirichlet = strict_dirichlet
        self.dirichlet_tolerance = 1.0e-6
        self._tag_ids = {}
        self._warned = set()

    # -- spec ------------------------------------------------------------
    @property
    def types(self) -> BoundaryTypeSpec:
        return self.spec.types

    @property
    def values(self) -> BoundaryValueSpec:
        return self.spec.values

    def sync(self, boundary_conditions=None) -> BoundaryValueBC:
        """(Re)read the legacy ``boundary_conditions`` dicts into the spec.

        Called on every ``init_bc_rhs`` because models mutate the dicts in
        place at runtime (e.g. the Mandel north-boundary displacement).
        """
        if boundary_conditions is None:
            boundary_conditions = self.reservoir.boundary_conditions
        self.spec = BoundarySpec.from_legacy_dict(
            boundary_conditions, thermal=bool(self.reservoir.thermoporoelasticity)
        )
        return self

    # -- value writes ----------------------------------------------------
    def write_mech_discretizer(self, tag_ids: Mapping, normals, t: float = 0.0):
        """Fill ``reservoir.bc_rhs`` in the ``mech_discretizer`` layout.

        :param tag_ids: ``{tag: boundary-element indices}``.
        :param normals: ``(n_bounds, 3)`` outward facet normals (only the rows
            listed in ``tag_ids`` are read).
        """
        self._write_bc_rhs(tag_ids, normals, t)

    def write_pm_discretizer(self, prop_ids, normals, t: float = 0.0):
        """Fill ``reservoir.bc_rhs`` in the ``pm_discretizer`` layout.

        :param prop_ids: boundary tag of every boundary face, in face order
            (the pm path indexes boundaries by face, not by tag).
        """
        tag_ids = {}
        for face_id, tag in enumerate(prop_ids):
            tag_ids.setdefault(tag, []).append(face_id)
        self._write_bc_rhs(tag_ids, normals, t)

    def _write_bc_rhs(self, tag_ids: Mapping, normals, t: float):
        """Batched per-tag scatter of the values into ``reservoir.bc_rhs``.

        The two layouts differ only in where ``p_bc_var``/``u_bc_var`` sit and
        in whether there is a thermal channel, both of which the reservoir
        already carries.
        """
        res = self.reservoir
        bc_rhs = res.bc_rhs
        n_bc_vars = res.n_bc_vars
        normals = np.asarray(normals, dtype=float)
        self._tag_ids = {
            tag: np.asarray(ids, dtype=np.int64) for tag, ids in tag_ids.items()
        }
        for tag, ids in self._tag_ids.items():
            if not ids.size:
                continue
            vals = self.values[tag].evaluate(t)
            bc_rhs[n_bc_vars * ids + res.p_bc_var] = vals.flow
            if res.thermoporoelasticity:
                if vals.thermal is None:
                    raise KeyError(
                        f"boundary tag {tag!r} has no 'temp' condition, which a "
                        'thermoporoelastic run requires.'
                    )
                bc_rhs[n_bc_vars * ids + res.t_bc_var] = vals.thermal
            traction = vals.mech_normal * normals[ids] + np.asarray(
                vals.mech_tangential, dtype=float
            )
            for k in range(res.n_dim):
                bc_rhs[n_bc_vars * ids + res.u_bc_var + k] = traction[:, k]

    # -- lifecycle -------------------------------------------------------
    def push(self, check: bool = True):
        """Zero-copy view writes of ``bc_rhs`` / ``bc_rhs_prev`` into the mesh.

        This is the ``n+1`` / ``n`` pair the C++ engine reads; the matching
        ``tran_biot_n`` rotation is done by C++ in ``post_newtonloop``, so the
        two halves of the time lag stay split between the languages (S8).
        """
        res = self.reservoir
        if res.discretizer_name == 'pm_discretizer':
            # mesh.bc also holds the fracture boundaries past the matrix ones
            n = res.n_bc_vars * res.unstr_discr.bound_faces_tot
            res.bc[:n] = res.bc_rhs
            res.bc_prev[:n] = res.bc_rhs_prev
        else:
            res.bc[:] = res.bc_rhs
            res.bc_prev[:] = res.bc_rhs_prev
        if check:
            self.check_dirichlet_coherence()

    def on_timestep_start(self, dt: float, t: float):
        """Rotate ``n+1 -> n``: the values pushed last step become the old ones."""
        res = self.reservoir
        res.bc_rhs_prev = np.copy(res.bc_rhs)
        if res.discretizer_name == 'pm_discretizer' and hasattr(res, 'pm'):
            res.pm.bc_prev = res.pm.bc

    def apply(self, ctx):
        """No-op: this item writes mesh arrays, never the assembled residual."""
        return

    # -- Dirichlet coherence (S9) ---------------------------------------
    def check_dirichlet_coherence(self) -> bool:
        """Check that Dirichlet ``bc_rhs`` values match ``mesh.pz_bounds``.

        The engine reads a prescribed boundary pressure / temperature twice:
        from ``mesh.bc`` (flux reconstruction) and from ``mesh.pz_bounds``
        (boundary property evaluation).  Nothing keeps the two in sync -- the
        assert that used to catch a mismatch is commented out in
        ``engines/src/mech/mech_operators.cpp``.  This restores it on the
        Python side; set :attr:`strict_dirichlet` to raise instead of warn.
        """
        res = self.reservoir
        bc_rhs = getattr(res, 'bc_rhs', None)
        pz_bounds = getattr(res, 'pz_bounds', None)
        if bc_rhs is None or pz_bounds is None or not len(pz_bounds):
            return True
        ok = True
        for tag, ids in self._tag_ids.items():
            if not ids.size:
                continue
            facets = self.types[tag]
            channels = [('pressure', facets.flow, res.p_bc_var, res.p_var)]
            if res.thermoporoelasticity and facets.thermal is not None:
                channels.append(
                    ('temperature', facets.thermal, res.t_bc_var, res.t_var)
                )
            for name, face, bc_var, state_var in channels:
                if not face.is_dirichlet:
                    continue
                pz_idx = self._pz_index(ids, state_var)
                if pz_idx is None or pz_idx.max() >= len(pz_bounds):
                    continue
                prescribed = bc_rhs[res.n_bc_vars * ids + bc_var] / face.a
                dev = np.max(np.abs(prescribed - pz_bounds[pz_idx]))
                if dev > self.dirichlet_tolerance:
                    ok = False
                    self._report_incoherence(tag, name, dev)
        return ok

    def _pz_index(self, ids, state_var):
        res = self.reservoir
        if res.discretizer_name == 'pm_discretizer':
            # pz_bounds holds one pressure per boundary face
            return ids
        if state_var is None:
            return None
        return res.n_state * ids + state_var

    def _report_incoherence(self, tag, channel, dev):
        key = (tag, channel)
        message = (
            f'Dirichlet boundary {channel} of tag {tag!r} differs between '
            f'mesh.bc and mesh.pz_bounds by up to {dev:.3e}. The engine reads '
            'the prescribed value from both; set them consistently (see the '
            'commented-out assert in engines/src/mech/mech_operators.cpp).'
        )
        if self.strict_dirichlet:
            raise AssertionError(message)
        if key not in self._warned:
            self._warned.add(key)
            warnings.warn(message, stacklevel=3)
