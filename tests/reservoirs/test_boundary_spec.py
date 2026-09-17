"""Unit tests for the declarative boundary spec (``darts.reservoirs.boundary_spec``).

Everything here is pure Python: the spec, the vocabulary and the array-building
half of the compilers need neither a model nor the compiled extensions.

* the vocabulary constructors reproduce the exact ``a``/``b``/``r`` triplets the
  deprecated ``bound_cond`` class hands out (compared against that class);
* :class:`FaceBoundary` is the declaration a model writes; it splits into the
  type/value halves and round-trips through the deprecated dict form;
* :meth:`BoundarySpec.from_boundaries` splits a representative THM declaration
  into a type spec + value spec, and :meth:`BoundarySpec.from_legacy_dict`
  produces the identical spec from the dicts it deprecates (warning once);
* :func:`mech_discretizer_arrays` scatters the per-tag types onto
  per-boundary-element arrays exactly like the packing it replaced, including
  the legacy defaults for elements no tag covers;
* :func:`pm_discretizer_row` produces the ``[an, bn, at, bt, a, b]`` packing and
  :func:`compile_pm_discretizer` rejects a thermal spec (thermoporoelasticity is
  unsupported on that path);
* :class:`BoundaryValueBC` writes the same ``bc_rhs`` the per-id loop in
  ``init_bc_rhs`` used to, in both layouts, rotates ``n+1 -> n`` and flags an
  incoherent Dirichlet ``bc`` / ``pz_bounds`` pair -- all against a stub
  reservoir carrying only the attributes the item reads.
"""

import warnings

import numpy as np
import pytest

import darts.reservoirs.boundary_spec as boundary_spec
from darts.reservoirs.boundary_spec import (
    BoundaryFacetSpec,
    BoundarySpec,
    BoundaryTypeSpec,
    BoundaryValues,
    BoundaryValueSpec,
    FaceBC,
    FaceBoundary,
    aquifer,
    compile_pm_discretizer,
    dirichlet,
    flow_rate,
    free,
    load,
    mech_discretizer_arrays,
    neumann,
    no_flow,
    pm_discretizer_row,
    robin,
    roller,
    stuck,
    stuck_roller,
    stuck_t_load_n,
)

# the legacy vocabulary this module must reproduce bit for bit
from darts.reservoirs.unstruct_reservoir_mech import bound_cond

X_MINUS, X_PLUS, Y_MINUS = 991, 992, 993


def assert_legacy_equal(produced, expected):
    """Compare a ``to_legacy()`` dict against a ``bound_cond`` entry."""
    assert set(produced) == set(expected)
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(np.asarray(produced[key]), value)
        else:
            assert produced[key] == value


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------


def test_scalar_vocabulary_matches_bound_cond():
    legacy = bound_cond()
    assert_legacy_equal(no_flow().to_legacy(), legacy.NO_FLOW)
    assert_legacy_equal(aquifer(123.4).to_legacy(), legacy.AQUIFER(123.4))
    assert_legacy_equal(flow_rate(-7.0).to_legacy(), legacy.FLOW(-7.0))


def test_mech_vocabulary_matches_bound_cond():
    legacy = bound_cond()
    assert_legacy_equal(roller().to_legacy(), legacy.ROLLER)
    assert_legacy_equal(free().to_legacy(), legacy.FREE)
    assert_legacy_equal(
        stuck(0.5, [1.0, 2.0, 3.0]).to_legacy(), legacy.STUCK(0.5, [1.0, 2.0, 3.0])
    )
    assert_legacy_equal(stuck_roller(-0.25).to_legacy(), legacy.STUCK_ROLLER(-0.25))
    assert_legacy_equal(
        load(-100.0, [0.0, 0.0, 0.0]).to_legacy(), legacy.LOAD(-100.0, [0.0, 0.0, 0.0])
    )
    assert_legacy_equal(
        stuck_t_load_n(-100.0, [0.0, 1.0, 0.0]).to_legacy(),
        legacy.STUCK_T_LOAD_N(-100.0, [0.0, 1.0, 0.0]),
    )


def test_generic_scalar_constructors():
    assert dirichlet(5.0).to_legacy() == {'a': 1.0, 'b': 0.0, 'r': 5.0}
    assert neumann(5.0).to_legacy() == {'a': 0.0, 'b': 1.0, 'r': 5.0}
    assert robin(2.0, 3.0, 5.0).to_legacy() == {'a': 2.0, 'b': 3.0, 'r': 5.0}
    assert dirichlet().face.is_dirichlet and not dirichlet().face.is_neumann
    assert neumann().face.is_neumann and not neumann().face.is_dirichlet


# --------------------------------------------------------------------------
# FaceBoundary -- the declaration a model writes
# --------------------------------------------------------------------------


@pytest.fixture
def unreported_deprecation(monkeypatch):
    """Start the test with the once-per-process deprecation not yet reported."""
    monkeypatch.setattr(boundary_spec, '_LEGACY_SCHEMA_WARNED', False)


def thm_boundaries():
    """The typed declaration a thermoporoelastic model writes today."""
    nf_r = FaceBoundary(flow=no_flow(), mech=roller(), temp=no_flow())
    return {
        X_MINUS: nf_r,
        X_PLUS: FaceBoundary(
            flow=aquifer(200.0),
            mech=load(-1.0e-5, [0.0, 0.0, 0.0]),
            temp=aquifer(350.0),
        ),
        Y_MINUS: nf_r,
    }


def thm_boundary_conditions():
    """The same conditions in the DEPRECATED dict schema."""
    legacy = bound_cond()
    nf_r = {'flow': legacy.NO_FLOW, 'mech': legacy.ROLLER, 'temp': legacy.NO_FLOW}
    return {
        X_MINUS: nf_r,
        X_PLUS: {
            'flow': legacy.AQUIFER(200.0),
            'mech': legacy.LOAD(-1.0e-5, [0.0, 0.0, 0.0]),
            'temp': legacy.AQUIFER(350.0),
        },
        # the pm path adds this key in place; from_legacy_dict must ignore it
        Y_MINUS: {**nf_r, 'cells': []},
    }


def test_face_boundary_splits_into_type_and_value_halves():
    bc = FaceBoundary(flow=aquifer(200.0), mech=load(-1.0e-5, [0.0, 0.0, 0.0]))
    assert bc.facets() == BoundaryFacetSpec(
        flow=FaceBC(1.0, 0.0),
        mech_normal=FaceBC(0.0, 1.0),
        mech_tangential=FaceBC(0.0, 1.0),
        thermal=None,
    )
    values = bc.channel_values()
    assert values.flow == 200.0 and values.mech_normal == -1.0e-5
    assert values.thermal is None

    bc.temp = aquifer(350.0)
    assert bc.facets().thermal == FaceBC(1.0, 0.0)
    assert bc.channel_values().thermal == 350.0
    # a poroelastic run of the same model drops the thermal channel
    assert bc.facets(thermal=False).thermal is None
    assert bc.channel_values(thermal=False).thermal is None


def test_face_boundary_round_trips_through_the_legacy_form(unreported_deprecation):
    with pytest.warns(DeprecationWarning):
        for bc in thm_boundaries().values():
            assert FaceBoundary.from_legacy(bc.to_legacy()) == bc


def test_face_boundary_carries_the_discretizers_per_tag_face_list():
    """UnstructDiscretizer fills bc['cells'] in place while reading the mesh."""
    bc = FaceBoundary(flow=no_flow(), mech=roller())
    assert bc['cells'] == []
    bc['cells'].append(7)
    assert bc.cells == [7]
    bc['cells'] = []
    assert bc.cells == []
    with pytest.raises(KeyError, match='not the legacy dict'):
        bc['mech']
    with pytest.raises(KeyError, match='not the legacy dict'):
        bc['flow'] = no_flow()


def test_coerce_rejects_a_non_condition():
    with pytest.raises(TypeError, match='FaceBoundary'):
        FaceBoundary.coerce(('roller',), tag=X_MINUS)


def test_from_boundaries_splits_types_and_values():
    spec = BoundarySpec.from_boundaries(thm_boundaries())
    assert spec.thermal
    assert sorted(spec.tags) == [X_MINUS, X_PLUS, Y_MINUS]
    assert spec.types[X_PLUS].flow == FaceBC(1.0, 0.0)
    assert spec.values[X_PLUS].flow == 200.0
    assert spec.values[X_PLUS].thermal == 350.0


def test_from_boundaries_thermal_flag():
    typed = thm_boundaries()
    assert not BoundarySpec.from_boundaries(typed, thermal=False).thermal
    assert BoundarySpec.from_boundaries(typed, thermal=True).thermal
    poro = {tag: FaceBoundary(bc.flow, bc.mech) for tag, bc in typed.items()}
    assert not BoundarySpec.from_boundaries(poro).thermal
    with pytest.raises(KeyError, match='temp'):
        BoundarySpec.from_boundaries(poro, thermal=True)


# --------------------------------------------------------------------------
# from_legacy_dict -- the DEPRECATED adapter, kept for one cycle
# --------------------------------------------------------------------------


def spec_as_comparable(spec):
    """(types, values) in a form that compares without numpy ambiguity."""
    return (
        {tag: spec.types[tag] for tag in spec.types},
        {
            tag: (
                spec.values[tag].flow,
                spec.values[tag].mech_normal,
                tuple(np.asarray(spec.values[tag].mech_tangential).ravel()),
                spec.values[tag].thermal,
            )
            for tag in spec.values
        },
    )


def test_legacy_dicts_produce_the_same_spec_as_the_typed_declaration():
    from_dicts = BoundarySpec.from_legacy_dict(thm_boundary_conditions())
    from_typed = BoundarySpec.from_boundaries(thm_boundaries())
    assert spec_as_comparable(from_dicts) == spec_as_comparable(from_typed)


def test_legacy_dicts_warn_once(unreported_deprecation):
    with pytest.warns(DeprecationWarning, match='deprecated'):
        BoundarySpec.from_legacy_dict(thm_boundary_conditions())
    # once per process, not once per tag or per call
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        BoundarySpec.from_legacy_dict(thm_boundary_conditions())
        BoundarySpec.from_boundaries(thm_boundary_conditions())


def test_typed_declarations_do_not_warn(unreported_deprecation):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        BoundarySpec.from_boundaries(thm_boundaries())


def test_from_legacy_dict_splits_types_and_values():
    spec = BoundarySpec.from_legacy_dict(thm_boundary_conditions())
    assert spec.thermal
    assert sorted(spec.tags) == [X_MINUS, X_PLUS, Y_MINUS]

    facets = spec.types[X_PLUS]
    assert facets.flow == FaceBC(1.0, 0.0)
    assert facets.thermal == FaceBC(1.0, 0.0)
    assert facets.mech_normal == FaceBC(0.0, 1.0)
    assert facets.mech_tangential == FaceBC(0.0, 1.0)

    values = spec.values[X_PLUS]
    assert values.flow == 200.0
    assert values.thermal == 350.0
    assert values.mech_normal == -1.0e-5
    assert np.array_equal(np.asarray(values.mech_tangential), np.zeros(3))

    # no value leaked into the type half and vice versa
    assert not hasattr(facets.flow, 'r')


def test_from_legacy_dict_round_trips():
    legacy_in = thm_boundary_conditions()
    legacy_out = BoundarySpec.from_legacy_dict(legacy_in).to_legacy_dict()
    assert set(legacy_out) == set(legacy_in)
    for tag, entry in legacy_in.items():
        for channel in ('flow', 'mech', 'temp'):
            assert_legacy_equal(legacy_out[tag][channel], entry[channel])


def test_from_legacy_dict_thermal_flag():
    legacy_in = thm_boundary_conditions()
    # a poroelastic run is handed the same dicts and must ignore 'temp'
    assert not BoundarySpec.from_legacy_dict(legacy_in, thermal=False).thermal
    assert BoundarySpec.from_legacy_dict(legacy_in, thermal=True).thermal
    poro = {
        tag: {k: v for k, v in e.items() if k != 'temp'} for tag, e in legacy_in.items()
    }
    assert not BoundarySpec.from_legacy_dict(poro).thermal
    with pytest.raises(KeyError, match='temp'):
        BoundarySpec.from_legacy_dict(poro, thermal=True)


def test_from_legacy_dict_requires_flow_and_mech():
    with pytest.raises(KeyError, match='mech'):
        BoundarySpec.from_legacy_dict({X_MINUS: {'flow': no_flow().to_legacy()}})
    with pytest.raises(KeyError, match='flow'):
        BoundarySpec.from_legacy_dict({X_MINUS: {'mech': roller().to_legacy()}})


def test_values_evaluate_resolves_callables():
    values = BoundaryValues(flow=lambda t: 10.0 * t, mech_normal=2.0, thermal=None)
    resolved = values.evaluate(3.0)
    assert resolved.flow == 30.0
    assert resolved.mech_normal == 2.0
    assert resolved.thermal is None


# --------------------------------------------------------------------------
# mech_discretizer compiler
# --------------------------------------------------------------------------


def test_mech_discretizer_arrays_scatter_and_defaults():
    spec = BoundarySpec.from_legacy_dict(thm_boundary_conditions())
    tag_ids = {X_MINUS: np.array([0, 1]), X_PLUS: np.array([2]), Y_MINUS: np.array([3])}
    arrays = mech_discretizer_arrays(spec.types, tag_ids, n_bounds=5)

    # element 4 belongs to no tag -> the legacy defaults (flow a=1, rest 0)
    assert np.array_equal(arrays['flow_a'], [0.0, 0.0, 1.0, 0.0, 1.0])
    assert np.array_equal(arrays['flow_b'], [1.0, 1.0, 0.0, 1.0, 0.0])
    assert np.array_equal(arrays['mech_normal_a'], [1.0, 1.0, 0.0, 1.0, 0.0])
    assert np.array_equal(arrays['mech_normal_b'], [0.0, 0.0, 1.0, 0.0, 0.0])
    assert np.array_equal(arrays['mech_tangen_a'], np.zeros(5))
    assert np.array_equal(arrays['mech_tangen_b'], [1.0, 1.0, 1.0, 1.0, 0.0])
    assert np.array_equal(arrays['thermal_a'], [0.0, 0.0, 1.0, 0.0, 0.0])
    assert np.array_equal(arrays['thermal_b'], [1.0, 1.0, 0.0, 1.0, 0.0])


def test_mech_discretizer_arrays_matches_legacy_packing():
    """The arrays must equal the loop that init_arrays_boundary_condition ran."""
    legacy_bc = thm_boundary_conditions()
    spec = BoundarySpec.from_legacy_dict(legacy_bc)
    n_bounds = 7
    rng = np.random.default_rng(0)
    tags = rng.choice(list(legacy_bc), size=n_bounds)
    tag_ids = {tag: np.where(tags == tag)[0] for tag in legacy_bc}

    ap, bp = np.ones(n_bounds), np.zeros(n_bounds)
    amn, bmn = np.zeros(n_bounds), np.zeros(n_bounds)
    amt, bmt = np.zeros(n_bounds), np.zeros(n_bounds)
    at, bt = np.zeros(n_bounds), np.zeros(n_bounds)
    for tag, ids in tag_ids.items():
        bc = legacy_bc[tag]
        ap[ids], bp[ids] = bc['flow']['a'], bc['flow']['b']
        amn[ids], bmn[ids] = bc['mech']['an'], bc['mech']['bn']
        amt[ids], bmt[ids] = bc['mech']['at'], bc['mech']['bt']
        at[ids], bt[ids] = bc['temp']['a'], bc['temp']['b']

    arrays = mech_discretizer_arrays(spec.types, tag_ids, n_bounds)
    assert np.array_equal(arrays['flow_a'], ap)
    assert np.array_equal(arrays['flow_b'], bp)
    assert np.array_equal(arrays['mech_normal_a'], amn)
    assert np.array_equal(arrays['mech_normal_b'], bmn)
    assert np.array_equal(arrays['mech_tangen_a'], amt)
    assert np.array_equal(arrays['mech_tangen_b'], bmt)
    assert np.array_equal(arrays['thermal_a'], at)
    assert np.array_equal(arrays['thermal_b'], bt)


def test_mech_discretizer_arrays_no_thermal_keys_without_thermal_facet():
    spec = BoundarySpec.from_legacy_dict(thm_boundary_conditions(), thermal=False)
    arrays = mech_discretizer_arrays(spec.types, {X_MINUS: np.array([0])}, n_bounds=1)
    assert 'thermal_a' not in arrays and 'thermal_b' not in arrays


def test_mech_discretizer_arrays_rejects_partial_thermal():
    types = BoundaryTypeSpec(
        {
            X_MINUS: BoundaryFacetSpec(
                FaceBC(0.0, 1.0), FaceBC(1.0, 0.0), FaceBC(0.0, 1.0), FaceBC(0.0, 1.0)
            ),
            X_PLUS: BoundaryFacetSpec(
                FaceBC(0.0, 1.0), FaceBC(1.0, 0.0), FaceBC(0.0, 1.0), None
            ),
        }
    )
    with pytest.raises(ValueError, match='thermal facet'):
        mech_discretizer_arrays(types, {X_MINUS: [0], X_PLUS: [1]}, n_bounds=2)


# --------------------------------------------------------------------------
# pm_discretizer compiler
# --------------------------------------------------------------------------


def test_pm_discretizer_row_matches_legacy_packing():
    legacy = bound_cond()
    legacy_bc = {
        X_MINUS: {'flow': legacy.NO_FLOW, 'mech': legacy.ROLLER},
        X_PLUS: {
            'flow': legacy.AQUIFER(200.0),
            'mech': legacy.LOAD(-100.0, [0.0, 0.0, 0.0]),
        },
    }
    spec = BoundarySpec.from_legacy_dict(legacy_bc)
    for tag, entry in legacy_bc.items():
        mech, flow = entry['mech'], entry['flow']
        expected = [
            mech['an'],
            mech['bn'],
            mech['at'],
            mech['bt'],
            flow['a'],
            flow['b'],
        ]
        assert pm_discretizer_row(spec.types[tag]) == expected


def test_pm_discretizer_rejects_thermal():
    spec = BoundarySpec.from_legacy_dict(thm_boundary_conditions())
    with pytest.raises(AssertionError, match='thermoporoelasticity is not supported'):
        pm_discretizer_row(spec.types[X_MINUS])
    with pytest.raises(AssertionError, match='thermoporoelasticity is not supported'):
        compile_pm_discretizer(spec.types, [X_MINUS])


def test_compile_pm_discretizer_is_pure_over_the_spec():
    """The compiler needs nothing but the spec and the face->tag list."""
    pytest.importorskip('darts.engines')
    spec = BoundarySpec.from_legacy_dict(thm_boundary_conditions(), thermal=False)
    prop_ids = [X_PLUS, X_MINUS, X_PLUS]
    matrices = compile_pm_discretizer(spec.types, prop_ids)
    assert len(matrices) == len(prop_ids)
    for tag, packed in zip(prop_ids, matrices, strict=True):
        assert list(np.array(packed.values)) == pm_discretizer_row(spec.types[tag])


# --------------------------------------------------------------------------
# containers
# --------------------------------------------------------------------------


def test_specs_are_read_only_mappings():
    spec = BoundarySpec.from_legacy_dict(thm_boundary_conditions())
    assert isinstance(spec.types, BoundaryTypeSpec)
    assert isinstance(spec.values, BoundaryValueSpec)
    assert len(spec.types) == len(spec.values) == 3
    assert X_PLUS in spec.types
    with pytest.raises(TypeError):
        spec.types[X_PLUS] = None


# --------------------------------------------------------------------------
# BoundaryValueBC -- driven against a stub reservoir (no engine, no mesh)
# --------------------------------------------------------------------------


class StubReservoir:
    """The attribute surface BoundaryValueBC reads off a mech reservoir."""

    def __init__(
        self, discretizer_name, thermoporoelasticity, n_bounds, boundary_conditions
    ):
        self.discretizer_name = discretizer_name
        self.thermoporoelasticity = thermoporoelasticity
        self.boundary_conditions = boundary_conditions
        self.n_dim = 3
        self.n_bounds = n_bounds
        if discretizer_name == 'pm_discretizer':
            self.u_bc_var, self.p_bc_var, self.t_bc_var = 0, 3, None
            self.n_state, self.p_var, self.t_var = 1, 3, None
        else:
            self.p_bc_var = 0
            self.t_bc_var = 1 if thermoporoelasticity else None
            self.u_bc_var = 2 if thermoporoelasticity else 1
            self.n_state = 2 if thermoporoelasticity else 1
            self.p_var, self.t_var = 0, 1 if thermoporoelasticity else None
        self.n_bc_vars = 1 + int(thermoporoelasticity) + self.n_dim
        self.bc_rhs = np.zeros(self.n_bc_vars * n_bounds)
        self.bc_rhs_prev = np.zeros(self.n_bc_vars * n_bounds)
        self.bc = np.zeros(self.n_bc_vars * n_bounds)
        self.bc_prev = np.zeros(self.n_bc_vars * n_bounds)
        self.pz_bounds = np.zeros(self.n_state * n_bounds)


def make_item(
    discretizer_name='mech_discretizer',
    thermoporoelasticity=True,
    n_bounds=3,
    legacy=False,
):
    from darts.reservoirs.boundary_spec import BoundaryValueBC

    boundaries = thm_boundary_conditions() if legacy else thm_boundaries()
    if not thermoporoelasticity:
        if legacy:
            boundaries = {
                t: {k: v for k, v in e.items() if k != 'temp'}
                for t, e in boundaries.items()
            }
        else:
            boundaries = {
                t: FaceBoundary(e.flow, e.mech) for t, e in boundaries.items()
            }
    res = StubReservoir(discretizer_name, thermoporoelasticity, n_bounds, boundaries)
    return BoundaryValueBC(res).sync(), res


def test_boundary_value_bc_writes_mech_layout():
    item, res = make_item()
    normals = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    item.write_mech_discretizer(
        {X_MINUS: np.array([0]), X_PLUS: np.array([1]), Y_MINUS: np.array([2])},
        normals,
    )
    nbv = res.n_bc_vars
    # X_PLUS (element 1): aquifer 200 / temp 350 / LOAD(-1e-5) along its normal
    assert res.bc_rhs[nbv * 1 + res.p_bc_var] == 200.0
    assert res.bc_rhs[nbv * 1 + res.t_bc_var] == 350.0
    assert np.allclose(
        res.bc_rhs[nbv * 1 + res.u_bc_var : nbv * 1 + res.u_bc_var + 3],
        -1.0e-5 * normals[1],
    )
    # X_MINUS (element 0): NO_FLOW + ROLLER -> zero everywhere
    assert np.allclose(res.bc_rhs[nbv * 0 : nbv * 1], 0.0)


def test_boundary_value_bc_matches_the_legacy_loop():
    """Same values the per-id loop in init_bc_rhs used to write."""
    item, res = make_item()
    rng = np.random.default_rng(1)
    normals = rng.normal(size=(res.n_bounds, 3))
    tag_ids = {X_MINUS: np.array([0]), X_PLUS: np.array([1]), Y_MINUS: np.array([2])}
    item.write_mech_discretizer(tag_ids, normals)

    expected = np.zeros_like(res.bc_rhs)
    nbv = res.n_bc_vars
    for tag, ids in tag_ids.items():
        bc = res.boundary_conditions[tag]
        expected[nbv * ids + res.p_bc_var] = bc.flow.r
        expected[nbv * ids + res.t_bc_var] = bc.temp.r
        for i in ids:
            expected[nbv * i + res.u_bc_var : nbv * i + res.u_bc_var + 3] = (
                bc.mech.rn * normals[i] + bc.mech.rt
            )
    assert np.array_equal(res.bc_rhs, expected)


def test_boundary_value_bc_accepts_deprecated_dicts_identically(unreported_deprecation):
    """An out-of-tree model still on the dict schema writes the same bc_rhs."""
    normals = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    tag_ids = {X_MINUS: np.array([0]), X_PLUS: np.array([1]), Y_MINUS: np.array([2])}
    typed_item, typed_res = make_item()
    typed_item.write_mech_discretizer(tag_ids, normals)
    with pytest.warns(DeprecationWarning):
        legacy_item, legacy_res = make_item(legacy=True)
    legacy_item.write_mech_discretizer(tag_ids, normals)
    assert np.array_equal(typed_res.bc_rhs, legacy_res.bc_rhs)


def test_boundary_value_bc_pm_layout_and_face_ordering():
    item, res = make_item('pm_discretizer', thermoporoelasticity=False, n_bounds=4)
    normals = np.tile([1.0, 0.0, 0.0], (4, 1))
    prop_ids = [X_PLUS, X_MINUS, X_PLUS, Y_MINUS]
    item.write_pm_discretizer(prop_ids, normals)
    for face_id, tag in enumerate(prop_ids):
        flow_r = res.boundary_conditions[tag].flow.r
        assert res.bc_rhs[4 * face_id + 3] == flow_r
        rn = res.boundary_conditions[tag].mech.rn
        assert np.allclose(
            res.bc_rhs[4 * face_id : 4 * face_id + 3], rn * normals[face_id]
        )


def test_boundary_value_bc_push_and_rotation():
    item, res = make_item()
    item.write_mech_discretizer({X_PLUS: np.array([1])}, np.zeros((3, 3)))
    item.push(check=False)
    assert np.array_equal(res.bc, res.bc_rhs)
    assert np.array_equal(res.bc_prev, res.bc_rhs_prev)

    # rotation makes the pushed values the "previous" ones
    item.on_timestep_start(dt=1.0, t=0.0)
    assert np.array_equal(res.bc_rhs_prev, res.bc_rhs)
    assert res.bc_rhs_prev is not res.bc_rhs


def test_boundary_value_bc_apply_is_a_noop():
    item, _ = make_item()
    assert item.apply(ctx=None) is None
    # a value-array item: no residual/Jacobian contributions at all
    assert item.writes_mesh_arrays and not item.provides_jacobian


def test_dirichlet_coherence_warns_and_can_raise():
    item, res = make_item()
    normals = np.zeros((3, 3))
    tag_ids = {X_MINUS: np.array([0]), X_PLUS: np.array([1]), Y_MINUS: np.array([2])}
    item.write_mech_discretizer(tag_ids, normals)

    # pz_bounds still zero while X_PLUS prescribes p = 200, T = 350
    with pytest.warns(UserWarning) as caught:
        assert not item.check_dirichlet_coherence()
    reported = ' '.join(str(w.message) for w in caught)
    assert 'pressure' in reported and 'temperature' in reported

    # make them agree -> silent, and the check passes
    res.pz_bounds[res.n_state * 1 + res.p_var] = 200.0
    res.pz_bounds[res.n_state * 1 + res.t_var] = 350.0
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert item.check_dirichlet_coherence()

    res.pz_bounds[res.n_state * 1 + res.p_var] = 199.0
    item.strict_dirichlet = True
    with pytest.raises(AssertionError, match='pressure'):
        item.check_dirichlet_coherence()
