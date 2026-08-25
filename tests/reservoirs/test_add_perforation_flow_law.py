"""``add_perforation(flow_law=...)`` works on EVERY concrete reservoir family (R5).

``ReservoirBase.add_perforation`` advertises the ``flow_law`` keyword and
``ReservoirBase._translate_perforation_flow_law`` holds the shared translation
code (each family validates BEFORE mutating and rolls back on rejection), so
the keyword must be accepted -- and honored -- by every concrete family, not
only by ``StructReservoir``. Nothing about the engine-side connection law is
structured-grid-specific: it lives on an ``ms_well`` perforation, and the
reservoir's only job is to attach it to the perforation it just created.

Three properties are established per family:

1. the keyword exists in the signature (a user following the base-class
   documentation must not get an unexpected-keyword ``TypeError``);
2. a law passed with a successfully added perforation is attached to exactly
   that perforation (verified through the engine's own accessor);
3. the silent-drop paths (duplicate perforation, inactive block) RAISE when a
   flow law was given, because dropping the perforation would silently discard
   the law and leave the well uncoupled.

``StructReservoir`` is exercised through a real (tiny) construction; the
unstructured and CPG families are exercised against stub reservoirs -- their
constructors need mesh files / geometry arrays that are irrelevant to the
perforation bookkeeping under test -- with real ``ms_well`` objects, so the
attachment and its zero-well-index validation run in the real engine class.
"""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("darts.engines")

from darts.engines import perforation_flow_law_type  # noqa: E402
from darts.pipes.linear_dfm_well_ipr import LinearIPR, PI_Type  # noqa: E402
from darts.reservoirs.cpg_reservoir import CPG_Reservoir  # noqa: E402
from darts.reservoirs.reservoir_base import ReservoirBase  # noqa: E402
from darts.reservoirs.struct_reservoir import StructReservoir  # noqa: E402
from darts.reservoirs.unstruct_reservoir import UnstructReservoir  # noqa: E402


def _law():
    return LinearIPR(productivity=2.5, basis=PI_Type.MASS, offset=0.5)


def _assert_law_attached(well, perforation_index=0):
    law = well.get_perforation_flow_law(perforation_index)
    assert law.law == perforation_flow_law_type.LINEAR_IPR
    assert law.productivity == 2.5
    assert law.offset == 0.5


# ---------------------------------------------------------------- signature
@pytest.mark.parametrize(
    "reservoir_cls",
    [ReservoirBase, StructReservoir, UnstructReservoir, CPG_Reservoir],
    ids=lambda cls: cls.__name__,
)
def test_add_perforation_accepts_flow_law(reservoir_cls):
    """The advertised keyword exists on the base AND every concrete family."""
    signature = inspect.signature(reservoir_cls.add_perforation)
    assert "flow_law" in signature.parameters, (
        f"{reservoir_cls.__name__}.add_perforation() does not accept flow_law, "
        "which ReservoirBase.add_perforation() advertises"
    )
    assert signature.parameters["flow_law"].default is None


# ------------------------------------------------------------------ struct
@pytest.fixture
def struct_reservoir():
    """A real 3x1x1 structured reservoir: cells (1..2,1,1) active, (3,1,1) inactive."""
    from darts.engines import timer_node

    timer = timer_node()
    timer.node["initialization"] = timer_node()
    reservoir = StructReservoir(
        timer=timer,
        nx=3,
        ny=1,
        nz=1,
        dx=10.0,
        dy=10.0,
        dz=10.0,
        permx=100.0,
        permy=100.0,
        permz=10.0,
        poro=np.full(3, 0.2),
        actnum=np.array([1, 1, 0], dtype=np.int32),
    )
    reservoir.mesh = reservoir.discretize()
    return reservoir


def test_struct_attaches_the_law(struct_reservoir):
    struct_reservoir.add_well("W1")
    struct_reservoir.add_perforation(
        "W1", res_cell_idx=(1, 1, 1), well_index=0.0, well_indexD=0.0, flow_law=_law()
    )
    well = struct_reservoir.get_well("W1")
    assert len(well.perforations) == 1
    _assert_law_attached(well)


def test_struct_duplicate_with_a_law_raises(struct_reservoir):
    struct_reservoir.add_well("W1")
    struct_reservoir.add_perforation(
        "W1", res_cell_idx=(1, 1, 1), well_index=0.0, well_indexD=0.0
    )
    with pytest.raises(ValueError, match="flow law"):
        struct_reservoir.add_perforation(
            "W1",
            res_cell_idx=(1, 1, 1),
            well_index=0.0,
            well_indexD=0.0,
            flow_law=_law(),
        )


def test_struct_inactive_block_with_a_law_raises(struct_reservoir):
    struct_reservoir.add_well("W1")
    with pytest.raises(ValueError, match="inactive"):
        struct_reservoir.add_perforation(
            "W1",
            res_cell_idx=(3, 1, 1),
            well_index=0.0,
            well_indexD=0.0,
            flow_law=_law(),
        )


# ---------------------------------------------------------------- unstruct
@pytest.fixture
def unstruct_reservoir():
    """A stub unstructured reservoir: real ``ms_well`` bookkeeping, no mesh file.

    ``UnstructReservoir.add_perforation`` touches ``self.wells`` (via
    ``get_well``/``add_well``) and ``self.mesh.depth``; with explicit
    ``well_index``/``well_indexD`` the discretizer is never consulted, so a
    depth array is the entire mesh surface under test.
    """
    reservoir = object.__new__(UnstructReservoir)
    reservoir.cache = False  # ReservoirBase.__del__ reads it
    reservoir.wells = []
    reservoir.mesh = SimpleNamespace(depth=np.linspace(1000.0, 1090.0, 10))
    return reservoir


def test_unstruct_attaches_the_law(unstruct_reservoir):
    unstruct_reservoir.add_well("W1")
    unstruct_reservoir.add_perforation(
        "W1", res_cell_idx=3, well_index=0.0, well_indexD=0.0, flow_law=_law()
    )
    well = unstruct_reservoir.get_well("W1")
    assert len(well.perforations) == 1
    assert well.perforations[0][1] == 3
    _assert_law_attached(well)


def test_unstruct_duplicate_with_a_law_raises(unstruct_reservoir):
    unstruct_reservoir.add_well("W1")
    unstruct_reservoir.add_perforation(
        "W1", res_cell_idx=3, well_index=0.0, well_indexD=0.0
    )
    # the duplicate path normally terminates the process; with a law it must
    # raise a diagnosable error instead of silently discarding the law first
    with pytest.raises(ValueError, match="flow law"):
        unstruct_reservoir.add_perforation(
            "W1", res_cell_idx=3, well_index=0.0, well_indexD=0.0, flow_law=_law()
        )


# --------------------------------------------------------------------- cpg
@pytest.fixture
def cpg_reservoir():
    """A stub CPG reservoir: real ``ms_well`` bookkeeping, no geometry arrays.

    ``CPG_Reservoir.add_perforation`` resolves the block through
    ``self.calc_well_index`` and reads ``self.depth_all_cells`` and
    ``self.discr_mesh.calc_cell_sizes``; those three are stubbed, everything
    downstream (perforation list, attachment, validation) is the real code.
    """
    reservoir = object.__new__(CPG_Reservoir)
    reservoir.cache = False  # ReservoirBase.__del__ reads it
    reservoir.wells = []
    reservoir.depth_all_cells = np.linspace(1000.0, 1090.0, 10)
    reservoir.discr_mesh = SimpleNamespace(
        calc_cell_sizes=lambda i, j, k: (10.0, 10.0, 5.0)
    )
    # block (1,1,1) is active local block 4; block (2,1,1) is inactive
    reservoir.calc_well_index = lambda i, j, k, **kwargs: (
        (4, 12.5, 3.5) if (i, j, k) == (1, 1, 1) else (-1, 0.0, 0.0)
    )
    return reservoir


def test_cpg_attaches_the_law(cpg_reservoir):
    cpg_reservoir.add_well("W1")
    cpg_reservoir.add_perforation(
        "W1", res_cell_idx=(1, 1, 1), well_index=0.0, well_indexD=0.0, flow_law=_law()
    )
    well = cpg_reservoir.get_well("W1")
    assert len(well.perforations) == 1
    assert well.perforations[0][1] == 4
    _assert_law_attached(well)


def test_cpg_duplicate_with_a_law_raises(cpg_reservoir):
    cpg_reservoir.add_well("W1")
    cpg_reservoir.add_perforation(
        "W1", res_cell_idx=(1, 1, 1), well_index=0.0, well_indexD=0.0
    )
    with pytest.raises(ValueError, match="flow law"):
        cpg_reservoir.add_perforation(
            "W1",
            res_cell_idx=(1, 1, 1),
            well_index=0.0,
            well_indexD=0.0,
            flow_law=_law(),
        )


def test_cpg_inactive_block_with_a_law_raises(cpg_reservoir):
    cpg_reservoir.add_well("W1")
    with pytest.raises(ValueError, match="inactive"):
        cpg_reservoir.add_perforation(
            "W1",
            res_cell_idx=(2, 1, 1),
            well_index=0.0,
            well_indexD=0.0,
            flow_law=_law(),
        )


def test_cpg_inactive_block_without_a_law_is_still_dropped(cpg_reservoir):
    """The historical silent-drop behavior without a law is unchanged."""
    cpg_reservoir.add_well("W1")
    cpg_reservoir.add_perforation(
        "W1", res_cell_idx=(2, 1, 1), well_index=0.0, well_indexD=0.0
    )
    assert len(cpg_reservoir.get_well("W1").perforations) == 0


# ------------------------------------------------- the engine's own guard
def test_attachment_enforces_zero_well_index_everywhere():
    """The shared attachment path runs the engine's zero-WI validation, so a
    non-Darcy law on a Peaceman-coupled perforation is refused in every family
    (here through the stub unstructured path, which reuses the shared code)."""
    reservoir = object.__new__(UnstructReservoir)
    reservoir.cache = False  # ReservoirBase.__del__ reads it
    reservoir.wells = []
    reservoir.mesh = SimpleNamespace(depth=np.linspace(1000.0, 1090.0, 10))
    reservoir.add_well("W1")
    with pytest.raises(RuntimeError, match="non-zero well index"):
        reservoir.add_perforation(
            "W1", res_cell_idx=3, well_index=12.5, well_indexD=0.0, flow_law=_law()
        )


# ------------------------------------- positional compatibility (finding F2)
#
# Every add_perforation signature ended in ``verbose`` before flow_law
# existed, so flow_law must come AFTER verbose: a historical positional call
# whose last argument is the verbose switch must keep meaning verbose, not be
# silently rebound to a flow law.


@pytest.mark.parametrize(
    "reservoir_cls",
    [ReservoirBase, StructReservoir, UnstructReservoir, CPG_Reservoir],
    ids=lambda cls: cls.__name__,
)
def test_flow_law_is_the_last_parameter_after_verbose(reservoir_cls):
    params = list(inspect.signature(reservoir_cls.add_perforation).parameters)
    assert params[-1] == "flow_law", (
        f"{reservoir_cls.__name__}.add_perforation() must keep flow_law LAST "
        "(after verbose) so historical positional calls keep their meaning"
    )
    assert params[-2] == "verbose"


def test_struct_historical_positional_call_still_binds_verbose(
    struct_reservoir, capsys
):
    """The full pre-flow-law positional form of StructReservoir, ending in the
    verbose switch."""
    struct_reservoir.add_well("W1")
    struct_reservoir.add_perforation(
        "W1", (1, 1, 1), None, 0.1524, 0.0, 0.0, "z_axis", 0.0, None, False, True
    )
    well = struct_reservoir.get_well("W1")
    assert len(well.perforations) == 1
    assert well.get_perforation_flow_law(0).law == perforation_flow_law_type.DARCY, (
        "the trailing True must mean verbose, never a flow law"
    )
    assert "Added perforation for well W1" in capsys.readouterr().out


def test_unstruct_historical_positional_call_still_binds_verbose(
    unstruct_reservoir, capsys
):
    unstruct_reservoir.add_well("W1")
    unstruct_reservoir.add_perforation(
        "W1", 3, None, 0.3048, 0.0, 0.0, "z_axis", 0.0, False, True
    )
    well = unstruct_reservoir.get_well("W1")
    assert len(well.perforations) == 1
    assert well.get_perforation_flow_law(0).law == perforation_flow_law_type.DARCY
    assert "Added perforation for well W1" in capsys.readouterr().out


def test_cpg_historical_positional_call_still_binds_verbose(cpg_reservoir, capsys):
    cpg_reservoir.centroids_all_cells = [
        SimpleNamespace(values=np.zeros(3)) for _ in range(10)
    ]
    cpg_reservoir.add_well("W1")
    cpg_reservoir.add_perforation(
        "W1", (1, 1, 1), None, 0.3048, 0.0, 0.0, "z_axis", 0.0, False, True
    )
    well = cpg_reservoir.get_well("W1")
    assert len(well.perforations) == 1
    assert well.get_perforation_flow_law(0).law == perforation_flow_law_type.DARCY
    assert "Added perforation for well W1" in capsys.readouterr().out


# ------------------------------------------------- atomicity (finding F3)
#
# A perforation whose flow law is rejected must leave the well EXACTLY as it
# was: no appended Darcy completion, no shifted depths, no rescaled segment
# volume. A caller that catches the configuration error to repair or retry
# must not inherit half-mutated state.


def _well_state(well):
    """Every well field add_perforation can mutate."""
    return (
        list(well.perforations),
        well.well_head_depth,
        well.well_body_depth,
        well.segment_depth_increment,
        well.segment_volume,
    )


class _ExplodingLaw:
    def to_engine(self):
        raise RuntimeError("translation boom")


class _JunkLaw:
    def to_engine(self):
        return 42


_FAMILY_PERF_TARGET = {
    "struct_reservoir": (1, 1, 1),
    "unstruct_reservoir": 3,
    "cpg_reservoir": (1, 1, 1),
}

_FAMILIES = list(_FAMILY_PERF_TARGET)


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize(
    "bad_law, expected_error",
    [
        (object(), TypeError),
        (_ExplodingLaw(), RuntimeError),
        (_JunkLaw(), TypeError),
    ],
    ids=["not-a-law", "to_engine-raises", "to_engine-returns-junk"],
)
def test_untranslatable_law_mutates_nothing(family, bad_law, expected_error, request):
    reservoir = request.getfixturevalue(family)
    reservoir.add_well("W1")
    well = reservoir.get_well("W1")
    before = _well_state(well)
    with pytest.raises(expected_error):
        reservoir.add_perforation(
            "W1",
            res_cell_idx=_FAMILY_PERF_TARGET[family],
            well_index=0.0,
            well_indexD=0.0,
            flow_law=bad_law,
        )
    assert _well_state(well) == before


@pytest.mark.parametrize("family", _FAMILIES)
def test_nonzero_well_index_rejection_rolls_the_well_back(family, request):
    """The engine refuses a law on a Peaceman-coupled perforation; the
    just-appended completion (and every depth/segment mutation) must be rolled
    back rather than left behind (finding F3's reproduction)."""
    reservoir = request.getfixturevalue(family)
    reservoir.add_well("W1")
    well = reservoir.get_well("W1")
    before = _well_state(well)
    with pytest.raises(RuntimeError, match="non-zero well index"):
        reservoir.add_perforation(
            "W1",
            res_cell_idx=_FAMILY_PERF_TARGET[family],
            well_index=12.5,
            well_indexD=0.0,
            flow_law=_law(),
        )
    assert _well_state(well) == before, (
        "a rejected flow law left a Darcy perforation or mutated depths behind"
    )
    # THE property the rollback exists for: the caller repairs the argument
    # and retries, and the retry must succeed (a leftover completion would
    # trip the duplicate-with-law refusal instead)
    reservoir.add_perforation(
        "W1",
        res_cell_idx=_FAMILY_PERF_TARGET[family],
        well_index=0.0,
        well_indexD=0.0,
        flow_law=_law(),
    )
    assert len(well.perforations) == 1
    _assert_law_attached(well)


@pytest.mark.parametrize("family", _FAMILIES)
def test_duplicate_with_a_law_mutates_nothing(family, request):
    reservoir = request.getfixturevalue(family)
    reservoir.add_well("W1")
    reservoir.add_perforation(
        "W1",
        res_cell_idx=_FAMILY_PERF_TARGET[family],
        well_index=0.0,
        well_indexD=0.0,
    )
    well = reservoir.get_well("W1")
    before = _well_state(well)
    with pytest.raises(ValueError, match="flow law"):
        reservoir.add_perforation(
            "W1",
            res_cell_idx=_FAMILY_PERF_TARGET[family],
            well_index=0.0,
            well_indexD=0.0,
            flow_law=_law(),
        )
    assert _well_state(well) == before


@pytest.mark.parametrize(
    "family, inactive_target",
    [("struct_reservoir", (3, 1, 1)), ("cpg_reservoir", (2, 1, 1))],
)
def test_inactive_with_a_law_mutates_nothing(family, inactive_target, request):
    reservoir = request.getfixturevalue(family)
    reservoir.add_well("W1")
    well = reservoir.get_well("W1")
    before = _well_state(well)
    with pytest.raises(ValueError, match="inactive"):
        reservoir.add_perforation(
            "W1",
            res_cell_idx=inactive_target,
            well_index=0.0,
            well_indexD=0.0,
            flow_law=_law(),
        )
    assert _well_state(well) == before


@pytest.mark.parametrize("family", _FAMILIES)
def test_negative_index_with_a_law_mutates_nothing_and_permits_retry(family, request):
    """A negative well index (geometric or thermal) with a flow law used to
    abort AFTER the completion was appended -- and, for a negative THERMAL
    index, after the law had already been attached (the engine validates only
    the geometric index), permanently blocking a repaired retry behind the
    duplicate-with-law refusal (2026-08-25 adversarial verification). The
    index validation now runs before any mutation."""
    reservoir = request.getfixturevalue(family)
    reservoir.add_well("W1")
    well = reservoir.get_well("W1")
    before = _well_state(well)
    for bad_kwargs in (
        {"well_index": -1.0, "well_indexD": 0.0},
        {"well_index": 0.0, "well_indexD": -1.0},
    ):
        with pytest.raises(AssertionError):
            reservoir.add_perforation(
                "W1",
                res_cell_idx=_FAMILY_PERF_TARGET[family],
                flow_law=_law(),
                **bad_kwargs,
            )
        assert _well_state(well) == before
        assert len(well.perforations) == 0
    # the repaired retry attaches the law to a clean first perforation
    reservoir.add_perforation(
        "W1",
        res_cell_idx=_FAMILY_PERF_TARGET[family],
        well_index=0.0,
        well_indexD=0.0,
        flow_law=_law(),
    )
    _assert_law_attached(well)


def test_unstruct_negative_index_preserves_depths_of_prior_perforations(
    unstruct_reservoir,
):
    """The unstructured depth update ran BEFORE the index validation, so a
    negative-index abort used to leave well_head_depth moved to the rejected
    cell's depth (2026-08-25 adversarial verification)."""
    unstruct_reservoir.add_well("W1")
    unstruct_reservoir.add_perforation(
        "W1", res_cell_idx=9, well_index=0.0, well_indexD=0.0
    )
    well = unstruct_reservoir.get_well("W1")
    depth_before = well.well_head_depth
    with pytest.raises(AssertionError):
        unstruct_reservoir.add_perforation(
            "W1", res_cell_idx=0, well_index=-1.0, well_indexD=0.0, flow_law=_law()
        )
    assert well.well_head_depth == depth_before
    assert well.well_body_depth == depth_before
    assert len(well.perforations) == 1
