"""``add_perforation(flow_law=...)`` works on EVERY concrete reservoir family (R5).

``ReservoirBase.add_perforation`` advertises the ``flow_law`` keyword and
``ReservoirBase._attach_perforation_flow_law`` holds the shared attachment code,
so the keyword must be accepted -- and honored -- by every concrete family, not
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
