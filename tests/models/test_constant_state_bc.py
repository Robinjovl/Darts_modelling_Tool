"""Unit tests for :class:`darts.models.conditions.ConstantStateBC`.

The "huge boundary volume" open / constant-state far field is a MESH property:
the reservoir writes the volumes, the engine caches ``PV = volume * poro`` once
in ``engine.init()``, and a volume written after that is silently ignored.
:class:`ConstantStateBC` is the declarative item that reads
``reservoir.boundary_volumes`` and validates that the trick actually took
effect. These tests run against pure-numpy stubs -- no mesh, no engine:

* the item reads the reservoir's ``boundary_volumes`` dict (explicit faces and
  the "whichever faces are set" default) and records volume / cell counts;
* the ordering validation fires when the volumes never reached ``mesh.volume``
  (they were set too late to be applied) and passes when they did;
* declaring a face that carries no volume, or a reservoir without the dict
  at all, is an error;
* ``mode="dirichlet"`` raises :class:`NotImplementedError` (the DirichletPin
  follow-up), unknown modes/faces raise :class:`ValueError`;
* ``apply()`` writes nothing;
* the reservoir-side latch: once ``freeze_pore_volumes()`` was called, the
  boundary-volume dict and the guarded reservoir entry points refuse writes.
"""

import numpy as np
import pytest

pytest.importorskip("darts.engines")

from darts.models.conditions import (  # noqa: E402
    AssemblyContext,
    ConstantStateBC,
)
from darts.reservoirs.reservoir_base import BoundaryVolumeDict  # noqa: E402

FAR_FIELD = 1e8


class _StubMesh:
    def __init__(self, volume, n_res_blocks=None):
        self.volume = np.asarray(volume, dtype=float)
        self.n_res_blocks = (
            len(self.volume) if n_res_blocks is None else int(n_res_blocks)
        )


class _StubReservoir:
    """Minimal stand-in for the structured reservoir family.

    ``boundary_volumes`` is the real :class:`BoundaryVolumeDict`; the volumes
    reach ``mesh.volume`` only when :meth:`discretize` is called -- which is
    exactly the ordering the item validates.
    """

    def __init__(self, n_cells=6):
        self.boundary_volumes = BoundaryVolumeDict(dict.fromkeys(ConstantStateBC.FACES))
        self.cell_volume = np.full(n_cells, 100.0)
        self.mesh = _StubMesh(self.cell_volume.copy())

    def discretize(self):
        """Apply the boundary volumes, as StructReservoir.discretize() does."""
        volume = self.cell_volume.copy()
        for index, face in enumerate(ConstantStateBC.FACES):
            value = self.boundary_volumes[face]
            if value is not None:
                volume[index] = value
        self.mesh = _StubMesh(volume)


class _StubPhysics:
    n_vars = 2

    def __init__(self):
        self.engine = object()


class _StubModel:
    def __init__(self, reservoir):
        self.reservoir = reservoir
        self.physics = _StubPhysics()
        self.platform = "cpu"


def _model(faces_with_volume=("yz_minus",), discretize=True, n_cells=6):
    reservoir = _StubReservoir(n_cells)
    for face in faces_with_volume:
        reservoir.boundary_volumes[face] = FAR_FIELD
    if discretize:
        reservoir.discretize()
    return _StubModel(reservoir)


# ------------------------------------------------------------------ reading
def test_bind_reads_boundary_volumes_of_declared_face():
    model = _model(("yz_minus",))
    item = ConstantStateBC(faces="yz_minus")
    item.bind(model)
    assert item.volumes == {"yz_minus": FAR_FIELD}
    assert item.n_cells == {"yz_minus": 1}


def test_bind_defaults_to_every_face_that_carries_a_volume():
    model = _model(("yz_minus", "xz_plus"))
    item = ConstantStateBC()
    item.bind(model)
    assert sorted(item.volumes) == ["xz_plus", "yz_minus"]
    assert all(volume == FAR_FIELD for volume in item.volumes.values())


def test_bind_counts_every_cell_carrying_the_far_field_volume():
    model = _model((), discretize=False)
    model.reservoir.mesh = _StubMesh([100.0, FAR_FIELD, FAR_FIELD, 100.0])
    model.reservoir.boundary_volumes["yz_minus"] = FAR_FIELD
    item = ConstantStateBC(faces=["yz_minus"])
    item.bind(model)
    assert item.n_cells["yz_minus"] == 2


def test_bind_ignores_well_blocks_beyond_n_res_blocks():
    model = _model((), discretize=False)
    # the huge volume only exists in a well block -> not a reservoir far field
    model.reservoir.mesh = _StubMesh([100.0, 100.0, FAR_FIELD], n_res_blocks=2)
    model.reservoir.boundary_volumes["yz_minus"] = FAR_FIELD
    with pytest.raises(RuntimeError, match="never reached the mesh"):
        ConstantStateBC(faces="yz_minus").bind(model)


# --------------------------------------------------------- ordering guard
def test_bind_rejects_volumes_applied_too_late():
    # the dict was populated but discretize() already ran without it: the
    # engine would have cached PV from the small volumes
    model = _model((), discretize=True)
    model.reservoir.boundary_volumes["yz_minus"] = FAR_FIELD
    item = ConstantStateBC(faces="yz_minus")
    with pytest.raises(RuntimeError) as excinfo:
        item.bind(model)
    message = str(excinfo.value)
    assert "never reached the mesh" in message
    assert "PV = volume * poro" in message
    assert "before model.init()" in message


def test_bind_accepts_volumes_applied_in_time():
    model = _model(("yz_minus",), discretize=True)
    ConstantStateBC(faces="yz_minus").bind(model)  # must not raise


def test_bind_rejects_declared_face_without_volume():
    model = _model(("yz_minus",))
    with pytest.raises(RuntimeError, match="xz_plus"):
        ConstantStateBC(faces=("yz_minus", "xz_plus")).bind(model)


def test_bind_rejects_empty_declaration():
    model = _model(())
    with pytest.raises(RuntimeError, match="no face"):
        ConstantStateBC().bind(model)


def test_bind_rejects_reservoir_without_boundary_volumes():
    model = _model(("yz_minus",))
    del model.reservoir.boundary_volumes
    with pytest.raises(RuntimeError, match="boundary_volumes"):
        ConstantStateBC(faces="yz_minus").bind(model)


# ------------------------------------------------------------------- modes
def test_dirichlet_mode_is_not_implemented():
    with pytest.raises(NotImplementedError, match="DirichletPin"):
        ConstantStateBC(mode="dirichlet")


def test_unknown_mode_and_face_are_rejected():
    with pytest.raises(ValueError, match="unknown mode"):
        ConstantStateBC(mode="volumes")
    with pytest.raises(ValueError, match="unknown face"):
        ConstantStateBC(faces="north")


# ----------------------------------------------------------------- no-op apply
def test_apply_writes_nothing():
    model = _model(("yz_minus",))
    item = ConstantStateBC(faces="yz_minus")
    item.bind(model)
    ctx = AssemblyContext(
        rhs=np.zeros(12),
        jac=None,
        X=np.zeros(12),
        Xn=np.zeros(12),
        dt=1.0,
        t=0.0,
        iteration=0,
        n_vars=2,
        n_res_blocks=6,
    )
    item.apply(ctx)
    assert np.all(ctx.rhs == 0.0)


# ------------------------------------------------------------ reservoir latch
def test_boundary_volume_dict_refuses_writes_once_frozen():
    volumes = BoundaryVolumeDict(dict.fromkeys(ConstantStateBC.FACES))
    volumes["yz_minus"] = FAR_FIELD
    volumes.frozen = True
    with pytest.raises(RuntimeError, match="SILENTLY IGNORED"):
        volumes["yz_plus"] = FAR_FIELD
    with pytest.raises(RuntimeError, match="SILENTLY IGNORED"):
        volumes.update({"yz_plus": FAR_FIELD})
    # reading is always fine
    assert volumes["yz_minus"] == FAR_FIELD


def test_freeze_pore_volumes_latches_reservoir_and_dict():
    from darts.reservoirs.reservoir_base import ReservoirBase

    reservoir = object.__new__(ReservoirBase)
    reservoir.cache = False  # ReservoirBase.__del__ reads it
    reservoir.boundary_volumes = BoundaryVolumeDict(
        dict.fromkeys(ConstantStateBC.FACES)
    )
    assert reservoir.pore_volumes_frozen is False
    reservoir.assert_pore_volumes_mutable("set_boundary_volume")  # no-op

    reservoir.freeze_pore_volumes()
    assert reservoir.boundary_volumes.frozen is True
    with pytest.raises(RuntimeError, match="set_boundary_volume"):
        reservoir.assert_pore_volumes_mutable("set_boundary_volume")

    reservoir.allow_pore_volume_updates()
    assert reservoir.boundary_volumes.frozen is False
    reservoir.assert_pore_volumes_mutable("set_boundary_volume")  # no-op again


def test_condition_set_compile_freezes_only_after_engine_init():
    from darts.models.conditions import ConditionSet
    from darts.reservoirs.reservoir_base import ReservoirBase

    class _Engine:
        def __init__(self, n_state):
            self.X = np.zeros(n_state)

    class _Physics:
        def __init__(self, engine):
            self.engine = engine

    class _Model:
        def __init__(self, engine):
            self.reservoir = object.__new__(ReservoirBase)
            self.reservoir.cache = False  # ReservoirBase.__del__ reads it
            self.reservoir.boundary_volumes = BoundaryVolumeDict(
                dict.fromkeys(ConstantStateBC.FACES)
            )
            self.physics = _Physics(engine)

    uninitialized = _Model(_Engine(0))
    ConditionSet().compile(uninitialized)
    assert uninitialized.reservoir.pore_volumes_frozen is False

    initialized = _Model(_Engine(4))
    ConditionSet().compile(initialized)
    assert initialized.reservoir.pore_volumes_frozen is True
    assert initialized.reservoir.boundary_volumes.frozen is True

    # a model without a reservoir (the empty-set fast path) is untouched
    ConditionSet().compile(object())
