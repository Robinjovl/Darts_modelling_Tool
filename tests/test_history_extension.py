"""Test the standalone OBL history-state support object."""

from types import SimpleNamespace

import numpy as np

from darts.physics.base.history_extension import HistoryField, HistoryStateSupport
from darts.physics.base.physics import PhysicsBase


def test_history_support_extends_axes_and_updates_engine_buffers() -> None:
    """Keep axis ordering and engine-buffer access consistent after extraction."""
    history = HistoryStateSupport(
        [
            HistoryField("sg_max", axes_origin=0.0, axes_step=0.1, default=0.2),
            HistoryField("cycle", axes_origin=1.0, axes_step=1.0, default=3.0),
        ]
    )
    engine = SimpleNamespace(
        X=np.array([100.0, 0.1, 200.0, 0.2]),
        Xhistory=np.array([0.4, 4.0, 0.5, 5.0]),
    )

    assert history.extend_axes([1.0, 0.0], [10.0, 0.01]) == (
        [1.0, 0.0, 0.0, 1.0],
        [10.0, 0.01, 0.1, 1.0],
    )
    np.testing.assert_allclose(history.get_engine_array(engine, "sg_max"), [0.4, 0.5])
    history.set_engine_array(engine, "cycle", 7.0)
    np.testing.assert_allclose(engine.Xhistory, [0.4, 7.0, 0.5, 7.0])
    np.testing.assert_allclose(
        history.get_interpolator_state(engine, n_vars=2),
        [100.0, 0.1, 0.4, 7.0, 200.0, 0.2, 0.5, 7.0],
    )


def test_history_support_propagates_defaults() -> None:
    """Populate boundary and well fallback values in cell-major order."""
    history = HistoryStateSupport(
        [HistoryField("sg_max", default=0.2), HistoryField("cycle", default=3.0)]
    )
    mesh = SimpleNamespace(n_bounds=2)
    control = SimpleNamespace()
    constraint = SimpleNamespace()
    well = SimpleNamespace(control=control, constraint=constraint)

    history.populate_boundary_defaults(mesh)
    history.populate_well_defaults([well])

    np.testing.assert_allclose(np.asarray(mesh.Xhistory_bounds), [0.2, 3.0, 0.2, 3.0])
    np.testing.assert_allclose(np.asarray(well.Xhistory_well_default), [0.2, 3.0])
    np.testing.assert_allclose(np.asarray(control.Xhistory_well_default), [0.2, 3.0])
    np.testing.assert_allclose(np.asarray(constraint.Xhistory_well_default), [0.2, 3.0])


def test_history_support_is_inert_without_fields() -> None:
    """An empty descriptor list disables every history operation without guards."""
    history = HistoryStateSupport()
    engine = SimpleNamespace(X=np.array([1.0, 2.0, 3.0, 4.0]))
    mesh = SimpleNamespace(n_bounds=2)
    well = SimpleNamespace()

    assert history.n_fields == 0
    assert history.labels == [] and history.defaults == []
    assert history.extend_axes([0.0], [1.0]) == ([0.0], [1.0])
    np.testing.assert_allclose(
        history.get_interpolator_state(engine, n_vars=2), [1.0, 2.0, 3.0, 4.0]
    )
    # No-ops rather than errors, so PhysicsBase never needs an `if history` guard.
    history.populate_boundary_defaults(mesh)
    history.populate_well_defaults([well])
    assert not hasattr(mesh, "Xhistory_bounds")
    assert not hasattr(well, "Xhistory_well_default")


def test_history_support_rejects_unknown_label_and_missing_fields() -> None:
    """Report a bad label and an unconfigured physics distinctly."""
    import pytest

    history = HistoryStateSupport([HistoryField("sg_max")])
    engine = SimpleNamespace(Xhistory=np.zeros(3))
    with pytest.raises(KeyError):
        history.default("nope")
    with pytest.raises(KeyError):
        history.get_engine_array(engine, "nope")

    empty = HistoryStateSupport()
    with pytest.raises(RuntimeError):
        empty.get_engine_array(engine, "sg_max")
    with pytest.raises(RuntimeError):
        empty.set_engine_array(engine, "sg_max", 1.0)


def test_history_field_validates_axis_metadata() -> None:
    """Reject descriptors that would produce an unusable OBL axis."""
    import pytest

    for kwargs in (
        {"label": ""},
        {"label": "h", "axes_step": 0.0},
        {"label": "h", "axes_step": float("inf")},
        {"label": "h", "axes_origin": float("nan")},
        {"label": "h", "default": float("inf")},
    ):
        with pytest.raises(ValueError):
            HistoryField(**kwargs)


def test_physics_history_fields_round_trip() -> None:
    """`PhysicsBase.history_fields` stays assignable after the extraction."""
    physics = PhysicsBase.__new__(PhysicsBase)

    physics.history_fields = []
    assert physics.n_history == 0
    assert physics.history_fields == []

    fields = [HistoryField("sg_max", default=0.2), HistoryField("cycle", default=3.0)]
    physics.history_fields = fields
    assert physics.n_history == 2
    assert physics.history.labels == ["sg_max", "cycle"]
    assert physics.get_history_default("cycle") == 3.0

    physics.history_fields = None
    assert physics.n_history == 0
