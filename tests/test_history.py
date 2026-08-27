"""Test the standalone OBL history-state support object."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import darts.physics.base.history as history_module
from darts.physics.base.history import HistoryField, HistoryStateSupport


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

    with patch.object(
        history_module, "_to_value_vector", side_effect=lambda values: values
    ):
        history.populate_boundary_defaults(mesh)
        history.populate_well_defaults([well])

    np.testing.assert_allclose(np.asarray(mesh.Xhistory_bounds), [0.2, 3.0, 0.2, 3.0])
    np.testing.assert_allclose(np.asarray(well.Xhistory_well_default), [0.2, 3.0])
    np.testing.assert_allclose(np.asarray(control.Xhistory_well_default), [0.2, 3.0])
    np.testing.assert_allclose(np.asarray(constraint.Xhistory_well_default), [0.2, 3.0])
