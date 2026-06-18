import pickle

from darts.physics.base.physics_base import PhysicsBase


class FakeDirtyItor:
    def __init__(self):
        self.point_data = {}
        self._dirty = set()

    def add(self, key, value):
        self.point_data[key] = value
        self._dirty.add(key)

    def point_data_size(self):
        return len(self.point_data)

    def point_data_delta(self):
        return {key: self.point_data[key] for key in self._dirty}

    def clear_point_data_delta(self):
        self._dirty.clear()


class FakeEpochItor:
    """Dirty-tracking itor that also stamps each point with an evaluation epoch,
    mirroring the native adaptive interpolators' point_data_epoch_delta()."""

    def __init__(self):
        self.point_data = {}
        self._dirty = set()
        self._epochs = {}

    def add(self, key, value, epoch):
        self.point_data[key] = value
        self._dirty.add(key)
        self._epochs[key] = epoch

    def point_data_size(self):
        return len(self.point_data)

    def point_data_delta(self):
        return {key: self.point_data[key] for key in self._dirty}

    def point_data_epoch_delta(self):
        return {key: self._epochs[key] for key in self._dirty}

    def clear_point_data_delta(self):
        self._dirty.clear()


class FakeFallbackItor:
    def __init__(self):
        self.point_data = {}

    def add(self, key, value):
        self.point_data[key] = value


def make_physics(itor, path):
    physics = object.__new__(PhysicsBase)
    physics.created_itors = [(itor, str(path))]
    physics._last_flushed_sizes = {}
    physics._flushed_point_keys = {}
    physics.cache = False
    return physics


def test_obl_cache_appends_only_dirty_points(tmp_path):
    path = tmp_path / "obl_point_data_test.pkl"
    itor = FakeDirtyItor()
    physics = make_physics(itor, path)

    itor.add(1, (1.0, 2.0))
    itor.add(2, (3.0, 4.0))
    physics.write_cache()
    base_bytes = path.read_bytes()
    assert pickle.loads(base_bytes) == {1: (1.0, 2.0), 2: (3.0, 4.0)}

    physics.write_cache()
    assert path.read_bytes() == base_bytes

    itor.add(3, (5.0, 6.0))
    physics.write_cache()
    appended_bytes = path.read_bytes()
    assert appended_bytes.startswith(base_bytes)
    assert len(appended_bytes) > len(base_bytes)
    assert physics._safe_pickle_load(str(path)) == {
        1: (1.0, 2.0),
        2: (3.0, 4.0),
        3: (5.0, 6.0),
    }


def test_obl_cache_fallback_tracks_flushed_keys(tmp_path):
    path = tmp_path / "obl_point_data_test.pkl"
    itor = FakeFallbackItor()
    physics = make_physics(itor, path)

    itor.add(1, (1.0,))
    physics.write_cache()
    base_bytes = path.read_bytes()

    itor.add(2, (2.0,))
    physics.write_cache()
    first_append = path.read_bytes()
    assert first_append.startswith(base_bytes)
    assert physics._safe_pickle_load(str(path)) == {1: (1.0,), 2: (2.0,)}

    physics.write_cache()
    assert path.read_bytes() == first_append


def test_obl_cache_ignores_truncated_delta_tail(tmp_path):
    path = tmp_path / "obl_point_data_test.pkl"
    itor = FakeDirtyItor()
    physics = make_physics(itor, path)

    physics._atomic_pickle_dump({1: (1.0,)}, str(path))
    physics._append_pickle_delta({2: (2.0,)}, str(path))
    with open(path, "ab") as fp:
        fp.write(physics._OBL_DELTA_MAGIC)
        fp.write(b"\x01")

    assert physics._safe_pickle_load(str(path)) == {1: (1.0,), 2: (2.0,)}


def test_obl_cache_records_point_epochs(tmp_path):
    path = tmp_path / "obl_point_data_test.pkl"
    itor = FakeEpochItor()
    physics = make_physics(itor, path)

    # First flush: base pickle (points) + epoch sidecar for those points.
    itor.add(1, (1.0, 2.0), epoch=0)
    itor.add(2, (3.0, 4.0), epoch=0)
    physics.write_cache()  # also clears the itor's dirty trackers

    # Subsequent points evaluated at a later nonlinear iteration -> appended delta + epochs.
    itor.add(3, (5.0, 6.0), epoch=7)
    physics.write_cache()

    # Point data still loads as a flat dict; epoch frames are skipped on the load path.
    assert physics._safe_pickle_load(str(path)) == {
        1: (1.0, 2.0),
        2: (3.0, 4.0),
        3: (5.0, 6.0),
    }

    # Epochs are recoverable for offline OBL-sampling analysis, for both the base and
    # the appended points.
    assert PhysicsBase.load_point_epochs(str(path)) == {1: 0, 2: 0, 3: 7}


def test_obl_cache_no_epoch_frames_without_support(tmp_path):
    # Interpolators without native epoch tracking must not gain epoch frames, keeping
    # the cache file byte-identical to the legacy append-only layout.
    path = tmp_path / "obl_point_data_test.pkl"
    itor = FakeDirtyItor()
    physics = make_physics(itor, path)

    itor.add(1, (1.0,))
    physics.write_cache()
    assert PhysicsBase.load_point_epochs(str(path)) == {}
