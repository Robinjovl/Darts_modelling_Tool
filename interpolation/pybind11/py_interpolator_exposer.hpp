#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals_interpolation.h"
#include "recursive_exposers.h"
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <type_traits>
#include <cstring>
#include <vector>

#include "multilinear_adaptive_cpu_interpolator.hpp"

#include "linear_adaptive_cpu_interpolator.hpp"
#ifdef WITH_GPU
#include "multilinear_adaptive_gpu_interpolator.hpp"
#endif //WITH_GPU

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Bulk array cache I/O for the adaptive interpolators.
//
// The tuple-keyed `point_data_full` property is correct but spends O(N) Python
// objects on both read and write (one boxed tuple per key and per value, plus a
// dict entry). For multi-GB OBL caches that boxing dominates the load time. These
// helpers move the entire cache as two contiguous numpy arrays instead:
//   keys : int32   [N, N_DIMS]   (signed multi-index of each supporting point)
//   vals : float64 [N, N_OPS]    (operator values at that point)
// so Python can persist/restore the cache via np.save / np.load(mmap_mode='r')
// with a single bulk memcpy and no per-point boxing. The value scalar type is
// taken from the map's mapped_type so this works for both the value-templated
// multilinear interpolators (float/double) and the always-double linear one.
// ---------------------------------------------------------------------------
template <typename interpolator_class, uint8_t N_DIMS, uint16_t N_OPS>
py::tuple bulk_get_point_data_arrays(const interpolator_class &self)
{
  // point_data.size() is exact; with an arena attached, the iterator pass below
  // is kept as an additional guard because it is the allocation size for the
  // following export and the iterator skips overlay-shadowed arena slots.
  size_t n;
  if (self.point_data.has_arena())
  {
    n = 0;
    for (auto it = self.point_data.begin(); it != self.point_data.end(); ++it)
      ++n;
  }
  else
  {
    n = self.point_data.size();
  }
  py::array_t<int32_t> keys({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(N_DIMS)});
  py::array_t<double> vals({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(N_OPS)});
  int32_t *kp = keys.mutable_data();
  double *vp = vals.mutable_data();
  size_t i = 0;
  for (const auto &kv : self.point_data)
  {
    int32_t *krow = kp + i * static_cast<size_t>(N_DIMS);
    for (uint8_t d = 0; d < N_DIMS; ++d)
      krow[d] = kv.first.idx[d];
    double *vrow = vp + i * static_cast<size_t>(N_OPS);
    for (uint16_t op = 0; op < N_OPS; ++op)
      vrow[op] = static_cast<double>(kv.second[op]);
    ++i;
  }
  return py::make_tuple(std::move(keys), std::move(vals));
}

template <typename interpolator_class, uint8_t N_DIMS, uint16_t N_OPS>
void bulk_set_point_data_arrays(interpolator_class &self,
                                py::array_t<int32_t, py::array::c_style | py::array::forcecast> keys,
                                py::array_t<double, py::array::c_style | py::array::forcecast> vals)
{
  py::buffer_info kb = keys.request();
  py::buffer_info vb = vals.request();
  if (kb.ndim != 2 || kb.shape[1] != static_cast<py::ssize_t>(N_DIMS))
    throw std::invalid_argument("set_point_data_arrays: keys must have shape (N, N_DIMS)");
  if (vb.ndim != 2 || vb.shape[1] != static_cast<py::ssize_t>(N_OPS))
    throw std::invalid_argument("set_point_data_arrays: vals must have shape (N, N_OPS)");
  if (kb.shape[0] != vb.shape[0])
    throw std::invalid_argument("set_point_data_arrays: keys and vals must have equal row counts");

  using key_t = typename interpolator_class::key_t;
  using mapped_t = typename std::decay_t<decltype(self.point_data)>::mapped_type;
  using scalar_t = typename mapped_t::value_type;

  const size_t n = static_cast<size_t>(kb.shape[0]);
  const int32_t *kp = static_cast<const int32_t *>(kb.ptr);
  const double *vp = static_cast<const double *>(vb.ptr);

  // Reload semantics mirror the point_data_full setter: replace the cache wholesale
  // and reset the dirty trackers (a fresh reload has no unpersisted points yet).
  self.point_data.clear();
  self.dirty_point_data.clear();
  self.dirty_point_epochs.clear();
  self.point_data.reserve(n);
  for (size_t i = 0; i < n; ++i)
  {
    key_t k;
    const int32_t *krow = kp + i * static_cast<size_t>(N_DIMS);
    for (uint8_t d = 0; d < N_DIMS; ++d)
      k.idx[d] = krow[d];
    mapped_t v;
    const double *vrow = vp + i * static_cast<size_t>(N_OPS);
    for (uint16_t op = 0; op < N_OPS; ++op)
      v[op] = static_cast<scalar_t>(vrow[op]);
    self.point_data.emplace(k, std::move(v));
  }
}

// Merge extra supporting points on top of the existing cache WITHOUT clearing it and
// WITHOUT marking them dirty. Used to fold in the pickle's trailing delta frames (points
// appended after the array snapshot was written) on top of a snapshot loaded via
// set_point_data_arrays: those points are already persisted, so they must not re-enter
// the dirty/append-on-flush set.
template <typename interpolator_class, uint8_t N_DIMS, uint16_t N_OPS>
void bulk_add_point_data_arrays(interpolator_class &self,
                                py::array_t<int32_t, py::array::c_style | py::array::forcecast> keys,
                                py::array_t<double, py::array::c_style | py::array::forcecast> vals)
{
  py::buffer_info kb = keys.request();
  py::buffer_info vb = vals.request();
  if (kb.ndim != 2 || kb.shape[1] != static_cast<py::ssize_t>(N_DIMS))
    throw std::invalid_argument("add_point_data_arrays: keys must have shape (N, N_DIMS)");
  if (vb.ndim != 2 || vb.shape[1] != static_cast<py::ssize_t>(N_OPS))
    throw std::invalid_argument("add_point_data_arrays: vals must have shape (N, N_OPS)");
  if (kb.shape[0] != vb.shape[0])
    throw std::invalid_argument("add_point_data_arrays: keys and vals must have equal row counts");

  using key_t = typename interpolator_class::key_t;
  using mapped_t = typename std::decay_t<decltype(self.point_data)>::mapped_type;
  using scalar_t = typename mapped_t::value_type;

  const size_t n = static_cast<size_t>(kb.shape[0]);
  const int32_t *kp = static_cast<const int32_t *>(kb.ptr);
  const double *vp = static_cast<const double *>(vb.ptr);

  self.point_data.reserve(self.point_data.size() + n);
  for (size_t i = 0; i < n; ++i)
  {
    key_t k;
    const int32_t *krow = kp + i * static_cast<size_t>(N_DIMS);
    for (uint8_t d = 0; d < N_DIMS; ++d)
      k.idx[d] = krow[d];
    mapped_t v;
    const double *vrow = vp + i * static_cast<size_t>(N_OPS);
    for (uint16_t op = 0; op < N_OPS; ++op)
      v[op] = static_cast<scalar_t>(vrow[op]);
    self.point_data[k] = v; // assign (overwrite if already present); leaves dirty set untouched
  }
}

// ---------------------------------------------------------------------------
// Single-point O(1) cache access, bypassing the interpolation machinery entirely.
//
// The bulk helpers above move the WHOLE cache; callers that only need to check/read/
// write one exact supporting point (e.g. a co-located evaluator sharing this
// interpolator's axes, reusing its point_data as a plain keyed cache instead of
// duplicating work in a Python-side dict) pay for a single hash lookup instead of an
// O(N) export. try_get_point returns None on a miss (no exception on the hot path);
// set_point mirrors the bookkeeping the interpolator's own on-miss materialization
// path performs (dirty_point_data / dirty_point_epochs), so the normal cache-flush /
// persistence machinery (OblCacheCodec) picks up points inserted this way exactly as
// if they had been evaluated through interpolate().
// ---------------------------------------------------------------------------
template <typename interpolator_class, uint8_t N_DIMS, uint16_t N_OPS>
py::object single_try_get_point(const interpolator_class &self,
                                py::array_t<int32_t, py::array::c_style | py::array::forcecast> key)
{
  py::buffer_info kb = key.request();
  if (kb.size != static_cast<py::ssize_t>(N_DIMS))
    throw std::invalid_argument("try_get_point: key must have length N_DIMS");
  typename interpolator_class::key_t k;
  const int32_t *kp = static_cast<const int32_t *>(kb.ptr);
  for (uint8_t d = 0; d < N_DIMS; ++d)
    k.idx[d] = kp[d];
  if (self.point_data.find(k) == self.point_data.end())
    return py::none();
  // Fetch via at() rather than the find()-returned const_iterator's operator->():
  // that iterator materializes a proxy entry{} (a placement-new'd struct holding a
  // *reference member* aliasing the stored array) purely to support the general
  // begin()/end() iteration protocol. at() returns a direct reference to the stored
  // std::array with no intermediate proxy -- the safer path for a single-value fetch.
  const auto &v = self.point_data.at(k);
  // MUST be a std::vector<py::ssize_t>, not a bare {N_OPS} braced literal: array_t has
  // two single-argument ctors -- array_t(ShapeContainer) and array_t(ssize_t count) --
  // and a one-element {N_OPS} is a viable argument for BOTH (list-init of a scalar from
  // a single-element list is a plain identity conversion, beating the user-defined
  // conversion to ShapeContainer), so {N_OPS} silently binds to the count ctor and the
  // "shape" is never used. That ctor derives strides from the runtime dtype descriptor
  // (dtype.itemsize()), which the vendored pybind11 (2.12.0.dev1, pre-NumPy-2 descriptor
  // layout) misreads as 0 under NumPy >= 2.0, yielding a stride-0 array (every element
  // aliases slot 0). A std::vector has no conversion to ssize_t, so it rules out the
  // count ctor entirely and forces ShapeContainer, whose strides come from the
  // compile-time sizeof(T) -- correct on both ABIs.
  py::array_t<double> out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(N_OPS)});
  double *op = out.mutable_data();
  for (uint16_t j = 0; j < N_OPS; ++j)
    op[j] = static_cast<double>(v[j]);
  return out;
}

template <typename interpolator_class, uint8_t N_DIMS, uint16_t N_OPS>
void single_set_point(interpolator_class &self,
                      py::array_t<int32_t, py::array::c_style | py::array::forcecast> key,
                      py::array_t<double, py::array::c_style | py::array::forcecast> vals)
{
  py::buffer_info kb = key.request();
  py::buffer_info vb = vals.request();
  if (kb.size != static_cast<py::ssize_t>(N_DIMS))
    throw std::invalid_argument("set_point: key must have length N_DIMS");
  if (vb.size != static_cast<py::ssize_t>(N_OPS))
    throw std::invalid_argument("set_point: vals must have length N_OPS");
  typename interpolator_class::key_t k;
  const int32_t *kp = static_cast<const int32_t *>(kb.ptr);
  for (uint8_t d = 0; d < N_DIMS; ++d)
    k.idx[d] = kp[d];
  using mapped_t = typename std::decay_t<decltype(self.point_data)>::mapped_type;
  mapped_t v;
  const double *vp = static_cast<const double *>(vb.ptr);
  for (uint16_t j = 0; j < N_OPS; ++j)
    v[j] = static_cast<typename mapped_t::value_type>(vp[j]);
  self.point_data[k] = v; // insert-or-overwrite (arena-shadow aware via operator[])
  self.dirty_point_data.insert(k);
  self.dirty_point_epochs[k] = self.eval_index;
}

// Contiguous-array export of ONLY the supporting points materialized since the last
// clear_point_data_delta() (the dirty set). Mirrors bulk_get_point_data_arrays but over
// dirty_point_data, so the append-on-flush path never boxes one Python tuple per point
// (PhysicsBase._point_data_delta_arrays prefers this when present). Returns
//   keys : int32   [M, N_DIMS]
//   vals : float64 [M, N_OPS]
template <typename interpolator_class, uint8_t N_DIMS, uint16_t N_OPS>
py::tuple bulk_point_data_delta_arrays(const interpolator_class &self)
{
  std::vector<int32_t> kbuf;
  std::vector<double> vbuf;
  kbuf.reserve(self.dirty_point_data.size() * static_cast<size_t>(N_DIMS));
  vbuf.reserve(self.dirty_point_data.size() * static_cast<size_t>(N_OPS));
  for (const auto &key : self.dirty_point_data)
  {
    auto it = self.point_data.find(key);
    if (it == self.point_data.end())
      continue; // defensive: a dirty key with no payload is skipped (keeps arrays dense)
    for (uint8_t d = 0; d < N_DIMS; ++d)
      kbuf.push_back(key.idx[d]);
    for (uint16_t op = 0; op < N_OPS; ++op)
      vbuf.push_back(static_cast<double>(it->second[op]));
  }
  const size_t m = vbuf.size() / static_cast<size_t>(N_OPS);
  py::array_t<int32_t> keys({static_cast<py::ssize_t>(m), static_cast<py::ssize_t>(N_DIMS)});
  py::array_t<double> vals({static_cast<py::ssize_t>(m), static_cast<py::ssize_t>(N_OPS)});
  if (m)
  {
    std::memcpy(keys.mutable_data(), kbuf.data(), kbuf.size() * sizeof(int32_t));
    std::memcpy(vals.mutable_data(), vbuf.data(), vbuf.size() * sizeof(double));
  }
  return py::make_tuple(std::move(keys), std::move(vals));
}

// Contiguous-array export of the per-point evaluation epoch for the dirty points. Returns
//   keys   : int32  [M, N_DIMS]
//   epochs : uint64 [M]
// Key order matches bulk_point_data_delta_arrays is NOT guaranteed (separate maps), so each
// is self-describing (keys + values together).
template <typename interpolator_class, uint8_t N_DIMS, uint16_t N_OPS>
py::tuple bulk_point_data_epoch_delta_arrays(const interpolator_class &self)
{
  std::vector<int32_t> kbuf;
  std::vector<uint64_t> ebuf;
  kbuf.reserve(self.dirty_point_epochs.size() * static_cast<size_t>(N_DIMS));
  ebuf.reserve(self.dirty_point_epochs.size());
  for (const auto &kv : self.dirty_point_epochs)
  {
    for (uint8_t d = 0; d < N_DIMS; ++d)
      kbuf.push_back(kv.first.idx[d]);
    ebuf.push_back(kv.second);
  }
  const size_t m = ebuf.size();
  py::array_t<int32_t> keys({static_cast<py::ssize_t>(m), static_cast<py::ssize_t>(N_DIMS)});
  // std::vector<py::ssize_t>{m}, NOT a bare {m} braced literal -- see the comment on
  // single_try_get_point's output array above: a one-element {m} is ambiguous between
  // array_t's ShapeContainer and ssize_t-count ctors and silently binds to the latter
  // (stride-0 under NumPy >= 2.0 with this vendored pybind11). A std::vector argument
  // rules out the count ctor entirely.
  py::array_t<uint64_t> eps(std::vector<py::ssize_t>{static_cast<py::ssize_t>(m)});
  if (m)
  {
    std::memcpy(keys.mutable_data(), kbuf.data(), kbuf.size() * sizeof(int32_t));
    std::memcpy(eps.mutable_data(), ebuf.data(), ebuf.size() * sizeof(uint64_t));
  }
  return py::make_tuple(std::move(keys), std::move(eps));
}

template <uint8_t N_DIMS, uint16_t N_OPS>
struct interpolator_exposer
{
  // template function used to expose different interpolators with the same Python interface.
  // Exposed name pattern: <base_name>_<s|d>_<N_DIMS>_<N_OPS>. Two tokens that used to sit
  // in <base_name> are gone: the index-type letter (_i_/_l_), because storage is keyed on
  // cell_key_t and the index type is no longer part of the class identity, and the
  // "adaptive" qualifier, because the static interpolators were removed and every
  // interpolator is adaptive. physics.py builds exactly this name -- there is no fallback
  // to the old spellings, so a stale compiled module fails with a "rebuild" message.
  template <typename f_t, typename interpolator_class>
  void expose_class(py::module &m, std::string base_name)
  {
    using namespace pybind11::literals;

    std::string name = base_name + '_';

    if (typeid(f_t) == typeid(float))
    {
      name = name + "s_";
    }
    else if (typeid(f_t) == typeid(double))
    {
      name = name + "d_";
    }
    else
    {
      std::cout << "Error: Unexpected value type id (" << typeid(f_t).name() << ") specified while exposing " << name << std::endl;
      return;
    }

    name = name + std::to_string(N_DIMS) + "_" + std::to_string(N_OPS);
    std::string f_typename = typeid(f_t).name();
    std::string long_name = "Operator set interpolator with " + f_typename + " value type for " + std::to_string(N_OPS) + " operators in " + std::to_string(N_DIMS) + "-dimensional parameter space";
    try
    {
      if constexpr (std::is_same_v<interpolator_class, multilinear_adaptive_cpu_interpolator<f_t, N_DIMS, N_OPS>>)
      {
        using point_data_t = typename interpolator_class::point_data_t;
        py::class_<interpolator_class,
          operator_set_gradient_evaluator_iface>(m, name.c_str(), long_name.c_str())
          .def(py::init<operator_set_evaluator_iface*, std::vector<value_t> &, std::vector<value_t> &>(), py::keep_alive<1, 2>()) /* (evaluator, axes_origin, axes_step) */
          .def("evaluate_with_derivatives", &interpolator_class::evaluate_with_derivatives,
            "Evaluate operators and derivatives (v)", "state"_a, "block_idx"_a, "values"_a, "derivatives"_a)
          .def("init_timer_node", &interpolator_class::init_timer_node,
            "Initialize timer", "timer_node"_a)
          .def("init", &interpolator_class::init, "Initialize interpolator")
          .def("write_to_file", &interpolator_class::write_to_file, "Write interpolator data to file")
          .def("evaluate", &interpolator_class::evaluate,
            "Evaluate operators", "state"_a, "values"_a)
          // point_data_full: lossless export of the entire cell-key-indexed cache as a
          // dict keyed on tuple-of-ints. This is the canonical cache I/O format now that
          // the grid is unbounded (there is no integer-key packing reach any more).
          .def_property("point_data_full",
            [](const interpolator_class& self) {
              py::dict out;
              for (const auto& kv : self.point_data) {
                py::tuple tk(N_DIMS);
                for (uint8_t d = 0; d < N_DIMS; ++d)
                  tk[d] = kv.first.idx[d];
                py::tuple tv(N_OPS);
                for (uint8_t op = 0; op < N_OPS; ++op)
                  tv[op] = kv.second[op];
                out[tk] = tv;
              }
              return out;
            },
            [](interpolator_class& self, const py::dict& d) {
              self.point_data.clear();
              // Keep dirty tracker in sync with the cache it shadows: a fresh reload
              // through point_data_full means there are no unpersisted points yet.
              self.dirty_point_data.clear();
              self.dirty_point_epochs.clear();
              self.point_data.reserve(d.size());
              for (auto item : d) {
                py::tuple tk = item.first.cast<py::tuple>();
                if (tk.size() != N_DIMS)
                  throw std::invalid_argument("point_data_full key tuple length mismatch");
                typename interpolator_class::key_t k;
                for (uint8_t dim = 0; dim < N_DIMS; ++dim)
                  k.idx[dim] = tk[dim].cast<int32_t>();
                py::sequence tv = item.second.cast<py::sequence>();
                if (static_cast<uint8_t>(tv.size()) != N_OPS)
                  throw std::invalid_argument("point_data_full value length mismatch");
                point_data_t v;
                for (uint8_t op = 0; op < N_OPS; ++op)
                  v[op] = tv[op].cast<double>();
                self.point_data.emplace(k, v);
              }
            })
          // Fast bulk array cache I/O (see bulk_*_point_data_arrays above): same cache
          // contents as point_data_full, but moved as contiguous numpy arrays so large
          // OBL caches load/save without per-point Python boxing.
          .def("get_point_data_arrays",
            [](const interpolator_class &self) {
              return bulk_get_point_data_arrays<interpolator_class, N_DIMS, N_OPS>(self);
            },
            "Export the whole cache as (keys:int32[N,N_DIMS], vals:float64[N,N_OPS]) arrays")
          .def("set_point_data_arrays",
            [](interpolator_class &self,
               py::array_t<int32_t, py::array::c_style | py::array::forcecast> keys,
               py::array_t<double, py::array::c_style | py::array::forcecast> vals) {
              bulk_set_point_data_arrays<interpolator_class, N_DIMS, N_OPS>(self, keys, vals);
            },
            "Bulk-load the cache from (keys, vals) numpy arrays (replaces existing cache)",
            "keys"_a, "vals"_a)
          .def("add_point_data_arrays",
            [](interpolator_class &self,
               py::array_t<int32_t, py::array::c_style | py::array::forcecast> keys,
               py::array_t<double, py::array::c_style | py::array::forcecast> vals) {
              bulk_add_point_data_arrays<interpolator_class, N_DIMS, N_OPS>(self, keys, vals);
            },
            "Merge (keys, vals) into the cache without clearing it or marking points dirty",
            "keys"_a, "vals"_a)
          .def("get_n_cached_points", &interpolator_class::get_n_cached_points,
            "Number of supporting points currently in the adaptive cache")
          .def("get_n_cached_hypercubes", &interpolator_class::get_n_cached_hypercubes,
            "Number of hypercubes currently in the adaptive cache")
          .def("get_axis_overflow_count", &interpolator_class::get_axis_overflow_count,
            "Cumulative count of per-axis cell indices that overflowed int32 and were "
            "saturation-clamped (all batches). Nonzero => some queried state fell so far "
            "outside the OBL grid that its multi-index did not fit in int32 (typically a "
            "diverging Newton step); a one-time host warning is also emitted.")
          .def("set_hypercube_cap", &interpolator_class::set_hypercube_cap,
            "Bound the in-memory derived hypercube cache to ~N most-recently-used "
            "entries (0 = unbounded). Caps peak RAM; the persisted supporting-point "
            "cache and its file format are untouched.", "cap"_a)
          .def("get_hypercube_cap", &interpolator_class::get_hypercube_cap,
            "Current hypercube cache cap (0 = unbounded)")
          .def("clear_hypercube_data", &interpolator_class::clear_hypercube_data,
            "Drop all cached hypercube payloads (rebuilt on demand from supporting "
            "points; no flash). Releases memory; supporting-point cache untouched.")
          // get_hypercube_keys: signed multi-index of every generated hypercube, as a
          // list of int tuples. Replaces the legacy integer get_hypercube_indexes()
          // (unbounded grid -> no integer packing). Used for body-path occupancy output.
          .def("get_hypercube_keys", [](const interpolator_class& self) {
              py::list out;
              for (const auto& k : self.get_hypercube_keys()) {
                py::tuple t(N_DIMS);
                for (uint8_t d = 0; d < N_DIMS; ++d) t[d] = k.idx[d];
                out.append(t);
              }
              return out;
            }, "Multi-index keys of all generated hypercubes (list of int tuples)")
          // Append-only cache hooks (from development): expose just the supporting
          // points materialized since the last clear_point_data_delta(), so Python
          // can persist only newly evaluated points. Keys are exported in the same
          // tuple-of-int shape as point_data_full.
          .def("point_data_size", [](const interpolator_class &self) {
            return self.point_data.size();
          })
          .def("point_data_delta", [](const interpolator_class &self) {
            py::dict delta;
            for (const auto &key : self.dirty_point_data) {
              auto item = self.point_data.find(key);
              if (item == self.point_data.end()) continue;
              py::tuple tk(N_DIMS);
              for (uint8_t d = 0; d < N_DIMS; ++d) tk[d] = key.idx[d];
              py::tuple tv(N_OPS);
              for (uint8_t op = 0; op < N_OPS; ++op) tv[op] = item->second[op];
              delta[tk] = tv;
            }
            return delta;
          })
          // Per-point evaluation epoch (batch-interpolation / nonlinear-iteration index)
          // for the points materialized since the last clear_point_data_delta(); keys
          // match point_data_delta()/point_data_full so Python can persist the sampling
          // epoch alongside the append-only OBL cache delta.
          .def("point_data_epoch_delta", [](const interpolator_class &self) {
            py::dict epochs;
            for (const auto &kv : self.dirty_point_epochs) {
              py::tuple tk(N_DIMS);
              for (uint8_t d = 0; d < N_DIMS; ++d) tk[d] = kv.first.idx[d];
              epochs[tk] = kv.second;
            }
            return epochs;
          })
          // Contiguous-array twins of point_data_delta()/point_data_epoch_delta(): the
          // append-on-flush path uses these (when present) to avoid boxing one Python
          // tuple per dirty point. See bulk_point_data_*_delta_arrays above.
          .def("point_data_delta_arrays",
            [](const interpolator_class &self) {
              return bulk_point_data_delta_arrays<interpolator_class, N_DIMS, N_OPS>(self);
            },
            "Dirty supporting points since last clear as (keys:int32[M,N_DIMS], vals:float64[M,N_OPS])")
          .def("point_data_epoch_delta_arrays",
            [](const interpolator_class &self) {
              return bulk_point_data_epoch_delta_arrays<interpolator_class, N_DIMS, N_OPS>(self);
            },
            "Dirty-point epochs since last clear as (keys:int32[M,N_DIMS], epochs:uint64[M])")
          .def("clear_point_data_delta", [](interpolator_class &self) {
            self.dirty_point_data.clear();
            self.dirty_point_epochs.clear();
          })
          // ---- mmap-arena bindings (point_data_store hybrid) ----
          .def("obl_arena_hash_id", [](interpolator_class &self) {
            return std::decay_t<decltype(self.point_data)>::arena_hash_id();
          }, "ABI/placement fingerprint; an arena with a mismatching id is rebuilt, never mis-probed")
          .def("has_arena", [](interpolator_class &self) { return self.point_data.has_arena(); },
            "True if an mmap'd base arena is currently attached")
          .def("build_arena_file", [](interpolator_class &self, const std::string &path, uint64_t hash_id) {
            self.point_data.build_arena_file(path, hash_id);
          }, "Write a complete arena file (open-addressing hash table) from the live union (overlay + arena)",
             "path"_a, "hash_id"_a)
          .def("mmap_arena", [](interpolator_class &self, const std::string &path, size_t bitmap_off,
                                size_t keys_off, size_t vals_off, size_t capacity, size_t count) {
            self.point_data.mmap_arena_at(path, bitmap_off, keys_off, vals_off, capacity, count);
          }, "mmap an arena in place (O(1) load, no per-point rebuild); offsets parsed from the header by Python",
             "path"_a, "bitmap_off"_a, "keys_off"_a, "vals_off"_a, "capacity"_a, "count"_a)
          // ---- single-point O(1) cache access (see single_try_get_point/single_set_point) ----
          .def("try_get_point",
            [](const interpolator_class &self, py::array_t<int32_t, py::array::c_style | py::array::forcecast> key) {
              return single_try_get_point<interpolator_class, N_DIMS, N_OPS>(self, key);
            },
            "Look up one supporting point by its int32[N_DIMS] multi-index key; "
            "returns a float64[N_OPS] array on hit, None on miss. No interpolation, "
            "no hypercube materialization -- a plain cache lookup.", "key"_a)
          .def("set_point",
            [](interpolator_class &self, py::array_t<int32_t, py::array::c_style | py::array::forcecast> key,
               py::array_t<double, py::array::c_style | py::array::forcecast> vals) {
              single_set_point<interpolator_class, N_DIMS, N_OPS>(self, key, vals);
            },
            "Insert or overwrite one supporting point (int32[N_DIMS] key, float64[N_OPS] "
            "values), marking it dirty for the next incremental cache flush -- same "
            "bookkeeping as a normal on-miss materialization.", "key"_a, "vals"_a)
          ;
      }
      else if constexpr (std::is_same_v<interpolator_class, linear_adaptive_cpu_interpolator<N_DIMS, N_OPS>>)
      {
        // Linear adaptive: storage is keyed on multi-index (cell_key_t). Python
        // cache I/O goes through the tuple-keyed point_data_full / point_data_delta
        // exports below; the legacy integer-keyed point_data view is retired.
        using point_value_t = std::array<double, N_OPS>;
        py::class_<interpolator_class,
          operator_set_gradient_evaluator_iface>(m, name.c_str(), long_name.c_str())
          .def(py::init<operator_set_evaluator_iface*, std::vector<value_t> &, std::vector<value_t> &, bool>(), py::keep_alive<1, 2>()) /* (evaluator, axes_origin, axes_step, is_barycentric) */
          .def("evaluate_with_derivatives", &interpolator_class::evaluate_with_derivatives,
            "Evaluate operators and derivatives (v)", "state"_a, "block_idx"_a, "values"_a, "derivatives"_a)
          .def("init_timer_node", &interpolator_class::init_timer_node,
            "Initialize timer", "timer_node"_a)
          .def("init", &interpolator_class::init, "Initialize interpolator")
          .def("write_to_file", &interpolator_class::write_to_file, "Write interpolator data to file")
          .def("evaluate", &interpolator_class::evaluate,
            "Evaluate operators", "state"_a, "values"_a)
          // point_data_full: lossless export keyed on tuple-of-ints. Canonical cache I/O
          // format (unbounded grid -> no integer-key packing). See multilinear branch.
          .def_property("point_data_full",
            [](const interpolator_class& self) {
              py::dict out;
              for (const auto& kv : self.point_data) {
                py::tuple tk(N_DIMS);
                for (uint8_t d = 0; d < N_DIMS; ++d)
                  tk[d] = kv.first.idx[d];
                py::tuple tv(N_OPS);
                for (uint8_t op = 0; op < N_OPS; ++op)
                  tv[op] = kv.second[op];
                out[tk] = tv;
              }
              return out;
            },
            [](interpolator_class& self, const py::dict& d) {
              self.point_data.clear();
              // Keep dirty tracker in sync with the cache it shadows: a fresh reload
              // through point_data_full means there are no unpersisted points yet.
              self.dirty_point_data.clear();
              self.dirty_point_epochs.clear();
              self.point_data.reserve(d.size());
              for (auto item : d) {
                py::tuple tk = item.first.cast<py::tuple>();
                if (tk.size() != N_DIMS)
                  throw std::invalid_argument("point_data_full key tuple length mismatch");
                typename interpolator_class::key_t k;
                for (uint8_t dim = 0; dim < N_DIMS; ++dim)
                  k.idx[dim] = tk[dim].cast<int32_t>();
                py::sequence tv = item.second.cast<py::sequence>();
                if (static_cast<uint8_t>(tv.size()) != N_OPS)
                  throw std::invalid_argument("point_data_full value length mismatch");
                point_value_t v;
                for (uint8_t op = 0; op < N_OPS; ++op)
                  v[op] = tv[op].cast<double>();
                self.point_data.emplace(k, v);
              }
            })
          // Fast bulk array cache I/O (see bulk_*_point_data_arrays above).
          .def("get_point_data_arrays",
            [](const interpolator_class &self) {
              return bulk_get_point_data_arrays<interpolator_class, N_DIMS, N_OPS>(self);
            },
            "Export the whole cache as (keys:int32[N,N_DIMS], vals:float64[N,N_OPS]) arrays")
          .def("set_point_data_arrays",
            [](interpolator_class &self,
               py::array_t<int32_t, py::array::c_style | py::array::forcecast> keys,
               py::array_t<double, py::array::c_style | py::array::forcecast> vals) {
              bulk_set_point_data_arrays<interpolator_class, N_DIMS, N_OPS>(self, keys, vals);
            },
            "Bulk-load the cache from (keys, vals) numpy arrays (replaces existing cache)",
            "keys"_a, "vals"_a)
          .def("add_point_data_arrays",
            [](interpolator_class &self,
               py::array_t<int32_t, py::array::c_style | py::array::forcecast> keys,
               py::array_t<double, py::array::c_style | py::array::forcecast> vals) {
              bulk_add_point_data_arrays<interpolator_class, N_DIMS, N_OPS>(self, keys, vals);
            },
            "Merge (keys, vals) into the cache without clearing it or marking points dirty",
            "keys"_a, "vals"_a)
          .def("get_n_cached_points", &interpolator_class::get_n_cached_points,
            "Number of supporting points currently in the adaptive cache (in-bounds + out-of-bounds)")
          // Append-only cache hooks (from development): persist only newly evaluated
          // supporting points. Keys exported as tuple-of-ints matching point_data_full.
          .def("point_data_size", [](const interpolator_class &self) {
            return self.point_data.size();
          })
          .def("point_data_delta", [](const interpolator_class &self) {
            py::dict delta;
            for (const auto &key : self.dirty_point_data) {
              auto item = self.point_data.find(key);
              if (item == self.point_data.end()) continue;
              py::tuple tk(N_DIMS);
              for (uint8_t d = 0; d < N_DIMS; ++d) tk[d] = key.idx[d];
              py::tuple tv(N_OPS);
              for (uint8_t op = 0; op < N_OPS; ++op) tv[op] = item->second[op];
              delta[tk] = tv;
            }
            return delta;
          })
          // Per-point evaluation epoch (batch-interpolation / nonlinear-iteration index)
          // for the points materialized since the last clear_point_data_delta(); keys
          // match point_data_delta()/point_data_full so Python can persist the sampling
          // epoch alongside the append-only OBL cache delta.
          .def("point_data_epoch_delta", [](const interpolator_class &self) {
            py::dict epochs;
            for (const auto &kv : self.dirty_point_epochs) {
              py::tuple tk(N_DIMS);
              for (uint8_t d = 0; d < N_DIMS; ++d) tk[d] = kv.first.idx[d];
              epochs[tk] = kv.second;
            }
            return epochs;
          })
          // Contiguous-array twins of point_data_delta()/point_data_epoch_delta(): the
          // append-on-flush path uses these (when present) to avoid boxing one Python
          // tuple per dirty point. See bulk_point_data_*_delta_arrays above.
          .def("point_data_delta_arrays",
            [](const interpolator_class &self) {
              return bulk_point_data_delta_arrays<interpolator_class, N_DIMS, N_OPS>(self);
            },
            "Dirty supporting points since last clear as (keys:int32[M,N_DIMS], vals:float64[M,N_OPS])")
          .def("point_data_epoch_delta_arrays",
            [](const interpolator_class &self) {
              return bulk_point_data_epoch_delta_arrays<interpolator_class, N_DIMS, N_OPS>(self);
            },
            "Dirty-point epochs since last clear as (keys:int32[M,N_DIMS], epochs:uint64[M])")
          .def("clear_point_data_delta", [](interpolator_class &self) {
            self.dirty_point_data.clear();
            self.dirty_point_epochs.clear();
          })
          // ---- mmap-arena bindings (point_data_store hybrid) ----
          .def("obl_arena_hash_id", [](interpolator_class &self) {
            return std::decay_t<decltype(self.point_data)>::arena_hash_id();
          }, "ABI/placement fingerprint; an arena with a mismatching id is rebuilt, never mis-probed")
          .def("has_arena", [](interpolator_class &self) { return self.point_data.has_arena(); },
            "True if an mmap'd base arena is currently attached")
          .def("build_arena_file", [](interpolator_class &self, const std::string &path, uint64_t hash_id) {
            self.point_data.build_arena_file(path, hash_id);
          }, "Write a complete arena file (open-addressing hash table) from the live union (overlay + arena)",
             "path"_a, "hash_id"_a)
          .def("mmap_arena", [](interpolator_class &self, const std::string &path, size_t bitmap_off,
                                size_t keys_off, size_t vals_off, size_t capacity, size_t count) {
            self.point_data.mmap_arena_at(path, bitmap_off, keys_off, vals_off, capacity, count);
          }, "mmap an arena in place (O(1) load, no per-point rebuild); offsets parsed from the header by Python",
             "path"_a, "bitmap_off"_a, "keys_off"_a, "vals_off"_a, "capacity"_a, "count"_a)
          // ---- single-point O(1) cache access (see single_try_get_point/single_set_point) ----
          .def("try_get_point",
            [](const interpolator_class &self, py::array_t<int32_t, py::array::c_style | py::array::forcecast> key) {
              return single_try_get_point<interpolator_class, N_DIMS, N_OPS>(self, key);
            },
            "Look up one supporting point by its int32[N_DIMS] multi-index key; "
            "returns a float64[N_OPS] array on hit, None on miss. No interpolation, "
            "no hypercube materialization -- a plain cache lookup.", "key"_a)
          .def("set_point",
            [](interpolator_class &self, py::array_t<int32_t, py::array::c_style | py::array::forcecast> key,
               py::array_t<double, py::array::c_style | py::array::forcecast> vals) {
              single_set_point<interpolator_class, N_DIMS, N_OPS>(self, key, vals);
            },
            "Insert or overwrite one supporting point (int32[N_DIMS] key, float64[N_OPS] "
            "values), marking it dirty for the next incremental cache flush -- same "
            "bookkeeping as a normal on-miss materialization.", "key"_a, "vals"_a)
          .def_readwrite("use_barycentric_interpolation", &interpolator_class::use_barycentric_interpolation);
      }
#ifdef WITH_GPU
      else if constexpr (std::is_same_v<interpolator_class, multilinear_adaptive_gpu_interpolator<f_t, N_DIMS, N_OPS>>)
      {
        // GPU adaptive multilinear: unbounded grid keyed on cell_key_t; cache I/O via
        // the tuple-keyed point_data_full view (no integer-key packing).
        using point_data_t = typename interpolator_class::point_data_t;
        py::class_<interpolator_class,
          operator_set_gradient_evaluator_iface>(m, name.c_str(), long_name.c_str())
          .def(py::init<operator_set_evaluator_iface*, std::vector<value_t> &, std::vector<value_t> &>(), py::keep_alive<1, 2>()) /* (evaluator, axes_origin, axes_step) */
          .def("evaluate_with_derivatives", &interpolator_class::evaluate_with_derivatives,
            "Evaluate operators and derivatives (v)", "state"_a, "block_idx"_a, "values"_a, "derivatives"_a)
          .def("init_timer_node", &interpolator_class::init_timer_node,
            "Initialize timer", "timer_node"_a)
          .def("init", &interpolator_class::init, "Initialize interpolator")
          .def("write_to_file", &interpolator_class::write_to_file, "Write interpolator data to file")
          .def("evaluate", &interpolator_class::evaluate,
            "Evaluate operators", "state"_a, "values"_a)
          .def_property("point_data_full",
            [](const interpolator_class& self) {
              py::dict out;
              for (const auto& kv : self.point_data) {
                py::tuple tk(N_DIMS);
                for (uint8_t d = 0; d < N_DIMS; ++d)
                  tk[d] = kv.first.idx[d];
                py::tuple tv(N_OPS);
                for (uint8_t op = 0; op < N_OPS; ++op)
                  tv[op] = kv.second[op];
                out[tk] = tv;
              }
              return out;
            },
            [](interpolator_class& self, const py::dict& d) {
              self.point_data.clear();
              // Keep dirty tracker in sync with the cache it shadows: a fresh reload
              // through point_data_full means there are no unpersisted points yet.
              self.dirty_point_data.clear();
              self.dirty_point_epochs.clear();
              self.point_data.reserve(d.size());
              for (auto item : d) {
                py::tuple tk = item.first.cast<py::tuple>();
                if (tk.size() != N_DIMS)
                  throw std::invalid_argument("point_data_full key tuple length mismatch");
                typename interpolator_class::key_t k;
                for (uint8_t dim = 0; dim < N_DIMS; ++dim)
                  k.idx[dim] = tk[dim].cast<int32_t>();
                py::sequence tv = item.second.cast<py::sequence>();
                if (static_cast<uint8_t>(tv.size()) != N_OPS)
                  throw std::invalid_argument("point_data_full value length mismatch");
                point_data_t v;
                for (uint8_t op = 0; op < N_OPS; ++op)
                  v[op] = tv[op].cast<double>();
                self.point_data.emplace(k, v);
              }
            })
          // Fast bulk array cache I/O (see bulk_*_point_data_arrays above).
          .def("get_point_data_arrays",
            [](const interpolator_class &self) {
              return bulk_get_point_data_arrays<interpolator_class, N_DIMS, N_OPS>(self);
            },
            "Export the whole cache as (keys:int32[N,N_DIMS], vals:float64[N,N_OPS]) arrays")
          .def("set_point_data_arrays",
            [](interpolator_class &self,
               py::array_t<int32_t, py::array::c_style | py::array::forcecast> keys,
               py::array_t<double, py::array::c_style | py::array::forcecast> vals) {
              bulk_set_point_data_arrays<interpolator_class, N_DIMS, N_OPS>(self, keys, vals);
            },
            "Bulk-load the cache from (keys, vals) numpy arrays (replaces existing cache)",
            "keys"_a, "vals"_a)
          .def("add_point_data_arrays",
            [](interpolator_class &self,
               py::array_t<int32_t, py::array::c_style | py::array::forcecast> keys,
               py::array_t<double, py::array::c_style | py::array::forcecast> vals) {
              bulk_add_point_data_arrays<interpolator_class, N_DIMS, N_OPS>(self, keys, vals);
            },
            "Merge (keys, vals) into the cache without clearing it or marking points dirty",
            "keys"_a, "vals"_a)
          // ---- single-point O(1) cache access (see single_try_get_point/single_set_point) ----
          .def("try_get_point",
            [](const interpolator_class &self, py::array_t<int32_t, py::array::c_style | py::array::forcecast> key) {
              return single_try_get_point<interpolator_class, N_DIMS, N_OPS>(self, key);
            },
            "Look up one supporting point by its int32[N_DIMS] multi-index key; "
            "returns a float64[N_OPS] array on hit, None on miss. No interpolation, "
            "no hypercube materialization -- a plain cache lookup. NOTE: does not "
            "invalidate any already-materialized device hypercube containing this "
            "point; call clear_hypercube_data() if that matters for your use case "
            "(same caveat as the CPU adaptive interpolator).", "key"_a)
          .def("set_point",
            [](interpolator_class &self, py::array_t<int32_t, py::array::c_style | py::array::forcecast> key,
               py::array_t<double, py::array::c_style | py::array::forcecast> vals) {
              single_set_point<interpolator_class, N_DIMS, N_OPS>(self, key, vals);
            },
            "Insert or overwrite one supporting point (int32[N_DIMS] key, float64[N_OPS] "
            "values), marking it dirty for the next incremental cache flush -- same "
            "bookkeeping as evaluating through interpolate().", "key"_a, "vals"_a)
          .def("get_n_cached_points", &interpolator_class::get_n_cached_points)
          .def("get_n_cached_hypercubes", &interpolator_class::get_n_cached_hypercubes)
          .def("get_axis_overflow_count", &interpolator_class::get_axis_overflow_count,
            "Cumulative count of per-axis cell indices that overflowed int32 and were "
            "saturation-clamped (all batches); nonzero indicates states driven far outside "
            "the OBL grid (typically a diverging Newton step).")
          .def("set_hypercube_cap", &interpolator_class::set_hypercube_cap,
            "Bound the device hypercube cache to ~N hypercubes (0 = unbounded); on "
            "overflow the device map + host key tracker are dropped and rebuilt on "
            "demand from supporting points (no flash). On-disk cache untouched.",
            "cap"_a)
          .def("get_hypercube_cap", &interpolator_class::get_hypercube_cap,
            "Current hypercube cache cap (0 = unbounded)")
          .def("clear_hypercube_data", &interpolator_class::clear_hypercube_data,
            "Drop all device hypercube payloads + host key tracker (rebuilt on demand "
            "from supporting points; no flash). Supporting-point cache untouched.")
          // Append-only cache hooks (from development): mirror the CPU adaptive
          // interpolators so Python persists only newly evaluated supporting points.
          .def("point_data_size", [](const interpolator_class &self) {
            return self.point_data.size();
          })
          .def("point_data_delta", [](const interpolator_class &self) {
            py::dict delta;
            for (const auto &key : self.dirty_point_data) {
              auto item = self.point_data.find(key);
              if (item == self.point_data.end()) continue;
              py::tuple tk(N_DIMS);
              for (uint8_t d = 0; d < N_DIMS; ++d) tk[d] = key.idx[d];
              py::tuple tv(N_OPS);
              for (uint8_t op = 0; op < N_OPS; ++op) tv[op] = item->second[op];
              delta[tk] = tv;
            }
            return delta;
          })
          // Per-point evaluation epoch (batch-interpolation / nonlinear-iteration index)
          // for the points materialized since the last clear_point_data_delta(); keys
          // match point_data_delta()/point_data_full so Python can persist the sampling
          // epoch alongside the append-only OBL cache delta.
          .def("point_data_epoch_delta", [](const interpolator_class &self) {
            py::dict epochs;
            for (const auto &kv : self.dirty_point_epochs) {
              py::tuple tk(N_DIMS);
              for (uint8_t d = 0; d < N_DIMS; ++d) tk[d] = kv.first.idx[d];
              epochs[tk] = kv.second;
            }
            return epochs;
          })
          // Contiguous-array twins of point_data_delta()/point_data_epoch_delta(): the
          // append-on-flush path uses these (when present) to avoid boxing one Python
          // tuple per dirty point. See bulk_point_data_*_delta_arrays above.
          .def("point_data_delta_arrays",
            [](const interpolator_class &self) {
              return bulk_point_data_delta_arrays<interpolator_class, N_DIMS, N_OPS>(self);
            },
            "Dirty supporting points since last clear as (keys:int32[M,N_DIMS], vals:float64[M,N_OPS])")
          .def("point_data_epoch_delta_arrays",
            [](const interpolator_class &self) {
              return bulk_point_data_epoch_delta_arrays<interpolator_class, N_DIMS, N_OPS>(self);
            },
            "Dirty-point epochs since last clear as (keys:int32[M,N_DIMS], epochs:uint64[M])")
          .def("clear_point_data_delta", [](interpolator_class &self) {
            self.dirty_point_data.clear();
            self.dirty_point_epochs.clear();
          })
          // ---- mmap-arena bindings (point_data_store hybrid) ----
          .def("obl_arena_hash_id", [](interpolator_class &self) {
            return std::decay_t<decltype(self.point_data)>::arena_hash_id();
          }, "ABI/placement fingerprint; an arena with a mismatching id is rebuilt, never mis-probed")
          .def("has_arena", [](interpolator_class &self) { return self.point_data.has_arena(); },
            "True if an mmap'd base arena is currently attached")
          .def("build_arena_file", [](interpolator_class &self, const std::string &path, uint64_t hash_id) {
            self.point_data.build_arena_file(path, hash_id);
          }, "Write a complete arena file (open-addressing hash table) from the live union (overlay + arena)",
             "path"_a, "hash_id"_a)
          .def("mmap_arena", [](interpolator_class &self, const std::string &path, size_t bitmap_off,
                                size_t keys_off, size_t vals_off, size_t capacity, size_t count) {
            self.point_data.mmap_arena_at(path, bitmap_off, keys_off, vals_off, capacity, count);
          }, "mmap an arena in place (O(1) load, no per-point rebuild); offsets parsed from the header by Python",
             "path"_a, "bitmap_off"_a, "keys_off"_a, "vals_off"_a, "capacity"_a, "count"_a)
          ;
      }
#endif
      else {
        // Fallback branch: static multilinear (dense bounded grid) — (origin, step, points).
        py::class_<interpolator_class,
          operator_set_gradient_evaluator_iface>(m, name.c_str(), long_name.c_str())
          .def(py::init<operator_set_evaluator_iface*, std::vector<value_t> &, std::vector<value_t> &, std::vector<index_t> &>(), py::keep_alive<1, 2>()) /* (evaluator, axes_origin, axes_step, axes_points) */
          .def("evaluate_with_derivatives", &interpolator_class::evaluate_with_derivatives,
            "Evaluate operators and derivatives (v)", "state"_a, "block_idx"_a, "values"_a, "derivatives"_a)
          .def("init_timer_node", &interpolator_class::init_timer_node,
            "Initialize timer", "timer_node"_a)
          .def("init", &interpolator_class::init, "Initialize interpolator")
          .def("write_to_file", &interpolator_class::write_to_file, "Write interpolator data to file")
          .def("evaluate", &interpolator_class::evaluate,
            "Evaluate operators", "state"_a, "values"_a)
          .def_readwrite("point_data", &interpolator_class::point_data);
      }
    }
    catch (const std::exception& e)
    {
      // expected for overlapping templated registrations; flag unexpected errors loudly
      // so we don't silently drop attributes from a class that registered partially.
      std::string what(e.what());
      const bool is_duplicate =
          what.find("already registered") != std::string::npos ||
          what.find("already defined") != std::string::npos;
      if (!is_duplicate)
      {
        std::cerr << "pybind registration error for " << name << ": " << what << std::endl;
      }
    }
    catch (...)
    {
      std::cerr << "pybind registration unknown error for " << name << std::endl;
    }
  }

  // here we specify which types and what interpolators are going to be exposed.
  // the specification seriously affects build time and binary size.
  //
  // CPU variant selection is driven by the OPENDARTS_INTERPOLATOR_PROFILE cmake
  // variable, forwarded as one of OD_INTERP_PROFILE_MINIMAL / OD_INTERP_PROFILE_FULL.
  // Default (no macro defined) behaves like FULL so older build scripts keep
  // working.
  //   MINIMAL: only multilinear_adaptive. Drops linear; callers that set
  //            itor_type='linear' must switch to 'multilinear'.
  //   FULL:    multilinear_adaptive + linear_adaptive.
  // One exposed class per (algorithm, platform, precision): the adaptive classes carry
  // no index-type template parameter any more (storage is keyed on cell_key_t), so the
  // former uint32/uint64 duplicates are gone and names carry no index-type letter. The
  // exposed names also drop the "adaptive" token: with the static interpolators removed
  // it no longer distinguishes anything (the C++ class names keep it).
  void expose(py::module &m)
  {
    // do not expose multilinear for higher dimensions, as it becomes inefficient
    if constexpr (N_DIMS <= 12)
    {
      expose_class<double, multilinear_adaptive_cpu_interpolator<double, N_DIMS, N_OPS>>(m, "multilinear_cpu_interpolator");
    }
    // expose_class<float, multilinear_adaptive_cpu_interpolator<float, N_DIMS, N_OPS>>(m, "multilinear2_cpu_interpolator");

#if !defined(OD_INTERP_PROFILE_MINIMAL)
    // Linear adaptive — exposed under FULL. Like the multilinear adaptive classes it
    // carries no index-type template parameter (int32-native vertex enumeration).
    expose_class<double, linear_adaptive_cpu_interpolator<N_DIMS, N_OPS>>(m, "linear_cpu_interpolator");
#endif
    //#ifdef WITH_GPU
//#endif
#ifdef WITH_GPU


    expose_class<double, multilinear_adaptive_gpu_interpolator<double, N_DIMS, N_OPS>>(m, "multilinear_gpu_interpolator");

    // expose_class<float, multilinear_adaptive_gpu_interpolator<float, N_DIMS, N_OPS>>(m, "multilinear_gpu_interpolator");

#endif //WITH_GPU
  }
};

#endif //PYBIND11_ENABLED
