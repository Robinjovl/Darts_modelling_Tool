#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals_interpolation.h"
#include "recursive_exposers.h"
#include <pybind11/stl.h>

#include "multilinear_static_cpu_interpolator.hpp"
#include "multilinear_adaptive_cpu_interpolator.hpp"

#include "linear_static_cpu_interpolator.hpp"
#include "linear_adaptive_cpu_interpolator.hpp"
#ifdef WITH_GPU
#include "multilinear_static_gpu_interpolator.hpp"
#include "multilinear_adaptive_gpu_interpolator.hpp"
#endif //WITH_GPU

namespace py = pybind11;

template <uint8_t N_DIMS, uint16_t N_OPS>
struct interpolator_exposer
{
  // template function used to expose different interpolators with the same Python interface
  template <typename i_t, typename f_t, typename interpolator_class>
  void expose_class(py::module &m, std::string base_name)
  {
    using namespace pybind11::literals;

    std::string name = base_name + '_';

    if (typeid(i_t) == typeid(int) || typeid(i_t) == typeid(uint32_t))
    {
      name += "i_";
    }
    else if (typeid(i_t) == typeid(long long) || typeid(i_t) == typeid(uint64_t))
    {
      name += "l_";
    }
    else
    {
      std::cout << "Error: Unexpected index type id (" << typeid(i_t).name() << ") specified while exposing " << name << std::endl;
      return;
    }

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
      std::cout << "Error: Unexpected index type id (" << typeid(f_t).name() << ") specified while exposing " << name << std::endl;
      return;
    }

    name = name + std::to_string(N_DIMS) + "_" + std::to_string(N_OPS);
    std::string i_typename = typeid(i_t).name();
    std::string f_typename = typeid(f_t).name();
    std::string long_name = "Operator set interpolator with " + i_typename + " index type and " + f_typename + " value type for " + std::to_string(N_OPS) + " operators in " + std::to_string(N_DIMS) + "-dimensional parameter space";
    try
    {
      if constexpr (std::is_same_v<interpolator_class, multilinear_adaptive_cpu_interpolator<i_t, f_t, N_DIMS, N_OPS>>)
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
          .def("get_n_cached_points", &interpolator_class::get_n_cached_points,
            "Number of supporting points currently in the adaptive cache")
          .def("get_n_cached_hypercubes", &interpolator_class::get_n_cached_hypercubes,
            "Number of hypercubes currently in the adaptive cache")
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
          .def("clear_point_data_delta", [](interpolator_class &self) {
            self.dirty_point_data.clear();
          });
      }
      else if constexpr (std::is_same_v<interpolator_class, linear_adaptive_cpu_interpolator<i_t, N_DIMS, N_OPS>>)
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
          .def("clear_point_data_delta", [](interpolator_class &self) {
            self.dirty_point_data.clear();
          })
          .def_readwrite("use_barycentric_interpolation", &interpolator_class::use_barycentric_interpolation);
      }
      else if constexpr (std::is_same_v<interpolator_class, linear_static_cpu_interpolator<i_t, N_DIMS, N_OPS>>)
      {
        py::class_<interpolator_class,
          operator_set_gradient_evaluator_iface>(m, name.c_str(), long_name.c_str())
          .def(py::init<operator_set_evaluator_iface*, std::vector<value_t> &, std::vector<value_t> &, std::vector<index_t> &, bool>(), py::keep_alive<1, 2>()) /* (evaluator, axes_origin, axes_step, axes_points, is_barycentric) */
          .def("evaluate_with_derivatives", &interpolator_class::evaluate_with_derivatives,
            "Evaluate operators and derivatives (v)", "state"_a, "block_idx"_a, "values"_a, "derivatives"_a)
          .def("init_timer_node", &interpolator_class::init_timer_node,
            "Initialize timer", "timer_node"_a)
          .def("init", &interpolator_class::init, "Initialize interpolator")
          .def("write_to_file", &interpolator_class::write_to_file, "Write interpolator data to file")
          .def("evaluate", &interpolator_class::evaluate,
            "Evaluate operators", "state"_a, "values"_a)
          // linear_static_cpu_interpolator stores point_data as std::vector<double>
          // (dense supporting-point payload) and has no dirty_point_data tracker;
          // the append-only delta hooks therefore do not apply to this branch.
          .def_readwrite("point_data", &interpolator_class::point_data)
          .def_readwrite("use_barycentric_interpolation", &interpolator_class::use_barycentric_interpolation);
      }
#ifdef WITH_GPU
      else if constexpr (std::is_same_v<interpolator_class, multilinear_adaptive_gpu_interpolator<i_t, f_t, N_DIMS, N_OPS>>)
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
          .def("get_n_cached_points", &interpolator_class::get_n_cached_points)
          .def("get_n_cached_hypercubes", &interpolator_class::get_n_cached_hypercubes)
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
          .def("clear_point_data_delta", [](interpolator_class &self) {
            self.dirty_point_data.clear();
          });
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
  //   MINIMAL: only multilinear_adaptive uint64. Drops linear; callers that set
  //            itor_type='linear' must switch to 'multilinear'.
  //   FULL:    multilinear_adaptive uint64 + linear_adaptive uint64.
  void expose(py::module &m)
  {
    // do not expose multilinear for higher dimensions, as it becomes inefficient
    if constexpr (N_DIMS <= 12)
    {
      // Multilinear adaptive uint64 — exposed under all profiles. Python's
      // physics_base.py first tries the *_i_* (uint32) name and falls back to
      // *_l_* (uint64) on NameError; uint32 is no longer compiled, so the
      // fallback path is now the only path. One-time try/except cost per
      // interpolator at construction, zero runtime cost after.
      // expose_class<uint32_t, double, multilinear_adaptive_cpu_interpolator<uint32_t, double, N_DIMS, N_OPS>>(m, "multilinear_adaptive_cpu_interpolator");
      expose_class<uint64_t, double, multilinear_adaptive_cpu_interpolator<uint64_t, double, N_DIMS, N_OPS>>(m, "multilinear_adaptive_cpu_interpolator");
      // __uint128_t exposure removed alongside the move to cell_key_t multi-index keys —
      // see interpolation_config.h. uint64_t legacy_index is still enough for diagnostic
      // counters; out-of-uint64 cells live in the multi-index map directly.
    }
    // expose_class<uint64_t, float, multilinear_adaptive_cpu_interpolator<uint64_t, float, N_DIMS, N_OPS>>(m, "multilinear_adaptive2_cpu_interpolator");

#if !defined(OD_INTERP_PROFILE_MINIMAL)
    // Linear adaptive with 64-bit legacy index and 64-bit data — exposed under FULL.
    expose_class<uint64_t, double, linear_adaptive_cpu_interpolator<uint64_t, N_DIMS, N_OPS>>(m, "linear_adaptive_cpu_interpolator");
    // expose_class<__uint128_t, double, linear_adaptive_cpu_interpolator<__uint128_t, N_DIMS, N_OPS>>(m, "linear_adaptive_cpu_interpolator");
#endif
    //expose_class<uint64_t, double, linear_static_cpu_interpolator<uint64_t, N_DIMS, N_OPS>>(m, "linear_static_cpu_interpolator");
    // we expose static versions only when needed
    //#ifdef WITH_GPU
    //expose_class<uint32_t, double, multilinear_static_cpu_interpolator<uint32_t, double, N_DIMS, N_OPS>>(m, "multilinear_static_cpu_interpolator");
//#endif
// we expose static GPU versions only when GPU build is active
#ifdef WITH_GPU

    //expose_class<uint32_t, double, multilinear_static_gpu_interpolator<uint32_t, double, N_DIMS, N_OPS>>(m, "multilinear_static_gpu_interpolator");
    //expose_class<uint32_t, float, multilinear_static_gpu_interpolator<uint32_t, float, N_DIMS, N_OPS>>(m, "multilinear_static_gpu_interpolator");

    expose_class<uint32_t, double, multilinear_adaptive_gpu_interpolator<uint32_t, double, N_DIMS, N_OPS>>(m, "multilinear_adaptive_gpu_interpolator");
    expose_class<uint64_t, double, multilinear_adaptive_gpu_interpolator<uint64_t, double, N_DIMS, N_OPS>>(m, "multilinear_adaptive_gpu_interpolator");

   // expose_class<uint32_t, float, multilinear_adaptive_gpu_interpolator<uint32_t, float, N_DIMS, N_OPS>>(m, "multilinear_adaptive_gpu_interpolator");
    //expose_class<uint64_t, float, multilinear_adaptive_gpu_interpolator<uint64_t, float, N_DIMS, N_OPS>>(m, "multilinear_adaptive_gpu_interpolator");

#endif //WITH_GPU
  }
};

#endif //PYBIND11_ENABLED
