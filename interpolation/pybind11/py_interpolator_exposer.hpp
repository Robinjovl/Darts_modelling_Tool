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

template <uint8_t N_DIMS, uint8_t N_OPS>
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
          .def(py::init<operator_set_evaluator_iface*, std::vector<index_t> &, std::vector<value_t> &, std::vector<value_t> &>(), py::keep_alive<1, 2>()) /*.def("benchmark", &interpolator_class::benchmark, "Init by nc and rate operators") \*/
          .def("evaluate_with_derivatives", &interpolator_class::evaluate_with_derivatives,
            "Evaluate operators and derivatives (v)", "state"_a, "block_idx"_a, "values"_a, "derivatives"_a)
          .def("init_timer_node", &interpolator_class::init_timer_node,
            "Initialize timer", "timer_node"_a)
          .def("init", &interpolator_class::init, "Initialize interpolator")
          .def("write_to_file", &interpolator_class::write_to_file, "Write interpolator data to file")
          .def("evaluate", &interpolator_class::evaluate,
            "Evaluate operators", "state"_a, "values"_a)
          // point_data: legacy integer-keyed view of the multi-index-keyed cache.
          // Out-of-bounds cells (per-axis index outside [0, axes_points[i]-1]) are skipped on read
          // and cannot be supplied on write. This preserves the existing pickle cache format.
          .def_property("point_data",
            [](const interpolator_class& self) {
              std::unordered_map<i_t, point_data_t> result;
              result.reserve(self.point_data.size());
              for (const auto& kv : self.point_data) {
                if (self.is_in_bounds_point(kv.first))
                  result.emplace(self.to_int_key_point(kv.first), kv.second);
              }
              return result;
            },
            [](interpolator_class& self, const std::unordered_map<i_t, point_data_t>& d) {
              self.point_data.clear();
              self.point_data.reserve(d.size());
              for (const auto& kv : d) {
                self.point_data.emplace(self.from_int_key_point(kv.first), kv.second);
              }
            })
          // point_data_full: lossless export of the entire cell-key-indexed cache as a
          // dict keyed on tuple-of-ints. Use this to save/restore caches that contain
          // out-of-bounds cells (the legacy `point_data` filters those out).
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
            "Number of supporting points currently in the adaptive cache (in-bounds + out-of-bounds)")
          .def("get_n_cached_hypercubes", &interpolator_class::get_n_cached_hypercubes,
            "Number of hypercubes currently in the adaptive cache (in-bounds + out-of-bounds)")
          .def("get_hypercube_indexes", &interpolator_class::get_hypercube_indexes);
      }
      else if constexpr (std::is_same_v<interpolator_class, linear_adaptive_cpu_interpolator<i_t, N_DIMS, N_OPS>>)
      {
        // Linear adaptive: storage is keyed on multi-index; expose point_data as
        // a legacy integer-keyed dict for pickle cache compatibility.
        using point_value_t = std::array<double, N_OPS>;
        py::class_<interpolator_class,
          operator_set_gradient_evaluator_iface>(m, name.c_str(), long_name.c_str())
          .def(py::init<operator_set_evaluator_iface*, std::vector<index_t> &, std::vector<value_t> &, std::vector<value_t> &, bool>(), py::keep_alive<1, 2>())
          .def("evaluate_with_derivatives", &interpolator_class::evaluate_with_derivatives,
            "Evaluate operators and derivatives (v)", "state"_a, "block_idx"_a, "values"_a, "derivatives"_a)
          .def("init_timer_node", &interpolator_class::init_timer_node,
            "Initialize timer", "timer_node"_a)
          .def("init", &interpolator_class::init, "Initialize interpolator")
          .def("write_to_file", &interpolator_class::write_to_file, "Write interpolator data to file")
          .def("evaluate", &interpolator_class::evaluate,
            "Evaluate operators", "state"_a, "values"_a)
          .def_property("point_data",
            [](const interpolator_class& self) {
              std::unordered_map<i_t, point_value_t> result;
              result.reserve(self.point_data.size());
              for (const auto& kv : self.point_data) {
                if (self.is_in_bounds(kv.first))
                  result.emplace(self.to_int_key(kv.first), kv.second);
              }
              return result;
            },
            [](interpolator_class& self, const std::unordered_map<i_t, point_value_t>& d) {
              self.point_data.clear();
              self.point_data.reserve(d.size());
              for (const auto& kv : d) {
                self.point_data.emplace(self.from_int_key(kv.first), kv.second);
              }
            })
          // point_data_full: lossless export keyed on tuple-of-ints. See note above
          // on the multilinear adaptive branch.
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
          .def_readwrite("use_barycentric_interpolation", &interpolator_class::use_barycentric_interpolation);
      }
      else if constexpr (std::is_same_v<interpolator_class, linear_static_cpu_interpolator<i_t, N_DIMS, N_OPS>>)
      {
        py::class_<interpolator_class,
          operator_set_gradient_evaluator_iface>(m, name.c_str(), long_name.c_str())
          .def(py::init<operator_set_evaluator_iface*, std::vector<index_t> &, std::vector<value_t> &, std::vector<value_t> &, bool>(), py::keep_alive<1, 2>())
          .def("evaluate_with_derivatives", &interpolator_class::evaluate_with_derivatives,
            "Evaluate operators and derivatives (v)", "state"_a, "block_idx"_a, "values"_a, "derivatives"_a)
          .def("init_timer_node", &interpolator_class::init_timer_node,
            "Initialize timer", "timer_node"_a)
          .def("init", &interpolator_class::init, "Initialize interpolator")
          .def("write_to_file", &interpolator_class::write_to_file, "Write interpolator data to file")
          .def("evaluate", &interpolator_class::evaluate,
            "Evaluate operators", "state"_a, "values"_a)
          .def_readwrite("point_data", &interpolator_class::point_data)
          .def_readwrite("use_barycentric_interpolation", &interpolator_class::use_barycentric_interpolation);
      }
#ifdef WITH_GPU
      else if constexpr (std::is_same_v<interpolator_class, multilinear_adaptive_gpu_interpolator<i_t, f_t, N_DIMS, N_OPS>>)
      {
        // GPU adaptive multilinear: same point_data shim as the CPU variant —
        // legacy integer-keyed dict for pickle cache compatibility.
        using point_data_t = typename interpolator_class::point_data_t;
        py::class_<interpolator_class,
          operator_set_gradient_evaluator_iface>(m, name.c_str(), long_name.c_str())
          .def(py::init<operator_set_evaluator_iface*, std::vector<index_t> &, std::vector<value_t> &, std::vector<value_t> &>(), py::keep_alive<1, 2>())
          .def("evaluate_with_derivatives", &interpolator_class::evaluate_with_derivatives,
            "Evaluate operators and derivatives (v)", "state"_a, "block_idx"_a, "values"_a, "derivatives"_a)
          .def("init_timer_node", &interpolator_class::init_timer_node,
            "Initialize timer", "timer_node"_a)
          .def("init", &interpolator_class::init, "Initialize interpolator")
          .def("write_to_file", &interpolator_class::write_to_file, "Write interpolator data to file")
          .def("evaluate", &interpolator_class::evaluate,
            "Evaluate operators", "state"_a, "values"_a)
          .def_property("point_data",
            [](const interpolator_class& self) {
              std::unordered_map<i_t, point_data_t> result;
              result.reserve(self.point_data.size());
              for (const auto& kv : self.point_data) {
                if (self.is_in_bounds_point(kv.first))
                  result.emplace(self.to_int_key_point(kv.first), kv.second);
              }
              return result;
            },
            [](interpolator_class& self, const std::unordered_map<i_t, point_data_t>& d) {
              self.point_data.clear();
              self.point_data.reserve(d.size());
              for (const auto& kv : d) {
                self.point_data.emplace(self.from_int_key_point(kv.first), kv.second);
              }
            })
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
          .def("get_n_cached_hypercubes", &interpolator_class::get_n_cached_hypercubes);
      }
#endif
      else {
        py::class_<interpolator_class,
          operator_set_gradient_evaluator_iface>(m, name.c_str(), long_name.c_str())
          .def(py::init<operator_set_evaluator_iface*, std::vector<index_t> &, std::vector<value_t> &, std::vector<value_t> &>(), py::keep_alive<1, 2>()) /*.def("benchmark", &interpolator_class::benchmark, "Init by nc and rate operators") \*/
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

  // here we specify which types and what interpolators are going to be exposed
  // the specification seriously affect build time and binary size
  void expose(py::module &m)
  {
    // do not expose multilinear for higher dimensions, as it becomes inefficient
    if constexpr (N_DIMS <= 12)
    {
      // we expose uint32 and uint64 adaptive interpolators by default
      expose_class<uint32_t, double, multilinear_adaptive_cpu_interpolator<uint32_t, double, N_DIMS, N_OPS>>(m, "multilinear_adaptive_cpu_interpolator");
      expose_class<uint64_t, double, multilinear_adaptive_cpu_interpolator<uint64_t, double, N_DIMS, N_OPS>>(m, "multilinear_adaptive_cpu_interpolator");
    }
    // expose_class<uint64_t, float, multilinear_adaptive_cpu_interpolator<uint64_t, float, N_DIMS, N_OPS>>(m, "multilinear_adaptive2_cpu_interpolator");

    // linear adaptive with 64-bit legacy index and 64-bit data
    expose_class<uint64_t, double, linear_adaptive_cpu_interpolator<uint64_t, N_DIMS, N_OPS>>(m, "linear_adaptive_cpu_interpolator");
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
