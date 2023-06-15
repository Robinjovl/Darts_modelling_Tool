//*************************************************************************
//    Copyright (c) 2018
//            Mark Khait         M.Khait@tudelft.nl
//            Denis Voskov    D.V.Voskov@tudelft.nl
//    Delft University of Technology, the Netherlands
//
//    This file is part of the Delft Advanced Research Terra Simulator (DARTS)
//
//    DARTS is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as
//    published by the Free Software Foundation, either version 3 of the
//    License, or (at your option) any later version.
//
//    DARTS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public
//    License along with DARTS. If not, see <http://www.gnu.org/licenses/>.
// *************************************************************************

#ifdef PYBIND11_ENABLED
#include <pybind11.h>
#include "py_globals.h"
#include <stl.h>

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
      py::class_<interpolator_class,
                 operator_set_gradient_evaluator_iface>(m, name.c_str(), long_name.c_str())
          .def(py::init<operator_set_evaluator_iface *, std::vector<index_t> &, std::vector<value_t> &, std::vector<value_t> &>(), py::keep_alive<1, 2>()) /*.def("benchmark", &interpolator_class::benchmark, "Init by nc and rate operators") \*/
          .def("evaluate_with_derivatives", &interpolator_class::evaluate_with_derivatives,
               "Evaluate operators and derivatives (v)", "state"_a, "block_idx"_a, "values"_a, "derivatives"_a)
          .def("init_timer_node", &interpolator_class::init_timer_node,
               "Initialize timer", "timer_node"_a)
          .def("init", &interpolator_class::init, "Initialize interpolator")
          .def("write_to_file", &interpolator_class::write_to_file, "Write interpolator data to file")
          .def("evaluate", &interpolator_class::evaluate,
               "Evaluate operators", "state"_a, "values"_a)
          .def_readwrite("point_data", &interpolator_class::point_data);
      // .def("clear", &interpolator_class::clear) \
        // .def("clear_body_data", &interpolator_class::clear_body_data) \
        // .def_readwrite("base_points_generator", &interpolator_class::base_points_generator) \
        // .def_readwrite("axis_points", &interpolator_class::axis_points) \
        // .def_readwrite("axis_min", &interpolator_class::axis_min) \
        // .def_readwrite("axis_max", &interpolator_class::axis_max) \
        // .def_readwrite("body_data", &interpolator_class::body_data);
    }
    catch (...)
    {
      // exceptions are expected - due to template-based exposure some n_dims-n_ops pairs can overlap
      // std::cout << "Exception occurred: " << name << " was laready defined"<< std::endl ;
    }
  }

  // here we specify which types and what interpolators are going to be exposed
  // the specification seriously affect build time and binary size
  void expose(py::module &m)
  {
    // we expose uint32 and uint64 adaptive interpolators by default
    expose_class<uint32_t, double, multilinear_adaptive_cpu_interpolator<uint32_t, double, N_DIMS, N_OPS>>(m, "multilinear_adaptive_cpu_interpolator");
    expose_class<uint64_t, double, multilinear_adaptive_cpu_interpolator<uint64_t, double, N_DIMS, N_OPS>>(m, "multilinear_adaptive_cpu_interpolator");
    // expose_class<uint64_t, float, multilinear_adaptive_cpu_interpolator<uint64_t, float, N_DIMS, N_OPS>>(m, "multilinear_adaptive2_cpu_interpolator");

    // linear adaptive with 64 bit index and 64 bit data
    //expose_class<uint64_t, double, linear_adaptive_cpu_interpolator<uint64_t, N_DIMS, N_OPS>>(m, "linear_adaptive_cpu_interpolator");
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
