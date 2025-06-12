#ifdef PYBIND11_ENABLED
#include <pybind11/stl_bind.h>
#include "py_globals.h"
#include "globals.h"
#include "engines_build_info.h"
#include <iostream>
#include <fstream>

#ifdef OPENDARTS_LINEAR_SOLVERS
#include "openDARTS/config/version.hpp"
#else
#include "linsolv_build_info.h"
#endif // OPENDARTS_LINEAR_SOLVERS

#ifdef OPENDARTS_LINEAR_SOLVERS
using namespace opendarts::config;
#endif // OPENDARTS_LINEAR_SOLVERS

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;


#if defined(__linux__) || defined(__APPLE__)
  // declaration of stream test main function
  // used to check the system bandwidth
  int stream_main();
#endif // defined(__linux__) || defined(__APPLE__)


void redirect_darts_output(std::string file_name) {
  // check if output stream was already opened - close it
  if (log_stream.is_open())
    log_stream.close();

  // if new name is empty, then all output will be suppressed
  if (file_name.length() != 0)
    log_stream.open(file_name.c_str());

  std::cout.rdbuf(log_stream.rdbuf());
}

#ifdef WITH_GPU
void set_gpu_device(int device_idx)
{
  cudaError_t err = cudaSetDevice(device_idx);
  device_num = device_idx;
  if (err == cudaSuccess)
    return;
  
  std::cerr << "CUDA set device error: " << cudaGetErrorString (err) << "(" << err << ") " << std::endl;
};

void cuda_device_reset()
{
  cudaDeviceReset();
}
#endif

void print_build_info()
{
  std::cout << "darts-linear-solvers built on " << LINSOLV_BUILD_DATE << " by " << LINSOLV_BUILD_MACHINE << " from " << LINSOLV_BUILD_GIT_HASH << std::endl;
  std::cout << "darts-engines built on " << ENGINES_BUILD_DATE << " by " << ENGINES_BUILD_MACHINE << " from " << ENGINES_BUILD_GIT_HASH << std::endl;
}

void pybind_globals(py::module &m)
{
  using namespace pybind11::literals;

  // ---- begin uint128 binding ----
  py::class_<__uint128_t>(m, "uint128", "128-bit unsigned integer")
    .def(py::init<>())
    .def(py::init([](py::int_ i){
      const py::int_ two64 = py::int_(1) << 64;
      const py::int_ hi_py = i / two64;
      const py::int_ lo_py = i % two64;
      // now cast each half to uint64_t
      const uint64_t hi = hi_py.cast<uint64_t>();
      const uint64_t lo = lo_py.cast<uint64_t>();
      // rebuild the 128-bit value: (hi<<64) | lo
      __uint128_t result = static_cast<__uint128_t>(hi);
      result <<= 64;
      result |= static_cast<__uint128_t>(lo);
      return result;
    }), "value"_a)

  // conversion to Python int & use in slicing/indexing
    .def("__int__", [](const __uint128_t& v) {
#ifdef _MSC_VER
      uint64_t lo = v._Word[0];
      uint64_t hi = v._Word[1];
#else
      uint64_t lo = static_cast<uint64_t>(v);
      uint64_t hi = static_cast<uint64_t>(v >> 64);
#endif
      py::int_ py_hi = py::int_(hi);
      py::int_ py_lo = py::int_(lo);
      return (py_hi << 64) | py_lo;
    })
    .def("__index__", [](const __uint128_t &v){
#ifdef _MSC_VER
      uint64_t lo = v._Word[0];
      uint64_t hi = v._Word[1];
#else
      uint64_t lo = static_cast<uint64_t>(v);
      uint64_t hi = static_cast<uint64_t>(v >> 64);
#endif
      py::int_ py_hi = py::int_(hi);
      py::int_ py_lo = py::int_(lo);
      return (py_hi << 64) | py_lo;
    })
    .def("__repr__", [](const __uint128_t &v){
      std::ostringstream oss;
      oss << "uint128(" << std::to_string(v) << ")";
      return oss.str();
    })

    // make it picklable: store as two 64-bit words
    .def(py::pickle(
      /*__getstate__*/ [](const __uint128_t &v){
#ifdef _MSC_VER
        uint64_t lo = v._Word[0];
        uint64_t hi = v._Word[1];
#else
        uint64_t lo = static_cast<uint64_t>(v);
        uint64_t hi = static_cast<uint64_t>(v >> 64);
#endif
        return py::make_tuple(lo, hi);
      },
      /*__setstate__*/ [](py::tuple t){
        if (t.size() != 2)
          throw std::runtime_error("Invalid state for uint128");
        uint64_t lo = t[0].cast<uint64_t>();
        uint64_t hi = t[1].cast<uint64_t>();
        __uint128_t result = static_cast<__uint128_t>(hi);
        result <<= 64;
        result |= static_cast<__uint128_t>(lo);
        return result;
      }
    ));
  // ---- end uint128 binding ----

  py::class_<sim_params> sim_params(m, "sim_params", "Class simulation parameters");

  sim_params.def(py::init<>())
    //properties
    .def_readwrite("first_ts", &sim_params::first_ts, "Length of the first time step (days)")
    .def_readwrite("max_ts", &sim_params::max_ts)
    .def_readwrite("mult_ts", &sim_params::mult_ts)
    .def_readwrite("min_ts", &sim_params::min_ts)
    .def_readwrite("max_i_newton", &sim_params::max_i_newton)
    .def_readwrite("max_i_linear", &sim_params::max_i_linear)
    .def_readwrite("tolerance_newton", &sim_params::tolerance_newton)
    .def_readwrite("tolerance_linear", &sim_params::tolerance_linear)
    .def_readwrite("newton_type", &sim_params::newton_type)
    .def_readwrite("newton_params", &sim_params::newton_params)
    .def_readwrite("linear_type", &sim_params::linear_type)
    .def_readwrite("linear_params", &sim_params::linear_params)
    .def_readwrite("nonlinear_norm_type", &sim_params::nonlinear_norm_type)
    .def_readwrite("log_transform", &sim_params::log_transform)
    .def_readwrite("trans_mult_exp", &sim_params::trans_mult_exp)
    .def_readwrite("obl_min_fac", &sim_params::obl_min_fac)
    .def_readwrite("global_actnum", &sim_params::global_actnum)
    .def_readwrite("well_tolerance_coefficient", &sim_params::well_tolerance_coefficient)
    .def_readwrite("stationary_point_tolerance", &sim_params::stationary_point_tolerance)
    .def_readwrite("assembly_kernel", &sim_params::assembly_kernel)
    .def_readwrite("finalize_mpi", &sim_params::finalize_mpi)
    .def_readwrite("phase_existence_tolerance", &sim_params::phase_existence_tolerance)
    .def_readwrite("line_search", &sim_params::line_search);


  py::class_<linear_solver_params>(m, "linear_solver_params", "Class linear solver parameters") \
    .def(py::init<>())
    .def_readwrite("max_i_linear", &linear_solver_params::max_i_linear)
    .def_readwrite("tolerance_linear", &linear_solver_params::tolerance_linear)
    .def_readwrite("linear_type", &linear_solver_params::linear_type);
  py::bind_vector<std::vector<linear_solver_params>>(m, "vector_linear_solver_params", py::module_local());

  py::enum_<sim_params::newton_solver_t>(sim_params, "newton_solver_t", "Available types of newton solvers")
    .value("newton_std", sim_params::newton_solver_t::NEWTON_STD)
    .value("newton_global_chop", sim_params::newton_solver_t::NEWTON_GLOBAL_CHOP)
    .value("newton_local_chop", sim_params::newton_solver_t::NEWTON_LOCAL_CHOP)
    .value("newton_inflection_point", sim_params::newton_solver_t::NEWTON_INFLECTION_POINT)
    .export_values();

  py::enum_<sim_params::linear_solver_t>(sim_params, "linear_solver_t", "Available types of linear solvers")
    .value("cpu_gmres_cpr_amg", sim_params::linear_solver_t::CPU_GMRES_CPR_AMG)
    .value("cpu_gmres_ilu0", sim_params::linear_solver_t::CPU_GMRES_ILU0)
    .value("cpu_superlu", sim_params::linear_solver_t::CPU_SUPERLU)
    .value("cpu_gmres_cpr_amg1r5", sim_params::linear_solver_t::CPU_GMRES_CPR_AMG1R5)
    .value("cpu_gmres_fs_cpr", sim_params::linear_solver_t::CPU_GMRES_FS_CPR)
    .value("cpu_samg", sim_params::linear_solver_t::CPU_SAMG)
    .value("gpu_gmres_cpr_amg", sim_params::linear_solver_t::GPU_GMRES_CPR_AMG)
    .value("gpu_gmres_ilu0", sim_params::linear_solver_t::GPU_GMRES_ILU0)
    .value("gpu_gmres_cpr_aips", sim_params::linear_solver_t::GPU_GMRES_CPR_AIPS)
    .value("gpu_gmres_cpr_amgx_ilu", sim_params::linear_solver_t::GPU_GMRES_CPR_AMGX_ILU)
    .value("gpu_gmres_cpr_amgx_ilu_sp", sim_params::linear_solver_t::GPU_GMRES_CPR_AMGX_ILU_SP)
    .value("gpu_gmres_cpr_amgx_amgx", sim_params::linear_solver_t::GPU_GMRES_CPR_AMGX_AMGX)
    .value("gpu_gmres_amgx", sim_params::linear_solver_t::GPU_GMRES_AMGX)
    .value("gpu_amgx", sim_params::linear_solver_t::GPU_AMGX)
    .value("gpu_gmres_cpr_nf", sim_params::linear_solver_t::GPU_GMRES_CPR_NF)
    .value("gpu_bicgstab_cpr_amgx", sim_params::linear_solver_t::GPU_BICGSTAB_CPR_AMGX)
    .value("gpu_cusolver", sim_params::linear_solver_t::GPU_CUSOLVER)
    .export_values();

  py::enum_<sim_params::nonlinear_norm_t>(sim_params, "nonlinear_norm_t", "Available types of nonlinear norm")
    .value("L1", sim_params::nonlinear_norm_t::L1)
    .value("L2", sim_params::nonlinear_norm_t::L2)
    .value("LINF", sim_params::nonlinear_norm_t::LINF)
    .export_values();

  py::class_<sim_stat>(m, "sim_stat", "Class simulation statistics")
      .def(py::init<>())
      //properties
      .def_readwrite("n_newton_wasted", &sim_stat::n_newton_wasted)
      .def_readwrite("n_newton_total", &sim_stat::n_newton_total)
      .def_readwrite("n_linear_total", &sim_stat::n_linear_total)
      .def_readwrite("n_linear_wasted", &sim_stat::n_linear_wasted)
      .def_readwrite("n_timesteps_total", &sim_stat::n_timesteps_total)
      .def_readwrite("n_timesteps_wasted", &sim_stat::n_timesteps_wasted);

  py::class_<timer_node>(m, "timer_node", "Timers tree structure")
      .def(py::init<>())
      .def("start", &timer_node::start)
      .def("stop", &timer_node::stop)
      .def("get_timer", &timer_node::get_timer)
      .def("print", &timer_node::print)
      .def("reset_recursive", &timer_node::reset_recursive)
      //properties
      .def_readwrite("node", &timer_node::node);

  m.def("redirect_darts_output", &redirect_darts_output, "Redirect darts standard output to a file. \n"
                                                         "If empty filename is specified, then no output will be produced.",
        "file_name"_a);

  m.def("print_build_info", &print_build_info, "Print build information: date, user, machine, git hash");

#ifdef defined(__linux__) || defined(__APPLE__)
  m.def("stream", &stream_main, "Launch stream bandwidth test");
#endif // defined(__linux__) || defined(__APPLE__)

#ifdef _OPENMP
  m.def("get_num_threads", &omp_get_num_threads, "Get the number of OpenMP threads to be used");
  m.def("set_num_threads", &omp_set_num_threads, "Set the number of OpenMP threads to be used", "num_threads"_a);
  // if the amount of threads is not defined explicitly, use a half of available threads
  if (!std::getenv("OMP_NUM_THREADS"))
    {
      omp_set_num_threads(omp_get_max_threads() / 2);
    }
#endif

#ifdef WITH_GPU
  m.def("set_gpu_device", &set_gpu_device, "Set the index of GPU device to be used", "num_threads"_a);
  m.def("cuda_device_reset", &cuda_device_reset, "Reset gpu device for memory leak check");
#endif
  
}
#endif //PYBIND11_ENABLED
