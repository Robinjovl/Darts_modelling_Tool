#ifdef PYBIND11_ENABLED
#include <pybind11/stl_bind.h>
#include "py_globals.h"
#include "globals.h"
#include "engines_build_info.h"
#include <cctype>
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
  int device_count = 0;
  cudaError_t cnt_err = cudaGetDeviceCount(&device_count);
  if (cnt_err != cudaSuccess)
  {
    std::cerr << "CUDA get device count error: " << cudaGetErrorString(cnt_err) << "(" << cnt_err << ") " << std::endl;
    return;
  }

  if (device_idx < 0 || device_idx >= device_count)
  {
    std::cerr << "CUDA set device error: invalid device index " << device_idx
              << ", available indices: 0.." << (device_count - 1) << std::endl;
    return;
  }

  cudaError_t err = cudaSetDevice(device_idx);
  if (err == cudaSuccess)
  {
    device_num = device_idx;
    return;
  }

  std::cerr << "CUDA set device error: " << cudaGetErrorString(err) << "(" << err << ") " << std::endl;
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

// Write a string to the darts standard output stream — i.e. the same destination
// redirect_darts_output() points std::cout at (the log file, the terminal if not
// redirected, or suppressed if redirected to ""). Lets Python helpers send text to the
// redirected log instead of the Python-level stdout.
void write_to_darts_output(std::string s)
{
  std::cout << s << std::flush;
}

void pybind_globals(py::module &m)
{
  using namespace pybind11::literals;

  // The uint128 Python binding was removed alongside __uint128_t support across the
  // interpolators (which is what used to consume it for legacy mixed-radix
  // hypercube enumeration in N_DIMS=20 grids). Adaptive storage now keys on a
  // signed multi-index (cell_key_t), and the legacy point_data integer keys fit
  // in uint64_t.

  py::class_<sim_params> sim_params(m, "sim_params", "Class simulation parameters");

  sim_params.def(py::init<>())
    //properties
    .def_readwrite("max_i_linear", &sim_params::max_i_linear)
    .def_readwrite("tolerance_linear", &sim_params::tolerance_linear)
    .def_readwrite("linear_type", &sim_params::linear_type)
    .def_readwrite("linear_params", &sim_params::linear_params)
    .def_readwrite("enable_permporo", &sim_params::enable_permporo)
    .def_readwrite("sim_eps", &sim_params::sim_eps)
    .def_readwrite("global_actnum", &sim_params::global_actnum)
    .def_readwrite("assembly_kernel", &sim_params::assembly_kernel)
    .def_readwrite("finalize_mpi", &sim_params::finalize_mpi)
    .def_readwrite("phase_existence_tolerance", &sim_params::phase_existence_tolerance);


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

    // timer_node is registered by darts.interpolators (imported at module init).
  // Re-export it so that `from darts.engines import timer_node` still works.
  m.attr("timer_node") = py::module_::import("darts.interpolators").attr("timer_node");

  m.def("redirect_darts_output", &redirect_darts_output, "Redirect darts standard output to a file. \n"
                                                         "If empty filename is specified, then no output will be produced.",
        "file_name"_a);

  m.def("print_build_info", &print_build_info, "Print build information: date, user, machine, git hash");

  m.def("write_to_darts_output", &write_to_darts_output,
        "Write a string to the darts output stream (the file set by redirect_darts_output, "
        "the terminal if not redirected, or nothing if redirected to an empty filename).",
        "text"_a);


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
  m.def("set_gpu_device", &set_gpu_device, "Set the index of GPU device to be used", "device_idx"_a);
  m.def("cuda_device_reset", &cuda_device_reset, "Reset gpu device for memory leak check");
#endif

}


#endif //PYBIND11_ENABLED
