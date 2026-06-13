#ifndef GLOBALS_H
#define GLOBALS_H

#include "timer_node.h"

#include <fstream>
#include <vector>

#include <cstdint>
using namespace std;

typedef int index_t;
typedef double value_t;
typedef int interp_index_t;
typedef double interp_value_t;
#define INTERP_BLOCK_SIZE 64

// the following two are optimised for engine_nc, SPE10 with N_VARS=2
#define ASSEMBLY_N_VARS_N_VARS_BLOCK_SIZE 512
#define ASSEMBLY_N_VARS_BLOCK_SIZE 256

#define ASSEMBLY_BLOCK_SIZE 64
#define SIMPLE_OPS_BLOCK_SIZE 256

#define PORO_MIN 0.001
static const double LOWER_LIMIT = 1.0e-12;
static const double UPPER_LIMIT = 1.0 - LOWER_LIMIT;
static std::ofstream log_stream;

// Max number of components for engine template instantiation (engine_nc_*,
// engine_super_*). Recursive_instantiator_nc / nc_np loops cover NC ∈ [2, MAX_NC].
// Driven by the OPENDARTS_MAX_DIMS cmake variable (-DMAX_NC=N) so it stays in
// sync with MAX_DIMS in interpolation_config.h — for thermal physics the
// interpolator parameter-space dim is NC+1, so MAX_DIMS must be ≥ MAX_NC.
//
// Fail loudly rather than silently defaulting: a TU compiled without -DMAX_NC
// would land at a different value than the rest of the binary and produce
// ODR-incoherent template instantiations with silent runtime corruption.
#ifndef MAX_NC
#error "MAX_NC must be defined (typically via the OPENDARTS_MAX_DIMS CMake variable, propagated as -DMAX_NC=N)."
#endif

#define GET_RAND_I(START, END) \
  START + rand() / (RAND_MAX / (END - START + 1) + 1)

#define GET_RAND_F(START, END) \
  START + rand() / (RAND_MAX / (END - START))

// workaround for vscode grammar checker
#ifdef __INTELLISENSE__
#define __global__
#define __constant__
#endif

#ifdef WITH_GPU
extern int device_num;
#endif

/// Main simulation parameters including tolerances
class sim_params
{

public:
  enum newton_solver_t
  {
    NEWTON_STD = 0,
    NEWTON_GLOBAL_CHOP,
    NEWTON_LOCAL_CHOP,
    NEWTON_INFLECTION_POINT
  };

  enum linear_solver_t
  {
    CPU_GMRES_CPR_AMG = 0,
    CPU_GMRES_CPR_AMG1R5,
    CPU_GMRES_FS_CPR,
    CPU_SAMG,
    CPU_GMRES_ILU0,
    CPU_SUPERLU,
    GPU_GMRES_CPR_AMG, // <<<---- Should be the first GPU method for correct Jacobian treatment
    GPU_GMRES_ILU0,
    GPU_GMRES_CPR_AIPS,
    GPU_GMRES_CPR_AMGX_ILU,
    GPU_GMRES_CPR_AMGX_ILU_SP,
    GPU_GMRES_CPR_AMGX_AMGX,
    GPU_GMRES_AMGX,
    GPU_AMGX,
    GPU_GMRES_CPR_NF,
    GPU_BICGSTAB_CPR_AMGX,
    GPU_CUSOLVER
  };

  enum nonlinear_norm_t
  {
    L1 = 0,
    L2,
    LINF
  };

  sim_params()
  {
    // set default params
    first_ts = 1;
    max_ts = 10;
    mult_ts = 2;
    min_ts = 1e-12;

    max_i_linear = 50;
    tolerance_linear = 1e-5;
    max_i_newton = 20;
    min_i_newton = 0;
    tolerance_newton = 1e-3;
    well_tolerance_coefficient = 1e2;
    stationary_point_tolerance = 1e-3;
    newton_type = NEWTON_LOCAL_CHOP;
    newton_params.push_back(0.1);
    line_search = false;

#ifdef OPENDARTS_LINEAR_SOLVERS
    linear_type = CPU_SUPERLU;
#else
    linear_type = CPU_GMRES_CPR_AMG;
#endif
    nonlinear_norm_type = L2;

    //Added for debugging purposes:
    tot_newt_count = 0;
    log_transform = 0;
    interface_avg_tmult = 0;
    enable_permporo = false;
    // obl_min_fac = 10;
    sim_eps = 1e-12;
    assembly_kernel = 0;

    finalize_mpi = 1;

    phase_existence_tolerance = 1.e-6;
  }

  value_t first_ts; // first time step length (days)
  value_t max_ts;   // maximum time step length (days)
  value_t mult_ts;  // multiplication ts factor
  value_t min_ts;   // minimum time step length (days)

  index_t max_i_newton;     // maximum number of newton iterations
  index_t min_i_newton;     // minimum number of newton iterations
  index_t max_i_linear;     // maximum number of linear iterations
  value_t tolerance_newton; // tolerance for newton solver
  value_t tolerance_linear; // tolerance for linear solver
  value_t well_tolerance_coefficient; // tolerance multiplier for well newton tolerance
  value_t stationary_point_tolerance; // stationary point tolerance
  bool line_search;         // apply line search in newton iterations

  //Added for debugging purposes:
  index_t tot_newt_count;      // total number of newton iterations (wasted + non-wasted)
  index_t log_transform;       // 0 => normal comp (X=[P,Z1,...,Znc-1]), 1 => logtransform of comp (X=[P,log(Z1),...,log(Znc-1)])
  index_t interface_avg_tmult; // 0 => normal trans-multiplier (in operator), 1 => interface weighted trans-multiplier (in engine)
  bool enable_permporo;        // flag enabling transmissibility multiplier in assembly
  value_t obl_min_fac;         // factor used to determine z_min --> usually taken around 10, such that z_min = 10*z_OBL_min
  value_t sim_eps;             // offset from axes that solution should remain inside
  int assembly_kernel;         // select non-default assebly kernel (for GPU)

  newton_solver_t newton_type;          // Newton solver type (more precisely, nonlinear update type - chopping strategies)
  linear_solver_t linear_type;          // Linear solver type
  nonlinear_norm_t nonlinear_norm_type; // Nonlinear norm type, used to check for convergence

  std::vector<value_t> newton_params;
  std::vector<value_t> linear_params;

  // for NF solver
  std::vector<int> global_actnum;

  // Global chop: 0 - solution increment/value (dX/X) ratio threshold (default 1)
  // Local chop:  1 - composition increment is limited by max_dx (default 0.1)

  index_t finalize_mpi;         // flag to run MPI_Finalize in relevant solvers (required for multiple model run)

  value_t phase_existence_tolerance;    // tolerance defining presence of phase in a cell
};

class linear_solver_params
{
public:
  sim_params::linear_solver_t linear_type;          // Linear solver type
  index_t max_i_linear;                 // maximum number of linear iterations
  value_t tolerance_linear;             // tolerance for linear solver

  linear_solver_params()
  {
#ifdef OPENDARTS_LINEAR_SOLVERS
    linear_type = sim_params::CPU_SUPERLU;
#else
    linear_type = sim_params::CPU_GMRES_CPR_AMG;
#endif
    max_i_linear = 50;
    tolerance_linear = 1e-5;
  };
};

/// Main simulation statistics with active and wasted counts
class sim_stat
{
public:
  sim_stat()
  {
    n_newton_total = 0;     // total number of nonlinear iterations
    n_linear_total = 0;     // total number of linear iterations
    n_newton_wasted = 0;    // number of wasted nonlinear iterations
    n_linear_wasted = 0;    // number of wasted linear iterations
    n_timesteps_total = 0;  // total number of timetseps
    n_timesteps_wasted = 0; // number of wasted timetseps
  }

  index_t n_newton_total;
  index_t n_linear_total;
  index_t n_newton_wasted;
  index_t n_linear_wasted;
  index_t n_timesteps_total;
  index_t n_timesteps_wasted;
};

void write_vector_to_file(std::string file_name, std::vector<value_t> &v);

template <class T>
inline void numa_set(T *src, int value, index_t start, index_t end)
{
  memset(&src[start], value, (end - start) * sizeof(T));
}

template <class T>
inline void numa_cpy(T *dsc, T *src, index_t start, index_t end)
{
  memcpy(dsc + start, src + start, (end - start) * sizeof(T));
}

// instantiator helper class for <NC> template

template <template <uint8_t NC> class templated_t, uint8_t NC_START, uint8_t NC_STOP>
struct recursive_instantiator_nc
{
  static void instantiate()
  {
    templated_t<NC_START> a;
    recursive_instantiator_nc<templated_t, NC_START + 1, NC_STOP>::instantiate();
  }
};

// partial specialization to stop recusrion
template <template <uint8_t NC> class templated_t, uint8_t NC_STOP>
struct recursive_instantiator_nc<templated_t, NC_STOP, NC_STOP>
{
  static void instantiate()
  {
    templated_t<NC_STOP> a;
  }
};

// instantiator helper class for <NC, NP> template

template <template <uint8_t NC, uint8_t NP> class templated_t, uint8_t NC_START, uint8_t NC_STOP, uint8_t NP>
struct recursive_instantiator_nc_np
{
  static void instantiate()
  {
    templated_t<NC_START, NP> a;
    recursive_instantiator_nc_np<templated_t, NC_START + 1, NC_STOP, NP>::instantiate();
  }
};

// partial specialization to stop recusrion

template <template <uint8_t NC, uint8_t NP> class templated_t, uint8_t NC_STOP, uint8_t NP>
struct recursive_instantiator_nc_np<templated_t, NC_STOP, NC_STOP, NP>
{
  static void instantiate()
  {
    templated_t<NC_STOP, NP> a;
  }
};

// instantiator helper class for <NC, NP> template

template <template <uint8_t NC, uint8_t NP, bool... EFFECTS> class templated_t, uint8_t NC_START, uint8_t NC_STOP, uint8_t NP, bool... EFFECTS>
struct recursive_instantiator_nc_np_effects
{
  static void instantiate()
  {
    templated_t<NC_START, NP, EFFECTS...> a;
    recursive_instantiator_nc_np_effects<templated_t, NC_START + 1, NC_STOP, NP, EFFECTS...>::instantiate();
  }
};

// partial specialization to stop recusrion

template <template <uint8_t NC, uint8_t NP, bool... EFFECTS> class templated_t, uint8_t NC_STOP, uint8_t NP, bool... EFFECTS>
struct recursive_instantiator_nc_np_effects<templated_t, NC_STOP, NC_STOP, NP, EFFECTS...>
{
  static void instantiate()
  {
    templated_t<NC_STOP, NP, EFFECTS...> a;
  }
};

template <template <uint8_t NC> class exposer_t, typename pymodule_t, uint8_t NC, uint8_t NC_STOP>
struct recursive_exposer_nc
{
  static void expose(pymodule_t &m)
  {
    exposer_t<NC> e;
    e.expose(m);

    recursive_exposer_nc<exposer_t, pymodule_t, NC + 1, NC_STOP>::expose(m);
  }
};

// partial specialization to stop recusrion
template <template <uint8_t NC> class exposer_t, typename pymodule_t, uint8_t NC_STOP>
struct recursive_exposer_nc<exposer_t, pymodule_t, NC_STOP, NC_STOP>
{
  static void expose(pymodule_t &m)
  {
    exposer_t<NC_STOP> e;
    e.expose(m);
  }
};

// helper class to recursevely expose <NC, NP> templated classes

template <template <uint8_t NC, uint8_t NP> class exposer_t, typename pymodule_t, uint8_t NC, uint8_t NC_STOP, uint8_t NP>
struct recursive_exposer_nc_np
{
  static void expose(pymodule_t &m)
  {
    exposer_t<NC, NP> e;
    e.expose(m);

    recursive_exposer_nc_np<exposer_t, pymodule_t, NC + 1, NC_STOP, NP>::expose(m);
  }
};

// partial specialization to stop recusrion
template <template <uint8_t NC, uint8_t NP> class exposer_t, typename pymodule_t, uint8_t NC_STOP, uint8_t NP>
struct recursive_exposer_nc_np<exposer_t, pymodule_t, NC_STOP, NC_STOP, NP>
{
  static void expose(pymodule_t &m)
  {
    exposer_t<NC_STOP, NP> e;
    e.expose(m);
  }
};

// helper class to recursevely expose <NC, NP, THERMAL> templated classes

template <template <uint8_t NC, uint8_t NP, bool THERMAL> class exposer_t, typename pymodule_t, uint8_t NC, uint8_t NC_STOP, uint8_t NP, bool THERMAL>
struct recursive_exposer_nc_np_t
{
  static void expose(pymodule_t &m)
  {
    exposer_t<NC, NP, THERMAL> e;
    e.expose(m);

    recursive_exposer_nc_np_t<exposer_t, pymodule_t, NC + 1, NC_STOP, NP, THERMAL>::expose(m);
  }
};

// partial specialization to stop recusrion
template <template <uint8_t NC, uint8_t NP, bool THERMAL> class exposer_t, typename pymodule_t, uint8_t NC_STOP, uint8_t NP, bool THERMAL>
struct recursive_exposer_nc_np_t<exposer_t, pymodule_t, NC_STOP, NC_STOP, NP, THERMAL>
{
  static void expose(pymodule_t &m)
  {
    exposer_t<NC_STOP, NP, THERMAL> e;
    e.expose(m);
  }
};

// Recursive exposer helpers moved to interpolation library (recursive_exposers.h).
// Include that header directly where needed (e.g. via py_interpolator_exposer.hpp).

#endif
