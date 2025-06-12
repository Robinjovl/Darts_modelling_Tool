#ifndef GLOBALS_H
#define GLOBALS_H

#ifdef OPENDARTS_LINEAR_SOLVERS
#include "openDARTS/auxiliary/timer_node.hpp"
#else
#include "timer_node.h"
#endif // OPENDARTS_LINEAR_SOLVERS

#include <fstream>
#include <vector>

#ifdef OPENDARTS_LINEAR_SOLVERS
using namespace opendarts::auxiliary;
#endif // OPENDARTS_LINEAR_SOLVERS

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
#define MAX_NC 8

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

#ifdef _MSC_VER
#include <__msvc_int128.hpp>

struct __uint128_t : std::_Unsigned128 
{
  // Inherit constructors
  using std::_Unsigned128::_Unsigned128;

  // Define a constructor from int to handle assignment from integer literals
  constexpr __uint128_t(int x) : _Unsigned128(x) {};
  constexpr __uint128_t(double x) : _Unsigned128(static_cast<int>(x)) {};
  constexpr __uint128_t(const std::_Unsigned128& x) : _Unsigned128(x) {}

  __uint128_t& operator=(const _Unsigned128& other) {
    this->_Word[0] = other._Word[0];
    this->_Word[1] = other._Word[1];
    return *this;
  }

  template <typename T>
  operator T() const
  {
    return static_cast<T>(this->_Word[0]) + static_cast<T>(this->_Word[1] * std::pow(2, 64));
  };

  __uint128_t operator*(int x) const
  {
    return *this * static_cast<__uint128_t>(x);
  };

  __uint128_t operator*(uint64_t x) const
  {
    return *this * static_cast<__uint128_t>(x);
  };

  __uint128_t operator*(const __uint128_t& other) const
  {
    return __uint128_t(static_cast<const std::_Unsigned128&>(*this) * static_cast<const std::_Unsigned128&>(other));
  };
};

namespace std
{
  template<>
  class numeric_limits<__uint128_t>
  {
  public:
    static constexpr bool is_specialized = true;
    static constexpr __uint128_t min() noexcept { return __uint128_t(0); }
    static constexpr __uint128_t max() noexcept { return __uint128_t(~uint64_t(0), ~uint64_t(0)); }
    static constexpr __uint128_t lowest() noexcept { return min(); }
    static constexpr int digits = 128;
    static constexpr int digits10 = 38; // ceil(log10(2^128))
    static constexpr int max_digits10 = 0;
    static constexpr bool is_signed = false;
    static constexpr bool is_integer = true;
    static constexpr bool is_exact = true;
    static constexpr int radix = 2;
    static constexpr __uint128_t epsilon() noexcept { return __uint128_t(0); }
    static constexpr __uint128_t round_error() noexcept { return __uint128_t(0); }
    static constexpr int min_exponent = 0;
    static constexpr int min_exponent10 = 0;
    static constexpr int max_exponent = 0;
    static constexpr int max_exponent10 = 0;
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = false;
    static constexpr bool has_signaling_NaN = false;
    static constexpr bool has_denorm_loss = false;
    static constexpr bool has_denorm = false;
    static constexpr float_denorm_style has_denorm_style = std::denorm_absent;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = true;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;
    static constexpr float_round_style round_style = std::round_toward_zero;
  };
};

#elif defined(__GNUC__)
#endif

namespace std
{
  template <>
  struct hash<__uint128_t>
  {
    size_t operator()(const __uint128_t& x) const noexcept
    {
#ifdef _MSC_VER
      size_t h1 = std::hash<uint64_t>{}(x._Word[0]);
      size_t h2 = std::hash<uint64_t>{}(x._Word[1]);
#elif defined(__GNUC__)
      size_t h1 = std::hash<uint64_t>{}(static_cast<uint64_t>(x));
      size_t h2 = std::hash<uint64_t>{}(static_cast<uint64_t>(x >> 64));
#endif
      return h1 ^ (h2 * 0x9e3779b97f4a7c15 + 0x7f4a7c15);  // Use a large prime multiplier and a random offset
    }
  };

  // Custom to_string for __uint128_t
  std::string to_string(const __uint128_t& value);
};

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
    trans_mult_exp = 0;
    obl_min_fac = 10;
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
  index_t trans_mult_exp;      // exponent used for transmissibility multiplier => pow(phi_n/phi_0, trans_mult_exp)
  value_t obl_min_fac;         // factor used to determine z_min --> usually taken around 10, such that z_min = 10*z_OBL_min
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

// exposer helper class for <N_DiMS, N_OPS> template: N_OPS=N_DIMS*N_OPS_A + N_OPS_B

template <template <uint8_t N_DIMS, uint8_t N_OPS> class exposer_t, typename pymodule_t, uint8_t N_DIMS, uint8_t N_OPS_A, uint8_t N_OPS_B>
struct recursive_exposer_ndims_nops
{
  static void expose(pymodule_t &m)
  {
    exposer_t<N_DIMS, N_DIMS * N_OPS_A + N_OPS_B> e;

    e.expose(m);

    recursive_exposer_ndims_nops<exposer_t, pymodule_t, N_DIMS - 1, N_OPS_A, N_OPS_B>::expose(m);
  }
};

template <template <uint8_t N_DIMS, uint8_t N_OPS> class exposer_t, typename pymodule_t, uint8_t N_DIMS, uint8_t N_OPS>
struct recursive_exposer_ndims_nops2
{
    static void expose(pymodule_t& m)
    {
        exposer_t<N_DIMS, N_OPS> e;

        e.expose(m);

        recursive_exposer_ndims_nops2<exposer_t, pymodule_t, N_DIMS - 1, N_OPS>::expose(m);
        recursive_exposer_ndims_nops2<exposer_t, pymodule_t, N_DIMS, N_OPS - 1>::expose(m);
    }
};

// partial specialization to stop recusrion

template <template <uint8_t N_DIMS, uint8_t N_OPS> class exposer_t, typename pymodule_t, uint8_t N_OPS_A, uint8_t N_OPS_B>
struct recursive_exposer_ndims_nops<exposer_t, pymodule_t, 1, N_OPS_A, N_OPS_B>
{
  static void expose(pymodule_t &m)
  {
    exposer_t<1, 1 * N_OPS_A + N_OPS_B> e;

    e.expose(m);
  }
};

template <template <uint8_t N_DIMS, uint8_t N_OPS> class exposer_t, typename pymodule_t, uint8_t N_OPS>
struct recursive_exposer_ndims_nops2<exposer_t, pymodule_t, 1, N_OPS>
{
    static void expose(pymodule_t& m)
    {
        exposer_t<1, N_OPS> e;

        e.expose(m);
    }
};

template <template <uint8_t N_DIMS, uint8_t N_OPS> class exposer_t, typename pymodule_t, uint8_t N_DIMS>
struct recursive_exposer_ndims_nops2<exposer_t, pymodule_t, N_DIMS, 1>
{
    static void expose(pymodule_t& m)
    {
        exposer_t<N_DIMS, 1> e;

        e.expose(m);
    }
};

#endif
