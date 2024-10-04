#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl.h>
#include <tuple>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

// Define a structure to hold A and B values
template<int AVal, int BVal>
struct ABPair 
{
  static constexpr int A = AVal;
  static constexpr int B = BVal;
};

// Recursive variadic template to handle each pair
template<typename T, typename... Rest>
void expose_recursive_exposer(py::module& m) {
	// N_DIMS = 1, 2, ..., N_DIMS_MAX
	const int N_DIMS_MAX = MAX_NC;
  using ExposerType = recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, T::A, T::B>;
  ExposerType exposer;
  exposer.expose(m);

  if constexpr (sizeof...(Rest) > 0) {
    expose_recursive_exposer<Rest...>(m);
  }
}

void pybind_operator_set_interpolator_super(py::module &m)
{
  // N_OPS = A * NC + B
  expose_recursive_exposer<
    /*  engine_super_*
        N_OPS = NC * (2 * NP + 2) + 4 * NP + 4 
    */

    // NP = 1: A =  4, B =  8 (th)
    ABPair<4, 8>,     // single phase

    // NP = 2: A =  6, B = 12 (th)
    ABPair<6, 12>,    // two-phase

    // NP = 3: A =  8, B = 12 (th) ???
    ABPair<8, 13>,    // three phases

    // NP = 4: A = 10, B = 16 (th)
    ABPair<10, 16>,   // four phases

    // ???
    ABPair<4, 4>,     // geothermal problem, three phases

    /*  engine_super_elastic_*
        N_OPS = NC * (2 * NP + 2) + 3 * NP + 5
    */

    // NP = 1: A =  4, B = 8
    ABPair<4, 8>,     // poroelasticity, single phase

    // NP = 2: A =  6, B = 11
    ABPair<6, 11>     // poroelasticity, two-phase
  >(m);
}

#endif //PYBIND11_ENABLED