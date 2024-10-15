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
    //  engine_super_*
	//    NE /*acc*/ + NE * NP /*flux*/ + NP /*UPSAT*/ + NE * NP /*gradient*/ + NE /*kinetic*/ + 2 * NP /*gravpc*/ + 1 /*poro*/ + 2 /*temperature and pressure*/
    //    N_OPS = NE * (2 * NP + 2) + 3 * NP + 3 

    ABPair<4, 6>,     // thermal problem, single phase

    ABPair<6, 9>,    // thermal problem, two phase

    ABPair<8, 12>,    // three phases thermal

    ABPair<10, 15>,   // isothermal problem, 4 phases

    // ???
    //ABPair<4, 4>,     // geothermal problem, three phases

    //  engine_super_elastic_*
	//    NE /*acc*/ + NE * NP /*flux*/ + NP /*UPSAT*/ + NE * NP /*gradient*/ + NE /*kinetic*/ + 2 * NP /*gravpc*/ + 1 /*poro*/ + 2 /*temperature and pressure*/ + 1 /*weight*/
    //    N_OPS = NE * (2 * NP + 2) + 3 * NP + 3 + 1 

    // NP = 1: A =  4, B =  8
    ABPair<4, 7>,     // poroelasticity, single-phase

    // NP = 1: A =  6, B =  8
    ABPair<6, 10>     // poroelasticity, two-phase
  >(m);
}

#endif //PYBIND11_ENABLED