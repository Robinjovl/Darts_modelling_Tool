#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"

namespace py = pybind11;

// Forward declarations of split functions
void pybind_operator_set_interpolator_rates_part1(py::module &m);
void pybind_operator_set_interpolator_rates_part2(py::module &m);
void pybind_operator_set_interpolator_rates_part3(py::module &m);

void pybind_operator_set_interpolator_rates(py::module &m)
{
  // Call all split parts
  // Part 1: Single component and single phase
  pybind_operator_set_interpolator_rates_part1(m);

  // Part 2: Two phase and three phase
  pybind_operator_set_interpolator_rates_part2(m);

  // Part 3: Four phase
  pybind_operator_set_interpolator_rates_part3(m);
}

#endif //PYBIND11_ENABLED
