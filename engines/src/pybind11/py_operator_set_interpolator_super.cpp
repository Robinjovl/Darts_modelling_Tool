#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"

namespace py = pybind11;

// Forward declarations of split functions
void pybind_operator_set_interpolator_super_part1(py::module &m);
void pybind_operator_set_interpolator_super_part2(py::module &m);
void pybind_operator_set_interpolator_super_part3(py::module &m);
void pybind_operator_set_interpolator_super_part4(py::module &m);

void pybind_operator_set_interpolator_super(py::module &m)
{
  // Call all split parts
  // Part 1: Thermal single-phase and two-phase (A=4,B=10), (A=6,B=17)
  pybind_operator_set_interpolator_super_part1(m);

  // Part 2: Thermal three-phase and four-phase (A=8,B=24), (A=10,B=31)
  pybind_operator_set_interpolator_super_part2(m);

  // Part 3: Geothermal and poroelasticity PM (A=4,B=4), (A=2,B=0)
  pybind_operator_set_interpolator_super_part3(m);

  // Part 4: engine_super_elastic (A=4,B=11), (A=6,B=18)
  pybind_operator_set_interpolator_super_part4(m);
}

#endif //PYBIND11_ENABLED
