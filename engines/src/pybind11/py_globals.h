#ifndef PY_GLOBALS_H
#define PY_GLOBALS_H

#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "globals.h"
#include "ms_well.h"
#include "logging.h"
#include "py_globals_interpolation.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<ms_well*>);

PYBIND11_MAKE_OPAQUE(std::vector<linear_solver_params>);
//PYBIND11_MAKE_OPAQUE(std::unordered_map<std::string,timer_node>);

template <typename T>
py::array_t<T> get_raw_array(T* arr, size_t size) {
  return py::array_t<T>(
    { size }, // Number of elements in the array
    { sizeof(T) }, // Stride of the array in bytes
    arr, // Pointer to the raw array data
    py::capsule(arr, [](void* f) {}) // Capsule for memory management
  );
}

#endif





