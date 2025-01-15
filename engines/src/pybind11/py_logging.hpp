// This header declares the function to bind logging utilities to Python using
// Pybind11. It is conditionally included when PYBIND11_ENABLED is defined.

#pragma once
#ifdef PYBIND11_ENABLED

#include <pybind11/pybind11.h>

void pybind_logging(pybind11::module &m);

#endif
