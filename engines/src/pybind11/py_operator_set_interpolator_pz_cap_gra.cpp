//*************************************************************************
//    Copyright (c) 2018
//            Mark Khait         M.Khait@tudelft.nl
//            Denis Voskov    D.V.Voskov@tudelft.nl
//    Delft University of Technology, the Netherlands
//
//    This file is part of the Delft Advanced Research Terra Simulator (DARTS)
//
//    DARTS is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as 
//    published by the Free Software Foundation, either version 3 of the 
//    License, or (at your option) any later version.
//
//    DARTS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public 
//    License along with DARTS. If not, see <http://www.gnu.org/licenses/>.
// *************************************************************************

#ifdef PYBIND11_ENABLED
#include <pybind11.h>
#include "py_globals.h"
#include <stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

void pybind_operator_set_interpolator_pz_cap_gra(py::module &m)
{
  // nc, grav + pc, 2 phases  : n_ops = (1 + 2) * N_DIMS + 2 + 2

  // N_DIMS = 1, 2, ..., N_DIMS_MAX
  const int N_DIMS_MAX = MAX_NC;
  
  // N_OPS = A * N_DIMS + B
  const int A = 3;
  const int B = 4;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A, B> e;

  // nc, grav + pc, 3 phases  : n_ops = (1 + 3) * N_DIMS + 3 + 3

  // N_DIMS = 1, 2, ..., N_DIMS_MAX
  const int N_DIMS_MAX1 = 3; // used only for black-oil

  // N_OPS = A * N_DIMS + B
  const int A1 = 4;
  const int B1 = 6;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX1, A1, B1> e1;

 
  e.expose(m);
  e1.expose(m);
}

#endif //PYBIND11_ENABLED