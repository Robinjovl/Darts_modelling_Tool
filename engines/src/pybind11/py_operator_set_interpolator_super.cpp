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

void pybind_operator_set_interpolator_super(py::module &m)
{
	// N_OPT = NC * (2 * NP + 2) + 3 * NP + 4
	// NP = 1: A =  4, B =  7 (th)
	// NP = 2: A =  6, B = 10 (th)
	// NP = 3: A =  8, B = 12 (th)
	// NP = 4: A = 10, B = 16 (th)

	// N_DIMS = 1, 2, ..., N_DIMS_MAX
	const int N_DIMS_MAX = MAX_NC;

	// thermal problem: N_OPS = A * N_DIMS + B
	const int A1 = 4;
	const int B1 = 7;

	// thermal problem: N_OPS = A * N_DIMS + B, two phase
	const int A2 = 6;
	const int B2 = 10;

	// single phase thermal: N_OPS = A * N_DIMS + B
	const int A3 = 8;
	const int B3 = 13;

	// isothermal problem: N_OPS = A * N_DIMS + B, three phases
	const int A4 = 10;
	const int B4 = 16;

	// geothermal problem: N_OPS = A * N_DIMS + B, three phases
	const int A5 = 4;
	const int B5 = 4;

  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A1, B1> e1;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A2, B2> e2;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A3, B3> e3;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A4, B4> e4;
	recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A5, B5> e5;

  e1.expose(m);
  e2.expose(m);
  e3.expose(m);
  e4.expose(m);
	e5.expose(m);
}

#endif //PYBIND11_ENABLED