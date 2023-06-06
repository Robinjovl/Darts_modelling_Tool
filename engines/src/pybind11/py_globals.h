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
#ifndef PY_GLOBALS_H
#define PY_GLOBALS_H

#include <stl_bind.h>
#include "globals.h"
#include "ms_well.h"

namespace py = pybind11;
PYBIND11_MAKE_OPAQUE(std::vector<index_t>);
PYBIND11_MAKE_OPAQUE(std::vector<value_t>);
PYBIND11_MAKE_OPAQUE(std::vector<ms_well*>);
PYBIND11_MAKE_OPAQUE(std::vector<operator_set_gradient_evaluator_iface*>);
//PYBIND11_MAKE_OPAQUE(std::unordered_map<std::string,timer_node>);


#endif





