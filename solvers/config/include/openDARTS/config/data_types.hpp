//*************************************************************************
//    Copyright (c) 2022
//    Delft University of Technology, the Netherlands
//
//    This file is part of the open Delft Advanced Research Terra Simulator (opendarts)
//
//    opendarts is free software: you can redistribute it and/or modify
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

//--------------------------------------------------------------------------
#ifndef OPENDARTS_CONFIG_DATA_TYPES_H
#define OPENDARTS_CONFIG_DATA_TYPES_H
//--------------------------------------------------------------------------

#include <climits>

namespace opendarts
{
  namespace config
  {
    // NOTE: This should be in an opendarts data_types.hpp file of some sort, global to all opendarts
    typedef int index_t; // define the matrix index type
    const unsigned long long int INDEX_T_MAX = INT_MAX;
    typedef double mat_float; // define data type of matrix values

    opendarts::config::index_t static_cast_check(size_t size_t_value);
  } // namespace config
} // namespace opendarts
//--------------------------------------------------------------------------
#endif // OPENDARTS_CONFIG_DATA_TYPES_H
//--------------------------------------------------------------------------
