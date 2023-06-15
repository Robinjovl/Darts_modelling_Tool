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

#include <stdexcept>
#include <string>

#include "openDARTS/config/data_types.hpp"

namespace opendarts
{
  namespace config
  {
    // Casts size_t to int, but check first if  there will be no overflow
    opendarts::config::index_t static_cast_check(size_t size_t_value)
    {
      if (size_t_value >
          opendarts::config::INDEX_T_MAX) // check if it is larger than the maximum value of opendarts::config::index_t,
                                          // if so throw and exception since this cannot be handled in the loops
      {
        throw std::overflow_error("Size of vector is larger than what is supported by opendarts::config::index_t");
      }
      return static_cast<opendarts::config::index_t>(size_t_value);
    }

  } // namespace config
} // namespace opendarts
