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
#ifndef OPENDARTS_CONFIG_VERSION_H
#define OPENDARTS_CONFIG_VERSION_H
//--------------------------------------------------------------------------

#include <string>

namespace opendarts
{
  namespace config
  {
    // Variables for compatibility with older interface 
    extern const char *LINSOLV_BUILD_DATE;
    extern const char *LINSOLV_BUILD_MACHINE;
    extern const char *LINSOLV_BUILD_GIT_HASH;
    
    // Return the major version of openDARTS
    std::string get_version_major();

    // Return the minor version of openDARTS
    std::string get_version_minor();

    // Return the absolute path of the openDARTS source directory
    std::string get_cmake_openDARTS_source_dir();

    // The Git ref at compile time
    std::string get_git_Ref();

    // The raw git hash at compile time
    std::string get_git_hash();
    
    // The build date of this version of the code
    std::string get_build_date();
    
    // The name of the machine where the code was built
    std::string get_build_machine();

    // Whether the working directory was clean at compile time
    bool is_git_clean();

  } // namespace config
} // namespace opendarts
//--------------------------------------------------------------------------
#endif // OPENDARTS_CONFIG_VERSION_H
//--------------------------------------------------------------------------
