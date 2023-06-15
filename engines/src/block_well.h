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

#ifndef BLOCK_WELL_H
#define BLOCK_WELL_H

#include <vector>
#include <array>

#include "globals.h"
#include "well_controls.h"

#ifdef OPENDARTS_LINEAR_SOLVERS
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#else
#include "csr_matrix.h"
#endif // OPENDARTS_LINEAR_SOLVERS 

#ifdef OPENDARTS_LINEAR_SOLVERS
using namespace opendarts::linear_solvers;
#endif // OPENDARTS_LINEAR_SOLVERS 

enum
{
  INJECTOR_RATE1 = -5,
  INJECTOR_RATE2 = -4,
  INJECTOR_RATE3 = -3,
  INJECTOR_RATE12 = -2,
  INJECTOR_BHP = -1,
  SHUT_DOWN = 0,
  PRODUCER_BHP = 1,
  PRODUCER_RATE12 = 2,
  PRODUCER_RATE3 = 3,
  PRODUCER_RATE2 = 4,
  PRODUCER_RATE1 = 5
};

/// Structure for well perforation
struct block_perf
{
  index_t reservoir_block;    /// the block of reservoir which is perforated
  value_t well_index;         /// transmissibilty between reservoir block and well block

  block_perf() {};
  //block_perf (index_t res_block, value_t wi) :reservoir_block (res_block), well_index (wi) {};
};

/// Structure for well control 
struct well_control
{
  int                   control_type;          /// INJECTOR_BHP or PRODUCER_BHP
  value_t               control_param;         /// BHP (or rate) value
  std::vector <value_t> inj_stream;            /// composition of injection stream
  value_t               inj_temperature;       /// temperature of injection stream

  well_control() {};
  //well_control (int type, value_t param) :control_type (type), control_param (param) {};
};

/// Class for well definition (both segments and controls)
class block_well
{

public:
  block_well();

  void set_nc(int nc_)
  {
    NC = nc_;
    current_rates = new value_t[NC];
  };
  std::vector<block_perf>    block_perf_list;
  std::vector<std::pair <int, well_control>> control_list;    /// a list of well controls and correspondent report timestep
  well_control current_control;                               /// effective well control to be used in jacobian assembly

  index_t first_well_block_index;   /// index of the first well block, which connects to ghost well block
  index_t ghost_well_block_index;   /// index of the first well block, which connects to ghost well block
  index_t NC;

  value_t well_block_volume;    /// volume of well block
  value_t well_block_trans;     /// transmissibility bewteen well blocks

  value_t *current_rates;       /// current molar rates taken as influx/outflux of ghost block for production/injection well respectively
  value_t current_BHP;          /// current BHP for well
  value_t current_temp;         /// current temperature for well

  well_control_iface *control;    /// set of controls
  well_control_iface *constraint; /// set of constraints

  std::string name;               /// name of well
};

#endif
