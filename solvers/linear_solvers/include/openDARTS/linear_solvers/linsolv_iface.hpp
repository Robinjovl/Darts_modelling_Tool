//*************************************************************************
//    Copyright (c) 2022
//    Delft University of Technology, the Netherlands
//    Netherlands eScience Center
//
//    This file is part of the open Delft Advanced Research Terra Simulator (opendarts)
//
//    opendarts is free software: you can redistribute it and/or modify
//    it under the terms of the Apache License.
//
//    DARTS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// *************************************************************************

//--------------------------------------------------------------------------
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_IFACE_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_IFACE_HPP
//--------------------------------------------------------------------------

#include "openDARTS/auxiliary/timer_node.hpp"
#include "openDARTS/config/data_types.hpp"
#include "openDARTS/linear_solvers/csr_matrix_base.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    class linsolv_iface
    {
    public:
      linsolv_iface(){};

      virtual ~linsolv_iface(){};

      // generally linear solvers have a single preconditioner
      virtual int set_prec(opendarts::linear_solvers::linsolv_iface *prec_input) = 0;

      // CPR-type linear solvers have two: first for reduced system, second - for the full
      // specify here the one for reduced, while set_prec still works for the full system
      virtual int set_p_system_prec(opendarts::linear_solvers::linsolv_iface *prec_input)
      {
        (void)prec_input; // to suppress warning of unused parameter
        return 0;
      };
      
      // TODO: this is problematic since A will have templated blocks and this is
      // unaware of it, I imagine they pass this info via the data member in csr_matrix_base
      // e.g., n_row_size which seems to be the block size
      virtual int init(opendarts::linear_solvers::csr_matrix_base *A,
          opendarts::config::index_t max_iters,
          opendarts::config::mat_float tolerance) = 0;

      void init_timer_nodes(opendarts::auxiliary::timer_node *timer_setup_input,
          opendarts::auxiliary::timer_node *timer_solve_input)
      {
        this->timer_setup = timer_setup_input;
        this->timer_solve = timer_solve_input;
      };

      virtual int setup(opendarts::linear_solvers::csr_matrix_base *A_input) = 0;

      virtual int solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X) = 0;

      virtual int get_n_iters() = 0;

      virtual opendarts::config::mat_float get_residual() = 0;

      opendarts::auxiliary::timer_node *timer_setup;
      opendarts::auxiliary::timer_node *timer_solve;
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_IFACE_HPP
//--------------------------------------------------------------------------
