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

// *************************************************************************
// BOS AMG wrapper
// *************************************************************************

//--------------------------------------------------------------------------
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_BOS_AMG_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_BOS_AMG_HPP
//--------------------------------------------------------------------------

#include "openDARTS/config/data_types.hpp"
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#include "openDARTS/linear_solvers/linsolv_iface_bos.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    template <uint8_t N_BLOCK_SIZE>
    class linsolv_bos_amg : public opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>
    {
    public:
      linsolv_bos_amg();

      ~linsolv_bos_amg();

      virtual int set_prec(opendarts::linear_solvers::linsolv_iface *prec_in);

      virtual int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_in, 
        opendarts::config::index_t max_iters, 
        opendarts::config::mat_float tolerance);
      
      virtual int setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_in);

      virtual int solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X);

      virtual opendarts::config::index_t get_n_iters ();

      virtual opendarts::config::mat_float get_residual ();

      // amg_solver *solver;  // TODO: Not needed for a shell implementation, 
                             //        so we avoid including external libraries
      opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE> *prec;

      opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A;
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_BOS_AMG_HPP
//--------------------------------------------------------------------------
