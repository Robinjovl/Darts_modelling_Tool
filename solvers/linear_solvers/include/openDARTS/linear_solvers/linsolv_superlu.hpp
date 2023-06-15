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
// SUPERLU wrapper
// *************************************************************************

//--------------------------------------------------------------------------
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_SUPERLU_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_SUPERLU_HPP
//--------------------------------------------------------------------------

#include "openDARTS/config/data_types.hpp"
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#include "openDARTS/linear_solvers/linsolv_iface_bos.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    template <uint8_t N_BLOCK_SIZE>
    class linsolv_superlu : public opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>
    {

    public:
      linsolv_superlu(){};

      ~linsolv_superlu();

      virtual int set_prec(
          opendarts::linear_solvers::linsolv_iface *prec_input); // Implemented as do nothing

      virtual int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_input,
          opendarts::config::index_t max_iters,
          opendarts::config::mat_float tolerance);

      virtual int setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_update);

      virtual int solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X);

      virtual opendarts::config::index_t get_n_iters();

      virtual opendarts::config::mat_float get_residual();

      opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A; // the pointer to the matrix to use to solve the system
                                                              // (this is a pointer, so it can change outside the class)

      opendarts::config::index_t n_rows;
      opendarts::config::index_t nnz;

      opendarts::config::index_t *perm_r; /* row permutations from partial pivoting */
      opendarts::config::index_t *perm_c; /* column permutation vector */

      opendarts::config::index_t first;
      void *work;
      opendarts::config::index_t lwork;
      opendarts::config::mat_float *R, *C;

      opendarts::config::index_t *etree;
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_IFACE_HPP
//--------------------------------------------------------------------------
