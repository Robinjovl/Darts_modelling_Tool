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
// HYPRE AMG wrapper
// *************************************************************************

//--------------------------------------------------------------------------
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_HYPRE_ILU_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_HYPRE_ILU_HPP
//--------------------------------------------------------------------------

#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv.h"
#include "_hypre_parcsr_mv.h"

#include "openDARTS/config/data_types.hpp"
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#include "openDARTS/linear_solvers/linsolv_iface.hpp"
#include "openDARTS/linear_solvers/linsolv_iface_bos.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    template <uint8_t N_BLOCK_SIZE>
    class linsolv_hypre_ilu //: public opendarts::linear_solvers::linsolv_iface
    {
    public:
      linsolv_hypre_ilu();

      ~linsolv_hypre_ilu();

      int set_prec(opendarts::linear_solvers::linsolv_iface *prec_in);

      int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_in, 
        opendarts::config::index_t max_iters, 
        opendarts::config::mat_float tolerance);
      
      int setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_in);

      int solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X);

      opendarts::config::index_t get_n_iters ();

      opendarts::config::mat_float get_residual ();

      HYPRE_Solver solver;
      opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE> *prec;
      opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A;  // kept for backwards compatibility
      HYPRE_IJMatrix A_ij;  // this matrix is the matrix linked to the hypre amg solver
                            // it shares most of the data with the matrix A, to avoid 
                            // duplication, but some duplication is unavoidable because 
                            // A and A_ij use slightly different data formats, specifically 
                            // the way row data is encoded.
      HYPRE_ParCSRMatrix A_parcsr;  // we also need a Hypre ParCSRMatrix so we keep it here also 
                                    // this one shares all data with A_ij
      HYPRE_IJVector b_ij;  // right hand side vector as HYPRE_IJVector
      HYPRE_ParVector b_par;  // right hand side vector as HYPRE_ParVector
      HYPRE_IJVector x_ij;  // solution vector as HYPRE_IJVector
      HYPRE_ParVector x_par;  // solution vector as HYPRE_ParVector
                            
    private:
      static void csr_matrix_to_hypre_ij(
        opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> &A,
        HYPRE_IJMatrix &A_ij);
      // using opendarts::linear_solvers::linsolv_iface::init;
      // using opendarts::linear_solvers::linsolv_iface::setup;
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_HYPRE_ILU_HPP
//--------------------------------------------------------------------------
