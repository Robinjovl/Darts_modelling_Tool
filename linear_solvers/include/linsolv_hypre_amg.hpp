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
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_HYPRE_AMG_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_HYPRE_AMG_HPP
//--------------------------------------------------------------------------

#include <vector>

#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv.h"
#include "_hypre_parcsr_mv.h"

#include "data_types.hpp"
#include "csr_matrix.hpp"
#include "linsolv_iface.hpp"
#include "linsolv_iface_bos.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    template <uint8_t N_BLOCK_SIZE>
    class linsolv_hypre_amg //: public opendarts::linear_solvers::linsolv_iface
    {
    public:
      linsolv_hypre_amg();

      ~linsolv_hypre_amg();

      int set_prec(opendarts::linear_solvers::linsolv_iface *prec_in);

      int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_in,
        opendarts::config::index_t max_iters,
        opendarts::config::mat_float tolerance);

      int setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_in);

      // Value-only refresh of the cached IJ matrix without rebuilding the AMG
      // hierarchy / ILU factorization. Call this on Newton iterations where
      // the sparsity pattern is unchanged but matrix values have been refreshed
      // -- significantly cheaper than setup() because the AMG hierarchy / ILU
      // factorization is reused.
      void refresh(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A);

      int solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X);

      opendarts::config::index_t get_n_iters ();

      opendarts::config::mat_float get_residual ();

      HYPRE_Solver solver;
      opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE> *prec;
      opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A;  // kept for backwards compatibility
      HYPRE_IJMatrix A_ij;  // the matrix linked to the hypre amg solver. HYPRE_IJMatrixSetValues +
                            // HYPRE_IJMatrixAssemble COPY the scalar CSR values into HYPRE-owned
                            // ParCSR storage, so A_ij holds an independent copy of A's data (nothing
                            // is shared with A) -- the values must be re-set whenever A is reassembled.
      HYPRE_ParCSRMatrix A_parcsr;  // the ParCSR object fetched from A_ij (HYPRE_IJMatrixGetObject);
                                    // it is a handle into A_ij's HYPRE-owned storage
      HYPRE_IJVector b_ij;  // right hand side vector as HYPRE_IJVector
      HYPRE_ParVector b_par;  // right hand side vector as HYPRE_ParVector
      HYPRE_IJVector x_ij;  // solution vector as HYPRE_IJVector
      HYPRE_ParVector x_par;  // solution vector as HYPRE_ParVector

    private:
      void csr_matrix_to_hypre_ij(
        opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> &A,
        HYPRE_IJMatrix &A_ij);

      // Per-apply scratch -- the row index list passed to HYPRE_IJVectorSetValues
      // is just [0, n_rows). Once n_rows is known it never changes, so cache
      // it as a member instead of allocating + std::iota'ing per solve().
      // Mirrored in linsolv_hypre_ilu and linsolv_cpr (set_hypre_vector).
      std::vector<opendarts::config::index_t> row_indices_;
      std::vector<opendarts::config::index_t> n_cols_;
      // using opendarts::linear_solvers::linsolv_iface::init;
      // using opendarts::linear_solvers::linsolv_iface::setup;
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_HYPRE_AMG_HPP
//--------------------------------------------------------------------------
