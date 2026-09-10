//*************************************************************************
//    Copyright (c) 2022
//    Delft University of Technology, the Netherlands
//    Netherlands eScience Center
//
//    This file is part of the open Delft Advanced Research Terra Simulator (opendarts)
//
//    opendarts is free software: you can redistribute it and/or modify
//    it under the terms of the Apache License.
// *************************************************************************

//--------------------------------------------------------------------------
#ifndef OPENDARTS_LINEAR_SOLVERS_HYPRE_IJ_BUILDER_HPP
#define OPENDARTS_LINEAR_SOLVERS_HYPRE_IJ_BUILDER_HPP
//--------------------------------------------------------------------------

// Shared scalar-CSR -> HYPRE IJ (ParCSR) build/refresh helpers.
//
// The "create / set / assemble / get-object" sequence for a block-size-1 CSR
// matrix was copy-pasted across linsolv_cpr (the four Ap/As/Ap_T/As_T handles)
// and the two standalone wrappers linsolv_hypre_amg / linsolv_hypre_ilu. These
// free functions are the single source of truth for that sequence, so a
// HYPRE-API fix (or the sticky-error handling) is made in one place.
//
// Each caller owns its own row-index and per-row-degree caches (a matrix may
// have several IJ handles, each with its own degree cache) and passes them in
// by reference. Operates on a raw scalar-CSR triple (n_rows, row_ptr, col_ind,
// values) so a caller can feed a csr_matrix<1> OR a scalar_csr_adapter's borrowed
// arrays directly (no intermediate shell copy).

#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv.h"

#include "data_types.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    namespace hypre_ij
    {
      // Shared HYPRE return-code check: log, clear HYPRE's sticky global error
      // flag (so a solver recovered after a timestep cut does not keep tripping
      // on it), and throw -- engine_base::solve_linear_equation() converts the
      // throw into a timestep cut instead of killing the host process. Unifies
      // linsolv_cpr::check_hypre and the per-wrapper check_result.
      inline void check(int res, const char *where)
      {
        if (res)
        {
          char msg[256];
          HYPRE_DescribeError(res, msg);
          std::string err =
              std::string("hypre_ij: HYPRE error in ") + where + " -- " + msg;
          std::cerr << err << std::endl;
          HYPRE_ClearAllErrors();
          throw std::runtime_error(err);
        }
      }

      // Grow row_indices monotonically to cover [0, n_rows) and fill the new tail.
      inline void ensure_row_indices(opendarts::config::index_t n_rows,
          std::vector<opendarts::config::index_t> &row_indices)
      {
        if (static_cast<opendarts::config::index_t>(row_indices.size()) < n_rows)
        {
          const opendarts::config::index_t old =
              static_cast<opendarts::config::index_t>(row_indices.size());
          row_indices.resize(n_rows);
          std::iota(row_indices.begin() + old, row_indices.end(), old);
        }
      }

      // Fill n_cols with per-row column counts derived from the scalar row
      // pointer (grow-only; values are structure-stable across Newton iters).
      inline void compute_row_degrees(opendarts::config::index_t n_rows,
          const opendarts::config::index_t *row_ptr,
          std::vector<opendarts::config::index_t> &n_cols)
      {
        if (static_cast<opendarts::config::index_t>(n_cols.size()) < n_rows)
          n_cols.resize(n_rows);
        for (opendarts::config::index_t i = 0; i < n_rows; ++i)
          n_cols[i] = row_ptr[i + 1] - row_ptr[i];
      }

      // Create a fresh IJ matrix from a scalar-CSR triple and (if A_parcsr is
      // non-null) fetch its ParCSR object. Pass A_parcsr = nullptr to defer the
      // GetObject to the caller.
      inline void build(opendarts::config::index_t n_rows,
          const opendarts::config::index_t *row_ptr,
          const opendarts::config::index_t *col_ind,
          const opendarts::config::mat_float *values,
          std::vector<opendarts::config::index_t> &row_indices,
          std::vector<opendarts::config::index_t> &n_cols, HYPRE_IJMatrix &A_ij,
          HYPRE_ParCSRMatrix *A_parcsr)
      {
        const opendarts::config::index_t ilower = 0;
        const opendarts::config::index_t iupper = n_rows - 1;
        ensure_row_indices(n_rows, row_indices);
        compute_row_degrees(n_rows, row_ptr, n_cols);
        check(HYPRE_IJMatrixCreate(hypre_MPI_COMM_WORLD, ilower, iupper, ilower,
                  iupper, &A_ij),
            "IJMatrixCreate");
        check(HYPRE_IJMatrixSetPrintLevel(A_ij, 0), "IJMatrixSetPrintLevel");
        check(HYPRE_IJMatrixSetObjectType(A_ij, HYPRE_PARCSR),
            "IJMatrixSetObjectType");
        check(HYPRE_IJMatrixInitialize(A_ij), "IJMatrixInitialize");
        check(HYPRE_IJMatrixSetValues(A_ij, n_rows, n_cols.data(),
                  row_indices.data(), col_ind, values),
            "IJMatrixSetValues");
        check(HYPRE_IJMatrixAssemble(A_ij), "IJMatrixAssemble");
        if (A_parcsr)
          check(HYPRE_IJMatrixGetObject(A_ij, (void **) A_parcsr),
              "IJMatrixGetObject");
      }

      // Refresh values on an existing IJ matrix (sparsity preserved) and fetch
      // the ParCSR object. Recomputes n_cols only if it is too small (the
      // structure is stable across Newton iterations, so this is normally a
      // no-op) -- matching the linsolv_cpr fast path.
      inline void refresh(opendarts::config::index_t n_rows,
          const opendarts::config::index_t *row_ptr,
          const opendarts::config::index_t *col_ind,
          const opendarts::config::mat_float *values,
          std::vector<opendarts::config::index_t> &row_indices,
          std::vector<opendarts::config::index_t> &n_cols, HYPRE_IJMatrix &A_ij,
          HYPRE_ParCSRMatrix *A_parcsr)
      {
        ensure_row_indices(n_rows, row_indices);
        if (static_cast<opendarts::config::index_t>(n_cols.size()) < n_rows)
          compute_row_degrees(n_rows, row_ptr, n_cols);
        check(HYPRE_IJMatrixInitialize(A_ij), "IJMatrixInitialize(refresh)");
        check(HYPRE_IJMatrixSetValues(A_ij, n_rows, n_cols.data(),
                  row_indices.data(), col_ind, values),
            "IJMatrixSetValues(refresh)");
        check(HYPRE_IJMatrixAssemble(A_ij), "IJMatrixAssemble(refresh)");
        if (A_parcsr)
          check(HYPRE_IJMatrixGetObject(A_ij, (void **) A_parcsr),
              "IJMatrixGetObject(refresh)");
      }
    } // namespace hypre_ij
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_HYPRE_IJ_BUILDER_HPP
//--------------------------------------------------------------------------
