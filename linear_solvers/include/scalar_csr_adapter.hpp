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
#ifndef OPENDARTS_LINEAR_SOLVERS_SCALAR_CSR_ADAPTER_HPP
#define OPENDARTS_LINEAR_SOLVERS_SCALAR_CSR_ADAPTER_HPP
//--------------------------------------------------------------------------

// scalar_csr_adapter -- the backend adapter that presents a block_csr_matrix
// as scalar (point) CSR (see SOLVER_REFACTORING_PLAN.md section 12.6).
//
// The block-native backends (bos, cuSPARSE, AMGX) consume a block_csr_matrix
// directly -- no adapter object is needed. The scalar-CSR backends (HYPRE,
// Pardiso, and PETSc -- which builds a scalar AIJ, not a block BAIJ, so that
// PCFIELDSPLIT can split within a cell block) cannot, and this adapter bridges
// them: it owns the expanded scalar values and exposes the (row_ptr, col_ind,
// values) triple those libraries ingest.
//
// The expanded *structure* is a pure function of the block sparsity pattern,
// so it is computed once and cached on the sparsity_pattern (csr_expansion).
// refresh() then re-gathers only the values -- the per-Newton-iteration cost
// is a flat O(nnz) permutation gather, never a structural rebuild.

#include <vector>

#include "block_csr_matrix.hpp"
#include "csr_expansion.hpp"
#include "data_types.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** @brief Scalar-CSR view of a block_csr_matrix for HYPRE / Pardiso.

        Holds a reference to the block matrix and owns the gathered scalar
        values; the scalar structure is borrowed from the matrix's cached
        csr_expansion. */
    class scalar_csr_adapter
    {
    public:
      using index_t = opendarts::config::index_t;
      using mat_float = opendarts::config::mat_float;

      /** Binds to @p matrix and performs the first value gather. The matrix
          must outlive the adapter. */
      explicit scalar_csr_adapter(const block_csr_matrix &matrix);

      /** Re-gathers the scalar values from the (re-assembled) block matrix.
          Call once per Newton setup; the scalar structure is unchanged. */
      void refresh();

      [[nodiscard]] index_t n_rows() const noexcept { return expansion_->n_rows(); }
      [[nodiscard]] index_t n_cols() const noexcept { return expansion_->n_cols(); }
      [[nodiscard]] index_t nnz() const noexcept { return expansion_->nnz(); }

      // The scalar-CSR triple a HYPRE / Pardiso solver ingests.
      [[nodiscard]] const index_t *row_ptr() const noexcept { return expansion_->row_ptr(); }
      [[nodiscard]] const index_t *col_ind() const noexcept { return expansion_->col_ind(); }
      [[nodiscard]] const mat_float *values() const noexcept { return values_.data(); }

    private:
      const block_csr_matrix *matrix_;
      const csr_expansion *expansion_; // cached on matrix_->structure()
      std::vector<mat_float> values_;  // owned scalar-CSR values [nnz]
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_SCALAR_CSR_ADAPTER_HPP
//--------------------------------------------------------------------------
