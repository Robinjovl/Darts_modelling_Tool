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
#ifndef OPENDARTS_LINEAR_SOLVERS_CSR_EXPANSION_HPP
#define OPENDARTS_LINEAR_SOLVERS_CSR_EXPANSION_HPP
//--------------------------------------------------------------------------

// csr_expansion -- the scalar (point) CSR expansion of a block-CSR sparsity
// pattern for a fixed block size (see SOLVER_REFACTORING_PLAN.md section 12.6).
//
// The scalar-CSR backends (HYPRE, Pardiso) cannot consume block-CSR directly.
// The expanded *structure* (row_ptr, col_ind) is a pure function of the block
// pattern and the block size, so it is built once and cached on the
// sparsity_pattern. Each Newton iteration then only gathers the values:
// csr_values[k] = bsr_values[value_map[k]] -- a fixed permutation, never a
// structural rebuild.

#include "data_types.hpp"
#include "dual_array.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    class sparsity_pattern;

    /** @brief Scalar-CSR view of a block-CSR pattern at a fixed block size.

        Built from a sparsity_pattern; owns the expanded structure and the
        block-to-scalar value permutation. Move-only. */
    class csr_expansion
    {
    public:
      using index_t = opendarts::config::index_t;
      using mat_float = opendarts::config::mat_float;

      /** Expands @p bsr into scalar CSR for the given @p block_size. The block
          pattern's column indices are assumed sorted ascending per row, so the
          expanded column indices are sorted too. */
      csr_expansion(const sparsity_pattern &bsr, int block_size);

      csr_expansion(csr_expansion &&) noexcept = default;
      csr_expansion &operator=(csr_expansion &&) noexcept = default;
      csr_expansion(const csr_expansion &) = delete;
      csr_expansion &operator=(const csr_expansion &) = delete;

      [[nodiscard]] index_t n_rows() const noexcept { return n_rows_; }
      [[nodiscard]] index_t n_cols() const noexcept { return n_cols_; }
      [[nodiscard]] index_t nnz() const noexcept { return nnz_; }
      [[nodiscard]] int block_size() const noexcept { return block_size_; }

      [[nodiscard]] const index_t *row_ptr() const noexcept { return row_ptr_.host_data(); }
      [[nodiscard]] const index_t *col_ind() const noexcept { return col_ind_.host_data(); }
      /** value_map[k] is the index into the block-CSR values array of the
          source of scalar nonzero k. */
      [[nodiscard]] const index_t *value_map() const noexcept { return value_map_.host_data(); }

      /** Gathers block-CSR values into the scalar-CSR value order:
          @p csr_values[k] = @p bsr_values[value_map[k]] for k in [0, nnz()).
          The caller owns @p csr_values (length nnz()). */
      void gather_values(const mat_float *bsr_values, mat_float *csr_values) const;

    private:
      index_t n_rows_ = 0;
      index_t n_cols_ = 0;
      index_t nnz_ = 0;
      int block_size_ = 0;

      dual_array<index_t> row_ptr_;   // [n_rows + 1]
      dual_array<index_t> col_ind_;   // [nnz]
      dual_array<index_t> value_map_; // [nnz] -- block-CSR value index per scalar nonzero
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_CSR_EXPANSION_HPP
//--------------------------------------------------------------------------
