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
#ifndef OPENDARTS_LINEAR_SOLVERS_BLOCK_CSR_VIEW_HPP
#define OPENDARTS_LINEAR_SOLVERS_BLOCK_CSR_VIEW_HPP
//--------------------------------------------------------------------------

// block_csr_view<N> -- the zero-overhead, compile-time-block-size lens over a
// block_csr_matrix (see SOLVER_REFACTORING_PLAN.md section 12.4, layer 4).
//
// The engine is templated on the number of equations per cell, so the
// assembly hot path holds a block_csr_view<N_VARS>: block accessors with a
// compile-time N are fully inlined, with no virtual dispatch and no runtime
// block-size arithmetic. The view is non-owning -- it stores a pointer to a
// block_csr_matrix and adds no storage of its own.

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "block_csr_matrix.hpp"
#include "data_types.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** @brief Compile-time-block-size view over a block_csr_matrix.
        @tparam N block size (equations per cell). */
    template <uint8_t N>
    class block_csr_view
    {
    public:
      using index_t = opendarts::config::index_t;
      using mat_float = opendarts::config::mat_float;

      static constexpr int block_size = N;
      static constexpr int block_area = static_cast<int>(N) * static_cast<int>(N);

      /** Wraps @p matrix; its runtime block size must equal N. */
      explicit block_csr_view(block_csr_matrix &matrix) noexcept : matrix_(&matrix)
      {
        assert(matrix.block_size() == static_cast<int>(N)
          && "block_csr_view: matrix block size does not match N");
      }

      [[nodiscard]] block_csr_matrix &matrix() const noexcept { return *matrix_; }

      // --- structure ---------------------------------------------------------
      [[nodiscard]] index_t n_block_rows() const noexcept { return matrix_->n_block_rows(); }
      [[nodiscard]] index_t n_blocks() const noexcept { return matrix_->n_blocks(); }
      [[nodiscard]] const index_t *row_ptr() const noexcept { return matrix_->row_ptr(); }
      [[nodiscard]] const index_t *col_ind() const noexcept { return matrix_->col_ind(); }
      [[nodiscard]] const index_t *diag_ind() const noexcept { return matrix_->diag_ind(); }

      // --- block access (assembly hot path) ----------------------------------
      /** Pointer to the @p jb-th stored block (jb in [0, n_blocks())): N*N
          row-major doubles. */
      [[nodiscard]] mat_float *block(index_t jb) const noexcept
      {
        return matrix_->values() + static_cast<std::size_t>(jb) * static_cast<std::size_t>(block_area);
      }

      /** Element (row @p e, column @p v) of the @p jb-th stored block. */
      [[nodiscard]] mat_float &at(index_t jb, int e, int v) const noexcept
      {
        return block(jb)[static_cast<std::size_t>(e) * N + static_cast<std::size_t>(v)];
      }

      /** Pointer to the diagonal block of block row @p i_block. */
      [[nodiscard]] mat_float *diag_block(index_t i_block) const noexcept
      {
        return block(matrix_->diag_ind()[i_block]);
      }

    private:
      block_csr_matrix *matrix_;
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_BLOCK_CSR_VIEW_HPP
//--------------------------------------------------------------------------
