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
#ifndef OPENDARTS_LINEAR_SOLVERS_BLOCK_CSR_MATRIX_HPP
#define OPENDARTS_LINEAR_SOLVERS_BLOCK_CSR_MATRIX_HPP
//--------------------------------------------------------------------------

// block_csr_matrix -- the concrete, non-templated unified matrix of the
// solver layer (see SOLVER_REFACTORING_PLAN.md section 12.4, layer 3).
//
// It owns a shared block sparsity_pattern and a values dual_array; the block
// size is a runtime field. The engine assembles into it and every solver
// backend receives it -- block-size-agnostically -- through an adapter. The
// compile-time-block-size hot path is block_csr_view<N> (block_csr_view.hpp).
//
// This is introduced alongside the legacy templated csr_matrix<N>; the engine
// and solvers migrate onto it in later steps, after which csr_matrix is
// retired.

#include <memory>

#include "data_types.hpp"
#include "dual_array.hpp"
#include "sparsity_pattern.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** @brief Block-CSR matrix: a shared sparsity_pattern plus the nonzero
        block values, mirrored host/device through dual_array.

        Move-only; clone() gives an explicit deep copy of the values (the
        immutable structure is shared, not duplicated). */
    class block_csr_matrix
    {
    public:
      using index_t = opendarts::config::index_t;
      using mat_float = opendarts::config::mat_float;

      block_csr_matrix() noexcept = default;

      /** Builds over a shared block @p structure with the given @p block_size;
          the values buffer is allocated (n_blocks * block_size^2) and zeroed. */
      block_csr_matrix(std::shared_ptr<sparsity_pattern> structure, int block_size);

      block_csr_matrix(block_csr_matrix &&) noexcept = default;
      block_csr_matrix &operator=(block_csr_matrix &&) noexcept = default;
      block_csr_matrix(const block_csr_matrix &) = delete;
      block_csr_matrix &operator=(const block_csr_matrix &) = delete;

      /** Deep-copies the values; the (immutable) structure is shared. */
      [[nodiscard]] block_csr_matrix clone() const;

      /** (Re)builds over a structure and block size; values zeroed. */
      void reset(std::shared_ptr<sparsity_pattern> structure, int block_size);

      [[nodiscard]] bool empty() const noexcept { return block_size_ == 0; }
      [[nodiscard]] int block_size() const noexcept { return block_size_; }

      [[nodiscard]] const sparsity_pattern &structure() const noexcept { return *structure_; }
      [[nodiscard]] const std::shared_ptr<sparsity_pattern> &structure_ptr() const noexcept
      {
        return structure_;
      }

      // --- dimensions (0 when empty) -----------------------------------------
      [[nodiscard]] index_t n_block_rows() const noexcept;
      [[nodiscard]] index_t n_blocks() const noexcept;  // nonzero blocks (nnzb)
      [[nodiscard]] index_t n_rows() const noexcept;    // scalar rows = n_block_rows * nb
      [[nodiscard]] index_t n_values() const noexcept;  // nnzb * nb * nb

      // --- structure (host) --------------------------------------------------
      [[nodiscard]] const index_t *row_ptr() const noexcept;
      [[nodiscard]] const index_t *col_ind() const noexcept;
      [[nodiscard]] const index_t *diag_ind() const noexcept;

      // --- values ------------------------------------------------------------
      // Row-major nb x nb blocks, n_values() doubles in block order.
      [[nodiscard]] mat_float *values() noexcept { return values_.host_data(); }
      [[nodiscard]] const mat_float *values() const noexcept { return values_.host_data(); }
      void set_zero();

#ifdef WITH_GPU
      [[nodiscard]] mat_float *values_device() { return values_.device_data(); }
      [[nodiscard]] const mat_float *values_device() const { return values_.device_data(); }
      [[nodiscard]] const index_t *row_ptr_device() const;
      [[nodiscard]] const index_t *col_ind_device() const;
      [[nodiscard]] const index_t *diag_ind_device() const;

      /** Mirrors structure + values to device (host -> device). */
      void sync_to_device() const;
      /** Brings the values back from device (device -> host). */
      void sync_to_host();
#endif

    private:
      std::shared_ptr<sparsity_pattern> structure_;
      dual_array<mat_float> values_;
      int block_size_ = 0;
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_BLOCK_CSR_MATRIX_HPP
//--------------------------------------------------------------------------
