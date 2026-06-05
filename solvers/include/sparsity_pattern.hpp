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
#ifndef OPENDARTS_LINEAR_SOLVERS_SPARSITY_PATTERN_HPP
#define OPENDARTS_LINEAR_SOLVERS_SPARSITY_PATTERN_HPP
//--------------------------------------------------------------------------

// sparsity_pattern -- the block-level sparse structure of a block-CSR matrix
// (see SOLVER_REFACTORING_PLAN.md section 12.4).
//
// It describes *which* blocks are nonzero -- independently of the block size
// and of the numerical values -- and is built once from the mesh
// connectivity. It is shared (via std::shared_ptr) by the Jacobian, the CPR
// pressure matrix, and the cached scalar-CSR expansion: the structure is
// static across a Newton solve while only the values are re-assembled.

#include <memory>
#include <string>

#include "data_types.hpp"
#include "dual_array.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    class csr_expansion; // scalar-CSR expansion; see csr_expansion.hpp

    /** @brief Block-CSR sparsity structure: row pointers, column indices,
        diagonal-block locations and a parallel row partition.

        Move-only -- it carries move-only dual_array storage and is meant to be
        shared through std::shared_ptr, not copied. */
    class sparsity_pattern
    {
    public:
      using index_t = opendarts::config::index_t;

      sparsity_pattern() noexcept = default;

      /** Builds directly from block-CSR structure arrays.
          @param n_block_rows number of block rows.
          @param n_block_cols number of block columns.
          @param row_ptr block row pointers, length @p n_block_rows + 1.
          @param col_ind block column indices, length row_ptr[n_block_rows]. */
      sparsity_pattern(index_t n_block_rows, index_t n_block_cols,
        const index_t *row_ptr, const index_t *col_ind);

      // Move-only. Defined in the .cpp where csr_expansion is complete (the
      // shared_ptr<csr_expansion> member needs the complete type to destroy).
      sparsity_pattern(sparsity_pattern &&) noexcept;
      sparsity_pattern &operator=(sparsity_pattern &&) noexcept;
      sparsity_pattern(const sparsity_pattern &) = delete;
      sparsity_pattern &operator=(const sparsity_pattern &) = delete;
      ~sparsity_pattern();

      /** (Re)builds the pattern from block-CSR structure arrays. The diagonal
          block index of every row is derived; the thread partition is reset. */
      void build(index_t n_block_rows, index_t n_block_cols,
        const index_t *row_ptr, const index_t *col_ind);

      /** Allocates the structure arrays (row_ptr, col_ind, diag_ind,
          row_thread_starts) sized for @p n_block_rows / @p nnzb but left
          zero-filled, for a caller that fills them in place (the engine's
          init_jacobian_structure). A single-partition row_thread_starts is
          installed; the diagonal indices are written by the caller. This is
          a migration bridge -- prefer build() once structure arrays exist. */
      void allocate(index_t n_block_rows, index_t n_block_cols, index_t nnzb);

      // --- dimensions --------------------------------------------------------
      [[nodiscard]] index_t n_block_rows() const noexcept { return n_block_rows_; }
      [[nodiscard]] index_t n_block_cols() const noexcept { return n_block_cols_; }
      [[nodiscard]] index_t n_blocks() const noexcept { return nnzb_; } // nnzb

      // --- structure (host) --------------------------------------------------
      [[nodiscard]] const index_t *row_ptr() const noexcept { return row_ptr_.host_data(); }
      [[nodiscard]] const index_t *col_ind() const noexcept { return col_ind_.host_data(); }
      [[nodiscard]] const index_t *diag_ind() const noexcept { return diag_ind_.host_data(); }

      // --- distributed-ready row ownership -----------------------------------
      [[nodiscard]] index_t global_row_start() const noexcept { return global_row_start_; }
      [[nodiscard]] index_t global_n_rows() const noexcept { return global_n_rows_; }
      void set_global_rows(index_t start, index_t n) noexcept
      {
        global_row_start_ = start;
        global_n_rows_ = n;
      }

      // --- parallel partition ------------------------------------------------
      /** Partitions the block rows across @p n_threads, balancing the nonzero
          blocks per thread, into row_thread_starts (length @p n_threads + 1). */
      void compute_row_thread_starts(int n_threads);
      [[nodiscard]] const index_t *row_thread_starts() const noexcept
      {
        return row_thread_starts_.host_data();
      }
      [[nodiscard]] int n_thread_partitions() const noexcept { return n_thread_partitions_; }

      // --- validation --------------------------------------------------------
      /** Checks the structural invariants: row_ptr non-decreasing and
          terminated by nnzb, every column index in range, and a diagonal
          block present in every row. @p message, if given, receives the
          first violation found. */
      [[nodiscard]] bool validate(std::string *message = nullptr) const;

      // --- scalar-CSR expansion ---------------------------------------------
      /** Lazily builds and caches the scalar-CSR expansion of this block
          pattern for the given @p block_size (see csr_expansion). Rebuilt only
          if requested with a different block size. */
      const csr_expansion &scalar_csr(int block_size) const;

#ifdef WITH_GPU
      /** Mirrors the structure arrays (row_ptr, col_ind, diag_ind) to device. */
      void sync_structure_to_device() const;

      // Device pointers to the structure arrays. Valid after a
      // sync_structure_to_device() call; the buffers are allocated on demand.
      [[nodiscard]] const index_t *row_ptr_device() const;
      [[nodiscard]] const index_t *col_ind_device() const;
      [[nodiscard]] const index_t *diag_ind_device() const;
#endif

    private:
      void compute_diag_ind();

      // Installs an even-row thread partition (row_thread_starts) sized to the
      // engine's OpenMP assembly team, so a freshly built/allocated pattern is
      // multi-threading-ready. See omp_partition.hpp.
      void install_even_row_partition();

      index_t n_block_rows_ = 0;
      index_t n_block_cols_ = 0;
      index_t nnzb_ = 0;

      dual_array<index_t> row_ptr_;           // [n_block_rows + 1]
      dual_array<index_t> col_ind_;           // [nnzb]
      dual_array<index_t> diag_ind_;          // [n_block_rows] -- diagonal block per row
      dual_array<index_t> row_thread_starts_; // [n_thread_partitions + 1]
      int n_thread_partitions_ = 0;

      index_t global_row_start_ = 0;
      index_t global_n_rows_ = 0;

      mutable std::shared_ptr<csr_expansion> csr_view_;
      mutable int csr_view_block_size_ = 0;
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_SPARSITY_PATTERN_HPP
//--------------------------------------------------------------------------
