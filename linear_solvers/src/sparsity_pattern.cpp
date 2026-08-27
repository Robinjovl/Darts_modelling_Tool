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

#include <algorithm>
#include <cassert>
#include <memory>
#include <string>

#include "csr_expansion.hpp"
#include "omp_partition.hpp"
#include "sparsity_pattern.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    // Defined here, where csr_expansion is a complete type, so the
    // shared_ptr<csr_expansion> member can be destroyed / reassigned.
    sparsity_pattern::sparsity_pattern(sparsity_pattern &&) noexcept = default;
    sparsity_pattern &sparsity_pattern::operator=(sparsity_pattern &&) noexcept = default;
    sparsity_pattern::~sparsity_pattern() = default;

    sparsity_pattern::sparsity_pattern(index_t n_block_rows, index_t n_block_cols,
      const index_t *row_ptr, const index_t *col_ind)
    {
      build(n_block_rows, n_block_cols, row_ptr, col_ind);
    }

    void sparsity_pattern::build(index_t n_block_rows, index_t n_block_cols,
      const index_t *row_ptr, const index_t *col_ind)
    {
      assert(n_block_rows >= 0 && n_block_cols >= 0);

      n_block_rows_ = n_block_rows;
      n_block_cols_ = n_block_cols;
      nnzb_ = (n_block_rows > 0) ? row_ptr[n_block_rows] : 0;

      row_ptr_.resize(static_cast<std::size_t>(n_block_rows) + 1);
      col_ind_.resize(static_cast<std::size_t>(nnzb_));
      diag_ind_.resize(static_cast<std::size_t>(n_block_rows));

      std::copy(row_ptr, row_ptr + n_block_rows + 1, row_ptr_.host_data());
      std::copy(col_ind, col_ind + nnzb_, col_ind_.host_data());

      compute_diag_ind();

      // The thread partition and the cached scalar-CSR expansion are derived
      // products; rebuild/reset them so a rebuild does not leave stale data.
      // Install an even-row partition sized to the assembly team so the matrix
      // is multi-threading-ready out of the box (see omp_partition.hpp).
      install_even_row_partition();
      global_row_start_ = 0;
      global_n_rows_ = n_block_rows;
      csr_view_.reset();
      csr_view_block_size_ = 0;
    }

    void sparsity_pattern::allocate(index_t n_block_rows, index_t n_block_cols, index_t nnzb)
    {
      assert(n_block_rows >= 0 && n_block_cols >= 0 && nnzb >= 0);

      n_block_rows_ = n_block_rows;
      n_block_cols_ = n_block_cols;
      nnzb_ = nnzb;

      row_ptr_.resize(static_cast<std::size_t>(n_block_rows) + 1);
      col_ind_.resize(static_cast<std::size_t>(nnzb));
      diag_ind_.resize(static_cast<std::size_t>(n_block_rows));

      // Even-row partition sized to the assembly team: the engine fills this
      // structure in place and then assembles into it under OpenMP, so the
      // partition must already be multi-thread-ready (see omp_partition.hpp).
      install_even_row_partition();

      global_row_start_ = 0;
      global_n_rows_ = n_block_rows;
      csr_view_.reset();
      csr_view_block_size_ = 0;
    }

    void sparsity_pattern::install_even_row_partition()
    {
      const int n_threads = omp_assembly_n_threads();
      n_thread_partitions_ = n_threads;
      row_thread_starts_.resize(static_cast<std::size_t>(n_threads) + 1);
      fill_even_row_partition(row_thread_starts_.host_data(), n_block_rows_, n_threads);
    }

    void sparsity_pattern::ensure_row_partition_current()
    {
      if (omp_assembly_n_threads() != n_thread_partitions_)
        install_even_row_partition();
    }

    void sparsity_pattern::compute_diag_ind()
    {
      const index_t *rp = row_ptr_.host_data();
      const index_t *ci = col_ind_.host_data();
      index_t *di = diag_ind_.host_data();
      for (index_t i = 0; i < n_block_rows_; ++i)
      {
        di[i] = -1; // -1 marks "no diagonal block"; flagged by validate()
        for (index_t jb = rp[i]; jb < rp[i + 1]; ++jb)
        {
          if (ci[jb] == i)
          {
            di[i] = jb;
            break;
          }
        }
      }
    }

    void sparsity_pattern::compute_row_thread_starts(int n_threads)
    {
      if (n_threads < 1)
        n_threads = 1;

      n_thread_partitions_ = n_threads;
      row_thread_starts_.resize(static_cast<std::size_t>(n_threads) + 1);
      index_t *ts = row_thread_starts_.host_data();
      const index_t *rp = row_ptr_.host_data();

      // Give each thread a contiguous block-row range holding roughly equal
      // numbers of nonzero blocks; row ranges stay write-disjoint.
      ts[0] = 0;
      index_t row = 0;
      for (int t = 1; t < n_threads; ++t)
      {
        const index_t target =
          static_cast<index_t>(static_cast<long long>(nnzb_) * t / n_threads);
        while (row < n_block_rows_ && rp[row + 1] <= target)
          ++row;
        ts[t] = row;
      }
      ts[n_threads] = n_block_rows_;
    }

    bool sparsity_pattern::validate(std::string *message) const
    {
      const auto fail = [message](const std::string &why) {
        if (message != nullptr)
          *message = why;
        return false;
      };

      const index_t *rp = row_ptr_.host_data();
      const index_t *ci = col_ind_.host_data();
      const index_t *di = diag_ind_.host_data();

      if (n_block_rows_ > 0 && rp[0] != 0)
        return fail("row_ptr[0] is not 0");

      for (index_t i = 0; i < n_block_rows_; ++i)
      {
        if (rp[i + 1] < rp[i])
          return fail("row_ptr is not non-decreasing at row " + std::to_string(i));

        for (index_t jb = rp[i]; jb < rp[i + 1]; ++jb)
        {
          if (ci[jb] < 0 || ci[jb] >= n_block_cols_)
            return fail("column index out of range at block " + std::to_string(jb));
        }

        if (di[i] < 0)
          return fail("no diagonal block in row " + std::to_string(i));
        if (ci[di[i]] != i)
          return fail("diag_ind does not point at the diagonal block in row "
            + std::to_string(i));
      }

      if (n_block_rows_ > 0 && rp[n_block_rows_] != nnzb_)
        return fail("row_ptr[n_block_rows] does not equal the block count");

      return true;
    }

    const csr_expansion &sparsity_pattern::scalar_csr(int block_size) const
    {
      if (!csr_view_ || csr_view_block_size_ != block_size)
      {
        csr_view_ = std::make_shared<csr_expansion>(*this, block_size);
        csr_view_block_size_ = block_size;
      }
      return *csr_view_;
    }

#ifdef WITH_GPU
    void sparsity_pattern::sync_structure_to_device() const
    {
      row_ptr_.sync_to_device();
      col_ind_.sync_to_device();
      diag_ind_.sync_to_device();
    }

    const sparsity_pattern::index_t *sparsity_pattern::row_ptr_device() const
    {
      return row_ptr_.device_data();
    }

    const sparsity_pattern::index_t *sparsity_pattern::col_ind_device() const
    {
      return col_ind_.device_data();
    }

    const sparsity_pattern::index_t *sparsity_pattern::diag_ind_device() const
    {
      return diag_ind_.device_data();
    }
#endif
  } // namespace linear_solvers
} // namespace opendarts
