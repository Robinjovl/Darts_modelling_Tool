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
#include <utility>

#include "block_csr_matrix.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    block_csr_matrix::block_csr_matrix(std::shared_ptr<sparsity_pattern> structure, int block_size)
    {
      reset(std::move(structure), block_size);
    }

    void block_csr_matrix::reset(std::shared_ptr<sparsity_pattern> structure, int block_size)
    {
      assert(structure != nullptr && "block_csr_matrix: null sparsity_pattern");
      assert(block_size >= 1 && "block_csr_matrix: block size must be >= 1");

      structure_ = std::move(structure);
      block_size_ = block_size;

      // Fresh, zero-initialised values buffer (nnzb * nb * nb).
      const std::size_t n = static_cast<std::size_t>(structure_->n_blocks())
        * static_cast<std::size_t>(block_size) * static_cast<std::size_t>(block_size);
      values_ = dual_array<mat_float>(n);
    }

    block_csr_matrix block_csr_matrix::clone() const
    {
      block_csr_matrix copy;
      copy.structure_ = structure_; // structure is immutable -> shared, not duplicated
      copy.block_size_ = block_size_;
      copy.values_ = values_.clone();
      return copy;
    }

    block_csr_matrix::index_t block_csr_matrix::n_block_rows() const noexcept
    {
      return structure_ ? structure_->n_block_rows() : 0;
    }

    block_csr_matrix::index_t block_csr_matrix::n_blocks() const noexcept
    {
      return structure_ ? structure_->n_blocks() : 0;
    }

    block_csr_matrix::index_t block_csr_matrix::n_rows() const noexcept
    {
      return n_block_rows() * block_size_;
    }

    block_csr_matrix::index_t block_csr_matrix::n_values() const noexcept
    {
      return n_blocks() * block_size_ * block_size_;
    }

    const block_csr_matrix::index_t *block_csr_matrix::row_ptr() const noexcept
    {
      return structure_->row_ptr();
    }

    const block_csr_matrix::index_t *block_csr_matrix::col_ind() const noexcept
    {
      return structure_->col_ind();
    }

    const block_csr_matrix::index_t *block_csr_matrix::diag_ind() const noexcept
    {
      return structure_->diag_ind();
    }

    void block_csr_matrix::set_zero()
    {
      mat_float *v = values_.host_data();
      std::fill(v, v + values_.size(), static_cast<mat_float>(0));
    }

#ifdef WITH_GPU
    const block_csr_matrix::index_t *block_csr_matrix::row_ptr_device() const
    {
      return structure_->row_ptr_device();
    }

    const block_csr_matrix::index_t *block_csr_matrix::col_ind_device() const
    {
      return structure_->col_ind_device();
    }

    const block_csr_matrix::index_t *block_csr_matrix::diag_ind_device() const
    {
      return structure_->diag_ind_device();
    }

    void block_csr_matrix::sync_to_device() const
    {
      structure_->sync_structure_to_device();
      values_.sync_to_device();
    }

    void block_csr_matrix::sync_to_host()
    {
      values_.sync_to_host();
    }
#endif
  } // namespace linear_solvers
} // namespace opendarts
