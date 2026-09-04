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
#include "block_csr_io.hpp"

#ifdef WITH_GPU
#include "gpu_bsr_spmv.hpp"
#endif

namespace opendarts
{
  namespace linear_solvers
  {
    block_csr_matrix::block_csr_matrix() noexcept { refresh_base_fields(); }

    block_csr_matrix::block_csr_matrix(std::shared_ptr<sparsity_pattern> structure, int block_size)
    {
      reset(std::move(structure), block_size);
    }

    // Move-only. gpu_spmv_ is deliberately NOT transferred: the adapter holds a
    // back-pointer to the matrix, so it is dropped and lazily recreated against
    // the moved-to object.
    block_csr_matrix::block_csr_matrix(block_csr_matrix &&other) noexcept
      : csr_matrix_base(), structure_(std::move(other.structure_)),
        values_(std::move(other.values_)), block_size_(other.block_size_)
    {
      other.block_size_ = 0;
      refresh_base_fields();
    }

    block_csr_matrix &block_csr_matrix::operator=(block_csr_matrix &&other) noexcept
    {
      if (this != &other)
      {
        structure_ = std::move(other.structure_);
        values_ = std::move(other.values_);
        block_size_ = other.block_size_;
        other.block_size_ = 0;
#ifdef WITH_GPU
        gpu_spmv_.reset();
#endif
        refresh_base_fields();
      }
      return *this;
    }

    block_csr_matrix::~block_csr_matrix() = default;

    void block_csr_matrix::reset(std::shared_ptr<sparsity_pattern> structure, int block_size)
    {
      assert(structure != nullptr && "block_csr_matrix: null sparsity_pattern");
      assert(block_size >= 1 && "block_csr_matrix: block size must be >= 1");

      structure_ = std::move(structure);
      block_size_ = block_size;
#ifdef WITH_GPU
      gpu_spmv_.reset(); // stale once the structure / values change
#endif

      // Fresh, zero-initialised values buffer (nnzb * nb * nb).
      const std::size_t n = static_cast<std::size_t>(structure_->n_blocks())
        * static_cast<std::size_t>(block_size) * static_cast<std::size_t>(block_size);
      values_ = dual_array<mat_float>(n);

      refresh_base_fields();
    }

    void block_csr_matrix::init(index_t n_block_rows, index_t n_block_cols, int block_size,
      index_t nnzb)
    {
      auto sp = std::make_shared<sparsity_pattern>();
      sp->allocate(n_block_rows, n_block_cols, nnzb);
      reset(std::move(sp), block_size);
    }

    block_csr_matrix block_csr_matrix::clone() const
    {
      block_csr_matrix copy;
      copy.structure_ = structure_; // structure is immutable -> shared, not duplicated
      copy.block_size_ = block_size_;
      copy.values_ = values_.clone();
      copy.refresh_base_fields();
      return copy;
    }

    void block_csr_matrix::refresh_base_fields() noexcept
    {
      // Keep the csr_matrix_base data members consistent with the structure so
      // legacy consumers that read ->n_rows etc. directly still see the truth.
      this->n_rows = structure_ ? structure_->n_block_rows() : 0;
      this->n_cols = structure_ ? structure_->n_block_cols() : 0;
      this->n_non_zeros = structure_ ? structure_->n_blocks() : 0;
      this->n_row_size = block_size_;
      this->type = opendarts::linear_solvers::MATRIX_TYPE_CSR;
      this->is_square = (this->n_rows == this->n_cols) ? 1 : 0;
    }

    block_csr_matrix::index_t block_csr_matrix::n_block_rows() const noexcept
    {
      return structure_ ? structure_->n_block_rows() : 0;
    }

    block_csr_matrix::index_t block_csr_matrix::n_blocks() const noexcept
    {
      return structure_ ? structure_->n_blocks() : 0;
    }

    block_csr_matrix::index_t block_csr_matrix::scalar_n_rows() const noexcept
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

    // --- csr_matrix_base interface -------------------------------------------
    block_csr_matrix::index_t *block_csr_matrix::get_rows_ptr()
    {
      return const_cast<index_t *>(structure_->row_ptr());
    }

    block_csr_matrix::index_t *block_csr_matrix::get_cols_ind()
    {
      return const_cast<index_t *>(structure_->col_ind());
    }

    block_csr_matrix::index_t *block_csr_matrix::get_diag_ind()
    {
      return const_cast<index_t *>(structure_->diag_ind());
    }

    block_csr_matrix::index_t *block_csr_matrix::get_row_thread_starts()
    {
      // The engines fetch the partition once, right before opening their
      // OpenMP assembly region; re-derive it here when the team size changed
      // since the Jacobian was built (a set_num_threads() call after model
      // init previously left tail rows unassembled or read out of bounds).
      structure_->ensure_row_partition_current();
      return const_cast<index_t *>(structure_->row_thread_starts());
    }

    int block_csr_matrix::export_matrix_to_file(const std::string &filename,
      opendarts::linear_solvers::sparse_matrix_export_format /*export_format*/)
    {
      return write_block_csr_matrix(*this, filename);
    }

    int block_csr_matrix::import_matrix_from_file(const std::string & /*filename*/,
      opendarts::linear_solvers::sparse_matrix_import_format /*import_format*/)
    {
      // The unified matrix is built from a sparsity_pattern, not imported.
      return 1;
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

    void block_csr_matrix::sync_to_host() { values_.sync_to_host(); }

    int block_csr_matrix::matrix_vector_product_d(const double *v, double *r)
    {
      if (!gpu_spmv_)
        gpu_spmv_ = std::make_unique<gpu_bsr_spmv>(*this);
      return gpu_spmv_->matrix_vector_product_d(v, r);
    }

    int block_csr_matrix::matrix_vector_product_d0(const double *v, double *r)
    {
      if (!gpu_spmv_)
        gpu_spmv_ = std::make_unique<gpu_bsr_spmv>(*this);
      return gpu_spmv_->matrix_vector_product_d0(v, r);
    }

    int block_csr_matrix::matrix_vector_product_d_ell(const double *v, double *r)
    {
      // The cuSPARSE HYB/ELL path was removed in CUDA 11; fall back to block SpMV.
      return matrix_vector_product_d(v, r);
    }

    int block_csr_matrix::calc_lin_comb_d(const double alpha, const double beta,
      double *u, double *v, double *r)
    {
      if (!gpu_spmv_)
        gpu_spmv_ = std::make_unique<gpu_bsr_spmv>(*this);
      return gpu_spmv_->calc_lin_comb_d(alpha, beta, u, v, r);
    }

    int block_csr_matrix::matrix_vector_product_t_d0(const double *v, double *r)
    {
      if (!gpu_spmv_)
        gpu_spmv_ = std::make_unique<gpu_bsr_spmv>(*this);
      return gpu_spmv_->matrix_vector_product_t_d0(v, r);
    }

    int block_csr_matrix::calc_lin_comb_t_d(const double alpha, const double beta,
      double *u, double *v, double *r)
    {
      if (!gpu_spmv_)
        gpu_spmv_ = std::make_unique<gpu_bsr_spmv>(*this);
      return gpu_spmv_->calc_lin_comb_t_d(alpha, beta, u, v, r);
    }

    int block_csr_matrix::refresh_transpose_spmv_d()
    {
      // The transposed products run over the adapter's scalar-CSR view;
      // rebuilding it (values-only for fixed sparsity) re-anchors them to the
      // current matrix values.
      return build_scalar_csr_device();
    }

    int block_csr_matrix::copy_struct_to_device()
    {
      // Explicit copy request: the caller has just filled or edited the host
      // structure through get_rows_ptr()/get_cols_ind()/get_diag_ind(), whose
      // const_cast route cannot set the dual_array dirty flag. Mark it dirty
      // here so the upload is unconditional rather than relying on the
      // first-time !device_populated_ path.
      structure_->mark_structure_modified();
      structure_->sync_structure_to_device();
      return 0;
    }

    int block_csr_matrix::copy_values_to_device()
    {
      values_.sync_to_device();
      return 0;
    }

    int block_csr_matrix::build_scalar_csr_device()
    {
      if (!gpu_spmv_)
        gpu_spmv_ = std::make_unique<gpu_bsr_spmv>(*this);
      return gpu_spmv_->build_scalar_csr_device();
    }

    block_csr_matrix::index_t block_csr_matrix::scalar_csr_nnz() const
    {
      return gpu_spmv_ ? gpu_spmv_->scalar_csr_nnz() : 0;
    }

    const block_csr_matrix::index_t *block_csr_matrix::scalar_csr_row_ptr_device() const
    {
      return gpu_spmv_ ? gpu_spmv_->scalar_csr_row_ptr_device() : nullptr;
    }

    const block_csr_matrix::index_t *block_csr_matrix::scalar_csr_col_ind_device() const
    {
      return gpu_spmv_ ? gpu_spmv_->scalar_csr_col_ind_device() : nullptr;
    }

    const block_csr_matrix::mat_float *block_csr_matrix::scalar_csr_values_device() const
    {
      return gpu_spmv_ ? gpu_spmv_->scalar_csr_values_device() : nullptr;
    }
#endif // WITH_GPU
  } // namespace linear_solvers
} // namespace opendarts
