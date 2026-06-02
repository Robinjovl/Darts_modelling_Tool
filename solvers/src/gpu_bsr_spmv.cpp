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

#ifdef WITH_GPU

#include <cstdio>
#include <utility>

#include <cuda_runtime.h>

#include "gpu_bsr_spmv.hpp"

// CUDA 12+ deprecates the legacy block-CSR cuSPARSE SpMV (cusparseDbsrmv).
// It remains functional and has no drop-in generic-API replacement, so the
// deprecation diagnostic is silenced for this wrapper; migrating to the
// generic cuSPARSE API is tracked separately.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

namespace opendarts
{
  namespace linear_solvers
  {
    gpu_bsr_spmv::gpu_bsr_spmv(block_csr_matrix &matrix) : matrix_(&matrix)
    {
      if (cusparseCreate(&handle_) != CUSPARSE_STATUS_SUCCESS)
        printf("gpu_bsr_spmv: cuSPARSE handle creation failed\n");
      if (cusparseCreateMatDescr(&descr_) != CUSPARSE_STATUS_SUCCESS)
        printf("gpu_bsr_spmv: cuSPARSE matrix descriptor creation failed\n");
      else
      {
        cusparseSetMatType(descr_, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr_, CUSPARSE_INDEX_BASE_ZERO);
      }
    }

    gpu_bsr_spmv::~gpu_bsr_spmv()
    {
      free_scalar_csr_device();
      if (descr_ != nullptr)
        cusparseDestroyMatDescr(descr_);
      if (handle_ != nullptr)
        cusparseDestroy(handle_);
    }

    void gpu_bsr_spmv::free_scalar_csr_device() noexcept
    {
      if (scalar_csr_row_ptr_d_ != nullptr)
      {
        cudaFree(scalar_csr_row_ptr_d_);
        scalar_csr_row_ptr_d_ = nullptr;
      }
      if (scalar_csr_col_ind_d_ != nullptr)
      {
        cudaFree(scalar_csr_col_ind_d_);
        scalar_csr_col_ind_d_ = nullptr;
      }
      if (scalar_csr_val_d_ != nullptr)
      {
        cudaFree(scalar_csr_val_d_);
        scalar_csr_val_d_ = nullptr;
      }
      if (scalar_csr_descr_ != nullptr)
      {
        cusparseDestroyMatDescr(scalar_csr_descr_);
        scalar_csr_descr_ = nullptr;
      }
      scalar_csr_n_rows_ = 0;
      scalar_csr_nnz_ = 0;
      scalar_csr_block_size_ = 0;
    }

    gpu_bsr_spmv::gpu_bsr_spmv(gpu_bsr_spmv &&other) noexcept
      : matrix_(other.matrix_), handle_(other.handle_), descr_(other.descr_),
        scalar_csr_descr_(other.scalar_csr_descr_),
        scalar_csr_row_ptr_d_(other.scalar_csr_row_ptr_d_),
        scalar_csr_col_ind_d_(other.scalar_csr_col_ind_d_),
        scalar_csr_val_d_(other.scalar_csr_val_d_),
        scalar_csr_n_rows_(other.scalar_csr_n_rows_),
        scalar_csr_nnz_(other.scalar_csr_nnz_),
        scalar_csr_block_size_(other.scalar_csr_block_size_)
    {
      other.handle_ = nullptr;
      other.descr_ = nullptr;
      other.scalar_csr_descr_ = nullptr;
      other.scalar_csr_row_ptr_d_ = nullptr;
      other.scalar_csr_col_ind_d_ = nullptr;
      other.scalar_csr_val_d_ = nullptr;
      other.scalar_csr_n_rows_ = 0;
      other.scalar_csr_nnz_ = 0;
      other.scalar_csr_block_size_ = 0;
    }

    gpu_bsr_spmv &gpu_bsr_spmv::operator=(gpu_bsr_spmv &&other) noexcept
    {
      if (this != &other)
      {
        free_scalar_csr_device();
        if (descr_ != nullptr)
          cusparseDestroyMatDescr(descr_);
        if (handle_ != nullptr)
          cusparseDestroy(handle_);
        matrix_ = other.matrix_;
        handle_ = other.handle_;
        descr_ = other.descr_;
        scalar_csr_descr_ = other.scalar_csr_descr_;
        scalar_csr_row_ptr_d_ = other.scalar_csr_row_ptr_d_;
        scalar_csr_col_ind_d_ = other.scalar_csr_col_ind_d_;
        scalar_csr_val_d_ = other.scalar_csr_val_d_;
        scalar_csr_n_rows_ = other.scalar_csr_n_rows_;
        scalar_csr_nnz_ = other.scalar_csr_nnz_;
        scalar_csr_block_size_ = other.scalar_csr_block_size_;
        other.handle_ = nullptr;
        other.descr_ = nullptr;
        other.scalar_csr_descr_ = nullptr;
        other.scalar_csr_row_ptr_d_ = nullptr;
        other.scalar_csr_col_ind_d_ = nullptr;
        other.scalar_csr_val_d_ = nullptr;
        other.scalar_csr_n_rows_ = 0;
        other.scalar_csr_nnz_ = 0;
        other.scalar_csr_block_size_ = 0;
      }
      return *this;
    }

    // Build (or refresh) the scalar-CSR device mirror of the bound block
    // matrix via cusparseDbsr2csr. The structure is shape-stable across
    // Newton iterations, so the buffers are allocated on the first call and
    // reused on subsequent calls -- only the bsr2csr output is rewritten.
    int gpu_bsr_spmv::build_scalar_csr_device()
    {
      const int mb = static_cast<int>(matrix_->n_block_rows());
      const int nnzb = static_cast<int>(matrix_->n_blocks());
      const int bs = matrix_->block_size();
      const index_t scalar_rows = static_cast<index_t>(mb) * bs;
      const index_t scalar_nnz = static_cast<index_t>(nnzb) * bs * bs;

      // (Re)allocate if the shape changed; idempotent on stable sparsity.
      if (scalar_csr_n_rows_ != scalar_rows ||
          scalar_csr_nnz_ != scalar_nnz ||
          scalar_csr_block_size_ != bs)
      {
        free_scalar_csr_device();
        if (cudaMalloc(reinterpret_cast<void **>(&scalar_csr_row_ptr_d_),
              sizeof(index_t) * (scalar_rows + 1)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&scalar_csr_col_ind_d_),
              sizeof(index_t) * scalar_nnz) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&scalar_csr_val_d_),
              sizeof(double) * scalar_nnz) != cudaSuccess)
        {
          printf("gpu_bsr_spmv: scalar-CSR device allocation failed\n");
          free_scalar_csr_device();
          return 1;
        }
        if (cusparseCreateMatDescr(&scalar_csr_descr_) != CUSPARSE_STATUS_SUCCESS)
        {
          printf("gpu_bsr_spmv: scalar-CSR descriptor creation failed\n");
          free_scalar_csr_device();
          return 1;
        }
        cusparseSetMatType(scalar_csr_descr_, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(scalar_csr_descr_, CUSPARSE_INDEX_BASE_ZERO);
        scalar_csr_n_rows_ = scalar_rows;
        scalar_csr_nnz_ = scalar_nnz;
        scalar_csr_block_size_ = bs;
      }

      // bsr2csr writes both structure and values; for fixed sparsity the
      // structure portion is stable across calls.
      const cusparseStatus_t status = cusparseDbsr2csr(handle_, CUSPARSE_DIRECTION_ROW,
        mb, mb, descr_, matrix_->values_device(), matrix_->row_ptr_device(),
        matrix_->col_ind_device(), bs, scalar_csr_descr_, scalar_csr_val_d_,
        scalar_csr_row_ptr_d_, scalar_csr_col_ind_d_);
      if (status != CUSPARSE_STATUS_SUCCESS)
      {
        printf("gpu_bsr_spmv: cusparseDbsr2csr failed (status %d)\n",
          static_cast<int>(status));
        return 1;
      }
      return 0;
    }

    // y_d = alpha * A * x_d + beta * y_d, A held in block-CSR on the device.
    int gpu_bsr_spmv::bsrmv(double alpha, const double *x_d, double beta, double *y_d) const
    {
      const int mb = static_cast<int>(matrix_->n_block_rows());
      const int nnzb = static_cast<int>(matrix_->n_blocks());
      const cusparseStatus_t status = cusparseDbsrmv(handle_, CUSPARSE_DIRECTION_ROW,
        CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &alpha, descr_,
        matrix_->values_device(), matrix_->row_ptr_device(), matrix_->col_ind_device(),
        matrix_->block_size(), x_d, &beta, y_d);
      if (status != CUSPARSE_STATUS_SUCCESS)
      {
        printf("gpu_bsr_spmv: cusparseDbsrmv failed (status %d)\n", static_cast<int>(status));
        return 1;
      }
      return 0;
    }

    int gpu_bsr_spmv::matrix_vector_product_d(const double *v_d, double *r_d) const
    {
      return bsrmv(1.0, v_d, 1.0, r_d); // r += A * v
    }

    int gpu_bsr_spmv::matrix_vector_product_d0(const double *v_d, double *r_d) const
    {
      return bsrmv(1.0, v_d, 0.0, r_d); // r = A * v
    }

    int gpu_bsr_spmv::calc_lin_comb_d(double alpha, double beta, const double *u_d,
      const double *v_d, double *r_d) const
    {
      // r = beta * v, then r = alpha * A * u + beta * r  ->  alpha*A*u + beta*v.
      const std::size_t bytes = static_cast<std::size_t>(matrix_->scalar_n_rows()) * sizeof(double);
      if (r_d != v_d)
        cudaMemcpy(r_d, v_d, bytes, cudaMemcpyDeviceToDevice);
      return bsrmv(alpha, u_d, beta, r_d);
    }
  } // namespace linear_solvers
} // namespace opendarts

#pragma GCC diagnostic pop

#endif // WITH_GPU
