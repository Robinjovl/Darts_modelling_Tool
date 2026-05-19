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
      if (descr_ != nullptr)
        cusparseDestroyMatDescr(descr_);
      if (handle_ != nullptr)
        cusparseDestroy(handle_);
    }

    gpu_bsr_spmv::gpu_bsr_spmv(gpu_bsr_spmv &&other) noexcept
      : matrix_(other.matrix_), handle_(other.handle_), descr_(other.descr_)
    {
      other.handle_ = nullptr;
      other.descr_ = nullptr;
    }

    gpu_bsr_spmv &gpu_bsr_spmv::operator=(gpu_bsr_spmv &&other) noexcept
    {
      if (this != &other)
      {
        if (descr_ != nullptr)
          cusparseDestroyMatDescr(descr_);
        if (handle_ != nullptr)
          cusparseDestroy(handle_);
        matrix_ = other.matrix_;
        handle_ = other.handle_;
        descr_ = other.descr_;
        other.handle_ = nullptr;
        other.descr_ = nullptr;
      }
      return *this;
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
      const std::size_t bytes = static_cast<std::size_t>(matrix_->n_rows()) * sizeof(double);
      if (r_d != v_d)
        cudaMemcpy(r_d, v_d, bytes, cudaMemcpyDeviceToDevice);
      return bsrmv(alpha, u_d, beta, r_d);
    }
  } // namespace linear_solvers
} // namespace opendarts

#endif // WITH_GPU
