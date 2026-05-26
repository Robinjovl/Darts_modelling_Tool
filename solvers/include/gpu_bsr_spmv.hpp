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
#ifndef OPENDARTS_LINEAR_SOLVERS_GPU_BSR_SPMV_HPP
#define OPENDARTS_LINEAR_SOLVERS_GPU_BSR_SPMV_HPP
//--------------------------------------------------------------------------

// gpu_bsr_spmv -- cuSPARSE block-CSR sparse matrix-vector products over a
// block_csr_matrix (see SOLVER_REFACTORING_PLAN.md section 12.10, phase A3).
//
// It is the device-SpMV adapter the GPU solver wrappers use; it replaces the
// per-matrix device layer that was bolted onto the legacy csr_matrix<N>. The
// block_csr_matrix already owns the device storage (dual_array); this adapter
// adds only the cuSPARSE handle and descriptor and the BSR SpMV calls.

#ifdef WITH_GPU

#include <cusparse.h>

#include "block_csr_matrix.hpp"
#include "data_types.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** @brief cuSPARSE block-CSR SpMV adapter over a block_csr_matrix.

        Owns a cuSPARSE handle and matrix descriptor. The matrix's device
        mirror must be current -- call sync() (or block_csr_matrix::
        sync_to_device()) after each re-assembly. Move-only. */
    class gpu_bsr_spmv
    {
    public:
      using index_t = opendarts::config::index_t;

      /** Binds to @p matrix and creates the cuSPARSE handle / descriptor.
          The matrix must outlive the adapter. */
      explicit gpu_bsr_spmv(block_csr_matrix &matrix);

      ~gpu_bsr_spmv();

      gpu_bsr_spmv(gpu_bsr_spmv &&) noexcept;
      gpu_bsr_spmv &operator=(gpu_bsr_spmv &&) noexcept;
      gpu_bsr_spmv(const gpu_bsr_spmv &) = delete;
      gpu_bsr_spmv &operator=(const gpu_bsr_spmv &) = delete;

      /** Mirrors the bound matrix (structure + values) to the device. */
      void sync() const { matrix_->sync_to_device(); }

      /** r_d += A * v_d (all device pointers). Returns 0 on success. */
      int matrix_vector_product_d(const double *v_d, double *r_d) const;

      /** r_d = A * v_d (result overwritten). Returns 0 on success. */
      int matrix_vector_product_d0(const double *v_d, double *r_d) const;

      /** r_d = alpha * A * u_d + beta * v_d. Returns 0 on success. */
      int calc_lin_comb_d(double alpha, double beta, const double *u_d,
        const double *v_d, double *r_d) const;

      [[nodiscard]] cusparseHandle_t handle() const noexcept { return handle_; }

    private:
      int bsrmv(double alpha, const double *x_d, double beta, double *y_d) const;

      block_csr_matrix *matrix_;
      cusparseHandle_t handle_ = nullptr;
      cusparseMatDescr_t descr_ = nullptr;
    };
  } // namespace linear_solvers
} // namespace opendarts

#endif // WITH_GPU

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_GPU_BSR_SPMV_HPP
//--------------------------------------------------------------------------
