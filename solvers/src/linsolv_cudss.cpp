//*************************************************************************
//    Copyright (c) 2026
//    Delft University of Technology, the Netherlands
//
//    This file is part of the open Delft Advanced Research Terra Simulator
//    (open-DARTS). It is distributed under the Apache License.
//*************************************************************************

#if defined(WITH_GPU) && defined(WITH_CUDSS)

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cudss.h>

#include "linsolv_cudss.hpp"
#include "block_csr_matrix.hpp"
#include "csr_matrix.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    namespace
    {
      // Pull the scalar-CSR device triple out of either matrix subclass.
      // For block_csr_matrix this requires that build_scalar_csr_device()
      // was called this Newton iteration (init/setup do that). Identical to
      // the dispatch in linsolv_cusolv.cpp -- cuDSS consumes the same
      // scalar-CSR device view the cuSOLVER QR wrapper does.
      template <uint8_t N_BLOCK_SIZE>
      bool fetch_scalar_csr_device(opendarts::linear_solvers::csr_matrix_base *A,
        opendarts::config::index_t &n_scalar_rows,
        opendarts::config::index_t &n_scalar_nnz,
        const double *&vals_d,
        const opendarts::config::index_t *&row_ptr_d,
        const opendarts::config::index_t *&col_ind_d)
      {
        if (auto *A_typed = dynamic_cast<opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *>(A))
        {
          n_scalar_rows = A_typed->n_rows * N_BLOCK_SIZE;
          n_scalar_nnz = A_typed->get_n_non_zeros() * N_BLOCK_SIZE * N_BLOCK_SIZE;
          if (N_BLOCK_SIZE > 1)
          {
            vals_d = A_typed->csrValC;
            row_ptr_d = A_typed->csrRowPtrC;
            col_ind_d = A_typed->csrColIndC;
          }
          else
          {
            vals_d = A_typed->values_d;
            row_ptr_d = A_typed->rows_ptr_d;
            col_ind_d = A_typed->cols_ind_d;
          }
          return true;
        }
        if (auto *A_block = dynamic_cast<opendarts::linear_solvers::block_csr_matrix *>(A))
        {
          n_scalar_rows = A_block->n_rows * A_block->block_size();
          n_scalar_nnz = A_block->scalar_csr_nnz();
          vals_d = A_block->scalar_csr_values_device();
          row_ptr_d = A_block->scalar_csr_row_ptr_device();
          col_ind_d = A_block->scalar_csr_col_ind_device();
          return true;
        }
        return false;
      }

      inline bool check_cudss(cudssStatus_t st, const char *where)
      {
        if (st != CUDSS_STATUS_SUCCESS)
        {
          fprintf(stderr, "linsolv_cudss: %s failed with status %d\n", where,
              static_cast<int>(st));
          return false;
        }
        return true;
      }
    } // namespace

    template <uint8_t N_BLOCK_SIZE>
    linsolv_cudss<N_BLOCK_SIZE>::linsolv_cudss()
    {
      opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::solver = this;
    }

    template <uint8_t N_BLOCK_SIZE>
    linsolv_cudss<N_BLOCK_SIZE>::~linsolv_cudss()
    {
      if (A_cudss_)
        cudssMatrixDestroy(A_cudss_);
      if (b_cudss_)
        cudssMatrixDestroy(b_cudss_);
      if (x_cudss_)
        cudssMatrixDestroy(x_cudss_);
      if (data_)
        cudssDataDestroy(handle_, data_);
      if (config_)
        cudssConfigDestroy(config_);
      if (handle_)
        cudssDestroy(handle_);

      cudaFree(d_B);
      cudaFree(d_X);
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cudss<N_BLOCK_SIZE>::bind_matrix_pointers()
    {
      const double *vals_d = nullptr;
      const opendarts::config::index_t *row_ptr_d = nullptr;
      const opendarts::config::index_t *col_ind_d = nullptr;
      if (!fetch_scalar_csr_device<N_BLOCK_SIZE>(A_matrix, n_scalar_rows,
            n_scalar_nnz, vals_d, row_ptr_d, col_ind_d))
      {
        fprintf(stderr, "linsolv_cudss: scalar-CSR device view unavailable\n");
        return -1;
      }

      if (!A_cudss_)
      {
        // rowEnd = nullptr selects the standard 3-array CSR convention
        // (rowEnd[i] == rowStart[i+1]).
        if (!check_cudss(cudssMatrixCreateCsr(&A_cudss_, n_scalar_rows,
                n_scalar_rows, n_scalar_nnz,
                const_cast<opendarts::config::index_t *>(row_ptr_d), nullptr,
                const_cast<opendarts::config::index_t *>(col_ind_d),
                const_cast<double *>(vals_d),
                CUDSS_R_32I /*offsetType*/, CUDSS_R_32I /*indexType*/,
                CUDSS_R_64F /*valueType*/,
                CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO),
              "cudssMatrixCreateCsr"))
          return -1;
      }
      else
      {
        // Structure is unchanged across Newton iterations, but the device
        // buffers can be reallocated by the matrix backend -- rebind all
        // three pointers (cheap descriptor update, no copy).
        if (!check_cudss(cudssMatrixSetCsrPointers(A_cudss_,
                const_cast<opendarts::config::index_t *>(row_ptr_d), nullptr,
                const_cast<opendarts::config::index_t *>(col_ind_d),
                const_cast<double *>(vals_d)),
              "cudssMatrixSetCsrPointers"))
          return -1;
      }
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cudss<N_BLOCK_SIZE>::init(opendarts::linear_solvers::csr_matrix_base *A_input,
      opendarts::config::index_t /*max_iters*/,
      opendarts::config::mat_float /*tolerance*/)
    {
      A_matrix = A_input;

      // Mirror the matrix on the device and build the scalar-CSR view cuDSS
      // consumes (same dispatch as linsolv_cusolv).
      if (auto *A_typed = dynamic_cast<opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *>(A_input))
      {
        A_typed->init_device(A_typed->n_rows, A_typed->n_non_zeros);
        A_typed->copy_struct_to_device();
        if (N_BLOCK_SIZE > 1)
          A_typed->convert_to_ELL();
      }
      else if (auto *A_block = dynamic_cast<opendarts::linear_solvers::block_csr_matrix *>(A_input))
      {
        if (A_block->build_scalar_csr_device() != 0)
          return -1;
      }
      else
      {
        fprintf(stderr, "linsolv_cudss: unsupported csr_matrix_base subclass\n");
        return -1;
      }

      if (!check_cudss(cudssCreate(&handle_), "cudssCreate"))
        return -1;
      if (!check_cudss(cudssConfigCreate(&config_), "cudssConfigCreate"))
        return -1;
      if (!check_cudss(cudssDataCreate(handle_, &data_), "cudssDataCreate"))
        return -1;

      if (bind_matrix_pointers() != 0)
        return -1;

      // Dense RHS / solution vectors (device-resident, n x 1).
      cudaError_t cudaStat;
      cudaStat = cudaMalloc((void **)&d_B, sizeof(opendarts::config::mat_float) * n_scalar_rows);
      if (cudaStat != cudaSuccess)
      {
        fprintf(stderr, "linsolv_cudss: can't allocate device RHS\n");
        return -1;
      }
      cudaStat = cudaMalloc((void **)&d_X, sizeof(opendarts::config::mat_float) * n_scalar_rows);
      if (cudaStat != cudaSuccess)
      {
        fprintf(stderr, "linsolv_cudss: can't allocate device solution\n");
        return -1;
      }
      if (!check_cudss(cudssMatrixCreateDn(&b_cudss_, n_scalar_rows, 1,
              n_scalar_rows, d_B, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR),
            "cudssMatrixCreateDn(b)"))
        return -1;
      if (!check_cudss(cudssMatrixCreateDn(&x_cudss_, n_scalar_rows, 1,
              n_scalar_rows, d_X, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR),
            "cudssMatrixCreateDn(x)"))
        return -1;

      // Reordering + symbolic factorization: structure-only, once per init.
      if (!check_cudss(cudssExecute(handle_, CUDSS_PHASE_ANALYSIS, config_,
              data_, A_cudss_, x_cudss_, b_cudss_),
            "cudssExecute(ANALYSIS)"))
        return -1;
      analyzed_ = true;

      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cudss<N_BLOCK_SIZE>::setup(opendarts::linear_solvers::csr_matrix_base *A_input)
    {
      if (this->timer_setup)
        this->timer_setup->node["CUDSS"].start();

      A_matrix = A_input;

      // Refresh the scalar-CSR device view from the (re-assembled) matrix.
      if (auto *A_typed = dynamic_cast<opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *>(A_input))
      {
        if (N_BLOCK_SIZE > 1)
          A_typed->convert_to_ELL();
      }
      else if (auto *A_block = dynamic_cast<opendarts::linear_solvers::block_csr_matrix *>(A_input))
      {
        if (A_block->build_scalar_csr_device() != 0)
        {
          if (this->timer_setup)
            this->timer_setup->node["CUDSS"].stop();
          return -1;
        }
      }
      else
      {
        fprintf(stderr, "linsolv_cudss: unsupported csr_matrix_base subclass\n");
        if (this->timer_setup)
          this->timer_setup->node["CUDSS"].stop();
        return -1;
      }

      int rc = 0;
      if (!analyzed_ || bind_matrix_pointers() != 0)
        rc = -1;

      // Numeric factorization on the refreshed values.
      if (rc == 0
          && !check_cudss(cudssExecute(handle_, CUDSS_PHASE_FACTORIZATION,
                config_, data_, A_cudss_, x_cudss_, b_cudss_),
              "cudssExecute(FACTORIZATION)"))
        rc = -1;

      // Propagate a singular/failed factorization as a setup failure so the
      // Newton loop cuts the timestep (engine checks the nonzero return).
      if (rc == 0)
      {
        int info = 0;
        std::size_t written = 0;
        if (cudssDataGet(handle_, data_, CUDSS_DATA_INFO, &info, sizeof(info),
                &written) == CUDSS_STATUS_SUCCESS
            && info != 0)
        {
          fprintf(stderr, "linsolv_cudss: factorization info = %d\n", info);
          rc = -1;
        }
      }

      if (this->timer_setup)
        this->timer_setup->node["CUDSS"].stop();
      return rc;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cudss<N_BLOCK_SIZE>::solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X)
    {
      if (this->timer_solve)
        this->timer_solve->node["CUDSS"].start();

      int rc = 0;
      cudaError_t cudaStat = cudaMemcpy(d_B, B,
          sizeof(opendarts::config::mat_float) * n_scalar_rows,
          cudaMemcpyHostToDevice);
      if (cudaStat != cudaSuccess)
      {
        fprintf(stderr, "linsolv_cudss: can't copy RHS to device\n");
        rc = -1;
      }

      if (rc == 0
          && !check_cudss(cudssExecute(handle_, CUDSS_PHASE_SOLVE, config_,
                data_, A_cudss_, x_cudss_, b_cudss_),
              "cudssExecute(SOLVE)"))
        rc = -1;

      if (rc == 0)
      {
        cudaDeviceSynchronize();
        cudaStat = cudaMemcpy(X, d_X,
            sizeof(opendarts::config::mat_float) * n_scalar_rows,
            cudaMemcpyDeviceToHost);
        if (cudaStat != cudaSuccess)
        {
          fprintf(stderr, "linsolv_cudss: can't copy solution to host\n");
          rc = -1;
        }
      }

      if (this->timer_solve)
        this->timer_solve->node["CUDSS"].stop();
      return rc;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cudss<N_BLOCK_SIZE>::get_n_iters()
    {
      return 1; // direct solve
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::mat_float linsolv_cudss<N_BLOCK_SIZE>::get_residual()
    {
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cudss<N_BLOCK_SIZE>::set_prec(opendarts::linear_solvers::linsolv_iface * /*prec_input*/)
    {
      return 0; // direct solver: no preconditioner
    }

    // Explicit instantiation for the supported block sizes.
    template class linsolv_cudss<1>;
    template class linsolv_cudss<2>;
    template class linsolv_cudss<3>;
    template class linsolv_cudss<4>;
    template class linsolv_cudss<5>;
    template class linsolv_cudss<6>;
    template class linsolv_cudss<7>;
    template class linsolv_cudss<8>;
    template class linsolv_cudss<9>;
    template class linsolv_cudss<10>;
    template class linsolv_cudss<11>;
    template class linsolv_cudss<12>;
    template class linsolv_cudss<13>;

  } // namespace linear_solvers
} // namespace opendarts

#endif // WITH_GPU && WITH_CUDSS
