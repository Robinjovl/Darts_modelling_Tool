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

#include <iostream>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <cmath>

#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_utilities.h"
#include "_hypre_parcsr_mv.h"

#include "data_types.hpp"
#include "csr_matrix.hpp"
#include "linsolv_iface.hpp"
#include "linsolv_hypre_ilu.hpp"

extern "C" {
HYPRE_Int HYPRE_Initialize(void);
HYPRE_Int HYPRE_Initialized(void);
}

namespace opendarts
{
  namespace linear_solvers
  {
    namespace
    {
      void check_result(int res);
      void check_result_nothrow(int res);
    }

    template <uint8_t N_BLOCK_SIZE>
    linsolv_hypre_ilu<N_BLOCK_SIZE>::linsolv_hypre_ilu()
    {
      // Initialize preconditioner (not used in this case, kept for compatibility)
      this->prec = 0;  // no preconditioner for this solver

      // Null-init all HYPRE handles so the destructor can safely skip Destroy
      // calls when init()/setup() were never reached (or were partially
      // executed). Together with the null-check in ~linsolv_hypre_ilu this
      // also makes destruction idempotent across pybind11 / interpreter
      // shutdown orderings.
      this->solver = nullptr;
      this->A_ij = nullptr;
      this->A_parcsr = nullptr;
      this->b_ij = nullptr;
      this->b_par = nullptr;
      this->x_ij = nullptr;
      this->x_par = nullptr;
    }

    template <uint8_t N_BLOCK_SIZE>
    linsolv_hypre_ilu<N_BLOCK_SIZE>::~linsolv_hypre_ilu()
    {
      // Null-safe destruction: pybind11 may destroy this wrapper at
      // interpreter shutdown after HYPRE-side state has already been torn
      // down (or after sibling Destroy calls), so calling HYPRE_*Destroy on
      // a stale / null handle is a use-after-free. Check each handle, Destroy
      // only if non-null, then null it to make any subsequent double-destroy
      // a no-op as well.
      if (this->solver != nullptr)
      {
        check_result_nothrow(HYPRE_ILUDestroy(this->solver));
        this->solver = nullptr;
      }
      if (this->A_ij != nullptr)
      {
        check_result_nothrow(HYPRE_IJMatrixDestroy(this->A_ij));
        this->A_ij = nullptr;
      }
      // check_result_nothrow(HYPRE_ParCSRMatrixDestroy(this->A_parcsr));  // gives error
      if (this->b_ij != nullptr)
      {
        check_result_nothrow(HYPRE_IJVectorDestroy(this->b_ij));
        this->b_ij = nullptr;
      }
      // check_result_nothrow(HYPRE_ParVectorDestroy(this->b_par));  // gives error
      if (this->x_ij != nullptr)
      {
        check_result_nothrow(HYPRE_IJVectorDestroy(this->x_ij));
        this->x_ij = nullptr;
      }
      // check_result_nothrow(HYPRE_ParVectorDestroy(this->x_par));  // gives error
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_hypre_ilu<N_BLOCK_SIZE>::set_prec(opendarts::linear_solvers::linsolv_iface *prec_in)
    {
      (void) prec_in;

      std::cout << "NOT IMPLEMENTED: linsolv_hypre_ilu::linsolv_hypre_ilu" << std::endl;

      return 1;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_hypre_ilu<N_BLOCK_SIZE>::init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_in,
      opendarts::config::index_t max_iters,
      opendarts::config::mat_float tolerance)
    {
      // max_iters: set to 1 if ILU is used as preconditioner
      // tolerance: set to 0.0 if ILU is used as preconditioner

      // Setup Hypre solver -- using HYPRE-ILU's local-only sequential mode.
      const int print_level = 0;  // 0 = quiet (was 2 = HYPRE diagnostics)

      // Ensure HYPRE is initialised before any HYPRE_* call. The engine calls
      // init() before setup(), so the setup() guard is not sufficient on its
      // own when this wrapper is used as a sub-prec (e.g. inside FS-CPR).
      if (!HYPRE_Initialized())
        HYPRE_Initialize();
      HYPRE_ClearAllErrors();

      check_result(HYPRE_ILUCreate(&(this->solver)));
    	check_result(HYPRE_ILUSetPrintLevel(this->solver, print_level));
    	check_result(HYPRE_ILUSetLogging(this->solver, print_level));

      check_result(HYPRE_ILUSetMaxIter(this->solver, max_iters));
      check_result(HYPRE_ILUSetTol(this->solver, tolerance));

      (void) A_in; // not used, kept to keep same interface

      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_hypre_ilu<N_BLOCK_SIZE>::setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_in)
    {
      // Ensure HYPRE is initialised; when this wrapper is used as a sub-prec
      // (e.g. inside FS-CPR) without a prior CPR call, HYPRE_Initialize would
      // otherwise never have been called and HYPRE_ILUSetup would fail with
      // HYPRE "[Generic error]".
      if (!HYPRE_Initialized())
        HYPRE_Initialize();
      HYPRE_ClearAllErrors();

      // linsolv_iface::timer_setup->node["AMG"].start();

      try
      {
      // Re-entrant setup: destroy the previous call's handles first (they
      // were unconditionally re-created, leaking one IJ matrix and two IJ
      // vectors per Newton iteration).
      if (this->b_ij != nullptr)
      {
        check_result_nothrow(HYPRE_IJVectorDestroy(this->b_ij));
        this->b_ij = nullptr;
        this->b_par = nullptr;
      }
      if (this->x_ij != nullptr)
      {
        check_result_nothrow(HYPRE_IJVectorDestroy(this->x_ij));
        this->x_ij = nullptr;
        this->x_par = nullptr;
      }
      if (this->A_ij != nullptr)
      {
        check_result_nothrow(HYPRE_IJMatrixDestroy(this->A_ij));
        this->A_ij = nullptr;
        this->A_parcsr = nullptr;
      }

      // Store input system matrix
      this->A = A_in;

      // Setup right hand side and solution vectors
      const int print_level = 0;  // 0 = quiet (was 2 = HYPRE diagnostics) in Hypre
      opendarts::config::index_t n_rows = this->A->n_cols;;  // number of rows in vector must
                                                             // be the same as number of columns
                                                             // of system matrix

      opendarts::config::index_t ilower, iupper;
      ilower = 0;
      iupper = n_rows - 1;

      check_result(HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, ilower, iupper, &(this->b_ij)));
    	check_result(HYPRE_IJVectorSetPrintLevel(this->b_ij, print_level));
    	check_result(HYPRE_IJVectorSetObjectType(this->b_ij, HYPRE_PARCSR));

      check_result(HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, ilower, iupper, &(this->x_ij)));
    	check_result(HYPRE_IJVectorSetPrintLevel(this->x_ij, print_level));
    	check_result(HYPRE_IJVectorSetObjectType(this->x_ij, HYPRE_PARCSR));

      // Convert system matrix into Hypre required formats and store them
      linsolv_hypre_ilu<N_BLOCK_SIZE>::csr_matrix_to_hypre_ij(*(this->A), this->A_ij);  // convert input matrix to hypre ij matrix
      check_result(HYPRE_IJMatrixGetObject(this->A_ij, (void **)&(this->A_parcsr)));  // convert to hypre parCSR, needed by the solver

      // Setup Hypre ILU solver
      // Note that in this function b_par and x_par are ignored
      check_result(HYPRE_ILUSetup(this->solver, this->A_parcsr, this->b_par, this->x_par));

      // linsolv_iface::timer_setup->node["AMG"].stop();

      return 0;
      }
      catch (const std::exception &e)
      {
        std::cout << e.what() << std::endl;
        return -1;
      }
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_hypre_ilu<N_BLOCK_SIZE>::solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X)
    {
      try
      {

      // Generate Hypre right hand side vector b_ij
      opendarts::config::index_t n_rows = this->A->n_cols;;  // number of rows in vector must
                                                             // be the same as number of columns
                                                             // of system matrix

      if (static_cast<opendarts::config::index_t>(this->row_indices_.size()) < n_rows)
      {
        const opendarts::config::index_t old_size =
            static_cast<opendarts::config::index_t>(this->row_indices_.size());
        this->row_indices_.resize(n_rows);
        std::iota(this->row_indices_.begin() + old_size,
            this->row_indices_.end(), old_size);
      }
      opendarts::config::index_t *rows_data = this->row_indices_.data();

      check_result(HYPRE_IJVectorInitialize(b_ij));
    	check_result(HYPRE_IJVectorSetValues(b_ij, n_rows, rows_data, B));
    	check_result(HYPRE_IJVectorAssemble(b_ij));
    	check_result(HYPRE_IJVectorGetObject(b_ij, (void **)&b_par));

      // Generate Hypre solution vector x_ij
      check_result(HYPRE_IJVectorInitialize(x_ij));
    	check_result(HYPRE_IJVectorSetValues(x_ij, n_rows, rows_data, X));
    	check_result(HYPRE_IJVectorAssemble(x_ij));
    	check_result(HYPRE_IJVectorGetObject(x_ij, (void **)&x_par));

      // Solve the system
      check_result(HYPRE_ILUSolve(this->solver, this->A_parcsr, b_par, x_par));

      // CRITICAL: retrieve the solution from HYPRE's internal vector back into
      // the caller-supplied X buffer. HYPRE_IJVectorSetValues *copies* values
      // in, so HYPRE_IJVectorGetValues is required to copy them back out --
      // otherwise X stays at whatever it was on entry. Mirrors the linsolv_cpr
      // pattern (HYPRE_IJVectorGetValues after ILUSolve).
      check_result(HYPRE_IJVectorGetValues(x_ij, n_rows, rows_data, X));

      return 0;
      }
      catch (const std::exception &e)
      {
        std::cout << e.what() << std::endl;
        return -1;
      }
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::index_t linsolv_hypre_ilu<N_BLOCK_SIZE>::get_n_iters()
    {
      std::cout << "NOT IMPLEMENTED: linsolv_hypre_ilu::get_n_iters" << std::endl;

      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::mat_float linsolv_hypre_ilu<N_BLOCK_SIZE>::get_residual()
    {
      std::cout << "NOT IMPLEMENTED: linsolv_hypre_ilu::get_residual" << std::endl;

      return 1000.0;
    }

    template<>
    void linsolv_hypre_ilu<1>::csr_matrix_to_hypre_ij(
      opendarts::linear_solvers::csr_matrix<1> &A,
      HYPRE_IJMatrix &A_ij)
    {
      // NOTE: This function works only for N_BLOCK_SIZE = 1
      //       For other values of the block size a full copy of the data must be
      //       done and some temporary storage needs to be arranged.

      const int print_level = 0;  // 0 = quiet (was 2 = HYPRE diagnostics)

      // Convert csr_matrix A to Hypre ij_matrix
      opendarts::config::index_t ilower, iupper;
      ilower = 0;
      iupper = A.n_rows - 1;

      if (static_cast<opendarts::config::index_t>(this->row_indices_.size()) < A.n_rows)
      {
        const opendarts::config::index_t old_size =
            static_cast<opendarts::config::index_t>(this->row_indices_.size());
        this->row_indices_.resize(A.n_rows);
        std::iota(this->row_indices_.begin() + old_size,
            this->row_indices_.end(), old_size);
      }
      this->n_cols_.resize(A.n_rows);
      for (opendarts::config::index_t row_idx = 0; row_idx < A.n_rows; row_idx++)
        this->n_cols_[row_idx] = A.rows_ptr[row_idx + 1] - A.rows_ptr[row_idx];

      check_result(HYPRE_IJMatrixCreate(hypre_MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A_ij));
    	check_result(HYPRE_IJMatrixSetPrintLevel(A_ij, print_level));
    	check_result(HYPRE_IJMatrixSetObjectType(A_ij, HYPRE_PARCSR));
      check_result(HYPRE_IJMatrixInitialize(A_ij));
    	check_result(HYPRE_IJMatrixSetValues(A_ij, A.n_rows, this->n_cols_.data(), this->row_indices_.data(), A.get_cols_ind(), A.get_values()));
    	check_result(HYPRE_IJMatrixAssemble(A_ij));
    }

    template <>
    void linsolv_hypre_ilu<1>::refresh(opendarts::linear_solvers::csr_matrix<1> *A)
    {
      // Update values on an existing IJMatrix without destroying it.
      // HYPRE_IJMatrixInitialize re-opens the matrix for SetValues; the
      // sparsity pattern is preserved across calls. The ILU factorization
      // (built by setup() via HYPRE_ILUSetup) is intentionally NOT rebuilt
      // here -- it will be reused by the next HYPRE_ILUSolve call.
      const opendarts::config::index_t n_rows = A->n_rows;

      // Cached row-index iota -- grow only if needed.
      if (static_cast<opendarts::config::index_t>(this->row_indices_.size()) < n_rows)
      {
        const opendarts::config::index_t old_size =
            static_cast<opendarts::config::index_t>(this->row_indices_.size());
        this->row_indices_.resize(n_rows);
        std::iota(this->row_indices_.begin() + old_size,
            this->row_indices_.end(), old_size);
      }
      // n_cols_ was already filled by csr_matrix_to_hypre_ij during setup();
      // recompute defensively only if the cache is too small.
      if (static_cast<opendarts::config::index_t>(this->n_cols_.size()) < n_rows)
      {
        this->n_cols_.resize(n_rows);
        for (opendarts::config::index_t row_idx = 0; row_idx < n_rows; row_idx++)
          this->n_cols_[row_idx] = A->rows_ptr[row_idx + 1] - A->rows_ptr[row_idx];
      }

      check_result(HYPRE_IJMatrixInitialize(this->A_ij));
      check_result(HYPRE_IJMatrixSetValues(this->A_ij, n_rows, this->n_cols_.data(),
          this->row_indices_.data(), A->get_cols_ind(), A->get_values()));
      check_result(HYPRE_IJMatrixAssemble(this->A_ij));
      check_result(HYPRE_IJMatrixGetObject(this->A_ij, (void **)&(this->A_parcsr)));
    }

    namespace
    {
      // Internal linkage: the AMG sibling wrapper defines its own helper of
      // the same name; keeping both file-local avoids any ODR interaction.
      inline void check_result(int res)
      {
        char err_msg_char[256];
        if (res)
        {
          HYPRE_DescribeError(res, err_msg_char);
          std::string err_msg(err_msg_char);
          std::cout << "\n" << err_msg << std::endl;
          // Throw instead of exit(-1); setup()/solve() translate to a
          // nonzero return so the engine cuts the timestep.
          throw std::runtime_error("linsolv_hypre_ilu: HYPRE error -- " + err_msg);
        }
      }

      inline void check_result_nothrow(int res)
      {
        char err_msg_char[256];
        if (res)
        {
          HYPRE_DescribeError(res, err_msg_char);
          std::cout << "\nlinsolv_hypre_ilu (dtor): " << err_msg_char << std::endl;
        }
      }
    } // namespace

    template class linsolv_hypre_ilu<1>;
    // Note that for values of block size larger than 1 a matrix copy must be carried
    // because conversion to a block matrix of size 1 is required
    // template class linsolv_hypre_amg<3>;
  } // namespace linear_solvers
} // namespace opendarts
