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
#include "hypre_ij_builder.hpp"
#include "linsolv_iface.hpp"
#include "linsolv_hypre_amg.hpp"

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
    linsolv_hypre_amg<N_BLOCK_SIZE>::linsolv_hypre_amg()
    {
      // Initialize preconditioner (not used in this case, kept for compatibility)
      this->prec = 0;  // no preconditioner for this solver

      // Null-init all HYPRE handles so the destructor can safely skip Destroy
      // calls when init()/setup() were never reached (or were partially
      // executed). Together with the null-check in ~linsolv_hypre_amg this
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
    linsolv_hypre_amg<N_BLOCK_SIZE>::~linsolv_hypre_amg()
    {
      // Null-safe destruction: pybind11 may destroy this wrapper at
      // interpreter shutdown after HYPRE-side state has already been torn
      // down (or after sibling Destroy calls), so calling HYPRE_*Destroy on
      // a stale / null handle is a use-after-free. Check each handle, Destroy
      // only if non-null, then null it to make any subsequent double-destroy
      // a no-op as well.
      if (this->solver != nullptr)
      {
        check_result_nothrow(HYPRE_BoomerAMGDestroy(this->solver));
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
    int linsolv_hypre_amg<N_BLOCK_SIZE>::set_prec(opendarts::linear_solvers::linsolv_iface *prec_in)
    {
      (void) prec_in;

      std::cout << "NOT IMPLEMENTED: linsolv_hypre_amg::linsolv_hypre_amg" << std::endl;

      return 1;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_hypre_amg<N_BLOCK_SIZE>::init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_in,
      opendarts::config::index_t max_iters,
      opendarts::config::mat_float tolerance)
    {
      // BoomerAMG configured to be used as a preconditioner -- mirrors the
      // proven LinearSolver::setupAMGPreconditioner sequence (PMIS coarsening,
      // direct interpolation, hybrid GS relaxation). Without these explicit
      // choices HYPRE's defaults make BoomerAMGSetup hang on the pressure
      // subsystem from the engine.
      const int print_level = 0;

      // Ensure HYPRE is initialised before any HYPRE_* call. When this
      // wrapper is used as a sub-prec (e.g. inside FS-CPR) without a prior
      // CPR call, HYPRE_Initialize would otherwise never have been called and
      // HYPRE_BoomerAMGCreate / the setters below would fail with HYPRE
      // "[Generic error]". The engine calls init() before setup(), so the
      // setup() guard is not sufficient on its own.
      if (!HYPRE_Initialized())
        HYPRE_Initialize();
      HYPRE_ClearAllErrors();

      check_result(HYPRE_BoomerAMGCreate(&(this->solver)));
      check_result(HYPRE_BoomerAMGSetPrintLevel(this->solver, print_level));
      check_result(HYPRE_BoomerAMGSetLogging(this->solver, print_level));

      // When BoomerAMG is the outer iterative solver, the caller-supplied
      // (max_iters, tolerance) drive convergence; when it is a CPR stage
      // (set_amg_max_iters / amg_tolerance), the caller passes (1, 0.0).
      check_result(HYPRE_BoomerAMGSetMaxIter(this->solver, max_iters));
      check_result(HYPRE_BoomerAMGSetTol(this->solver, tolerance));

      // Coarsening / interpolation / relaxation -- the porous-media
      // recommended set, matching the in-tree MGR's AMG configuration.
      check_result(HYPRE_BoomerAMGSetCoarsenType(this->solver, 6));   // PMIS
      check_result(HYPRE_BoomerAMGSetInterpType(this->solver, 6));    // Direct
      check_result(HYPRE_BoomerAMGSetRelaxType(this->solver, 6));     // Hybrid GS

      // Systems AMG: when the caller declared n unknowns per node
      // (set_num_functions -- the displacement block of FS-CPR), tell BoomerAMG
      // so that coarsening and interpolation respect the vector structure
      // instead of treating each component as an independent scalar field.
      // HYPRE's default dof_func is row % num_functions, which matches the
      // interleaved storage the FS-CPR U block uses.
      if (num_functions_ > 1)
        check_result(HYPRE_BoomerAMGSetNumFunctions(this->solver, num_functions_));

      (void) A_in;  // matrix is consumed in setup()
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_hypre_amg<N_BLOCK_SIZE>::setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_in)
    {
      // Ensure HYPRE is initialised; when this wrapper is used as a sub-prec
      // (e.g. inside FS-CPR) without a prior CPR call, HYPRE_Initialize would
      // otherwise never have been called and BoomerAMGSetup would fail with
      // HYPRE "[Generic error]".
      if (!HYPRE_Initialized())
        HYPRE_Initialize();
      HYPRE_ClearAllErrors();

      try
      {
      // Store input system matrix
      this->A = A_in;

      // Re-entrant setup: destroy the handles from the previous call first.
      // They were unconditionally re-created on every call, leaking one IJ
      // matrix and two IJ vectors per Newton iteration when this wrapper
      // serves as an FS-CPR sub-preconditioner.
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

      // Setup right hand side and solution vectors
      const int print_level = 0;  // 0 = quiet (was 2 = HYPRE diagnostics)
      opendarts::config::index_t n_rows = this->A->n_cols;  // number of rows in vector must
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

      // Convert system matrix into Hyper required formats and store them
      linsolv_hypre_amg<N_BLOCK_SIZE>::csr_matrix_to_hypre_ij(*(this->A), this->A_ij);  // convert input matrix to hypre ij matrix
      check_result(HYPRE_IJMatrixGetObject(this->A_ij, (void **)&(this->A_parcsr)));  // convert to hypre parCSR, needed by the solver

      // Setup Hypre AMG solver
      // Note that in this function b_par and x_par are ignored
      check_result(HYPRE_BoomerAMGSetup(this->solver, this->A_parcsr, this->b_par, this->x_par));

      // linsolv_iface::timer_setup->node["AMG"].stop();

      return 0;
      }
      catch (const std::exception &e)
      {
        // A HYPRE failure becomes a nonzero return -> the caller (FS-CPR /
        // engine) cuts the timestep instead of the process dying (the old
        // check_result called exit(-1)).
        std::cout << e.what() << std::endl;
        return -1;
      }
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_hypre_amg<N_BLOCK_SIZE>::solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X)
    {
      try
      {

      // Generate Hypre right hand side vector b_ij
      opendarts::config::index_t n_rows = this->A->n_cols;;  // number of rows in vector must
                                                             // be the same as number of columns
                                                             // of system matrix

      // Cached [0, n_rows) index buffer -- grown once and reused across solves.
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
      check_result(HYPRE_BoomerAMGSolve(this->solver, this->A_parcsr, this->b_par, this->x_par));

      // CRITICAL: retrieve the solution from HYPRE's internal vector back into
      // the caller-supplied X buffer. HYPRE_IJVectorSetValues *copies* values
      // in, so HYPRE_IJVectorGetValues is required to copy them back out --
      // otherwise X stays at whatever it was on entry (typically zero from a
      // caller's fill_n), and BoomerAMG appears to produce a no-op preconditioner.
      // Mirrors the linsolv_cpr pattern (HYPRE_IJVectorGetValues after ILUSolve).
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
    opendarts::config::index_t linsolv_hypre_amg<N_BLOCK_SIZE>::get_n_iters()
    {
      std::cout << "NOT IMPLEMENTED: linsolv_hypre_amg::get_n_iters" << std::endl;

      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::mat_float linsolv_hypre_amg<N_BLOCK_SIZE>::get_residual()
    {
      std::cout << "NOT IMPLEMENTED: linsolv_hypre_amg::get_residual" << std::endl;

      return 1000.0;
    }

    template<>
    void linsolv_hypre_amg<1>::csr_matrix_to_hypre_ij(
      opendarts::linear_solvers::csr_matrix<1> &A,
      HYPRE_IJMatrix &A_ij)
    {
      // NOTE: This function works only for N_BLOCK_SIZE = 1
      //       For other values of the block size a full copy of the data must be
      //       done and some temporary storage needs to be arranged.
      // The scalar-CSR -> HYPRE-IJ create/set/assemble sequence and the cached
      // row-index / row-degree buffers live in the shared hypre_ij::build helper
      // (also used by linsolv_cpr and linsolv_hypre_ilu). GetObject is deferred
      // to setup() (hence the nullptr), preserving the previous flow.
      opendarts::linear_solvers::hypre_ij::build(A.n_rows, A.rows_ptr.data(),
          A.get_cols_ind(), A.get_values(), this->row_indices_, this->n_cols_,
          A_ij, nullptr);
    }

    template<>
    void linsolv_hypre_amg<1>::refresh(opendarts::linear_solvers::csr_matrix<1> *A)
    {
      // Value-only refresh of the cached IJ matrix. Re-opens the existing
      // HYPRE IJ matrix via HYPRE_IJMatrixInitialize, pushes new values using
      // the cached n_cols_ / row_indices_ buffers, re-assembles, and re-fetches
      // the ParCSR object. Does NOT call HYPRE_BoomerAMGSetup -- the AMG
      // hierarchy is reused, which is the whole point of this fast path.
      this->A = A;

      // Value-only refresh via the shared helper (re-open, SetValues, assemble,
      // GetObject) using the cached row-index / row-degree buffers. Does NOT
      // rebuild the AMG hierarchy -- the whole point of this fast path.
      opendarts::linear_solvers::hypre_ij::refresh(A->n_rows, A->rows_ptr.data(),
          A->get_cols_ind(), A->get_values(), this->row_indices_, this->n_cols_,
          this->A_ij, &this->A_parcsr);
    }

    namespace
    {
      // Internal linkage: the ILU sibling wrapper defines its own helper of
      // the same name; keeping both file-local avoids any ODR interaction.
      inline void check_result(int res)
      {
        char err_msg_char[256];
        if (res)
        {
          HYPRE_DescribeError(res, err_msg_char);
          std::string err_msg(err_msg_char);
          std::cout << "\n" << err_msg << std::endl;
          // Throw instead of the previous exit(-1): the public setup()/solve()
          // entry points translate this into a nonzero return so the engine
          // can cut the timestep instead of the whole host process dying.
          throw std::runtime_error("linsolv_hypre_amg: HYPRE error -- " + err_msg);
        }
      }

      // Destructor-safe variant: report but never throw (throwing from a
      // destructor terminates the process).
      inline void check_result_nothrow(int res)
      {
        char err_msg_char[256];
        if (res)
        {
          HYPRE_DescribeError(res, err_msg_char);
          std::cout << "\nlinsolv_hypre_amg (dtor): " << err_msg_char << std::endl;
        }
      }
    } // namespace

    // Explicit template instantiations — generated by CMake's
    // od_emit_template_instantiations(); see solvers/src/CMakeLists.txt.
    // Edit the block-size range there, not here.
#include "linsolv_hypre_amg_instantiations.inc"
  } // namespace linear_solvers
} // namespace opendarts
