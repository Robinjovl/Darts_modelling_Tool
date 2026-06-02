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

#include "slu_ddefs.h"

#include "block_csr_matrix.hpp"
#include "data_types.hpp"
#include "csr_matrix.hpp"
#include "linsolv_iface.hpp"
#include "linsolv_superlu.hpp"
#include "scalar_csr_adapter.hpp"

#define SLU_SIMPLE

#ifndef SLU_SIMPLE
#define SLU_PREALLOC_WORK
#endif // SLU_SIMPLE

namespace opendarts
{
  namespace linear_solvers
  {
    template <uint8_t N_BLOCK_SIZE> linsolv_superlu<N_BLOCK_SIZE>::~linsolv_superlu()
    {
#ifndef SLU_SIMPLE
      delete this->etree delete this->R delete this->C
#endif // SLU_SIMPLE
          ;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_superlu<N_BLOCK_SIZE>::set_prec(opendarts::linear_solvers::linsolv_iface *prec_input)
    {
      (void)prec_input; // to suppress warning of unused parameter
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_superlu<N_BLOCK_SIZE>::init(opendarts::linear_solvers::csr_matrix_base *A_input,
        opendarts::config::index_t max_iters,
        opendarts::config::mat_float tolerance)
    {
      (void)max_iters; // not used in this solver, but required for other solvers, so added this to suppress warning of
                       // unused parameter
      (void)tolerance; // same here

      this->A_base = A_input;
      this->A = dynamic_cast<opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *>(A_input);

      // Bind a scalar_csr_adapter when the engine hands us the new
      // block_csr_matrix Jacobian. The adapter shares the scalar-CSR
      // *structure* across Newton iterations (cached on the matrix's
      // sparsity_pattern) and refreshes only the values per setup() --
      // bypasses the per-solve `new csr_matrix<1>; to_nb_1; delete`
      // cycle the legacy fallback in solve() still uses.
      auto *A_block = dynamic_cast<opendarts::linear_solvers::block_csr_matrix *>(A_input);
      if (A_block != nullptr)
        this->scalar_adapter_ = std::make_unique<
            opendarts::linear_solvers::scalar_csr_adapter>(*A_block);
      else
        this->scalar_adapter_.reset();

      const opendarts::config::index_t block_size =
          A_input != nullptr ? A_input->n_row_size : N_BLOCK_SIZE;
      this->n_rows = A_input->n_rows * block_size;
      this->perm_r = new int[this->n_rows];
      this->perm_c = new int[this->n_rows];

#ifndef SLU_SIMPLE
      this->etree = new int[this->n_rows];
      this->R = new double[this->n_rows];
      this->C = new double[this->n_rows];
      this->lwork = 0;

#ifdef SLU_PREALLOC_WORK
      // Why is there an allocation of 150 times the size of the memory of A?
      // I guess it is because it must be able to store L and U (and maybe B),
      // and this is just guesswork, this can go quite badly
      this->lwork = A_input->n_non_zeros * block_size * block_size * 150 * sizeof(double);

      this->work = SUPERLU_MALLOC(this->lwork);
      if (!(this->work))
      {
        ABORT("linsolv_superlu: cannot allocate work[]");
      }

#endif // SLU_PREALLOC_WORK

      this->first = 1;
#endif // SLU_SIMPLE

      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_superlu<N_BLOCK_SIZE>::init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_input,
        opendarts::config::index_t max_iters,
        opendarts::config::mat_float tolerance)
    {
      return this->init(static_cast<opendarts::linear_solvers::csr_matrix_base *>(A_input), max_iters, tolerance);
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_superlu<N_BLOCK_SIZE>::setup(opendarts::linear_solvers::csr_matrix_base *A_update)
    {
      this->timer_setup->node["SUPERLU"].start();

      this->A_base = A_update;
      this->A = dynamic_cast<opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *>(A_update);

      // Re-gather the scalar values from the (re-assembled) block Jacobian.
      // The structure is unchanged across Newton iterations -- refresh() is
      // an O(nnz) value-only gather, no allocations. If init() bound the
      // adapter to a different matrix (rare; would happen only if the
      // engine swaps Jacobians), rebuild it.
      if (this->scalar_adapter_)
      {
        auto *A_block = dynamic_cast<opendarts::linear_solvers::block_csr_matrix *>(A_update);
        if (A_block == nullptr)
        {
          // Setup-time switch back to a legacy csr_matrix<N> -- discard the
          // adapter so solve() takes the fallback path.
          this->scalar_adapter_.reset();
        }
        else
        {
          this->scalar_adapter_->refresh();
        }
      }
      else
      {
        auto *A_block = dynamic_cast<opendarts::linear_solvers::block_csr_matrix *>(A_update);
        if (A_block != nullptr)
          this->scalar_adapter_ = std::make_unique<
              opendarts::linear_solvers::scalar_csr_adapter>(*A_block);
      }

      this->timer_setup->node["SUPERLU"].stop();
      return 0;
    };

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_superlu<N_BLOCK_SIZE>::setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_update)
    {
      return this->setup(static_cast<opendarts::linear_solvers::csr_matrix_base *>(A_update));
    };

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_superlu<N_BLOCK_SIZE>::solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X)
    {
      SuperMatrix A_superlu, B_superlu, L_superlu, U_superlu;
      superlu_options_t options_superlu;
      SuperLUStat_t stat_superlu;
      int info_superlu;

#ifndef SLU_SIMPLE
      SuperMatrix X_superlu;
      char equed[1];
      double rpg, rcond;
      GlobalLU_t Glu;
      mem_usage_t mem_usage;
      double ferr, berr;
#endif // SLU_SIMPLE

      set_default_options(&options_superlu);
      options_superlu.IterRefine = SLU_DOUBLE;
      options_superlu.ColPerm = COLAMD;

      StatInit(&stat_superlu);

      // Resolve the scalar-CSR triple SuperLU consumes. Preferred path
      // (block Jacobian + cached adapter): zero-alloc, structure-shared,
      // values gathered once in setup(). Fallback path (legacy
      // csr_matrix<N> -- proprietary build, tests, GPU): the existing
      // per-solve `new csr_matrix<1>; to_nb_1; delete` cycle.
      opendarts::linear_solvers::csr_matrix<1> *A_as_nb_1 = nullptr;
      bool delete_A_as_nb_1 = false;

      // Pointers to the scalar arrays handed to dCreate_CompCol_Matrix:
      // populated either from the adapter (preferred) or A_as_nb_1
      // (fallback). int_t/opendarts::index_t = int (see data_types.hpp).
      opendarts::config::index_t n_scalar_rows = 0;
      opendarts::config::index_t n_scalar_cols = 0;
      opendarts::config::index_t n_scalar_nnz = 0;
      double *vals_ptr = nullptr;
      // SuperLU is non-const for these pointers but never writes through
      // them in COLAMD/SamePattern paths -- safe to const_cast.
      opendarts::config::index_t *cols_ptr = nullptr;
      opendarts::config::index_t *rows_ptr = nullptr;

      if (this->A_base == nullptr)
      {
        return -1;
      }

      if (this->scalar_adapter_)
      {
        auto *adapter = this->scalar_adapter_.get();
        n_scalar_rows = adapter->n_rows();
        n_scalar_cols = adapter->n_cols();
        n_scalar_nnz = adapter->nnz();
        vals_ptr = const_cast<double *>(adapter->values());
        cols_ptr = const_cast<opendarts::config::index_t *>(adapter->col_ind());
        rows_ptr = const_cast<opendarts::config::index_t *>(adapter->row_ptr());
      }
      else
      {
        if (this->A_base->n_row_size == 1)
        {
          A_as_nb_1 = dynamic_cast<opendarts::linear_solvers::csr_matrix<1> *>(this->A_base);
        }
        if (A_as_nb_1 == nullptr)
        {
          A_as_nb_1 = new opendarts::linear_solvers::csr_matrix<1>;
          A_as_nb_1->to_nb_1(this->A_base);
          delete_A_as_nb_1 = true;
        }
        n_scalar_rows = A_as_nb_1->n_rows;
        n_scalar_cols = A_as_nb_1->n_cols;
        n_scalar_nnz = A_as_nb_1->n_non_zeros;
        vals_ptr = A_as_nb_1->values.data();
        cols_ptr = A_as_nb_1->cols_ind.data();
        rows_ptr = A_as_nb_1->rows_ptr.data();
      }

      dCreate_CompCol_Matrix(&A_superlu, n_scalar_rows, n_scalar_cols, n_scalar_nnz,
          vals_ptr, cols_ptr, rows_ptr, SLU_NR, SLU_D, SLU_GE);
      this->timer_solve->node["SUPERLU"].start();

#ifdef SLU_SIMPLE
      memcpy(X, B, A_superlu.nrow * sizeof(opendarts::config::mat_float));
      dCreate_Dense_Matrix(&B_superlu, A_superlu.nrow, 1, X, A_superlu.nrow, SLU_DN, SLU_D, SLU_GE);

      dgssv(&options_superlu, &A_superlu, perm_c, perm_r, &L_superlu, &U_superlu, &B_superlu, &stat_superlu,
          &info_superlu);
#else  // SLU_SIMPLE

      if (this->first)
      {
        this->first = 0;
      }
      else
      {
        options_superlu.Fact = SamePattern;
      }

      dCreate_Dense_Matrix(&B_superlu, A_superlu.nrow, 1, B, A_superlu.nrow, SLU_DN, SLU_D, SLU_GE);
      dCreate_Dense_Matrix(&X_superlu, A_superlu.nrow, 1, X, A_superlu.nrow, SLU_DN, SLU_D, SLU_GE);

      dgssvx(&options_superlu, &A_superlu, this->perm_c, this->perm_r, this->etree, equed, this->R, this->C, &L_superlu,
          &U_superlu, this->work, this->lwork, &B_superlu, &X_superlu, &rpg, &rcond, &ferr, &berr, &Glu, &mem_usage,
          &stat_superlu, &info_superlu);

      Destroy_SuperMatrix_Store(&X_superlu);
#endif // SLU_SIMPLE

      this->timer_solve->node["SUPERLU"].stop();

      Destroy_SuperMatrix_Store(&B_superlu);

#ifndef SLU_PREALLOC_WORK
      Destroy_SuperNode_Matrix(&L_superlu);
      Destroy_CompCol_Matrix(&U_superlu);
#endif // SLU_PREALLOC_WORK

      if (delete_A_as_nb_1)
      {
        delete A_as_nb_1;
      }

      return 0;
    }

    template <uint8_t N_BLOCK_SIZE> opendarts::config::index_t linsolv_superlu<N_BLOCK_SIZE>::get_n_iters() { return 1; }

    template <uint8_t N_BLOCK_SIZE> opendarts::config::mat_float linsolv_superlu<N_BLOCK_SIZE>::get_residual() { return 0.0; }

    template class linsolv_superlu<1>;
    template class linsolv_superlu<2>;
    template class linsolv_superlu<3>;
    template class linsolv_superlu<4>;
    template class linsolv_superlu<5>;
    template class linsolv_superlu<6>;
    template class linsolv_superlu<7>;
    template class linsolv_superlu<8>;
    template class linsolv_superlu<9>;
    template class linsolv_superlu<10>;
    template class linsolv_superlu<11>;
    template class linsolv_superlu<12>;
    template class linsolv_superlu<13>;

  } // namespace linear_solvers
} // namespace opendarts
