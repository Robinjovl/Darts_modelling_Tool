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

#include "openDARTS/config/data_types.hpp"
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#include "openDARTS/linear_solvers/linsolv_iface.hpp"
#include "openDARTS/linear_solvers/linsolv_superlu.hpp"

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
    int linsolv_superlu<N_BLOCK_SIZE>::init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_input,
        opendarts::config::index_t max_iters,
        opendarts::config::mat_float tolerance)
    {
      (void)max_iters; // not used in this solver, but required for other solvers, so added this to suppress warning of
                       // unused parameter
      (void)tolerance; // same here

      this->n_rows = A_input->n_rows * N_BLOCK_SIZE;
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
      this->lwork = A_input->n_total_non_zeros * 150 * sizeof(double);

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
    int linsolv_superlu<N_BLOCK_SIZE>::setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_update)
    {
      this->timer_setup->node["SUPERLU"].start();

      this->A = A_update;

      this->timer_setup->node["SUPERLU"].stop();
      return 0;
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

      opendarts::linear_solvers::csr_matrix<1> *A_as_nb_1;
      if (N_BLOCK_SIZE > 1)
      {
        // If the system matrix A is not of block size 1 then we need to create a temporary
        // A matrix with block size 1 (A_as_nb_1). We use a pointer, because if the block
        // size is 1 (below) we do not do a conversion and we just assign the pointer
        // to the A matrix to A_as_nb_1, so that we can use it below no matter
        // what the block size is.
        A_as_nb_1 = new opendarts::linear_solvers::csr_matrix<1>;
      }
      this->A->as_nb_1(A_as_nb_1);

      dCreate_CompCol_Matrix(&A_superlu, A_as_nb_1->n_rows, A_as_nb_1->n_cols, A_as_nb_1->n_non_zeros,
          A_as_nb_1->values.data(), A_as_nb_1->cols_ind.data(), A_as_nb_1->rows_ptr.data(), SLU_NR, SLU_D, SLU_GE);
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

      if (this->A->n_block_size_ > 1)
      {
        // If the block size is not 1 then we need to delete the temporary block
        // size 1 copy of A, to free up memory.
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
