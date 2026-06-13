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

// *************************************************************************
// SUPERLU wrapper
// *************************************************************************

//--------------------------------------------------------------------------
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_SUPERLU_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_SUPERLU_HPP
//--------------------------------------------------------------------------

#include <memory>

#include "block_csr_matrix.hpp"
#include "data_types.hpp"
#include "csr_matrix.hpp"
#include "linsolv_iface_bos.hpp"
#include "scalar_csr_adapter.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    template <uint8_t N_BLOCK_SIZE>
    class linsolv_superlu : public opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>
    {

    public:
      linsolv_superlu(){};

      ~linsolv_superlu();

      // Keep the csr_matrix_base init()/setup() overloads visible: declaring
      // the csr_matrix<N>* overloads below otherwise hides them by name.
      using opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::init;
      using opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::setup;

      virtual int set_prec(
          opendarts::linear_solvers::linsolv_iface *prec_input); // Implemented as do nothing

      virtual int init(opendarts::linear_solvers::csr_matrix_base *A_input,
          opendarts::config::index_t max_iters,
          opendarts::config::mat_float tolerance) override;

      virtual int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_input,
          opendarts::config::index_t max_iters,
          opendarts::config::mat_float tolerance);

      virtual int setup(opendarts::linear_solvers::csr_matrix_base *A_update) override;

      virtual int setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_update);

      virtual int solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X);

      virtual opendarts::config::index_t get_n_iters();

      virtual opendarts::config::mat_float get_residual();

      opendarts::linear_solvers::csr_matrix_base *A_base = nullptr; // the matrix used to solve the system
      opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A = nullptr; // legacy typed view when available

      // Cached scalar-CSR view when A_base is a block_csr_matrix (the new
      // canonical Jacobian layout). Built once in init() / first setup() and
      // refreshed per Newton iteration -- avoids the per-solve
      //   `new csr_matrix<1>; to_nb_1; ... ; delete`
      // allocation cycle the legacy block_size > 1 path was running. Null
      // when the engine still hands a legacy csr_matrix<N> (tests + GPU);
      // in that case solve() takes the legacy to_nb_1 fallback below.
      std::unique_ptr<opendarts::linear_solvers::scalar_csr_adapter> scalar_adapter_;

      opendarts::config::index_t n_rows = 0;
      opendarts::config::index_t nnz = 0;

      // Owned workspaces (released in the destructor; init() re-allocates,
      // freeing any previous allocation so repeated init() does not leak).
      opendarts::config::index_t *perm_r = nullptr; /* row permutations from partial pivoting */
      opendarts::config::index_t *perm_c = nullptr; /* column permutation vector */

      opendarts::config::index_t first = 1;
      void *work = nullptr;
      opendarts::config::index_t lwork = 0;
      opendarts::config::mat_float *R = nullptr;
      opendarts::config::mat_float *C = nullptr;

      opendarts::config::index_t *etree = nullptr;
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_IFACE_HPP
//--------------------------------------------------------------------------
