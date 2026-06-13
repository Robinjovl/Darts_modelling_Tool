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
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_IFACE_BOS_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_IFACE_BOS_HPP
//--------------------------------------------------------------------------

#include <cassert>
#include <cstdio>

#include "data_types.hpp"
#include "csr_matrix_base.hpp"
#include "csr_matrix.hpp"
#include "linear_solver_base.hpp"
#include "linsolv_iface.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    template <uint8_t N_BLOCK_SIZE>
    class linsolv_iface_bos : public linsolv_iface
    {

      public:
        linsolv_iface_bos () {};

        virtual ~linsolv_iface_bos () {};

        virtual int init(opendarts::linear_solvers::csr_matrix_base *A,
          int max_iters,
          opendarts::config::mat_float tolerance)
        {
          // The down-cast below is valid only for an actual csr_matrix<N> of
          // the matching block size -- NOT for a block_csr_matrix or a
          // mismatched block size. Use a checked cast so a wrong matrix type
          // fails loudly (error return -> the Newton loop reports a linear-
          // solver failure) instead of silently corrupting memory in Release
          // builds; the assert keeps the immediate fail-fast in Debug.
          auto *A_typed = checked_cast(A, "init");
          if (A != nullptr && A_typed == nullptr)
            return -1;
          return this->init(A_typed, max_iters, tolerance);
        };

        virtual int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A, int max_iters, double tolerance) = 0;

        virtual int setup(opendarts::linear_solvers::csr_matrix_base *A)
        {
          // See init(): checked down-cast, loud failure on mismatch.
          auto *A_typed = checked_cast(A, "setup");
          if (A != nullptr && A_typed == nullptr)
            return -1;
          return this->setup(A_typed);
        };

        virtual int setup (opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A) = 0;

        opendarts::linear_solvers::linear_solver_base *get_bos_solver ()
        {
          return solver;
        };

        opendarts::linear_solvers::linear_solver_base *solver;

      private:
        // Checked replacement for the former bare static_cast: returns null
        // (after a clear diagnostic) when A is not a csr_matrix<N_BLOCK_SIZE>
        // -- e.g. a block_csr_matrix or a different block size reached a
        // solver that never overrode the csr_matrix_base entry points.
        opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *checked_cast(
          opendarts::linear_solvers::csr_matrix_base *A, const char *where)
        {
          if (A == nullptr)
            return nullptr;
          auto *A_typed = dynamic_cast<opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *>(A);
          assert(A_typed != nullptr
            && "matrix is not a csr_matrix of the solver's block size");
          if (A_typed == nullptr)
            fprintf(stderr,
              "linsolv_iface_bos<%d>::%s: matrix (n_row_size=%d) is not a "
              "csr_matrix<%d>; this solver lacks block_csr_matrix support\n",
              static_cast<int>(N_BLOCK_SIZE), where,
              static_cast<int>(A->n_row_size), static_cast<int>(N_BLOCK_SIZE));
          return A_typed;
        }

    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_IFACE_HPP
//--------------------------------------------------------------------------
