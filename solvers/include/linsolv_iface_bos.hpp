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
          // The down-cast below is valid only when the matrix block size
          // matches N_BLOCK_SIZE. Guard it so a mismatch fails loudly in Debug
          // builds instead of silently corrupting memory (compiled out under
          // NDEBUG, so Release builds are unaffected).
          assert((A == nullptr || A->n_row_size == N_BLOCK_SIZE)
            && "csr_matrix block size does not match the solver block size");
          return this->init(static_cast<opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *>(A), max_iters, tolerance);
        };

        virtual int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A, int max_iters, double tolerance) = 0;

        virtual int setup(opendarts::linear_solvers::csr_matrix_base *A)
        {
          // See init(): guard the down-cast against a block-size mismatch so
          // it fails loudly in Debug builds instead of corrupting memory.
          assert((A == nullptr || A->n_row_size == N_BLOCK_SIZE)
            && "csr_matrix block size does not match the solver block size");
          return this->setup(static_cast<opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *>(A));
        };

        virtual int setup (opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A) = 0;

        opendarts::linear_solvers::linear_solver_base *get_bos_solver ()
        {
          return solver;
        };

        opendarts::linear_solvers::linear_solver_base *solver;

    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_IFACE_HPP
//--------------------------------------------------------------------------
