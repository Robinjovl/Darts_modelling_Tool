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
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_AMGX_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_AMGX_HPP
//--------------------------------------------------------------------------

// AMGX support is opt-in (CMake option WITH_AMGX); the wrapper is only
// compiled when both the GPU build and AMGX are enabled.
#if defined(WITH_GPU) && defined(WITH_AMGX)

#include "data_types.hpp"
#include "csr_matrix.hpp"
#include "linear_solver_base.hpp"
#include "linsolv_iface_bos.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** @brief NVIDIA AMGX algebraic-multigrid solver on the GPU.

        Wraps the AMGX C API (thirdparty/AMGX). The matrix is uploaded either
        in native block form or, when convert_to_bs1 is set, expanded to
        scalar CSR via csr_matrix::convert_to_ELL. The AMGX solver itself is
        configured from amgx_config_bs<N>.json when present, otherwise from a
        built-in classical-AMG default.

        AMGX handles are kept as void* to avoid pulling the AMGX headers into
        this interface header (their typedefs are awkward to forward-declare);
        the implementation casts them back.

        Ported from the proprietary darts-linear-solvers linsolv_amgx.
    */
    template <uint8_t N_BLOCK_SIZE>
    class linsolv_amgx : public opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>,
                         public opendarts::linear_solvers::linear_solver_base
    {
    public:
      /** @param device_num - CUDA device AMGX should run on.
          @param convert_to_bs1 - expand block matrices to scalar CSR before
              handing them to AMGX. */
      linsolv_amgx(int device_num = 0, int convert_to_bs1 = 1);

      ~linsolv_amgx();

      // Keep the csr_matrix_base init()/setup() overloads visible: declaring
      // the csr_matrix<N>* overloads below otherwise hides them by name.
      using opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::init;
      using opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::setup;

      //////////////////////
      // linear_solver_base
      //////////////////////

      int solve(opendarts::linear_solvers::csr_matrix_base * /*matrix*/,
        opendarts::config::mat_float *v,
        opendarts::config::mat_float *r) override
      {
        return solve(v, r);
      }

      int setup(opendarts::linear_solvers::csr_matrix_base *A) override
      {
        return this->setup(static_cast<opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *>(A));
      }

      //////////////////////
      // linsolv_iface
      //////////////////////

      int set_prec(opendarts::linear_solvers::linsolv_iface *prec_input) override;

      int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_input,
        int max_iters,
        double tolerance) override;

      int setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_input) override;

      int solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X) override;

      int get_n_iters() override;

      opendarts::config::mat_float get_residual() override;

      int device_num;
      int convert_to_bs1;
      int n_rows;

      // AMGX handles (AMGX_solver_handle, AMGX_config_handle, AMGX_matrix_handle,
      // AMGX_vector_handle). Kept as void* -- they are opaque pointers anyway.
      void *solver;
      void *config;
      void *A;
      void *x, *b;
      int AMGX_mode;
    };
  } // namespace linear_solvers
} // namespace opendarts

#endif // WITH_GPU && WITH_AMGX

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_AMGX_HPP
//--------------------------------------------------------------------------
