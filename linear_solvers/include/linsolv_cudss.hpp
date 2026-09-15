//*************************************************************************
//    Copyright (c) 2026
//    Delft University of Technology, the Netherlands
//
//    This file is part of the open Delft Advanced Research Terra Simulator
//    (open-DARTS). It is distributed under the Apache License.
//*************************************************************************

//--------------------------------------------------------------------------
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_CUDSS_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_CUDSS_HPP
//--------------------------------------------------------------------------

#if defined(WITH_GPU) && defined(WITH_CUDSS)

#include <cstdio>

#include <cudss.h>

#include "data_types.hpp"
#include "csr_matrix.hpp"
#include "linear_solver_base.hpp"
#include "linsolv_iface_bos.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** @brief Direct sparse linear solver on the GPU via NVIDIA cuDSS.

        cuDSS (https://docs.nvidia.com/cuda/cudss/) is NVIDIA's sparse
        direct-solver library -- the successor of the deprecated cusolverSp
        QR routines wrapped by linsolv_cusolv. The solve is staged:
        ANALYSIS (reordering + symbolic factorization, structure-only) runs
        once in init(); FACTORIZATION re-runs per setup() on the refreshed
        Jacobian values; SOLVE runs per solve() call.

        The matrix is consumed in scalar-CSR form on the device, through the
        exact same dispatch linsolv_cusolv uses: the legacy csr_matrix<N>
        Jacobian via convert_to_ELL (csrValC/csrRowPtrC/csrColIndC) and the
        unified block_csr_matrix via build_scalar_csr_device()
        (cusparseDbsr2csr through gpu_bsr_spmv). A follow-up can move to
        cuDSS's batch/block API and skip the scalar expansion.

        Opt-in build: -D WITH_CUDSS=ON (see thirdparty/thirdparty_cudss.cmake;
        the library ships prebuilt, e.g. the nvidia-cudss-cu13 wheel or the
        NVIDIA redistributable). Select at runtime with
        params.linear_type = sim_params.gpu_cudss (CuDSSSolverSpec in Python).
    */
    template <uint8_t N_BLOCK_SIZE>
    class linsolv_cudss : public opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>,
                          public opendarts::linear_solvers::linear_solver_base
    {
    public:
      linsolv_cudss();

      ~linsolv_cudss();

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

      // Polymorphic csr_matrix_base entry points -- accept both the legacy
      // csr_matrix<N> Jacobian and the unified block_csr_matrix.
      int init(opendarts::linear_solvers::csr_matrix_base *A,
        opendarts::config::index_t max_iters,
        opendarts::config::mat_float tolerance) override;

      int setup(opendarts::linear_solvers::csr_matrix_base *A) override;

      //////////////////////
      // linsolv_iface
      //////////////////////

      int set_prec(opendarts::linear_solvers::linsolv_iface *prec_input) override;

      // Typed overrides forward to the csr_matrix_base entry points above.
      int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_input,
        int max_iters,
        double tolerance) override
      {
        return this->init(static_cast<opendarts::linear_solvers::csr_matrix_base *>(A_input),
            static_cast<opendarts::config::index_t>(max_iters),
            static_cast<opendarts::config::mat_float>(tolerance));
      }

      int setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_input) override
      {
        return this->setup(static_cast<opendarts::linear_solvers::csr_matrix_base *>(A_input));
      }

      int solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X) override;

      int get_n_iters() override;

      opendarts::config::mat_float get_residual() override;

      int device_num = 0;
      opendarts::config::index_t n_scalar_rows = 0;
      opendarts::config::index_t n_scalar_nnz = 0;

      /** Iterative-refinement steps after each solve (CUDSS_CONFIG_IR_N_STEPS;
          0 = library default, i.e. none). Refinement recovers the accuracy
          lost to the perturbed (static) pivots cuDSS applies to tiny pivots --
          on badly scaled Jacobians (rows scaled to unit max with entries down
          to 1e-40, well rows, contact rows) a plain factorization occasionally
          returns a solution with a large true residual while reporting
          success; two refinement steps bring it back to round-off. Set before
          init(). */
      int ir_n_steps = 0;
      /** CUDSS_CONFIG_PIVOT_EPSILON override (< 0 keeps the library default). */
      double pivot_epsilon = -1.0;
      /** CUDSS_CONFIG_HYBRID_MEMORY_MODE: keep part of the LU factors in host
          memory so systems whose factorization exceeds the free device memory
          still solve (slower); hybrid_device_memory_limit (bytes, 0 = let
          cuDSS choose) caps the device share. Set before init(). */
      bool hybrid_memory = false;
      long long hybrid_device_memory_limit = 0;

    private:
      // (Re)bind the cudssMatrix_t CSR descriptor to the matrix's current
      // scalar-CSR device pointers; creates the descriptor on first use.
      int bind_matrix_pointers();

      cudssHandle_t handle_ = nullptr;
      cudssConfig_t config_ = nullptr;
      cudssData_t data_ = nullptr;
      cudssMatrix_t A_cudss_ = nullptr; // CSR view of the Jacobian (device ptrs)
      cudssMatrix_t b_cudss_ = nullptr; // dense RHS (device, n x 1)
      cudssMatrix_t x_cudss_ = nullptr; // dense solution (device, n x 1)

      bool analyzed_ = false; // ANALYSIS (reordering/symbolic) done

      // Polymorphic matrix pointer -- supports both legacy csr_matrix<N>
      // and the unified block_csr_matrix.
      opendarts::linear_solvers::csr_matrix_base *A_matrix = nullptr;
      opendarts::config::mat_float *d_B = nullptr;
      opendarts::config::mat_float *d_X = nullptr;
    };
  } // namespace linear_solvers
} // namespace opendarts

#endif // WITH_GPU && WITH_CUDSS

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_CUDSS_HPP
//--------------------------------------------------------------------------
