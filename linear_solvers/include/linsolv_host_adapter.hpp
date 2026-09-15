//*************************************************************************
//    Copyright (c) 2026
//    Delft University of Technology, the Netherlands
//
//    This file is part of the open Delft Advanced Research Terra Simulator
//    (open-DARTS). It is distributed under the Apache License.
//*************************************************************************

//--------------------------------------------------------------------------
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_HOST_ADAPTER_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_HOST_ADAPTER_HPP
//--------------------------------------------------------------------------

#ifdef WITH_GPU

#include <memory>

#include "data_types.hpp"
#include "csr_matrix_base.hpp"
#include "linear_solver.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** @brief Runs a device-resident (GPU) linear solver on a host-assembled system.

        The GPU solvers (linsolv_gmres_gpu, linsolv_bicgstab, linsolv_cpr_gpu,
        linsolv_cusparse_ilu, linsolv_amgx, ...) read the Jacobian through the
        matrix's device mirror and take DEVICE pointers in solve(). The GPU
        engines satisfy both natively: they assemble on the device and hand in
        RHS_d / dX_d. A CPU engine (every mechanics engine, and any flow engine
        run on platform 'cpu') assembles into the host side of the
        block_csr_matrix dual_array and calls solve(&RHS[0], &dX[0]) with host
        vectors -- so a GPU solver cannot be injected into it directly.

        This wrapper closes that gap without touching either the engines or the
        GPU solver classes:

          * init()  -- uploads the sparsity structure once, allocates the
                       device RHS / solution buffers and forwards to the inner
                       solver (which builds its device views from the mirror);
          * setup() -- marks the host values authoritative and uploads them
                       (dual_array::sync_to_device: one nnz*N*N doubles H2D
                       copy per Newton iteration), then forwards;
          * solve() -- stages B to the device (zero initial guess, as the CPU
                       Krylov solvers), runs the inner solve on device
                       pointers, copies X back.
                       Inner solvers that already stage host vectors themselves
                       (linsolv_cudss, linsolv_cusolv) are called through
                       directly (inner_takes_host_vectors).

        Everything else (iteration counts, residual, reconfigure, adjoint
        transpose solve, outer-iteration feedback) is forwarded verbatim.
        The registry factories "gpu_*" (solver_factories.cpp) return the GPU
        chains wrapped in this adapter, which is what makes a GPUSolverSpec
        usable on platform 'cpu' from Python (engine.set_linear_solver).

        Ownership: the adapter shares the chain head (shared_ptr); the stages
        below it are owned by the chain itself, exactly as in the GPU engine
        factory (the GPU Krylov drivers and linsolv_cpr_gpu delete the
        preconditioners handed to set_prec / set_p_system_prec).
    */
    class linsolv_host_adapter : public opendarts::linear_solvers::linear_solver
    {
    public:
      /** @param inner - the device-resident solver (or solver chain head).
          @param inner_takes_host_vectors - true when inner->solve() already
              accepts host pointers (cuDSS / cuSOLVER wrappers); the adapter
              then only keeps the matrix mirror in sync.
          @param device_num - CUDA device to select before the first
              allocation (negative leaves the current device untouched). */
      explicit linsolv_host_adapter(std::shared_ptr<opendarts::linear_solvers::linear_solver> inner,
        bool inner_takes_host_vectors = false, int device_num = -1);

      ~linsolv_host_adapter() override;

      std::shared_ptr<opendarts::linear_solvers::linear_solver> inner() const { return inner_; }

      int set_prec(opendarts::linear_solvers::linear_solver *prec_input) override
      {
        return inner_->set_prec(prec_input);
      }

      int set_p_system_prec(opendarts::linear_solvers::linear_solver *prec_input) override
      {
        return inner_->set_p_system_prec(prec_input);
      }

      int init(opendarts::linear_solvers::csr_matrix_base *A,
        opendarts::config::index_t max_iters,
        opendarts::config::mat_float tolerance) override;

      int setup(opendarts::linear_solvers::csr_matrix_base *A_input) override;

      int refresh(opendarts::linear_solvers::csr_matrix_base *A_input) override;

      int solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X) override;

      int solve_transposed(opendarts::config::mat_float *B, opendarts::config::mat_float *X) override;

      bool requires_setup_for_transposed_solve() const override
      {
        return inner_->requires_setup_for_transposed_solve();
      }

      int reconfigure(const opendarts::linear_solvers::solver_config &config) override
      {
        return inner_->reconfigure(config);
      }

      void set_last_outer_iters(int n_iters) override { inner_->set_last_outer_iters(n_iters); }

      int get_n_iters() override { return inner_->get_n_iters(); }

      opendarts::config::mat_float get_residual() override { return inner_->get_residual(); }

      opendarts::linear_solvers::solver_stats stats() const override { return inner_->stats(); }

    private:
      /// Upload the (host-authoritative) matrix values to the device mirror.
      int sync_values_to_device(opendarts::linear_solvers::csr_matrix_base *A);
      /// (Re)allocate the device RHS / solution staging buffers for n unknowns.
      int ensure_device_vectors(opendarts::config::index_t n);
      void free_device_vectors();

      std::shared_ptr<opendarts::linear_solvers::linear_solver> inner_;
      bool inner_takes_host_vectors_ = false;
      int device_num_ = -1;

      opendarts::config::mat_float *B_d_ = nullptr;
      opendarts::config::mat_float *X_d_ = nullptr;
      opendarts::config::index_t n_ = 0; // scalar unknowns of the staged system
    };
  } // namespace linear_solvers
} // namespace opendarts

#endif // WITH_GPU

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_HOST_ADAPTER_HPP
//--------------------------------------------------------------------------
