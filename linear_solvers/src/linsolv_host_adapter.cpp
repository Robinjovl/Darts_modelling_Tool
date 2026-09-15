//*************************************************************************
//    Copyright (c) 2026
//    Delft University of Technology, the Netherlands
//
//    This file is part of the open Delft Advanced Research Terra Simulator
//    (open-DARTS). It is distributed under the Apache License.
//*************************************************************************

#ifdef WITH_GPU

#include <cstdio>

#include <cuda_runtime.h>

#include "linsolv_host_adapter.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    linsolv_host_adapter::linsolv_host_adapter(
      std::shared_ptr<opendarts::linear_solvers::linear_solver> inner,
      bool inner_takes_host_vectors, int device_num)
      : inner_(std::move(inner)), inner_takes_host_vectors_(inner_takes_host_vectors),
        device_num_(device_num)
    {
    }

    linsolv_host_adapter::~linsolv_host_adapter()
    {
      free_device_vectors();
    }

    void linsolv_host_adapter::free_device_vectors()
    {
      if (B_d_) { cudaFree(B_d_); B_d_ = nullptr; }
      if (X_d_) { cudaFree(X_d_); X_d_ = nullptr; }
      n_ = 0;
    }

    int linsolv_host_adapter::ensure_device_vectors(opendarts::config::index_t n)
    {
      if (inner_takes_host_vectors_)
        return 0;
      if (n == n_ && B_d_ && X_d_)
        return 0;
      free_device_vectors();
      if (cudaMalloc((void **)&B_d_, sizeof(opendarts::config::mat_float) * n) != cudaSuccess
          || cudaMalloc((void **)&X_d_, sizeof(opendarts::config::mat_float) * n) != cudaSuccess)
      {
        fprintf(stderr, "linsolv_host_adapter: can't allocate device RHS/solution buffers (%lld unknowns)\n",
          (long long)n);
        free_device_vectors();
        return -1;
      }
      n_ = n;
      return 0;
    }

    int linsolv_host_adapter::sync_values_to_device(opendarts::linear_solvers::csr_matrix_base *A)
    {
      // The engine assembles through a host pointer it may have taken once at
      // init (the dual_array only records host writes made through the
      // non-const accessor), so mark the host side authoritative explicitly
      // before the upload; sync_to_device() is then an unconditional H2D copy
      // of nnz * N * N doubles.
      (void)A->get_values();
      if (this->timer_setup)
        this->timer_setup->node["send_to_device"].start();
      const int rc = A->copy_values_to_device();
      if (this->timer_setup)
        this->timer_setup->node["send_to_device"].stop();
      return rc;
    }

    int linsolv_host_adapter::init(opendarts::linear_solvers::csr_matrix_base *A,
      opendarts::config::index_t max_iters,
      opendarts::config::mat_float tolerance)
    {
      if (!inner_ || !A)
        return -1;
      if (device_num_ >= 0 && cudaSetDevice(device_num_) != cudaSuccess)
      {
        fprintf(stderr, "linsolv_host_adapter: cudaSetDevice(%d) failed\n", device_num_);
        return -1;
      }
      // The engine binds its timer nodes to the adapter (init_timer_nodes is
      // called before init); hand them down so the inner stages report under
      // the same "linear solver setup/solve" nodes as the native GPU path.
      inner_->init_timer_nodes(this->timer_setup, this->timer_solve);

      // Structure once, values now (the inner init may build device views
      // from the mirror -- e.g. the scalar-CSR expansion of cuDSS / cuSOLVER).
      if (A->copy_struct_to_device() != 0)
        return -1;
      if (sync_values_to_device(A) != 0)
        return -1;
      if (ensure_device_vectors(static_cast<opendarts::config::index_t>(A->n_rows) * A->n_row_size) != 0)
        return -1;
      return inner_->init(A, max_iters, tolerance);
    }

    int linsolv_host_adapter::setup(opendarts::linear_solvers::csr_matrix_base *A_input)
    {
      if (sync_values_to_device(A_input) != 0)
        return -1;
      return inner_->setup(A_input);
    }

    int linsolv_host_adapter::refresh(opendarts::linear_solvers::csr_matrix_base *A_input)
    {
      if (sync_values_to_device(A_input) != 0)
        return -1;
      return inner_->refresh(A_input);
    }

    int linsolv_host_adapter::solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X)
    {
      if (inner_takes_host_vectors_)
        return inner_->solve(B, X);
      if (!B_d_ || !X_d_)
        return -1;

      ::timer_node *t = this->timer_solve ? &this->timer_solve->node["host<->device"] : nullptr;
      if (t) t->start();
      const size_t bytes = sizeof(opendarts::config::mat_float) * n_;
      // Zero initial guess: the device Krylov drivers start from the X they
      // are handed, whereas the CPU engines leave the previous Newton update
      // in dX and the CPU Krylov solvers (linsolv_gmres) zero it themselves.
      // Zeroing here keeps the GPU chains on the same convention, so a run is
      // independent of the stale dX contents.
      const bool up_ok = cudaMemcpy(B_d_, B, bytes, cudaMemcpyHostToDevice) == cudaSuccess
                      && cudaMemset(X_d_, 0, bytes) == cudaSuccess;
      if (t) t->stop();
      if (!up_ok)
      {
        fprintf(stderr, "linsolv_host_adapter: host->device copy of the RHS failed\n");
        return -1;
      }

      const int rc = inner_->solve(B_d_, X_d_);

      if (t) t->start();
      cudaDeviceSynchronize();
      const bool down_ok = cudaMemcpy(X, X_d_, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
      if (t) t->stop();
      if (!down_ok)
      {
        fprintf(stderr, "linsolv_host_adapter: device->host copy of the solution failed\n");
        return -1;
      }
      return rc;
    }

    int linsolv_host_adapter::solve_transposed(opendarts::config::mat_float *B,
      opendarts::config::mat_float *X)
    {
      // Every device-resident solver's solve_transposed() is the host-side
      // adjoint entry point (it stages host B/X itself), so pass through.
      return inner_->solve_transposed(B, X);
    }
  } // namespace linear_solvers
} // namespace opendarts

#endif // WITH_GPU
