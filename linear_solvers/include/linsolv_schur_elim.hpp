//*************************************************************************
//    Copyright (c) 2026
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
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_SCHUR_ELIM_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_SCHUR_ELIM_HPP
//--------------------------------------------------------------------------

#include <vector>

#include "data_types.hpp"
#include "csr_matrix.hpp"
#include "linear_solver_base.hpp"
#include "linsolv_iface_bos.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** @brief Exact block-local Schur-complement elimination (static
        condensation) of K cell-local equation/unknown pairs per block, wrapping
        an inner solver of block size M = N - K.

        This is a general linear-algebra transform, not tied to any physics. It
        applies to any block system that contains equations with a purely LOCAL
        stencil: equations whose Jacobian row is nonzero ONLY in the cell's own
        diagonal block (no coupling to neighbour cells / off-diagonal blocks).
        Such an equation and one of its unknowns can be removed *exactly* and
        *locally* by static condensation: the reduced (N-K) system keeps the
        SAME block sparsity, and the reduction / back-substitution are
        algebraically exact (the outer Newton iteration is unchanged to
        linear-solver tolerance). Because the leading equations of the reduced
        system are the retained (neighbour-coupled) ones, it also repairs the
        CPR True-IMPES pressure decoupling that a local-equation-first ordering
        would otherwise degrade.

        The motivating application is reactive-transport chemistry, where each
        mineral balance is exactly such a local equation (a precipitated solid
        does not flow, so its balance has no inter-cell flux term); but the same
        machinery serves any model whose equations exhibit a diagonal-block-only
        stencil.

        Per cell, with the K eliminated equation rows E and K eliminated unknown
        columns C (the kept rows F / kept columns G being the complements):

          P_i  = D_i[E, C]                    (K x K pivot block)
          Gm_i = P_i^-1 . D_i[E, keptCols]    (K x M)
          S_kj = A_kj[keptRows(k), keptCols]  - A_kj[keptRows(k), C] . Gm_j
          b_i  = b_i[keptRows] - sum_j A_ij[keptRows, C] . (P_j^-1 b_j[E])
          x_i[C] = P_i^-1 b_i[E] - Gm_i . x_i[keptCols]    (back-substitution)

        **Explicit selection, no hidden convention.** The eliminated rows and
        columns are supplied by the caller (spec / engine params); this class
        contains NO built-in assumption about which row/column is local (e.g. no
        hardcoded "row 0 / column 1"). The eliminated COLUMNS are global (the
        same K columns in every cell), so the kept-column map is global; only the
        eliminated ROWS may vary per cell: at some cells the preferred rows have
        no usable pivot (e.g. a well-head control equation), so a per-cell pivot
        search selects an alternative row set whose K x K block is invertible
        (typically the well-head state-copy rows). A selected row's off-diagonal
        entries (e.g. well-head to well-body couplings) form one-level dependency
        chains that are resolved exactly.

        Robustness: the row/column selection and chain topology are detected
        once (structural), but the pivot blocks and chain coefficients are
        re-read and re-validated on every Newton iteration, so a pivot that
        degenerates later (e.g. after a well control switch) fails the setup
        loudly rather than silently dividing by ~0.

        Both a host (CPU, OpenMP) and a device (GPU, CUDA) path are provided.

        @tparam N_BLOCK_SIZE full block size N.
        @tparam N_ELIM number of eliminated pairs K (1 <= K < N). M = N - K.
    */
    template <uint8_t N_BLOCK_SIZE, uint8_t N_ELIM>
    class linsolv_schur_elim : public opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>,
                               public opendarts::linear_solvers::linear_solver_base
    {
    public:
      static constexpr uint8_t K = N_ELIM;                     // eliminated pairs
      static constexpr uint8_t M_BLOCK_SIZE = N_BLOCK_SIZE - N_ELIM; // reduced block size
      static_assert(N_ELIM >= 1, "schur elimination needs at least one eliminated pair");
      static_assert(N_ELIM < N_BLOCK_SIZE, "cannot eliminate the whole block");

      /** @param on_device run the condensation / rhs / back-substitution on the
              GPU and hand the inner solver a device-resident reduced matrix.
          @param elim_rows preferred eliminated equation rows (size K). The
              i-th row is paired with the i-th eliminated column. At cells where
              this pairing is singular a per-cell fallback selects other rows.
          @param elim_cols eliminated unknown columns (size K), global. Must not
              contain the pressure column if a CPR-type inner is used (it is
              always kept so the pressure subsystem is preserved).
          @param pivot_eps pivots at or below this magnitude disqualify a
              candidate during detection and fail setup during a later solve. */
      linsolv_schur_elim(bool on_device,
                         const std::vector<int> &elim_rows,
                         const std::vector<int> &elim_cols,
                         double pivot_eps = 0.0);

      ~linsolv_schur_elim();

      using opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::init;
      using opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::setup;

      //////////////////////
      // linear_solver_base
      //////////////////////

      int solve(opendarts::linear_solvers::csr_matrix_base * /*matrix*/,
        opendarts::config::mat_float *v, opendarts::config::mat_float *r) override
      {
        return solve(v, r);
      }

      int init(opendarts::linear_solvers::csr_matrix_base *A,
        opendarts::config::index_t max_iters,
        opendarts::config::mat_float tolerance) override;

      int setup(opendarts::linear_solvers::csr_matrix_base *A) override;

      //////////////////////
      // linsolv_iface
      //////////////////////

      /// Inner solver on the reduced (N-K)-sized system. Set BEFORE init().
      /// Ownership is explicit: NOT owned by default (registry-built inners are
      /// shared_ptr-owned on the Python side); the engine-built GPU chain calls
      /// set_inner_owned(true) so the whole raw-pointer chain is released when
      /// the engine deletes its top-level solver (matching linsolv_gmres_gpu,
      /// which deletes its own preconditioner).
      int set_prec(opendarts::linear_solvers::linsolv_iface *prec_input) override
      {
        inner = prec_input;
        return 0;
      }

      /// Declare that this wrapper owns `inner` and must delete it (engine-built
      /// raw-pointer chains). Default false (Python/shared_ptr-owned inners).
      void set_inner_owned(bool owned) { inner_owned = owned; }

      int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_input, int max_iters, double tolerance) override
      {
        return this->init(static_cast<opendarts::linear_solvers::csr_matrix_base *>(A_input),
            static_cast<opendarts::config::index_t>(max_iters), static_cast<opendarts::config::mat_float>(tolerance));
      }
      int setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_input) override
      {
        return this->setup(static_cast<opendarts::linear_solvers::csr_matrix_base *>(A_input));
      }

      int solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X) override;

      int get_n_iters() override { return inner ? inner->get_n_iters() : 0; }
      opendarts::config::mat_float get_residual() override { return inner ? inner->get_residual() : 0; }
      void set_last_outer_iters(int n_iters) override { if (inner) inner->set_last_outer_iters(n_iters); }

    private:
      int detect_rows(const opendarts::config::mat_float *values_h);
      int condense_host(const opendarts::config::mat_float *values_h);
      int reduce_rhs_host(const opendarts::config::mat_float *B);
      int backsub_host(const opendarts::config::mat_float *B, opendarts::config::mat_float *X);
      void refresh_chain_coeffs(const opendarts::config::mat_float *values_at, bool values_on_device);
      void report_degenerate_cells(const opendarts::config::mat_float *values_h);
      void free_device();

      opendarts::linear_solvers::linsolv_iface *inner = nullptr;
      bool inner_owned = false;  // see set_inner_owned()
      opendarts::linear_solvers::csr_matrix<M_BLOCK_SIZE> *reduced = nullptr;
      opendarts::linear_solvers::csr_matrix_base *A_saved = nullptr;

      bool on_device = false;
      std::vector<uint8_t> elim_rows_pref;  // size K (spec preferred rows)
      std::vector<uint8_t> elim_cols;       // size K (global)
      std::vector<uint8_t> keep_cols;       // size M (global complement of elim_cols)
      double pivot_eps;
      bool detected = false;

      opendarts::config::index_t n_rows = 0;
      opendarts::config::index_t n_nnz = 0;

      // per block row: selected eliminated rows (K) and kept rows (M)
      std::vector<uint8_t> e_rows;  // n_rows * K
      std::vector<uint8_t> k_rows;  // n_rows * M

      // per block row factors: Pinv (K x K) and Gm (K x M); solve-time h (K)
      std::vector<opendarts::config::mat_float> Pinv;  // n_rows * K * K
      std::vector<opendarts::config::mat_float> Gm;    // n_rows * K * M
      std::vector<opendarts::config::mat_float> h_eff; // n_rows * K

      // one-level chains: eliminated row (cell i, local elim index e = 0..K-1)
      // carries an off-diagonal entry o at column dep_col of neighbour dep.
      struct chain_t
      {
        opendarts::config::index_t row;      // chained block row i
        opendarts::config::index_t dep;      // dependency block row j
        opendarts::config::index_t blk;      // nnz index of the (i, j) block
        opendarts::config::mat_float coeff;  // o = A_ij[e_rows(i)[eidx], dep_col]
        uint8_t eidx;                        // which eliminated row of i (0..K-1)
        uint8_t dep_col;                     // referenced column within dep's block
        bool dep_eliminated;                 // dep_col is an eliminated col of dep
        uint8_t dep_epos;                    // its index in elim_cols when eliminated
        uint8_t dep_kpos;                    // its index in keep_cols when kept
      };
      std::vector<chain_t> chains;
      // encoded (blk*N + row)*N + col offsets of ALL recorded chain entries,
      // sorted -- setup re-validates each Newton that no UNRECORDED nonzero has
      // appeared in an eliminated row (topology is value-detected once; a
      // structural entry that was 0.0 at detection and becomes nonzero later
      // would otherwise be silently dropped from the condensation)
      std::vector<unsigned long long> chain_offsets;
      // unique block rows whose per-cell factors / reduced solution the chain
      // handling touches (chain rows + dependencies): the GPU path copies ONLY
      // these rows across PCIe, not the full n_rows arrays
      std::vector<opendarts::config::index_t> chain_touched;
      struct chain_fix_t
      {
        opendarts::config::index_t src_blk;  // (k, i) block in A
        opendarts::config::index_t dst_blk;  // (k, dep) block in S
        opendarts::config::index_t row_k;    // block row k (for its k_rows)
        size_t chain_idx;                    // index into chains
      };
      std::vector<chain_fix_t> chain_fixes;

      std::vector<opendarts::config::mat_float> b_red;  // host scratch (host path)
      std::vector<opendarts::config::mat_float> x_red;

#ifdef WITH_GPU
      uint8_t *e_rows_d = nullptr;
      uint8_t *k_rows_d = nullptr;
      uint8_t *elim_cols_d = nullptr;
      uint8_t *keep_cols_d = nullptr;
      opendarts::config::mat_float *Pinv_d = nullptr;
      opendarts::config::mat_float *Gm_d = nullptr;
      opendarts::config::mat_float *h_eff_d = nullptr;
      opendarts::config::mat_float *b_red_d = nullptr;
      opendarts::config::mat_float *x_red_d = nullptr;
      int *flags_d = nullptr;             // [0] pivot degenerated, [1] unrecorded chain entry
      unsigned long long *chain_offsets_d = nullptr;
      std::vector<opendarts::config::mat_float> values_h_staging;
#endif
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_SCHUR_ELIM_HPP
//--------------------------------------------------------------------------
