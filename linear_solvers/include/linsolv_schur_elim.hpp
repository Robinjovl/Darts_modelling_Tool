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
    /** @brief Exact per-cell Schur elimination of one flux-free equation/unknown
        pair per block, wrapping an inner solver of block size N-1.

        Chemistry (reactive-transport) block systems carry mineral balance
        equations that have no flux terms: their block rows are nonzero ONLY in
        the diagonal block. Eliminating one unknown through such a row is a
        purely local static condensation: the reduced (N-1)x(N-1) system has the
        SAME block sparsity pattern, and the reduction/back-substitution are
        exact (no approximation). The inner solver then works on a smaller
        system whose first equation is a fluid balance -- which also repairs the
        True-IMPES pressure decoupling of CPR-type preconditioners that the
        mineral-first ordering degrades.

        Row selection: per block row i, the eliminated equation row r(i) is
        auto-detected at the first setup (zero flux rows are a property of the
        assembled values, not of the pattern):
          - default: row ``elim_row`` (the mineral balance, row 0);
          - fallback (well heads, where row 0 is the control equation): another
            row with a usable pivot, preferring rows without off-diagonal
            entries; a fallback row's off-diagonal entries (well-head state-copy
            rows) form one-level dependency chains that are resolved exactly.

        Column selection (the pivot): default ``elim_col = 1`` eliminates the
        mineral z (minerals-first ordering) -- the natural pairing, since the
        mineral balance is the equation that determines the mineral unknown.
        Although that pivot can be small in magnitude (kinetics-dominated
        rows), it is balanced by the equally small mineral COLUMN, so the
        condensation growth stays bounded. ``elim_col = -1`` is an EXPERIMENTAL
        per-row max-magnitude auto pivot; measured on carbonated_water it can
        leave the mineral unknown nearly decoupled in the reduced system
        (near-singular preconditioner setups) -- prefer the default. The
        pressure column (0) is always kept, so CPR-type inner preconditioners
        are unaffected.

        Both a host (CPU, OpenMP) and a device (GPU, CUDA) path are provided;
        the device path keeps the reduced matrix and all per-cell factors on
        the GPU and matches the engine's device-pointer solve(B_d, X_d) call
        convention. Single mineral only (one eliminated pair per block).
    */
    template <uint8_t N_BLOCK_SIZE>
    class linsolv_schur_elim : public opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>,
                               public opendarts::linear_solvers::linear_solver_base
    {
    public:
      static constexpr uint8_t M_BLOCK_SIZE = N_BLOCK_SIZE - 1; // reduced block size

      /** @param on_device run the condensation and rhs/back-substitution on the
              GPU and hand the inner solver a device-resident reduced matrix
              (engine GPU path); false = host path (CPU chains).
          @param elim_col eliminated unknown column within the block: default 1
              = the mineral z (the natural pairing, see class docs); -1 selects,
              per cell, the largest-magnitude pivot among the non-pressure
              columns of the eliminated row (experimental; column 0 = pressure
              is never eliminable).
          @param elim_row preferred eliminated equation row (default 0 = the
              mineral balance with minerals-first component ordering).
          @param pivot_eps pivots at or below this magnitude disqualify a
              row/column candidate during detection. */
      linsolv_schur_elim(bool on_device = false, int elim_col = 1, uint8_t elim_row = 0,
                         double pivot_eps = 0.0);

      ~linsolv_schur_elim();

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

      /// Build the reduced pattern from A's structure and init the inner solver
      /// on the reduced matrix (structure only; values are condensed in setup).
      int init(opendarts::linear_solvers::csr_matrix_base *A,
        opendarts::config::index_t max_iters,
        opendarts::config::mat_float tolerance) override;

      /// Condense A into the reduced matrix (detecting the per-row eliminated
      /// row/column on the first call) and set up the inner solver on it.
      int setup(opendarts::linear_solvers::csr_matrix_base *A) override;

      //////////////////////
      // linsolv_iface
      //////////////////////

      /// The inner solver operating on the reduced (N-1)-sized system. Must be
      /// set BEFORE init(). NOT owned: registry-built inners are shared_ptr-owned
      /// on the Python side, engine-built chains are never freed by convention.
      int set_prec(opendarts::linear_solvers::linsolv_iface *prec_input) override
      {
        inner = prec_input;
        return 0;
      }

      int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_input,
        int max_iters,
        double tolerance) override
      {
        return this->init(static_cast<opendarts::linear_solvers::csr_matrix_base *>(A_input),
            static_cast<opendarts::config::index_t>(max_iters), static_cast<opendarts::config::mat_float>(tolerance));
      }

      int setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_input) override
      {
        return this->setup(static_cast<opendarts::linear_solvers::csr_matrix_base *>(A_input));
      }

      /// Reduce B, solve the condensed system with the inner solver, and
      /// back-substitute the eliminated unknowns into X (full N-block layout).
      /// Pointers are device pointers when constructed with on_device=true,
      /// host pointers otherwise (matching the engine calling convention).
      int solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X) override;

      int get_n_iters() override { return inner ? inner->get_n_iters() : 0; }

      opendarts::config::mat_float get_residual() override
      {
        return inner ? inner->get_residual() : 0;
      }

      /// Forward the outer-iteration feedback (adaptive AMGX hierarchy reuse).
      void set_last_outer_iters(int n_iters) override
      {
        if (inner)
          inner->set_last_outer_iters(n_iters);
      }

    private:
      // ---- host-side detection + condensation helpers ----
      int detect_rows(const opendarts::config::mat_float *values_h);
      int condense_host(const opendarts::config::mat_float *values_h);
      int reduce_rhs_host(const opendarts::config::mat_float *B);
      int backsub_host(const opendarts::config::mat_float *B, opendarts::config::mat_float *X);
      void apply_chain_fixes(const opendarts::config::mat_float *values_at,
        opendarts::config::mat_float *red_values_at, bool values_on_device);
      // Re-read chain coefficients (ch.coeff) from the CURRENT values each setup:
      // detection (row/column selection + chain topology) is structural and runs
      // once, but the chain entry MAGNITUDES change with the Jacobian, so freezing
      // them would make the reduction inexact for state-dependent well rows.
      void refresh_chain_coeffs(const opendarts::config::mat_float *values_at, bool values_on_device);
      // Validate that every eliminated pivot is still usable on the current
      // values (detection only checked the first setup). Returns false + prints
      // if any |pivot| <= pivot_eps, so setup fails loudly instead of dividing by
      // ~0 and feeding NaN to the inner solver.
      bool check_pivots_host();
      void free_device();

      opendarts::linear_solvers::linsolv_iface *inner = nullptr;
      opendarts::linear_solvers::csr_matrix<M_BLOCK_SIZE> *reduced = nullptr;
      opendarts::linear_solvers::csr_matrix_base *A_saved = nullptr;

      bool on_device = false;
      int elim_col;       // -1 = per-row auto pivot column; >=1 fixed column
      uint8_t elim_row;   // preferred eliminated equation row
      double pivot_eps;
      bool detected = false;

      opendarts::config::index_t n_rows = 0;
      opendarts::config::index_t n_nnz = 0;

      // per block row: selected eliminated equation row + column, kept-row and
      // kept-column maps [M] (original index of each kept row/column)
      std::vector<uint8_t> r_sel;
      std::vector<uint8_t> c_sel;
      std::vector<uint8_t> rmap; // n_rows * M
      std::vector<uint8_t> cmap; // n_rows * M

      // per block row factors: pivot p_i = A_ii[r_sel, c_sel] and
      // g_i[M] = A_ii[r_sel, cmap]/p_i
      std::vector<opendarts::config::mat_float> pivot;
      std::vector<opendarts::config::mat_float> g;
      // solve-time affine constants h_eff[i] (built from B each solve)
      std::vector<opendarts::config::mat_float> h_eff;

      // one-level chains: eliminated row of block-row i carries an entry o at
      // column dep_col of neighbor dep (well-head state-copy rows). The entry
      // references either dep's ELIMINATED unknown (dep_col == c_sel[dep]) or
      // one of dep's KEPT unknowns (dep_pos = its position in cmap[dep]).
      struct chain_t
      {
        opendarts::config::index_t row;      // chained block row i
        opendarts::config::index_t dep;      // dependency block row j
        opendarts::config::index_t blk;      // nnz index of the (i, j) block
        opendarts::config::mat_float coeff;  // o = A_ij[r_sel(i), dep_col]
        uint8_t dep_col;                     // referenced column within dep's block
        bool dep_eliminated;                 // dep_col == c_sel[dep]
        uint8_t dep_pos;                     // kept-col position when !dep_eliminated
      };
      std::vector<chain_t> chains;
      // reduced-matrix corrections induced by chains, applied to block (k, dep)
      // for every pattern block (k, chained row)
      struct chain_fix_t
      {
        opendarts::config::index_t src_blk;  // (k, i) block in A
        opendarts::config::index_t dst_blk;  // (k, dep) block in S
        opendarts::config::index_t row_k;    // block row k (for its rmap)
        size_t chain_idx;                    // index into chains
      };
      std::vector<chain_fix_t> chain_fixes;

      // host scratch for the reduced rhs / solution (host path)
      std::vector<opendarts::config::mat_float> b_red;
      std::vector<opendarts::config::mat_float> x_red;

#ifdef WITH_GPU
      // device mirrors (GPU path)
      uint8_t *r_sel_d = nullptr;
      uint8_t *c_sel_d = nullptr;
      uint8_t *rmap_d = nullptr;
      uint8_t *cmap_d = nullptr;
      opendarts::config::mat_float *pivot_d = nullptr;
      opendarts::config::mat_float *g_d = nullptr;
      opendarts::config::mat_float *h_eff_d = nullptr;
      opendarts::config::mat_float *b_red_d = nullptr;
      opendarts::config::mat_float *x_red_d = nullptr;
      int *pivot_bad_d = nullptr;   // device flag: se_factors_kernel sets it if |pivot| <= eps
      // host staging buffer for value download during detection; released once
      // detection has run (it is only needed for the first setup)
      std::vector<opendarts::config::mat_float> values_h_staging;
#endif
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_SCHUR_ELIM_HPP
//--------------------------------------------------------------------------
