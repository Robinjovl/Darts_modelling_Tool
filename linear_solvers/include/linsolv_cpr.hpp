//*************************************************************************
//    Copyright (c) 2026
//    Delft University of Technology, the Netherlands
//
//    This file is part of the open Delft Advanced Research Terra Simulator
//    (open-DARTS). It is distributed under the Apache License.
// *************************************************************************

//--------------------------------------------------------------------------
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_CPR_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_CPR_HPP
//--------------------------------------------------------------------------

#include <memory>

#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv.h"

#include "block_csr_matrix.hpp"
#include "cpr_block_ilu0.hpp"
#include "csr_matrix.hpp"
#include "csr_matrix_base.hpp"
#include "data_types.hpp"
#include "linsolv_iface_bos.hpp"
#include "scalar_csr_adapter.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** @brief Open-source two-stage Constrained Pressure Residual (CPR) preconditioner.
     *
     * The in-tree replacement for the proprietary ``linsolv_bos_cpr``: paired
     * with an outer Krylov solver (typically :class:`linsolv_gmres`) it forms
     * the open-source equivalent of the legacy ``bos_gmres + bos_cpr`` stack.
     *
     * Algorithm (forward, true-IMPES variant; Wallis 1983):
     *   1. Extract the scalar pressure subsystem ``A_p = R A C`` from the
     *      block-CSR Jacobian (same sparsity pattern). The restriction
     *      ``R_i = w_i^T`` uses per-row true-IMPES weights
     *      ``w_i = [1, -D_pf D_ff^{-1}]`` that decouple the non-pressure
     *      unknowns from the diagonal block ``D = A[i,i]``, so
     *      ``A_p[i,j] = sum_v w_i[v] * A[i,j][v, P_VAR]`` (quasi-IMPES, the bare
     *      (0,0) entry, is the per-row fallback when ``D_ff`` is singular). The
     *      stronger decoupling is what makes CPR converge on wide-stencil MPFA
     *      Jacobians, where quasi-IMPES stalls.
     *   2. Apply HYPRE BoomerAMG to ``A_p`` for the pressure correction
     *      ``x_p``; prolong to ``x_g`` (pressure component, zero elsewhere).
     *   3. Apply HYPRE ILU(0) to the full-system residual ``r_m = b - A x_g``
     *      for the global update ``x_f``.
     *   4. Return ``x = x_g + x_f``.
     *
     * HYPRE is driven directly here (no ``linsolv_hypre_amg`` /
     * ``linsolv_hypre_ilu`` wrappers): those wrappers are tuned for the
     * elasticity engines (mechanical systems) and would be the wrong AMG
     * configuration for flow. The BoomerAMG config mirrors
     * ``mgr::CompositionalFlowStrategy::setupPressureAMG`` -- aggressive PMIS
     * coarsening + multipass interpolation + C-F relaxation ordering,
     * configured as a preconditioner (MaxIter=1, Tol=0). HYPRE_ILU follows the
     * same proven-on-flow defaults.
     *
     * Transpose (CPRA, Han et al. 2013) -- needed for the adjoint Newton step:
     * the same two-stage structure with the order reversed and each stage
     * applied as its transpose (``M̃^T`` then ``(A_p)^T``). Hooked up via the
     * ``linear_solver::solve_transposed`` entry point. Forward CPR and the
     * CPRA-transposed solve are both implemented: ``Ap_T_`` and ``As_T_``
     * are materialised on first setup with their own ``HYPRE_Solver``
     * handles (BoomerAMG, HYPRE_ILU) because ``HYPRE_BoomerAMGSolveT`` has
     * limited compatibility with our aggressive-coarsening flow config and
     * ``HYPRE_ILU`` has no transpose-solve entry point. See
     * :meth:`solve_transposed`.
     *
     * Canonical CPRA path for the open-source adjoint Newton step
     * ----------------------------------------------------------
     * ``linsolv_cpr::solve_transposed`` (above) is the open-DARTS canonical
     * CPRA implementation -- the one exposed through
     * ``CPRSolverSpec`` / ``GMRESSolverSpec(prec=CPRSolverSpec())`` and
     * driven by ``Adjoint_super_engine``'s ``cpra`` mode. The MGR-internal
     * ``applyBCSRCPRTransposePreconditioner`` (``solvers/src/mgr_linear_solver.cpp``)
     * is kept alongside it pending the SPE10 benchmark comparison Xiaoming
     * promised on MR #280; once that lands the MGR-internal path is to be
     * retired in favour of this one. See ``SOLVER_REFACTORING_PLAN.md``.
     *
     * The matrix is consumed through :class:`csr_matrix_base` accessors, so
     * this works against both the legacy ``csr_matrix<N>`` and the unified
     * ``block_csr_matrix`` Jacobian.
     */
    template <uint8_t N_BLOCK_SIZE>
    class linsolv_cpr : public opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>
    {
    public:
      linsolv_cpr();
      ~linsolv_cpr() override;

      using opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::init;
      using opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::setup;

      int set_prec(opendarts::linear_solvers::linsolv_iface * /*prec_input*/) override
      {
        return 0;  // CPR composes its own inner stages internally
      }

      int init(opendarts::linear_solvers::csr_matrix_base *A,
          int max_iters,
          opendarts::config::mat_float tolerance) override;

      int setup(opendarts::linear_solvers::csr_matrix_base *A_input) override;

      // iface_bos pure virtuals -- the csr_matrix_base* variants above are used.
      int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> * /*A*/,
          int /*max_iters*/,
          double /*tolerance*/) override
      {
        return 0;
      }
      int setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> * /*A*/) override
      {
        return 0;
      }

      int solve(opendarts::config::mat_float *B,
          opendarts::config::mat_float *X) override;

      int solve_transposed(opendarts::config::mat_float *B,
          opendarts::config::mat_float *X) override;

      int get_n_iters() override { return n_iters_; }
      opendarts::config::mat_float get_residual() override { return 0.0; }

      /** Maximum HYPRE-AMG iterations on the pressure subsystem per CPR apply.
       *  AMG is used as a preconditioner stage -- a few sweeps usually suffice. */
      void set_amg_max_iters(int n) { amg_max_iters_ = n; }
      void set_ilu_fill_level(int k) { ilu_fill_level_ = k; }

      /** Pressure-stage BoomerAMG configuration (see cpr_solver_config for the
       *  field semantics; negative values keep the HYPRE built-in default). */
      void set_pressure_amg_options(int coarsen_type, int interp_type,
          int relax_type, int relax_order, int num_sweeps,
          double strong_threshold, int agg_num_levels, int agg_interp_type,
          int agg_pmax_elmts, int pmax_elmts, double trunc_factor,
          int max_levels, int cycle_type, int max_coarse_size,
          int coarse_relax_type, double relax_wt)
      {
        amg_coarsen_type_ = coarsen_type;
        amg_interp_type_ = interp_type;
        amg_relax_type_ = relax_type;
        amg_relax_order_ = relax_order;
        amg_num_sweeps_ = num_sweeps;
        amg_strong_threshold_ = strong_threshold;
        amg_agg_num_levels_ = agg_num_levels;
        amg_agg_interp_type_ = agg_interp_type;
        amg_agg_pmax_elmts_ = agg_pmax_elmts;
        amg_pmax_elmts_ = pmax_elmts;
        amg_trunc_factor_ = trunc_factor;
        amg_max_levels_ = max_levels;
        amg_cycle_type_ = cycle_type;
        amg_max_coarse_size_ = max_coarse_size;
        amg_coarse_relax_type_ = coarse_relax_type;
        amg_relax_wt_ = relax_wt;
      }

      /** Build the CPRA transpose hierarchies on every setup() from the start
       *  (historical behaviour) instead of lazily on the first
       *  solve_transposed() call. */
      void set_eager_adjoint(bool eager) { eager_adjoint_ = eager; }

      /** Live reconfiguration (see linear_solver::reconfigure).
       *  Hot: amg_max_iters, weight_scheme, eager_adjoint, reuse/adaptive
       *  policy knobs. Warm: the pressure-AMG profile and ilu_fill_level --
       *  re-applied to the existing HYPRE handles with a forced hierarchy
       *  rebuild on the next setup() (same bound matrices). Structural
       *  (returns 1, caller rebuilds): changing stage2_type on a live
       *  solver. */
      int reconfigure(const opendarts::linear_solvers::solver_config &config) override;

      /** Pressure-decoupling weight scheme: 0 = diagonal-block (quasi-IMPES
       *  with local f-elimination, historical behaviour), 1 = column-sum
       *  (True-IMPES, Wallis 1983 -- what the proprietary linsolv_bos_cpr
       *  computes). */
      void set_weight_scheme(int scheme) { weight_scheme_ = scheme; }

      /** Full-system smoothing stage: 0 = HYPRE scalar ILU(k) on the
       *  expanded system (historical behaviour), 1 = in-tree block ILU(0)
       *  on the block-CSR system (dense NxN block inverses -- the
       *  proprietary csr_ilu_prec equivalent). */
      void set_stage2_type(int type) { stage2_type_ = type; }

    private:
      // Unguarded implementations of the public entry points. The public
      // setup()/solve()/solve_transposed() wrap these in try/catch and
      // translate a thrown HYPRE failure (see check_hypre in the .cpp) into
      // a nonzero return, which engine_base::solve_linear_equation() turns
      // into a timestep cut instead of aborting the host process.
      int setup_unguarded(opendarts::linear_solvers::csr_matrix_base *A_input);
      int solve_unguarded(opendarts::config::mat_float *B,
          opendarts::config::mat_float *X);
      int solve_transposed_unguarded(opendarts::config::mat_float *B,
          opendarts::config::mat_float *X);

      // Extract the scalar pressure subsystem A_p (block (0,0) of each block)
      // into a csr_matrix<1> sharing the block-CSR structure.
      void build_pressure_subsystem(opendarts::linear_solvers::csr_matrix_base *A);

      // (Re)populate the scalar expansion As_ of the full block system
      // (structure once, values every call). Used by setup() when the HYPRE
      // scalar-ILU stage or the adjoint chain needs it, and by
      // activate_adjoint_chain() when the block-ILU0 stage skipped it.
      void refresh_scalar_expansion(opendarts::linear_solvers::csr_matrix_base *A_input);

      // Build a HYPRE IJ matrix from a scalar csr_matrix<1> (sequential, single rank).
      void build_hypre_ij(opendarts::linear_solvers::csr_matrix<1> &A,
          HYPRE_IJMatrix &A_ij,
          HYPRE_ParCSRMatrix &A_parcsr);

      // Create the b/x HYPRE IJ vectors for a system of n_rows scalar unknowns.
      void create_hypre_vectors(opendarts::config::index_t n_rows,
          HYPRE_IJVector &b_ij,
          HYPRE_IJVector &x_ij);

      // Set the values of a HYPRE IJ vector from a raw scalar array.
      void set_hypre_vector(HYPRE_IJVector &v_ij,
          opendarts::config::index_t n_rows,
          opendarts::config::mat_float *vals,
          HYPRE_ParVector &v_par);

      // Refresh an already-created HYPRE IJ matrix with new values (same
      // sparsity). Returns the underlying ParCSR handle.
      void refresh_hypre_ij(opendarts::linear_solvers::csr_matrix<1> &A,
          HYPRE_IJMatrix &A_ij,
          HYPRE_ParCSRMatrix &A_parcsr);

      // Destroy HYPRE handles created in setup() (idempotent -- guarded by nullptr).
      void destroy_hypre();

      opendarts::linear_solvers::csr_matrix_base *A_;  // full-system matrix (kept by pointer)

      // Scalar-CSR view of A_ when the engine hands us the new block_csr_matrix
      // Jacobian (the canonical layout post-plan-§12 phase B). The adapter
      // owns the scalar value buffer (gathered from the block values per
      // setup() via a vectorisable O(nnz) permutation) and borrows the
      // expanded structure from the matrix's cached sparsity_pattern. As_
      // (below) is populated from the adapter on first setup and refreshed
      // from adapter->values() each subsequent setup, replacing the per-
      // Newton csr_matrix<1>::to_nb_1 nested-loop scalar expansion. Null
      // when A_ is a legacy csr_matrix<N> (proprietary build / tests); the
      // setup() path then falls back to the legacy to_nb_1.
      std::unique_ptr<opendarts::linear_solvers::scalar_csr_adapter> scalar_adapter_;

      // Pressure subsystem A_p as a scalar matrix + its HYPRE handles.
      std::unique_ptr<opendarts::linear_solvers::csr_matrix<1>> Ap_;
      HYPRE_IJMatrix Ap_ij_;
      HYPRE_ParCSRMatrix Ap_parcsr_;
      HYPRE_IJVector amg_b_ij_, amg_x_ij_;
      HYPRE_ParVector amg_b_par_, amg_x_par_;
      HYPRE_Solver amg_;
      bool amg_setup_done_;

      // Transposed pressure subsystem A_p^T and a second AMG hierarchy on it
      // (for CPRA solve_transposed). HYPRE_BoomerAMGSolveT has limited
      // relax/coarsening compatibility, so a separate setup on the transposed
      // matrix is more robust and reuses the same proven flow config.
      std::unique_ptr<opendarts::linear_solvers::csr_matrix<1>> Ap_T_;
      HYPRE_IJMatrix Ap_T_ij_;
      HYPRE_ParCSRMatrix Ap_T_parcsr_;
      HYPRE_IJVector amg_T_b_ij_, amg_T_x_ij_;
      HYPRE_ParVector amg_T_b_par_, amg_T_x_par_;
      HYPRE_Solver amg_T_;
      bool amg_T_setup_done_;

      // Scalar-expanded full-system A_s (size n_block_rows * N_BLOCK_SIZE) and
      // its HYPRE-ILU handles. HYPRE_ILU runs on the scalar expansion produced
      // by the polymorphic ``csr_matrix<1>::to_nb_1(csr_matrix_base*)`` helper.
      std::unique_ptr<opendarts::linear_solvers::csr_matrix<1>> As_;
      HYPRE_IJMatrix As_ij_;
      HYPRE_ParCSRMatrix As_parcsr_;
      HYPRE_IJVector ilu_b_ij_, ilu_x_ij_;
      HYPRE_ParVector ilu_b_par_, ilu_x_par_;
      HYPRE_Solver ilu_;
      bool ilu_setup_done_;

      // Transposed scalar expansion A_s^T and a second HYPRE_ILU on it --
      // HYPRE_ILU does not expose a transpose-solve entry point, so the
      // CPRA-transposed apply factorises A^T separately. BoomerAMG, in
      // contrast, exposes HYPRE_BoomerAMGSolveT and reuses the forward setup,
      // so there is no separate Ap_T_ matrix.
      std::unique_ptr<opendarts::linear_solvers::csr_matrix<1>> As_T_;
      HYPRE_IJMatrix As_T_ij_;
      HYPRE_ParCSRMatrix As_T_parcsr_;
      HYPRE_IJVector ilu_T_b_ij_, ilu_T_x_ij_;
      HYPRE_ParVector ilu_T_b_par_, ilu_T_x_par_;
      HYPRE_Solver ilu_T_;
      bool ilu_T_setup_done_;

      // Pressure-component index inside each block (always 0 in DARTS).
      static constexpr int P_VAR = 0;

      // Configuration.
      int amg_max_iters_;
      int ilu_fill_level_;
      int max_iters_;
      opendarts::config::mat_float tolerance_;
      int n_iters_;

      // Workspace (scalar arrays of total size n_block_rows * N).
      std::vector<opendarts::config::mat_float> wksp_;

      // True-IMPES pressure-decoupling weights, one row of N per block-row:
      // cpr_weights_[i*N + v] is the weight of variable v in block-row i's
      // decoupled pressure equation (cpr_weights_[i*N + P_VAR] == 1). Built in
      // build_pressure_subsystem() and applied to the residual restriction
      // (forward solve) and the prolongation (transposed CPRA solve).
      std::vector<opendarts::config::mat_float> cpr_weights_;

      // Cached HYPRE-IJ scratch buffers. The row index list passed to
      // HYPRE_IJVector*/HYPRE_IJMatrix* is just [0, n). Once init() bound the
      // matrices, both sizes (block-row and scalar-row) and the per-matrix
      // row degrees are stable, so caching avoids two heap allocations per
      // set_hypre_vector / build_hypre_ij / refresh_hypre_ij call.
      // row_indices_ is grown monotonically to the max of the two sizes.
      std::vector<opendarts::config::index_t> row_indices_;
      std::vector<opendarts::config::index_t> n_cols_Ap_;
      std::vector<opendarts::config::index_t> n_cols_As_;
      std::vector<opendarts::config::index_t> n_cols_Ap_T_;
      std::vector<opendarts::config::index_t> n_cols_As_T_;

      // Grow row_indices_ (monotonic) and fill the new tail with the iota.
      void ensure_row_indices(opendarts::config::index_t n);

      // Hierarchy reuse policy. Mirrors mgr::SolverParameters' BCSR-CPR knobs:
      //   reuse_amg_hierarchy:        skip BoomerAMG (and ILU) Setup if the
      //                               last solve converged within
      //                               adaptive_iter_threshold iterations.
      //   adaptive_amg_rebuild:       once adaptive_consecutive_bad solves in
      //                               a row exceed the threshold, force a
      //                               rebuild on the next setup() and clear
      //                               the bad-streak counter.
      //   adaptive_iter_threshold:    iteration-count threshold per solve.
      //   adaptive_consecutive_bad:   how many bad solves in a row before
      //                               forcing a rebuild.
      // Default: reuse OFF (preserves the current per-Newton rebuild behaviour
      // exactly). Setters live on the Spec class; flipping reuse_amg_hierarchy
      // is the single knob most flow runs benefit from.
      bool reuse_amg_hierarchy_;
      bool adaptive_amg_rebuild_;
      int adaptive_iter_threshold_;
      int adaptive_consecutive_bad_;
      // Number of outer-Krylov iterations the previous solve took. The outer
      // solver (linsolv_gmres) calls back via set_last_outer_iters() to feed
      // the adaptive rebuild policy; 0 disables the policy (used for the
      // first solve where no history exists).
      int last_outer_iters_;
      int consecutive_bad_streak_;
      // Set to true when the next setup() must rebuild rather than reuse.
      bool force_amg_rebuild_;

      // Pressure-stage BoomerAMG options (see cpr_solver_config; negative =
      // keep the HYPRE built-in default). Defaults reproduce the historical
      // configuration: HYPRE defaults + one aggressive-coarsening level with
      // multipass-family interpolation + C/F relaxation ordering.
      int amg_coarsen_type_ = -1;
      int amg_interp_type_ = -1;
      int amg_relax_type_ = -1;
      int amg_relax_order_ = 1;
      int amg_num_sweeps_ = -1;
      double amg_strong_threshold_ = -1.0;
      int amg_agg_num_levels_ = 1;
      int amg_agg_interp_type_ = 6;
      int amg_agg_pmax_elmts_ = 20;
      int amg_pmax_elmts_ = -1;
      double amg_trunc_factor_ = -1.0;
      int amg_max_levels_ = -1;
      int amg_cycle_type_ = -1;
      int amg_max_coarse_size_ = -1;
      int amg_coarse_relax_type_ = -1;
      double amg_relax_wt_ = -1.0;

      // CPRA transpose-chain policy: with eager_adjoint_ == false (default)
      // the A_p^T / A_s^T transposes and their BoomerAMG / HYPRE_ILU
      // hierarchies are not touched by setup() until the first
      // solve_transposed() call flips adjoint_active_; from then on every
      // setup() keeps them refreshed alongside the forward chain.
      bool eager_adjoint_ = false;
      bool adjoint_active_ = false;

      // Pressure-decoupling weight scheme (see set_weight_scheme): 1 =
      // column-sum True-IMPES (BOS parity, default), 0 = diagonal-block.
      int weight_scheme_ = 1;
      // Row sign normalisation of A_p (+1 / -1 per row so the pressure
      // diagonal is positive -- helps AMG's M-matrix-oriented heuristics;
      // mirrors linsolv_bos_cpr's rhs_mults). Applied to the restricted
      // residual on the forward path and to the prolonged solution on the
      // transposed path.
      std::vector<opendarts::config::mat_float> rhs_mults_;
      // Column-sum accumulator for the True-IMPES weights (n_rows x N x N).
      std::vector<opendarts::config::mat_float> colsum_;

      // Full-system stage selector (see set_stage2_type): 1 = in-tree block
      // ILU(0) (BOS parity, default), 0 = HYPRE scalar ILU(k).
      int stage2_type_ = 1;
      std::unique_ptr<opendarts::linear_solvers::cpr_block_ilu0<N_BLOCK_SIZE>>
          bilu0_;
      bool bilu0_ready_ = false;

      // Create + configure a BoomerAMG solver handle for the pressure stage
      // (shared by the forward and transpose hierarchies).
      void create_pressure_amg(HYPRE_Solver &amg, const char *tag);
      // (Re)apply the pressure-AMG options to an existing handle -- used at
      // creation and by reconfigure() on live handles (BoomerAMG reads its
      // parameters at Setup time, so Set* + a forced re-Setup is the
      // supported way to retune a hierarchy in place).
      void apply_pressure_amg_options(HYPRE_Solver &amg, const char *tag);
      // Create + configure a HYPRE_ILU handle for the full-system stage.
      void create_fullsystem_ilu(HYPRE_Solver &ilu, const char *tag);
      // Build the transpose matrices + hierarchies (first solve_transposed()
      // on the lazy path, or every first setup() on the eager path).
      void activate_adjoint_chain();
public:
      /// Hierarchy-reuse policy: skip BoomerAMG/ILU Setup on subsequent
      /// Newton iterations when the previous solve converged in fewer than
      /// adaptive_iter_threshold outer iterations. Halves the per-Newton CPR
      /// setup cost on well-converging timesteps.
      void set_reuse_amg_hierarchy(bool enable) { reuse_amg_hierarchy_ = enable; }
      /// Adaptive rebuild: after `consecutive_bad` solves over the threshold
      /// in a row, force a single rebuild on the next setup().
      void set_adaptive_amg_rebuild(bool enable, int iter_threshold = 15,
          int consecutive_bad = 2)
      {
        adaptive_amg_rebuild_ = enable;
        adaptive_iter_threshold_ = iter_threshold;
        adaptive_consecutive_bad_ = consecutive_bad;
      }
      /// Outer solver hook -- the outer Krylov (or external client) reports
      /// the iteration count of the previous solve so the policy can decide
      /// whether to reuse on the next setup. 0 means "no history".
      void set_last_outer_iters(int n) override { last_outer_iters_ = n; }
private:

      // True until the first setup() completes. After the first setup the
      // HYPRE handles are reused -- a destroy/create cycle on every Newton
      // iteration regressed `2ph_comp` (BoomerAMGSetup crashed on the second
      // call), so the post-first path only refreshes IJ matrix values and
      // re-runs `*Setup` on the same handles.
      bool first_setup_;
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_CPR_HPP
//--------------------------------------------------------------------------
