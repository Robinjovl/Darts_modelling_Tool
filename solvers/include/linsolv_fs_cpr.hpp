//*************************************************************************
//    Copyright (c) 2026
//    Delft University of Technology, the Netherlands
//
//    This file is part of the open Delft Advanced Research Terra Simulator
//    (open-DARTS). It is distributed under the Apache License.
// *************************************************************************

//--------------------------------------------------------------------------
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_FS_CPR_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_FS_CPR_HPP
//--------------------------------------------------------------------------

// linsolv_fs_cpr -- open-source port of the proprietary FS-CPR (Full-System
// CPR poromechanics) preconditioner.
//
// Two-stage poromechanics CPR: an HYPRE-BoomerAMG correction on the
// displacement subsystem (U) and a second correction on the flow / pressure
// subsystem (P or PPSS, depending on NE), driven via a Schur complement
// approximation.
//
// Algorithm (FS_UP variant):
//   setup():
//     - extract scalar U subsystem from A's block layout and refresh /
//       rebuild the U-preconditioner (HYPRE-AMG)
//     - build Schur complement approximation on the P (or PPSS) subsystem,
//       extract it as scalar / block CSR, and refresh / rebuild the
//       P-preconditioner
//   solve(B, X):
//     - solve for the pressure correction P_X via P-prec
//     - back-substitute the displacement residual U_B = B|U - A_UP*P_X
//       (- A_US*P_X if NE>1)
//     - solve for the displacement correction U_X via U-prec
//     - scatter (P_X, U_X) into X
//
// Only the FS_UP path is supported -- FS_UPG (contact-mechanics / gap
// subsystem) is deprecated and has been removed; init() returns -1 if
// n_fracs != 0.
//
// Sub-preconditioners are injected via set_prec() (shared ownership) -- the
// canonical choice for both stages is linsolv_hypre_amg<1>; on each Newton
// iteration the sub-prec is refresh()'d (value-only) when possible and
// fully re-setup() only on update_uu_ events or when the sub-prec doesn't
// expose refresh().

#include <cstdint>
#include <memory>
#include <vector>

#include "block_csr_matrix.hpp"
#include "csr_matrix.hpp"
#include "csr_matrix_base.hpp"
#include "data_types.hpp"
#include "linsolv_iface.hpp"
#include "linsolv_iface_bos.hpp"
#include "matrix_slice.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    template <std::uint8_t N_BLOCK_SIZE>
    class linsolv_fs_cpr : public opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>
    {
    public:
      using index_t = opendarts::config::index_t;
      using mat_float = opendarts::config::mat_float;

      // 4-arg constructor matching the existing stub linsolv_bos_fs_cpr
      // signature so the engine factory can drop-in this class as a
      // replacement. ND = 3 is hardcoded for 3D mechanics. NE =
      // N_BLOCK_SIZE - ND is the number of flow / composition equations.
      // NC is kept for API parity with the proprietary 4-arg overload.
      linsolv_fs_cpr(std::uint8_t P_VAR, std::uint8_t Z_VAR,
          std::uint8_t U_VAR, std::uint8_t NC);

      ~linsolv_fs_cpr() override = default;

      // Make hidden-overload warnings quiet.
      using opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::init;
      using opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::setup;

      // ----------------------------------------------------------------
      // Sub-preconditioner injection (FS_UP only; G overload removed).
      // ----------------------------------------------------------------

      int set_prec(std::shared_ptr<opendarts::linear_solvers::linsolv_iface> prec_p);
      int set_prec(std::shared_ptr<opendarts::linear_solvers::linsolv_iface> prec_p,
          std::shared_ptr<opendarts::linear_solvers::linsolv_iface> prec_u);

      // Required by linsolv_iface base -- raw-pointer set_prec stores
      // prec_input as the p-prec (matches proprietary single-prec
      // overload). Ownership is non-owning (a shared_ptr aliasing the raw
      // pointer with a no-op deleter).
      int set_prec(opendarts::linear_solvers::linsolv_iface *prec_input) override;

      // Engine-side partition injection. Must be called BEFORE init().
      void set_block_sizes(index_t n_res, index_t n_fracs, index_t n_wells);

      // External "U sparsity changed" signal -- forces a full AMG setup
      // (not just refresh) on the next setup() call.
      void do_update_uu()
      {
        update_uu_ = true;
      }

      // BoomerAMG asymmetric workaround toggle (default true to match
      // proprietary). When true, the first row of each extracted scalar
      // subsystem is doubled to defeat HYPRE's symmetric-detection
      // heuristic.
      void set_force_amg_asymmetric(bool b)
      {
        force_amg_asymmetric_ = b;
      }

      // ----------------------------------------------------------------
      // Polymorphic csr_matrix_base entry points. These bypass the UB
      // static_cast in linsolv_iface_bos<N> when A is a block_csr_matrix.
      // Mirror the linsolv_cusparse_ilu pattern.
      // ----------------------------------------------------------------

      int init(opendarts::linear_solvers::csr_matrix_base *A,
          opendarts::config::index_t max_iters,
          opendarts::config::mat_float tolerance) override;

      int setup(opendarts::linear_solvers::csr_matrix_base *A) override;

      // Typed overloads forward to the csr_matrix_base entry points.
      int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A,
          int max_iters, double tolerance) override
      {
        return this->init(static_cast<opendarts::linear_solvers::csr_matrix_base *>(A),
            static_cast<opendarts::config::index_t>(max_iters),
            static_cast<opendarts::config::mat_float>(tolerance));
      }

      int setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A) override
      {
        return this->setup(static_cast<opendarts::linear_solvers::csr_matrix_base *>(A));
      }

      int solve(opendarts::config::mat_float *B,
          opendarts::config::mat_float *X) override;

      int get_n_iters() override
      {
        return 1;
      }

      opendarts::config::mat_float get_residual() override
      {
        return 0.0;
      }

    private:
      // Constants
      static constexpr std::uint8_t ND = 3;
      const std::uint8_t P_VAR_;
      const std::uint8_t Z_VAR_;
      const std::uint8_t U_VAR_;
      const std::uint8_t NC_;
      const std::uint8_t NE_; // = N_BLOCK_SIZE - ND

      // Partition (set via set_block_sizes BEFORE init()).
      index_t n_res_ = 0;
      index_t n_fracs_ = 0;
      index_t n_wells_ = 0;

      // The 9 FS_UP-relevant sub-block slices + the joint PPSS slice.
      // The G-row/col slices (UG, PG, GU, GP, GG, GS, SG) belonged to
      // the deprecated FS_UPG path and have been removed.
      opendarts::linear_solvers::MatrixSlice UU_, UP_, US_;
      opendarts::linear_solvers::MatrixSlice PU_, PP_, PS_;
      opendarts::linear_solvers::MatrixSlice SU_, SP_, SS_;
      opendarts::linear_solvers::MatrixSlice PPSS_;

      // Block Jacobian (kept as polymorphic pointer; setup() updates per
      // Newton).
      opendarts::linear_solvers::block_csr_matrix *A_block_ = nullptr;

      // Extracted scalar subsystem matrices. The proprietary code uses
      // raw pointers; use unique_ptr here for RAII.
      std::unique_ptr<opendarts::linear_solvers::csr_matrix<1>> U_;
      // P matrix: branches on NE. Only the matching one is allocated.
      // Explicit NE dispatch via static branches; documented as a
      // deliberate design decision in lieu of a templated helper.
      std::unique_ptr<opendarts::linear_solvers::csr_matrix<1>> P_scalar_;   // NE == 1
      std::unique_ptr<opendarts::linear_solvers::csr_matrix<2>> P_block_2_;  // NE == 2
      std::unique_ptr<opendarts::linear_solvers::csr_matrix<3>> P_block_3_;  // NE == 3
      std::unique_ptr<opendarts::linear_solvers::csr_matrix<4>> P_block_4_;  // NE == 4
      std::unique_ptr<opendarts::linear_solvers::csr_matrix<5>> P_block_5_;  // NE == 5
      // NE > 1: scalar (nb=1) expansion of the PPSS block subsystem. The
      // default pressure sub-preconditioner (hypre_amg_adapter<1>, see
      // solver_factories) is a block-size-1 solver and cannot consume the
      // block-NE matrix directly -- the former bare static_cast silently
      // misread the block memory layout (caught by the checked
      // linsolv_iface_bos down-cast). Expansion preserves the scalar DOF
      // count (n_rows * NE), so the solve-side P_B_/P_X_ buffers and the
      // AMG vector lengths are unchanged.
      std::unique_ptr<opendarts::linear_solvers::csr_matrix<1>> P_scalar_ne_; // NE > 1

      // Expand the block PPSS subsystem into P_scalar_ne_ (structure +
      // values + diag_ind + diag-first column convention, matching the
      // NE == 1 path) and run the pressure preconditioner setup on it.
      int setup_p_prec_from_block_(opendarts::linear_solvers::csr_matrix_base *P_block);

      // Per-row sign flips from the positive-diagonal step.
      std::vector<mat_float> u_rhs_mults_;
      std::vector<mat_float> ps_rhs_mults_;

      // Per-solve scratch buffers (allocated once in init).
      std::vector<mat_float> xp_;         // constant -1 vector for Schur approx
      std::vector<mat_float> x_sch_u_;
      std::vector<mat_float> x_sch_u_inv_;
      std::vector<mat_float> x_sch_p_;
      std::vector<mat_float> x_sch_s_;    // only when NE > 1

      std::vector<mat_float> P_B_, P_X_;    // NE * PP.n_rows
      std::vector<mat_float> U_B_, U_X_;    // ND * UU.n_rows
      std::vector<mat_float> UP_B_, US_B_;  // ND * UU.n_rows (NE > 1 path)

      // Sub-preconditioners (shared ownership).
      std::shared_ptr<opendarts::linear_solvers::linsolv_iface> p_system_preconditioner_;
      std::shared_ptr<opendarts::linear_solvers::linsolv_iface> u_system_preconditioner_;

      // Newton iteration state.
      bool first_setup_ = true;
      bool update_uu_ = true;
      bool force_amg_asymmetric_ = true;
      bool fs_cpr_debug_ = false;
      bool debug_printed_setup_ = false;

      // ----------------------------------------------------------------
      // Helpers
      // ----------------------------------------------------------------

      // Build slices (one-time, called from init()).
      void build_slices_(const opendarts::linear_solvers::block_csr_matrix &A);

      // Allocate the extracted subsystem matrices and the scratch buffers
      // (one-time, called from init()).
      void build_subsystem_matrices_(
          const opendarts::linear_solvers::block_csr_matrix &A,
          opendarts::config::index_t max_iters,
          opendarts::config::mat_float tolerance);

      // Per-Newton Schur-complement approximation step.
      void approx_schur_complement_(
          const opendarts::linear_solvers::block_csr_matrix &A);

      // The FS_UP cascade (the actual preconditioner application).
      int run_FS_UP_solve_(opendarts::config::mat_float *B,
          opendarts::config::mat_float *X);

      // Sub-prec refresh helpers -- call linear_solver::refresh() on the
      // sub-preconditioner. HYPRE-backed adapters override that virtual to
      // do a value-only re-push (cheap); others fall through to setup() via
      // the base-class default. Returns 0 on success.
      int refresh_u_prec_();
      int refresh_p_prec_();
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_FS_CPR_HPP
//--------------------------------------------------------------------------
