//*************************************************************************
//    Copyright (c) 2026
//    Delft University of Technology, the Netherlands
//
//    This file is part of the open Delft Advanced Research Terra Simulator
//    (open-DARTS). It is distributed under the Apache License.
// *************************************************************************

// linsolv_fs_cpr -- implementation. See linsolv_fs_cpr.hpp for the design
// rationale and the reference to the reference code this file ports.

#include "linsolv_fs_cpr.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <iterator>
#include <memory>
#include <vector>

#include "block_csr_matrix.hpp"
#include "csr_matrix.hpp"
#include "csr_matrix_base.hpp"
#include "data_types.hpp"
#include "linear_solvers_data_types.hpp"
#include "linsolv_iface.hpp"
#include "matrix_slice.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    using opendarts::config::index_t;
    using opendarts::config::mat_float;

    // ====================================================================
    // Constructor
    // ====================================================================
    template <std::uint8_t N_BLOCK_SIZE>
    linsolv_fs_cpr<N_BLOCK_SIZE>::linsolv_fs_cpr(std::uint8_t _P_VAR,
        std::uint8_t _Z_VAR, std::uint8_t _U_VAR, std::uint8_t _NC)
      : P_VAR_(_P_VAR),
        Z_VAR_(_Z_VAR),
        U_VAR_(_U_VAR),
        NC_(_NC),
        NE_(static_cast<std::uint8_t>(N_BLOCK_SIZE - ND))
    {
      // The base linsolv_iface_bos<N> stores a linear_solver_base* pointer
      // for legacy callers. The reference class sets it to `this`; this
      // class is not a linear_solver_base, so leave it as nullptr -- our
      // entry points come through the csr_matrix_base / csr_matrix<N>
      // overrides on linsolv_iface_bos<N>.
      this->solver = nullptr;

      // Diagnostic instrumentation toggle via env var FS_CPR_DEBUG=1.
      const char *dbg = std::getenv("FS_CPR_DEBUG");
      fs_cpr_debug_ = (dbg != nullptr && dbg[0] == '1');

      // U matrix is always csr_matrix<1>.
      U_ = std::make_unique<opendarts::linear_solvers::csr_matrix<1>>();

      // P matrix: branch by NE_ (runtime, set in constructor).
      if (NE_ == 1)
        P_scalar_ = std::make_unique<opendarts::linear_solvers::csr_matrix<1>>();
      else if (NE_ == 2)
        P_block_2_ = std::make_unique<opendarts::linear_solvers::csr_matrix<2>>();
      else if (NE_ == 3)
        P_block_3_ = std::make_unique<opendarts::linear_solvers::csr_matrix<3>>();
      else if (NE_ == 4)
        P_block_4_ = std::make_unique<opendarts::linear_solvers::csr_matrix<4>>();
      else if (NE_ == 5)
        P_block_5_ = std::make_unique<opendarts::linear_solvers::csr_matrix<5>>();
      if (NE_ > 1)
        P_scalar_ne_ = std::make_unique<opendarts::linear_solvers::csr_matrix<1>>();
    }

    namespace
    {
      // to_nb_1 expands structure + values but leaves diag_ind untouched;
      // fill it by scanning each row (set_diag_first and the HYPRE IJ build
      // both consume it).
      void fill_scalar_diag_ind(opendarts::linear_solvers::csr_matrix<1> &M)
      {
        const opendarts::config::index_t n = M.n_rows;
        const opendarts::config::index_t *rows = M.get_rows_ptr();
        const opendarts::config::index_t *cols = M.get_cols_ind();
        opendarts::config::index_t *diag = M.get_diag_ind();
        for (opendarts::config::index_t i = 0; i < n; ++i)
        {
          diag[i] = rows[i];
          for (opendarts::config::index_t j = rows[i]; j < rows[i + 1]; ++j)
          {
            if (cols[j] == i)
            {
              diag[i] = j;
              break;
            }
          }
        }
      }
    } // namespace

    // ====================================================================
    // set_prec overloads (FS_UP only; G overload removed)
    // ====================================================================
    template <std::uint8_t N_BLOCK_SIZE>
    int linsolv_fs_cpr<N_BLOCK_SIZE>::set_prec(
        std::shared_ptr<opendarts::linear_solvers::linsolv_iface> prec_p)
    {
      p_system_preconditioner_ = std::move(prec_p);
      return 0;
    }

    template <std::uint8_t N_BLOCK_SIZE>
    int linsolv_fs_cpr<N_BLOCK_SIZE>::set_prec(
        std::shared_ptr<opendarts::linear_solvers::linsolv_iface> prec_p,
        std::shared_ptr<opendarts::linear_solvers::linsolv_iface> prec_u)
    {
      p_system_preconditioner_ = std::move(prec_p);
      u_system_preconditioner_ = std::move(prec_u);
      return 0;
    }

    template <std::uint8_t N_BLOCK_SIZE>
    int linsolv_fs_cpr<N_BLOCK_SIZE>::set_prec(
        opendarts::linear_solvers::linsolv_iface *prec_input)
    {
      // Non-owning shared_ptr (no-op deleter) -- matches the reference
      // single-prec overload which stored a raw pointer.
      p_system_preconditioner_ = std::shared_ptr<opendarts::linear_solvers::linsolv_iface>(
          prec_input, [](opendarts::linear_solvers::linsolv_iface *) {});
      return 0;
    }

    // ====================================================================
    // set_block_sizes
    // ====================================================================
    template <std::uint8_t N_BLOCK_SIZE>
    void linsolv_fs_cpr<N_BLOCK_SIZE>::set_block_sizes(
        index_t _n_res, index_t _n_fracs, index_t _n_wells)
    {
      n_res_ = _n_res;
      n_fracs_ = _n_fracs;
      n_wells_ = _n_wells;
      // Partition changed -- next setup() must do a full U setup, not a
      // refresh.
      update_uu_ = true;
    }

    // ====================================================================
    // Slice-building helper
    // ====================================================================
    template <std::uint8_t N_BLOCK_SIZE>
    void linsolv_fs_cpr<N_BLOCK_SIZE>::build_slices_(
        const opendarts::linear_solvers::block_csr_matrix &A)
    {
      const index_t n_rows = A.n_rows;
      constexpr std::uint8_t N_VARS = N_BLOCK_SIZE;
      const std::uint8_t NE = NE_;

      // FS_UP: n_fracs_ == 0 is enforced in init(). Define global slicing
      // for jacobian: reservoir and well blocks. Two ranges when
      // n_wells > 0 i.e. n_res != n_rows; one range otherwise.
      if (n_res_ != n_rows)
      {
        UU_.ranges.push_back({0, n_res_, {{0, n_res_}, {n_res_, n_rows}}});
        UU_.ranges.push_back({n_res_, n_rows, {{0, n_res_}, {n_res_, n_rows}}});
        UP_.ranges.push_back({0, n_res_, {{0, n_rows}}});
        UP_.ranges.push_back({n_res_, n_rows, {{0, n_rows}}});
        US_.ranges.push_back({0, n_res_, {{0, n_rows}}});
        US_.ranges.push_back({n_res_, n_rows, {{0, n_rows}}});
        UPS_.ranges.push_back({0, n_res_, {{0, n_rows}}});
        UPS_.ranges.push_back({n_res_, n_rows, {{0, n_rows}}});
      }
      else
      {
        UU_.ranges.push_back({0, n_res_, {{0, n_res_}}});
        UP_.ranges.push_back({0, n_res_, {{0, n_rows}}});
        US_.ranges.push_back({0, n_res_, {{0, n_rows}}});
        UPS_.ranges.push_back({0, n_res_, {{0, n_rows}}});
      }
      UU_.init(A);
      UP_.init_structural();
      US_.init_structural();
      UPS_.init_structural();

      PU_.ranges.push_back({0, n_rows, {{0, n_res_}, {n_res_, n_rows}}});
      PP_.ranges.push_back({0, n_rows, {{0, n_rows}}});
      PS_.ranges.push_back({0, n_rows, {{0, n_rows}}});
      PU_.init_structural();
      PP_.init(A);
      PS_.init_structural();

      SU_.ranges.push_back({0, n_rows, {{0, n_res_}, {n_res_, n_rows}}});
      SP_.ranges.push_back({0, n_rows, {{0, n_rows}}});
      SS_.ranges.push_back({0, n_rows, {{0, n_rows}}});
      SU_.init_structural();
      SP_.init_structural();
      SS_.init(A);

      // Define local slicing for jacobian: displacements (U), pressures
      // (P), composition (S). The NE-1 sizes for the S row/col strides
      // are kept verbatim from the reference code.
      UU_.pos = static_cast<std::uint8_t>(U_VAR_ * N_VARS + U_VAR_);
      UP_.pos = static_cast<std::uint8_t>(U_VAR_ * N_VARS + P_VAR_);
      US_.pos = static_cast<std::uint8_t>(U_VAR_ * N_VARS + Z_VAR_);
      UU_.sizes[0] = ND;   UU_.sizes[1] = ND;
      UP_.sizes[0] = ND;   UP_.sizes[1] = 1;
      US_.sizes[0] = ND;   US_.sizes[1] = static_cast<std::uint8_t>(NE - 1);
      UU_.strides[0] = N_VARS; UU_.strides[1] = 1;
      UP_.strides[0] = N_VARS; UP_.strides[1] = 1;
      US_.strides[0] = N_VARS; US_.strides[1] = 1;
      // Joint (P, S) columns of the displacement rows, matching PPSS_'s column
      // convention and P_X_'s stride-NE interleaving. See the UPS_ declaration.
      UPS_.pos = static_cast<std::uint8_t>(U_VAR_ * N_VARS + P_VAR_);
      UPS_.sizes[0] = ND;      UPS_.sizes[1] = NE;
      UPS_.strides[0] = N_VARS; UPS_.strides[1] = 1;

      PU_.pos = static_cast<std::uint8_t>(P_VAR_ * N_VARS + U_VAR_);
      PP_.pos = static_cast<std::uint8_t>(P_VAR_ * N_VARS + P_VAR_);
      PS_.pos = static_cast<std::uint8_t>(P_VAR_ * N_VARS + Z_VAR_);
      PU_.sizes[0] = 1;    PU_.sizes[1] = ND;
      PP_.sizes[0] = 1;    PP_.sizes[1] = 1;
      PS_.sizes[0] = 1;    PS_.sizes[1] = static_cast<std::uint8_t>(NE - 1);
      PU_.strides[0] = N_VARS; PU_.strides[1] = 1;
      PP_.strides[0] = N_VARS; PP_.strides[1] = 1;
      PS_.strides[0] = N_VARS; PS_.strides[1] = 1;

      SU_.pos = static_cast<std::uint8_t>(Z_VAR_ * N_VARS + U_VAR_);
      SP_.pos = static_cast<std::uint8_t>(Z_VAR_ * N_VARS + P_VAR_);
      SS_.pos = static_cast<std::uint8_t>(Z_VAR_ * N_VARS + Z_VAR_);
      SU_.sizes[0] = static_cast<std::uint8_t>(NE - 1); SU_.sizes[1] = ND;
      SP_.sizes[0] = static_cast<std::uint8_t>(NE - 1); SP_.sizes[1] = 1;
      SS_.sizes[0] = static_cast<std::uint8_t>(NE - 1); SS_.sizes[1] = static_cast<std::uint8_t>(NE - 1);
      SU_.strides[0] = N_VARS; SU_.strides[1] = 1;
      SP_.strides[0] = N_VARS; SP_.strides[1] = 1;
      SS_.strides[0] = N_VARS; SS_.strides[1] = 1;

      // Joint pressures (P) and compositions (S) slice.
      PPSS_.ranges.push_back({0, n_rows, {{0, n_rows}}});
      PPSS_.init(A);
      PPSS_.pos = static_cast<std::uint8_t>(P_VAR_ * N_VARS + P_VAR_);
      PPSS_.sizes[0] = NE;
      PPSS_.sizes[1] = NE;
      PPSS_.strides[0] = N_VARS;
      PPSS_.strides[1] = 1;
      PPSS_.global_to_local_rows.resize(static_cast<std::size_t>(PPSS_.sizes[0]) * PPSS_.n_rows);
      index_t arr_id = 0;
      for (const auto &range : PPSS_.ranges)
      {
        for (index_t row = range.rows_from; row < range.rows_to; row++)
        {
          for (std::uint8_t block_row = P_VAR_;
               block_row < static_cast<std::uint8_t>(P_VAR_ + NE);
               block_row++)
          {
            PPSS_.global_to_local_rows[arr_id++] =
                row * N_VARS + block_row;
          }
        }
      }

#ifndef NDEBUG
      // Sanity assertions (FS_UP variant: G subsystem removed).
      assert(UU_.n_cols * UU_.sizes[1] + UP_.n_cols * UP_.sizes[1]
             + US_.n_cols * US_.sizes[1]
             == N_VARS * n_rows);
      assert(PU_.n_cols * PU_.sizes[1] + PP_.n_cols * PP_.sizes[1]
             + PS_.n_cols * PS_.sizes[1]
             == N_VARS * n_rows);
      if constexpr (N_BLOCK_SIZE - ND > 1)
      {
        assert(SU_.n_cols * SU_.sizes[1] + SP_.n_cols * SP_.sizes[1]
               + SS_.n_cols * SS_.sizes[1]
               == N_VARS * n_rows);
      }
      assert(UU_.n_rows * UU_.sizes[0] + PU_.n_rows * PU_.sizes[0]
             + SU_.n_rows * SU_.sizes[0]
             == N_VARS * n_rows);
      assert(UP_.n_rows * UP_.sizes[0] + PP_.n_rows * PP_.sizes[0]
             + SP_.n_rows * SP_.sizes[0]
             == N_VARS * n_rows);
      if constexpr (N_BLOCK_SIZE - ND > 1)
      {
        assert(US_.n_rows * US_.sizes[0] + PS_.n_rows * PS_.sizes[0]
               + SS_.n_rows * SS_.sizes[0]
               == N_VARS * n_rows);
      }
#endif
    }

    // ====================================================================
    // Subsystem-matrix + scratch allocation
    // ====================================================================
    template <std::uint8_t N_BLOCK_SIZE>
    void linsolv_fs_cpr<N_BLOCK_SIZE>::build_subsystem_matrices_(
        const opendarts::linear_solvers::block_csr_matrix &A,
        opendarts::config::index_t max_iters,
        opendarts::config::mat_float tolerance)
    {
      const index_t n_rows = A.n_rows;
      const std::uint8_t NE = NE_;

      // ------------------------------------------------------------------
      // Pressure / pressure+composition matrix init.
      // ------------------------------------------------------------------
      ps_rhs_mults_.assign(static_cast<std::size_t>(NE) * n_rows, 0.0);

      if (NE == 1)
      {
        // NE == 1: extract PP as scalar (unit-block) matrix.
        P_scalar_->init(PP_.n_rows, PP_.n_rows, 1,
            static_cast<index_t>(PPSS_.nnz));
        P_scalar_->type = MATRIX_TYPE_CSR;
        P_scalar_->is_square = 1;
        init_rows_cols_to_unit_matrix(A, PP_, *P_scalar_);
        // Pressure preconditioner: 1 V-cycle, tolerance irrelevant
        // (preconditioner mode). Matches the U stage and the reference
        // FS-CPR convention -- using the outer max_iters (e.g. 5000) here
        // would configure BoomerAMG as a stand-alone solver and BoomerAMGSolve
        // can fail with HYPRE_ERROR_GENERIC on an indefinite Schur subsystem.
        (void) max_iters;
        (void) tolerance;
        p_system_preconditioner_->init(P_scalar_.get(),
            static_cast<index_t>(p_amg_max_iters_),
            static_cast<mat_float>(0.0));
      }
      else
      {
        // NE > 1: extract PPSS as a block-NE matrix sharing A's sparsity.
        const index_t nnzb = static_cast<index_t>(PPSS_.nnz);
        // Allocate the right matrix.
        opendarts::linear_solvers::csr_matrix_base *P_base = nullptr;
        index_t *P_rows_ptr = nullptr;
        index_t *P_cols_ind = nullptr;
        index_t *P_diag_ind = nullptr;
        if (NE == 2)
        {
          P_block_2_->init(PP_.n_rows, PP_.n_rows, nnzb);
          P_block_2_->type = MATRIX_TYPE_CSR;
          P_block_2_->is_square = 1;
          P_base = P_block_2_.get();
          P_rows_ptr = P_block_2_->get_rows_ptr();
          P_cols_ind = P_block_2_->get_cols_ind();
          P_diag_ind = P_block_2_->get_diag_ind();
        }
        else if (NE == 3)
        {
          P_block_3_->init(PP_.n_rows, PP_.n_rows, nnzb);
          P_block_3_->type = MATRIX_TYPE_CSR;
          P_block_3_->is_square = 1;
          P_base = P_block_3_.get();
          P_rows_ptr = P_block_3_->get_rows_ptr();
          P_cols_ind = P_block_3_->get_cols_ind();
          P_diag_ind = P_block_3_->get_diag_ind();
        }
        else if (NE == 4)
        {
          P_block_4_->init(PP_.n_rows, PP_.n_rows, nnzb);
          P_block_4_->type = MATRIX_TYPE_CSR;
          P_block_4_->is_square = 1;
          P_base = P_block_4_.get();
          P_rows_ptr = P_block_4_->get_rows_ptr();
          P_cols_ind = P_block_4_->get_cols_ind();
          P_diag_ind = P_block_4_->get_diag_ind();
        }
        else if (NE == 5)
        {
          P_block_5_->init(PP_.n_rows, PP_.n_rows, nnzb);
          P_block_5_->type = MATRIX_TYPE_CSR;
          P_block_5_->is_square = 1;
          P_base = P_block_5_.get();
          P_rows_ptr = P_block_5_->get_rows_ptr();
          P_cols_ind = P_block_5_->get_cols_ind();
          P_diag_ind = P_block_5_->get_diag_ind();
        }
        else
        {
          assert(false && "unsupported NE in linsolv_fs_cpr build_subsystem_matrices_");
        }

        // Copy A's sparsity (row_ptr, col_ind, diag_ind) into P. Mirrors
        // reference MEM_*_CPY at lines 260-262.
        const index_t *a_rows = A.row_ptr();
        const index_t *a_cols = A.col_ind();
        const index_t *a_diag = A.diag_ind();
        std::copy_n(a_rows, n_rows + 1, P_rows_ptr);
        std::copy_n(a_diag, n_rows, P_diag_ind);
        std::copy_n(a_cols, P_rows_ptr[n_rows], P_cols_ind);

        // Pressure(-flow)-system preconditioner: 1 V-cycle, tolerance
        // irrelevant (preconditioner mode). See NE == 1 branch above.
        if (p_prec_takes_block_)
        {
          // Nested block CPR (linsolv_cpr<NE>): consumes the block-NE matrix
          // directly, so there is nothing to scalar-expand. Its own init() is
          // a pure parameter store; the pressure-subsystem extraction, the
          // BoomerAMG hierarchy and the block-ILU(0) factor are all built in
          // its setup(), which runs from setup_p_prec_from_block_ once the
          // PPSS values exist. Mirrors the proprietary FS-CPR wiring
          // (linsolv_bos_fs_cpr::set_prec(new linsolv_bos_cpr<NE>, ...)).
          p_system_preconditioner_->init(P_base,
              static_cast<index_t>(p_amg_max_iters_),
              static_cast<mat_float>(0.0));
        }
        else
        {
          // The scalar sub-prec (hypre_amg_adapter<1>) is a block-size-1
          // solver; initialise it on the scalar (nb=1) expansion of the
          // block-NE structure with unit values (mirrors the NE == 1 branch's
          // init_rows_cols_to_unit_matrix -- to_nb_1 would copy uninitialised
          // block values here). setup() refreshes the real values per Newton.
          P_scalar_ne_->to_nb_1(P_base);
          fill_scalar_diag_ind(*P_scalar_ne_);
          {
            mat_float *sv = P_scalar_ne_->get_values();
            const index_t n_scalar_nnz =
                P_scalar_ne_->get_rows_ptr()[P_scalar_ne_->n_rows];
            std::fill_n(sv, n_scalar_nnz, static_cast<mat_float>(0.0));
            const index_t *sdiag = P_scalar_ne_->get_diag_ind();
            for (index_t i = 0; i < P_scalar_ne_->n_rows; ++i)
              sv[sdiag[i]] = static_cast<mat_float>(1.0);
          }
          p_system_preconditioner_->init(P_scalar_ne_.get(),
              static_cast<index_t>(p_amg_max_iters_),
              static_cast<mat_float>(0.0));
        }
      }

      if (this->timer_setup && p_system_preconditioner_)
      {
        p_system_preconditioner_->init_timer_nodes(
            &this->timer_setup->node["FS-CPR"],
            this->timer_solve ? &this->timer_solve->node["FS-CPR"] : nullptr);
      }

      // ------------------------------------------------------------------
      // Displacements matrix init.
      // ------------------------------------------------------------------
      const index_t u_scalar_rows =
          static_cast<index_t>(UU_.sizes[0]) * UU_.n_rows;
      U_->init(u_scalar_rows, static_cast<index_t>(UU_.sizes[1]) * UU_.n_rows,
          1,
          static_cast<index_t>(UU_.sizes[0]) * UU_.sizes[1]
              * static_cast<index_t>(UU_.nnz));
      U_->type = MATRIX_TYPE_CSR;
      U_->is_square = 1;
      init_rows_cols_to_unit_matrix(A, UU_, *U_);
      u_rhs_mults_.assign(static_cast<std::size_t>(u_scalar_rows), 0.0);
      // Mech preconditioner: u_amg_max_iters_ V-cycles (default 1), tolerance
      // irrelevant (preconditioner mode), matches the reference call at line 274.
      u_system_preconditioner_->init(U_.get(),
          static_cast<index_t>(u_amg_max_iters_),
          static_cast<mat_float>(0.0));
      if (this->timer_setup && u_system_preconditioner_)
      {
        u_system_preconditioner_->init_timer_nodes(
            &this->timer_setup->node["FS-CPR"],
            this->timer_solve ? &this->timer_solve->node["FS-CPR"] : nullptr);
      }

      // ------------------------------------------------------------------
      // Scratch allocations.
      // ------------------------------------------------------------------
      xp_.assign(static_cast<std::size_t>(PP_.sizes[0]) * PP_.n_rows, -1.0);
      x_sch_u_.assign(static_cast<std::size_t>(UU_.sizes[0]) * UU_.n_rows, 0.0);
      x_sch_u_inv_.assign(static_cast<std::size_t>(UU_.sizes[0]) * UU_.n_rows, 0.0);
      x_sch_p_.assign(static_cast<std::size_t>(PU_.sizes[0]) * PU_.n_rows, 0.0);
      if (NE > 1)
      {
        x_sch_s_.assign(static_cast<std::size_t>(SU_.sizes[0]) * SU_.n_rows, 0.0);
      }
      P_B_.assign(static_cast<std::size_t>(PPSS_.sizes[0]) * PPSS_.n_rows, 0.0);
      P_X_.assign(static_cast<std::size_t>(PPSS_.sizes[0]) * PPSS_.n_rows, 0.0);
      U_B_.assign(static_cast<std::size_t>(UU_.sizes[0]) * UU_.n_rows, 0.0);
      U_X_.assign(static_cast<std::size_t>(UU_.sizes[0]) * UU_.n_rows, 0.0);
      if (NE > 1)
      {
        UPS_B_.assign(static_cast<std::size_t>(UU_.sizes[0]) * UU_.n_rows, 0.0);
      }
      else
      {
        // Single-buffer path: U_B accumulates UP * P_X directly.
        UPS_B_.clear();
      }
    }

    // ====================================================================
    // init (csr_matrix_base*)
    // ====================================================================
    template <std::uint8_t N_BLOCK_SIZE>
    int linsolv_fs_cpr<N_BLOCK_SIZE>::init(
        opendarts::linear_solvers::csr_matrix_base *A,
        opendarts::config::index_t max_iters,
        opendarts::config::mat_float tolerance)
    {
      if (A == nullptr)
      {
        std::cerr << "linsolv_fs_cpr::init: null matrix pointer" << std::endl;
        return -1;
      }
      auto *A_block = dynamic_cast<opendarts::linear_solvers::block_csr_matrix *>(A);
      if (A_block == nullptr)
      {
        std::cerr << "linsolv_fs_cpr::init: matrix is not a block_csr_matrix"
                  << std::endl;
        return -1;
      }
      // FS_UPG (gap / contact-mechanics) path has been removed.
      if (n_fracs_ != 0)
      {
        std::cerr << "linsolv_fs_cpr::init: FS-CPR n_fracs must be 0 "
                     "(FS_UPG / contact mechanics support has been removed)"
                  << std::endl;
        return -1;
      }
      if (p_system_preconditioner_ == nullptr
          || u_system_preconditioner_ == nullptr)
      {
        std::cerr << "linsolv_fs_cpr::init: sub-preconditioners must be set "
                  << "via set_prec() before init()" << std::endl;
        return -1;
      }
      assert(A_block->n_rows == n_res_ + n_wells_);

      A_block_ = A_block;
      build_slices_(*A_block);
      build_subsystem_matrices_(*A_block, max_iters, tolerance);

      if (fs_cpr_debug_ && !debug_printed_setup_)
      {
        std::fprintf(stderr,
            "[FS-CPR] N_BLOCK_SIZE=%d NE=%d P_VAR=%d Z_VAR=%d U_VAR=%d NC=%d ND=%d\n",
            static_cast<int>(N_BLOCK_SIZE), static_cast<int>(NE_),
            static_cast<int>(P_VAR_), static_cast<int>(Z_VAR_),
            static_cast<int>(U_VAR_), static_cast<int>(NC_),
            static_cast<int>(ND));
        std::fprintf(stderr,
            "[FS-CPR] partition: n_res=%d n_fracs=%d n_wells=%d (sum=%d, A.n_rows=%d)\n",
            static_cast<int>(n_res_), static_cast<int>(n_fracs_),
            static_cast<int>(n_wells_),
            static_cast<int>(n_res_ + n_fracs_ + n_wells_),
            static_cast<int>(A_block->n_rows));
        std::fprintf(stderr,
            "[FS-CPR] UU: n_rows=%d nnz=%lu  PP: n_rows=%d nnz=%lu  PPSS: n_rows=%d nnz=%lu\n",
            static_cast<int>(UU_.n_rows), static_cast<unsigned long>(UU_.nnz),
            static_cast<int>(PP_.n_rows), static_cast<unsigned long>(PP_.nnz),
            static_cast<int>(PPSS_.n_rows), static_cast<unsigned long>(PPSS_.nnz));
        std::fprintf(stderr,
            "[FS-CPR] force_amg_asymmetric=%d\n",
            static_cast<int>(force_amg_asymmetric_));
        std::fflush(stderr);
        debug_printed_setup_ = true;
      }

      first_setup_ = true;
      update_uu_ = true;
      return 0;
    }

    // ====================================================================
    // approx_schur_complement_
    // ====================================================================
    template <std::uint8_t N_BLOCK_SIZE>
    void linsolv_fs_cpr<N_BLOCK_SIZE>::approx_schur_complement_(
        const opendarts::linear_solvers::block_csr_matrix &A)
    {
      const std::uint8_t NE = NE_;
      const std::size_t n_u_scalar =
          static_cast<std::size_t>(UU_.sizes[0]) * UU_.n_rows;
      const std::size_t n_pu_scalar =
          static_cast<std::size_t>(PU_.sizes[0]) * PU_.n_rows;

      // x_sch_u <- A_UP * xp  (xp is the constant -1 vector)
      std::fill(x_sch_u_.begin(), x_sch_u_.end(), 0.0);
      block_vector_product(A, UP_, xp_.data(), x_sch_u_.data());
      for (std::size_t i = 0; i < n_u_scalar; i++)
        x_sch_u_[i] *= u_rhs_mults_[i];
      if (fs_cpr_debug_)
      {
        mat_float max_xschu = 0.0;
        for (std::size_t i = 0; i < n_u_scalar; ++i)
          if (std::fabs(x_sch_u_[i]) > max_xschu) max_xschu = std::fabs(x_sch_u_[i]);
        std::fprintf(stderr,
            "[FS-CPR.schur] x_sch_u = u_rhs_mults * (A_UP * -1): max|x_sch_u|=%.3e\n",
            max_xschu);
        std::fflush(stderr);
      }

      // x_sch_u_inv <- U^{-1} * x_sch_u (one AMG V-cycle)
      std::fill(x_sch_u_inv_.begin(), x_sch_u_inv_.end(), 0.0);
      if (fs_cpr_debug_)
      {
        std::fprintf(stderr, "[FS-CPR.schur] BEFORE u_prec->solve()\n");
        std::fflush(stderr);
      }
      u_system_preconditioner_->solve(x_sch_u_.data(), x_sch_u_inv_.data());
      if (fs_cpr_debug_)
      {
        mat_float max_xschui = 0.0;
        for (std::size_t i = 0; i < n_u_scalar; ++i)
          if (std::fabs(x_sch_u_inv_[i]) > max_xschui)
            max_xschui = std::fabs(x_sch_u_inv_[i]);
        std::fprintf(stderr,
            "[FS-CPR.schur] AFTER u_prec->solve(): max|x_sch_u_inv|=%.3e\n",
            max_xschui);
        std::fflush(stderr);
      }

      // x_sch_p <- A_PU * x_sch_u_inv
      std::fill(x_sch_p_.begin(), x_sch_p_.end(), 0.0);
      block_vector_product(A, PU_, x_sch_u_inv_.data(), x_sch_p_.data());

      if (NE > 1)
      {
        std::fill(x_sch_s_.begin(), x_sch_s_.end(), 0.0);
        block_vector_product(A, SU_, x_sch_u_inv_.data(), x_sch_s_.data());

        // Flow preconditioner (NE > 1 path): build the PPSS block matrix
        // and augment the diagonal via the apply_ps_relaxation.
        if (NE == 2)
        {
          extract_sub_block_to_block_csr<2>(PPSS_, A, *P_block_2_,
              ps_rhs_mults_.data());
          apply_ps_relaxation<2>(*P_block_2_, x_sch_p_.data(),
              x_sch_s_.data(), ps_rhs_mults_.data());
          if (p_decouple_block_diag_)
            decouple_ppss_block_<2>(*P_block_2_);
          setup_p_prec_from_block_(P_block_2_.get());
        }
        else if (NE == 3)
        {
          extract_sub_block_to_block_csr<3>(PPSS_, A, *P_block_3_,
              ps_rhs_mults_.data());
          apply_ps_relaxation<3>(*P_block_3_, x_sch_p_.data(),
              x_sch_s_.data(), ps_rhs_mults_.data());
          if (p_decouple_block_diag_)
            decouple_ppss_block_<3>(*P_block_3_);
          setup_p_prec_from_block_(P_block_3_.get());
        }
        else if (NE == 4)
        {
          extract_sub_block_to_block_csr<4>(PPSS_, A, *P_block_4_,
              ps_rhs_mults_.data());
          apply_ps_relaxation<4>(*P_block_4_, x_sch_p_.data(),
              x_sch_s_.data(), ps_rhs_mults_.data());
          if (p_decouple_block_diag_)
            decouple_ppss_block_<4>(*P_block_4_);
          setup_p_prec_from_block_(P_block_4_.get());
        }
        else if (NE == 5)
        {
          extract_sub_block_to_block_csr<5>(PPSS_, A, *P_block_5_,
              ps_rhs_mults_.data());
          apply_ps_relaxation<5>(*P_block_5_, x_sch_p_.data(),
              x_sch_s_.data(), ps_rhs_mults_.data());
          if (p_decouple_block_diag_)
            decouple_ppss_block_<5>(*P_block_5_);
          setup_p_prec_from_block_(P_block_5_.get());
        }
      }
      else
      {
        // NE == 1 path: build scalar PP, augment diagonal, AMG setup with
        // the diagonal-first column layout.
        extract_sub_block_to_scalar_csr(PP_, A, *P_scalar_,
            ps_rhs_mults_.data(), /*positive_diagonal=*/true,
            /*asymmetric_hack=*/force_amg_asymmetric_);
        apply_relaxation<1>(*P_scalar_, x_sch_p_.data(),
            ps_rhs_mults_.data(), /*positive_diagonal=*/true);
        set_diag_first<1>(*P_scalar_);
        p_system_preconditioner_->setup(P_scalar_.get());
        set_diag_in_order<1>(*P_scalar_);
      }

      (void)n_pu_scalar;
    }

    // ====================================================================
    // setup (csr_matrix_base*)
    // ====================================================================
    template <std::uint8_t N_BLOCK_SIZE>
    int linsolv_fs_cpr<N_BLOCK_SIZE>::setup(
        opendarts::linear_solvers::csr_matrix_base *A)
    {
      if (A == nullptr)
      {
        std::cerr << "linsolv_fs_cpr::setup: null matrix pointer" << std::endl;
        return -1;
      }
      auto *A_block = dynamic_cast<opendarts::linear_solvers::block_csr_matrix *>(A);
      if (A_block == nullptr)
      {
        std::cerr << "linsolv_fs_cpr::setup: matrix is not a block_csr_matrix"
                  << std::endl;
        return -1;
      }
      A_block_ = A_block;

      if (fs_cpr_debug_)
      {
        std::fprintf(stderr,
            "[FS-CPR.setup] ENTRY (first_setup=%d update_uu=%d)\n",
            static_cast<int>(first_setup_), static_cast<int>(update_uu_));
        std::fflush(stderr);
      }

      if (this->timer_setup)
        this->timer_setup->node["FS-CPR"].start();

      // ----- mechanics preconditioner: refresh U scalar values -----
      extract_sub_block_to_scalar_csr(UU_, *A_block, *U_,
          u_rhs_mults_.data(), /*positive_diagonal=*/true,
          /*asymmetric_hack=*/force_amg_asymmetric_);

      if (update_uu_ || first_setup_)
      {
        if (fs_cpr_debug_)
        {
          std::fprintf(stderr,
              "[FS-CPR.setup] BEFORE u_prec->setup() (first_setup=%d update_uu=%d)\n",
              static_cast<int>(first_setup_), static_cast<int>(update_uu_));
          std::fflush(stderr);
        }
        int res = u_system_preconditioner_->setup(U_.get());
        if (fs_cpr_debug_)
        {
          std::fprintf(stderr,
              "[FS-CPR.setup] AFTER u_prec->setup() res=%d\n", res);
          std::fflush(stderr);
        }
        if (res)
        {
          if (this->timer_setup)
            this->timer_setup->node["FS-CPR"].stop();
          return res;
        }
        update_uu_ = false;
      }
      else
      {
        int res = refresh_u_prec_();
        if (res)
        {
          if (this->timer_setup)
            this->timer_setup->node["FS-CPR"].stop();
          return res;
        }
      }

      // ----- pressure (or PPSS) preconditioner via Schur approximation --
      approx_schur_complement_(*A_block);

      first_setup_ = false;

      if (this->timer_setup)
        this->timer_setup->node["FS-CPR"].stop();
      return 0;
    }

    // ====================================================================
    // refresh helpers
    // ====================================================================
    // Both helpers go through the polymorphic linear_solver::refresh()
    // entry point added to the base interface; HYPRE-backed sub-precs
    // (hypre_amg_adapter / hypre_ilu_adapter) override it to do a value-
    // only re-push of the cached IJ matrix and reuse the existing AMG /
    // ILU hierarchy, while other implementations (SuperLU, ...) inherit
    // the default that falls through to setup(). No dynamic_cast is
    // needed here -- the override knows the concrete type.
    template <std::uint8_t N_BLOCK_SIZE>
    int linsolv_fs_cpr<N_BLOCK_SIZE>::refresh_u_prec_()
    {
      return u_system_preconditioner_->refresh(U_.get());
    }

    // ====================================================================
    // Block-diagonal (ABF / quasi-IMPES) decoupling of the PPSS subsystem
    // ====================================================================
    // The NE > 1 flow block holds the NC component mass balances (plus the
    // energy balance when thermal), NOT a pressure equation. Its per-cell
    // diagonal block is strongly off-diagonal-dominant whenever a composition
    // unknown is present: d(R_mass)/dz is the accumulation term while
    // d(R_mass)/dp is only compressibility-small, a ratio of ~700 on
    // SPE10_mech dead_oil, and the oil balance's own diagonal is negative.
    // A point-wise smoother (BoomerAMG's hybrid Gauss-Seidel) therefore
    // AMPLIFIES the error -- measured Gauss-Seidel spectral radius ~400 -- and
    // no amount of interpolation tuning (NumFunctions included) can repair a
    // divergent smoother.
    //
    // Left-multiplying each block row by the inverse of its own diagonal block
    // (Behie & Vinsome's Alternate Block Factorization; the same decoupling
    // idea CPR applies to extract a pressure equation) makes every diagonal
    // block the identity and drops that spectral radius to ~0.998, without
    // discarding the multigrid treatment of the diffusive rows the way a
    // nested CPR's ILU(0) second stage would. It is a similarity-preserving
    // left preconditioner: the RHS is scaled by the same factors in
    // run_FS_UP_solve_, so the stage still approximates PPSS^{-1}.
    template <std::uint8_t N_BLOCK_SIZE>
    template <std::uint8_t NE_T>
    void linsolv_fs_cpr<N_BLOCK_SIZE>::decouple_ppss_block_(
        opendarts::linear_solvers::csr_matrix<NE_T> &P)
    {
      constexpr int NE_I = static_cast<int>(NE_T);
      constexpr int NB = NE_I * NE_I;
      const index_t n_rows = P.n_rows;
      const index_t *rows = P.get_rows_ptr();
      const index_t *diag = P.get_diag_ind();
      mat_float *vals = P.get_values();

      ps_decouple_inv_.assign(static_cast<std::size_t>(n_rows) * NB, 0.0);

      // Gauss-Jordan with partial pivoting on [D | I]; NE_T <= 5 so the dense
      // work is negligible next to one AMG setup.
      mat_float aug[NE_I][2 * NE_I];
      for (index_t i = 0; i < n_rows; ++i)
      {
        mat_float *inv = &ps_decouple_inv_[static_cast<std::size_t>(i) * NB];
        const mat_float *D = &vals[static_cast<std::size_t>(diag[i]) * NB];

        for (int r = 0; r < NE_I; ++r)
        {
          for (int c = 0; c < NE_I; ++c)
          {
            aug[r][c] = D[r * NE_I + c];
            aug[r][NE_I + c] = (r == c) ? 1.0 : 0.0;
          }
        }

        bool singular = false;
        for (int c = 0; c < NE_I && !singular; ++c)
        {
          int piv = c;
          for (int r = c + 1; r < NE_I; ++r)
            if (std::fabs(aug[r][c]) > std::fabs(aug[piv][c])) piv = r;
          if (!(std::fabs(aug[piv][c]) > 0.0) || !std::isfinite(aug[piv][c]))
          {
            singular = true;
            break;
          }
          if (piv != c)
            for (int k = 0; k < 2 * NE_I; ++k) std::swap(aug[c][k], aug[piv][k]);
          const mat_float inv_p = 1.0 / aug[c][c];
          for (int k = 0; k < 2 * NE_I; ++k) aug[c][k] *= inv_p;
          for (int r = 0; r < NE_I; ++r)
          {
            if (r == c) continue;
            const mat_float f = aug[r][c];
            if (f == 0.0) continue;
            for (int k = 0; k < 2 * NE_I; ++k) aug[r][k] -= f * aug[c][k];
          }
        }

        // A finite pivot at every step still does not guarantee a finite
        // inverse -- an ill-conditioned block can overflow during the
        // elimination. Check the result before it is allowed anywhere near the
        // preconditioner: a single NaN here would propagate through the AMG
        // setup and come back as a non-finite residual, i.e. a hard linear
        // failure, rather than the graceful fall-back this row deserves.
        if (!singular)
        {
          for (int k = 0; k < NE_I * NE_I && !singular; ++k)
            if (!std::isfinite(aug[k / NE_I][NE_I + (k % NE_I)])) singular = true;
        }

        if (singular)
        {
          // Leave this row alone: identity factor, matrix row unchanged. The
          // stage is then simply un-decoupled here, which is what the
          // pre-decoupling code did for every row.
          for (int r = 0; r < NE_I; ++r) inv[r * NE_I + r] = 1.0;
          continue;
        }
        for (int r = 0; r < NE_I; ++r)
          for (int c = 0; c < NE_I; ++c) inv[r * NE_I + c] = aug[r][NE_I + c];

        // Scale every block in this row: B <- D^{-1} B. The diagonal block
        // becomes the identity by construction.
        for (index_t k = rows[i]; k < rows[i + 1]; ++k)
        {
          mat_float *B = &vals[static_cast<std::size_t>(k) * NB];
          mat_float tmp[NB];
          for (int r = 0; r < NE_I; ++r)
          {
            for (int c = 0; c < NE_I; ++c)
            {
              mat_float acc = 0.0;
              for (int m = 0; m < NE_I; ++m)
                acc += inv[r * NE_I + m] * B[m * NE_I + c];
              tmp[r * NE_I + c] = acc;
            }
          }
          std::copy_n(tmp, NB, B);
        }
      }
    }

    template <std::uint8_t N_BLOCK_SIZE>
    int linsolv_fs_cpr<N_BLOCK_SIZE>::setup_p_prec_from_block_(
        opendarts::linear_solvers::csr_matrix_base *P_block)
    {
      // Nested block CPR (linsolv_cpr<NE>) consumes the block-NE PPSS matrix
      // as-is: it does its own true-IMPES pressure decoupling internally and
      // needs the natural (not diag-first) column order that
      // extract_sub_block_to_block_csr produces. No scalar expansion.
      if (p_prec_takes_block_)
        return p_system_preconditioner_->setup(P_block);

      // Otherwise scalar-expand the block-NE PPSS subsystem for the
      // block-size-1 pressure sub-preconditioner (hypre_amg_adapter<1>): same
      // scalar DOF count, memory-correct -- the former direct
      // setup(P_block_N) was a misread of the block layout through a bare
      // static_cast (caught by the checked linsolv_iface_bos down-cast).
      // Matches the NE == 1 path's diag-first column convention for the HYPRE
      // IJ build; P_scalar_ne_ is regenerated from the block matrix every
      // setup, so no set_diag_in_order restore is needed.
      P_scalar_ne_->to_nb_1(P_block);
      fill_scalar_diag_ind(*P_scalar_ne_);
      set_diag_first<1>(*P_scalar_ne_);
      return p_system_preconditioner_->setup(P_scalar_ne_.get());
    }

    template <std::uint8_t N_BLOCK_SIZE>
    int linsolv_fs_cpr<N_BLOCK_SIZE>::refresh_p_prec_()
    {
      // Only the NE == 1 scalar pressure subsystem owns a standalone
      // single-stage preconditioner that the FS-CPR class drives directly.
      // For NE > 1 the pressure stage is the multi-equation PPSS path,
      // which gets its setup() through approx_schur_complement_ -- the
      // caller invokes that separately and the refresh helper has nothing
      // to do here.
      if (NE_ == 1)
      {
        return p_system_preconditioner_->refresh(P_scalar_.get());
      }
      return 0;
    }

    // ====================================================================
    // solve
    // ====================================================================
    template <std::uint8_t N_BLOCK_SIZE>
    int linsolv_fs_cpr<N_BLOCK_SIZE>::solve(
        opendarts::config::mat_float *B, opendarts::config::mat_float *X)
    {
      if (this->timer_solve)
        this->timer_solve->node["FS-CPR"].start();
      int res = run_FS_UP_solve_(B, X);
      if (this->timer_solve)
        this->timer_solve->node["FS-CPR"].stop();
      return res;
    }

    // ====================================================================
    // Stage divergence guard
    // ====================================================================
    // FS-CPR's stages are single BoomerAMG V-cycles. A V-cycle is only a
    // contraction when the operator is close enough to an M-matrix for its
    // point smoother to converge; when it is not -- e.g. an MPFA pressure
    // block on a strongly heterogeneous full-tensor field, where the row can
    // carry positive off-diagonals large enough to destroy diagonal dominance
    // -- the V-cycle DIVERGES. Crucially it does so quietly: HYPRE reports no
    // error and the returned vector is finite, just enormous. The outer Krylov
    // then stalls at a relative residual of 1, classifies that as "budget
    // exhausted, residual did not regress" (solve_result::not_converged), and
    // the default on_linear_nonconvergence = 'accept' policy applies the
    // useless step -- so the run burns minutes per Newton iteration forever
    // instead of cutting the timestep. Catching it here turns an unbounded
    // silent stall into one diagnosable line plus a timestep cut.
    template <std::uint8_t N_BLOCK_SIZE>
    bool linsolv_fs_cpr<N_BLOCK_SIZE>::stage_diverged_(const char *stage,
        const std::vector<mat_float> &in,
        const std::vector<mat_float> &out) const
    {
      mat_float max_out = 0.0;
      for (const mat_float v : out)
      {
        if (!std::isfinite(v))
        {
          std::cerr << "linsolv_fs_cpr: " << stage
                    << " stage returned a non-finite value" << std::endl;
          return true;
        }
        const mat_float a = std::fabs(v);
        if (a > max_out) max_out = a;
      }
      if (!(stage_growth_cap_ > 0.0))
        return false;

      mat_float max_in = 0.0;
      for (const mat_float v : in)
      {
        const mat_float a = std::fabs(v);
        if (a > max_in) max_in = a;
      }
      // An all-zero RHS must produce an all-zero correction; scaling by a
      // vanishing max_in would otherwise make the ratio meaningless.
      const mat_float ref = std::max(max_in, std::numeric_limits<mat_float>::min());
      if (max_out > stage_growth_cap_ * ref)
      {
        std::cerr << "linsolv_fs_cpr: " << stage
                  << " stage diverged -- max|in|=" << max_in
                  << " max|out|=" << max_out
                  << " (amplification " << (max_out / ref)
                  << " exceeds cap " << stage_growth_cap_
                  << "). The sub-preconditioner is not a contraction on this "
                       "operator; failing the solve so the timestep is cut."
                  << std::endl;
        return true;
      }
      return false;
    }

    template <std::uint8_t N_BLOCK_SIZE>
    int linsolv_fs_cpr<N_BLOCK_SIZE>::run_FS_UP_solve_(
        opendarts::config::mat_float *B, opendarts::config::mat_float *X)
    {
      assert(A_block_ != nullptr);
      const opendarts::linear_solvers::block_csr_matrix &A = *A_block_;
      const std::uint8_t NE = NE_;

      // ----- diagnostic: max|B| / max|B[P]| / max|B[U]| at solve entry ------
      // Only print on the first ~5 solve calls to avoid swamping the log.
      static int solve_call_count = 0;
      const bool dbg_print_this_call = fs_cpr_debug_ && (solve_call_count < 5);
      ++solve_call_count;
      if (dbg_print_this_call)
      {
        const index_t n_rows_full =
            static_cast<index_t>(A.n_rows) * static_cast<index_t>(N_BLOCK_SIZE);
        mat_float max_B = 0.0;
        for (index_t i = 0; i < n_rows_full; ++i)
          if (std::fabs(B[i]) > max_B) max_B = std::fabs(B[i]);
        mat_float max_BP = 0.0;
        if (NE > 1)
        {
          for (const auto &i : PPSS_.global_to_local_rows)
            if (std::fabs(B[i]) > max_BP) max_BP = std::fabs(B[i]);
        }
        else
        {
          for (const auto &i : PP_.global_to_local_rows)
            if (std::fabs(B[i]) > max_BP) max_BP = std::fabs(B[i]);
        }
        mat_float max_BU = 0.0;
        for (const auto &i : UU_.global_to_local_rows)
          if (std::fabs(B[i]) > max_BU) max_BU = std::fabs(B[i]);
        std::fprintf(stderr,
            "[FS-CPR.solve#%d] max|B|=%.3e max|B[P]|=%.3e max|B[U]|=%.3e\n",
            solve_call_count, max_B, max_BP, max_BU);
        std::fflush(stderr);
      }

      // ----- assemble pressure (or PPSS) RHS ------
      std::fill(P_B_.begin(), P_B_.end(), 0.0);
      {
        std::size_t i_loc = 0;
        if (NE > 1)
        {
          for (const auto &i : PPSS_.global_to_local_rows)
          {
            P_B_[i_loc] = ps_rhs_mults_[i_loc] * B[i];
            ++i_loc;
          }
          // Match the left scaling applied to the matrix in
          // decouple_ppss_block_: b_i <- D_i^{-1} b_i, per cell.
          if (p_decouple_block_diag_ && !ps_decouple_inv_.empty())
          {
            const int ne = static_cast<int>(NE);
            const std::size_t nb = static_cast<std::size_t>(ne) * ne;
            const std::size_t n_cells = P_B_.size() / static_cast<std::size_t>(ne);
            std::vector<mat_float> tmp(static_cast<std::size_t>(ne));
            for (std::size_t c = 0; c < n_cells; ++c)
            {
              const mat_float *inv = &ps_decouple_inv_[c * nb];
              const mat_float *b = &P_B_[c * static_cast<std::size_t>(ne)];
              for (int r = 0; r < ne; ++r)
              {
                mat_float acc = 0.0;
                for (int m = 0; m < ne; ++m) acc += inv[r * ne + m] * b[m];
                tmp[static_cast<std::size_t>(r)] = acc;
              }
              std::copy_n(tmp.data(), ne, &P_B_[c * static_cast<std::size_t>(ne)]);
            }
          }
        }
        else
        {
          for (const auto &i : PP_.global_to_local_rows)
          {
            P_B_[i_loc] = ps_rhs_mults_[i_loc] * B[i];
            ++i_loc;
          }
        }
      }

      // ----- pressure (or PPSS) solve ------
      std::fill(P_X_.begin(), P_X_.end(), 0.0);
      int res = 0;
      if (NE > 1)
      {
        res = p_system_preconditioner_->solve(P_B_.data(), P_X_.data());
      }
      else
      {
        set_diag_first<1>(*P_scalar_);
        res = p_system_preconditioner_->solve(P_B_.data(), P_X_.data());
        set_diag_in_order<1>(*P_scalar_);
      }
      if (res) return res;
      if (stage_diverged_("pressure", P_B_, P_X_))
        return -8;

      if (dbg_print_this_call)
      {
        mat_float max_PX = 0.0;
        for (std::size_t k = 0; k < P_X_.size(); ++k)
          if (std::fabs(P_X_[k]) > max_PX) max_PX = std::fabs(P_X_[k]);
        std::fprintf(stderr,
            "[FS-CPR.solve#%d] after p_prec: max|P_X|=%.3e\n",
            solve_call_count, max_PX);
        std::fflush(stderr);
      }

      // ----- assemble displacement RHS ------
      if (NE > 1)
      {
        // A_U,flow * P_X in ONE product over the joint (P, S) column block.
        // P_X_ is the joint PPSS correction interleaved at stride NE (see the
        // scatter below, which walks PPSS_.global_to_local_rows), so the
        // source stride must be NE. Splitting this into UP_ (column stride 1)
        // and US_ (column stride NE - 1) products read the wrong entries of
        // P_X_ -- for NE == 2 both landed on P_X_[j] and the second half of
        // the vector was never read at all -- which left the displacement
        // stage correcting a residual that was not the true one.
        std::fill(UPS_B_.begin(), UPS_B_.end(), 0.0);
        block_vector_product(A, UPS_, P_X_.data(), UPS_B_.data());

        std::size_t i_loc = 0;
        for (const auto &i : UU_.global_to_local_rows)
        {
          U_B_[i_loc] = u_rhs_mults_[i_loc] * (B[i] - UPS_B_[i_loc]);
          ++i_loc;
        }
      }
      else
      {
        std::fill(U_B_.begin(), U_B_.end(), 0.0);
        block_vector_product(A, UP_, P_X_.data(), U_B_.data());
        std::size_t i_loc = 0;
        for (const auto &i : UU_.global_to_local_rows)
        {
          U_B_[i_loc] = u_rhs_mults_[i_loc] * (B[i] - U_B_[i_loc]);
          ++i_loc;
        }
      }

      if (dbg_print_this_call)
      {
        mat_float max_UB = 0.0;
        for (std::size_t k = 0; k < U_B_.size(); ++k)
          if (std::fabs(U_B_[k]) > max_UB) max_UB = std::fabs(U_B_[k]);
        std::fprintf(stderr,
            "[FS-CPR.solve#%d] U_B (after back-sub): max|U_B|=%.3e\n",
            solve_call_count, max_UB);
        std::fflush(stderr);
      }

      // ----- displacement solve ------
      std::fill(U_X_.begin(), U_X_.end(), 0.0);
      res = u_system_preconditioner_->solve(U_B_.data(), U_X_.data());
      if (res) return res;
      if (stage_diverged_("displacement", U_B_, U_X_))
        return -8;

      if (dbg_print_this_call)
      {
        mat_float max_UX = 0.0;
        for (std::size_t k = 0; k < U_X_.size(); ++k)
          if (std::fabs(U_X_[k]) > max_UX) max_UX = std::fabs(U_X_[k]);
        std::fprintf(stderr,
            "[FS-CPR.solve#%d] after u_prec: max|U_X|=%.3e\n",
            solve_call_count, max_UX);
        std::fflush(stderr);
      }

      // ----- scatter X ------
      {
        std::size_t i_loc = 0;
        if (NE > 1)
        {
          for (const auto &i : PPSS_.global_to_local_rows)
            X[i] = P_X_[i_loc++];
        }
        else
        {
          for (const auto &i : PP_.global_to_local_rows)
            X[i] = P_X_[i_loc++];
        }
      }
      {
        std::size_t i_loc = 0;
        for (const auto &i : UU_.global_to_local_rows)
          X[i] = U_X_[i_loc++];
      }

      if (dbg_print_this_call)
      {
        const index_t n_rows_full =
            static_cast<index_t>(A.n_rows) * static_cast<index_t>(N_BLOCK_SIZE);
        mat_float max_X = 0.0;
        for (index_t i = 0; i < n_rows_full; ++i)
          if (std::fabs(X[i]) > max_X) max_X = std::fabs(X[i]);
        std::fprintf(stderr,
            "[FS-CPR.solve#%d] after scatter: max|X|=%.3e\n",
            solve_call_count, max_X);
        std::fflush(stderr);
      }

      return 0;
    }

    // Explicit template instantiations — generated by CMake's
    // od_emit_template_instantiations(); see solvers/src/CMakeLists.txt.
    // Edit the block-size range there, not here.
#include "linsolv_fs_cpr_instantiations.inc"
  } // namespace linear_solvers
} // namespace opendarts
