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

#include "solver_factories.hpp"

#include <cstddef>
#include <typeinfo>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "csr_matrix.hpp"
#include "linear_solver.hpp"
#include "linsolv_cpr.hpp"
#include "linsolv_schur_elim.hpp"
#include "linsolv_fs_cpr.hpp"
#include "linsolv_gmres.hpp"
#include "linsolv_hypre_amg.hpp"
#include "linsolv_iface_bos.hpp"
#include "linsolv_mgr.hpp"
#include "linsolv_superlu.hpp"
#include "solver_config.hpp"
#include "solver_configs.hpp"
#include "solver_registry.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    namespace
    {
      // open-DARTS instantiates the block-templated solvers for N = 1..13.
      constexpr int MIN_BLOCK_SIZE = 1;
      constexpr int MAX_BLOCK_SIZE = 13;

      using solver_handle = std::shared_ptr<opendarts::linear_solvers::linear_solver>;

      // Resolve the typed configuration for a factory. Exact type passes
      // through; a plain base solver_config yields the solver's defaults
      // carrying the base tolerance/max_iterations; any OTHER derived config
      // type is a caller bug -- previously the entire user configuration was
      // silently discarded and the solver ran on defaults.
      template <class ConfigT>
      ConfigT resolve_config(const opendarts::linear_solvers::solver_config &config,
          const char *solver_name)
      {
        if (const auto *typed = dynamic_cast<const ConfigT *>(&config))
          return *typed;
        if (typeid(config) == typeid(opendarts::linear_solvers::solver_config))
        {
          ConfigT defaults;
          defaults.tolerance = config.tolerance;
          defaults.max_iterations = config.max_iterations;
          return defaults;
        }
        throw std::runtime_error(
            std::string("openDARTS linear solver registry: solver '") + solver_name +
            "' was given a configuration of mismatched type " +
            typeid(config).name() +
            "; pass the solver's own config class (or a plain SolverConfig "
            "for defaults).");
      }


      // ---- MGR (HYPRE Multigrid Reduction) ---------------------------------

      // Construct and configure an MGR solver for a compile-time block size.
      template <uint8_t N_BLOCK_SIZE>
      solver_handle build_mgr(const opendarts::linear_solvers::mgr_solver_config &config)
      {
        // reconfigure() is the single source of config application (shared
        // with live mid-run reconfiguration).
        auto solver = std::make_shared<opendarts::linear_solvers::linsolv_mgr<N_BLOCK_SIZE>>();
        solver->reconfigure(config);
        return solver;
      }

      // Runtime block-size dispatch for MGR.
      solver_handle build_mgr_for_block_size(int block_size,
          const opendarts::linear_solvers::mgr_solver_config &config)
      {
        switch (block_size)
        {
          case 1:  return build_mgr<1>(config);
          case 2:  return build_mgr<2>(config);
          case 3:  return build_mgr<3>(config);
          case 4:  return build_mgr<4>(config);
          case 5:  return build_mgr<5>(config);
          case 6:  return build_mgr<6>(config);
          case 7:  return build_mgr<7>(config);
          case 8:  return build_mgr<8>(config);
          case 9:  return build_mgr<9>(config);
          case 10: return build_mgr<10>(config);
          case 11: return build_mgr<11>(config);
          case 12: return build_mgr<12>(config);
          case 13: return build_mgr<13>(config);
          default:
            throw std::runtime_error("MGR solver: unsupported block size " +
                std::to_string(block_size) + " (supported: " +
                std::to_string(MIN_BLOCK_SIZE) + ".." + std::to_string(MAX_BLOCK_SIZE) + ").");
        }
      }

      // Factory registered under the name "mgr".
      solver_handle make_mgr_solver(const opendarts::linear_solvers::solver_config &config,
          int block_size)
      {
        // Use the MGR-specific configuration if one was supplied; a plain
        // solver_config falls back to the MGR defaults (carrying the base
        // fields); a mismatched derived config type throws.
        const auto resolved_config =
            resolve_config<opendarts::linear_solvers::mgr_solver_config>(config, "mgr");
        const opendarts::linear_solvers::mgr_solver_config *mgr_config = &resolved_config;

        return build_mgr_for_block_size(block_size, *mgr_config);
      }

      // ---- SuperLU (direct solver) -----------------------------------------

      template <uint8_t N_BLOCK_SIZE>
      solver_handle build_superlu()
      {
        return std::make_shared<opendarts::linear_solvers::linsolv_superlu<N_BLOCK_SIZE>>();
      }

      solver_handle build_superlu_for_block_size(int block_size)
      {
        switch (block_size)
        {
          case 1:  return build_superlu<1>();
          case 2:  return build_superlu<2>();
          case 3:  return build_superlu<3>();
          case 4:  return build_superlu<4>();
          case 5:  return build_superlu<5>();
          case 6:  return build_superlu<6>();
          case 7:  return build_superlu<7>();
          case 8:  return build_superlu<8>();
          case 9:  return build_superlu<9>();
          case 10: return build_superlu<10>();
          case 11: return build_superlu<11>();
          case 12: return build_superlu<12>();
          case 13: return build_superlu<13>();
          default:
            throw std::runtime_error("SuperLU solver: unsupported block size " +
                std::to_string(block_size) + " (supported: " +
                std::to_string(MIN_BLOCK_SIZE) + ".." + std::to_string(MAX_BLOCK_SIZE) + ").");
        }
      }

      // Factory registered under the name "superlu". SuperLU is a direct solver
      // and takes no parameters beyond the matrix, so the configuration is unused.
      solver_handle make_superlu_solver(
          const opendarts::linear_solvers::solver_config & /*config*/, int block_size)
      {
        return build_superlu_for_block_size(block_size);
      }
      // ---- GMRES (open-source restarted Krylov, see linsolv_gmres) ---------

      template <uint8_t N_BLOCK_SIZE>
      solver_handle build_gmres(const opendarts::linear_solvers::gmres_solver_config &config)
      {
        // reconfigure() is the single source of config application (shared
        // with live mid-run reconfiguration).
        auto solver = std::make_shared<opendarts::linear_solvers::linsolv_gmres<N_BLOCK_SIZE>>();
        solver->reconfigure(config);
        return solver;
      }

      solver_handle build_gmres_for_block_size(int block_size,
          const opendarts::linear_solvers::gmres_solver_config &config)
      {
        switch (block_size)
        {
          case 1:  return build_gmres<1>(config);
          case 2:  return build_gmres<2>(config);
          case 3:  return build_gmres<3>(config);
          case 4:  return build_gmres<4>(config);
          case 5:  return build_gmres<5>(config);
          case 6:  return build_gmres<6>(config);
          case 7:  return build_gmres<7>(config);
          case 8:  return build_gmres<8>(config);
          case 9:  return build_gmres<9>(config);
          case 10: return build_gmres<10>(config);
          case 11: return build_gmres<11>(config);
          case 12: return build_gmres<12>(config);
          case 13: return build_gmres<13>(config);
          default:
            throw std::runtime_error("GMRES solver: unsupported block size " +
                std::to_string(block_size) + " (supported: " +
                std::to_string(MIN_BLOCK_SIZE) + ".." + std::to_string(MAX_BLOCK_SIZE) + ").");
        }
      }

      // Factory registered under the name "gmres".
      solver_handle make_gmres_solver(
          const opendarts::linear_solvers::solver_config &config, int block_size)
      {
        const auto gmres_config =
            resolve_config<opendarts::linear_solvers::gmres_solver_config>(config, "gmres");
        return build_gmres_for_block_size(block_size, gmres_config);
      }

      // ---- CPR (open-source two-stage CPR preconditioner) ------------------

      template <uint8_t N_BLOCK_SIZE>
      solver_handle build_cpr(const opendarts::linear_solvers::cpr_solver_config &config)
      {
        // reconfigure() is the single source of config application (shared
        // with live mid-run reconfiguration).
        auto solver = std::make_shared<opendarts::linear_solvers::linsolv_cpr<N_BLOCK_SIZE>>();
        solver->reconfigure(config);
        return solver;
      }

      solver_handle build_cpr_for_block_size(int block_size,
          const opendarts::linear_solvers::cpr_solver_config &config)
      {
        switch (block_size)
        {
          case 1:  return build_cpr<1>(config);
          case 2:  return build_cpr<2>(config);
          case 3:  return build_cpr<3>(config);
          case 4:  return build_cpr<4>(config);
          case 5:  return build_cpr<5>(config);
          case 6:  return build_cpr<6>(config);
          case 7:  return build_cpr<7>(config);
          case 8:  return build_cpr<8>(config);
          case 9:  return build_cpr<9>(config);
          case 10: return build_cpr<10>(config);
          case 11: return build_cpr<11>(config);
          case 12: return build_cpr<12>(config);
          case 13: return build_cpr<13>(config);
          default:
            throw std::runtime_error("CPR solver: unsupported block size " +
                std::to_string(block_size));
        }
      }

      // Factory registered under the name "cpr".
      solver_handle make_cpr_solver(
          const opendarts::linear_solvers::solver_config &config, int block_size)
      {
        const auto cpr_config =
            resolve_config<opendarts::linear_solvers::cpr_solver_config>(config, "cpr");
        return build_cpr_for_block_size(block_size, cpr_config);
      }

      // ---- Schur mineral elimination (linsolv_schur_elim) ------------------
      //
      // Wrapper: exact per-cell condensation of one flux-free mineral
      // equation/unknown pair, then the inner solver (attached by the caller
      // via set_prec, built for block size N-1) runs on the reduced system.

      template <uint8_t N_BLOCK_SIZE>
      solver_handle build_schur_elim(const opendarts::linear_solvers::schur_elim_solver_config &config)
      {
        return std::make_shared<opendarts::linear_solvers::linsolv_schur_elim<N_BLOCK_SIZE>>(
            /*on_device=*/false, config.elim_col, (uint8_t)config.elim_row, config.pivot_eps);
      }

      solver_handle build_schur_elim_for_block_size(int block_size,
          const opendarts::linear_solvers::schur_elim_solver_config &config)
      {
        switch (block_size)
        {
          case 2:  return build_schur_elim<2>(config);
          case 3:  return build_schur_elim<3>(config);
          case 4:  return build_schur_elim<4>(config);
          case 5:  return build_schur_elim<5>(config);
          case 6:  return build_schur_elim<6>(config);
          case 7:  return build_schur_elim<7>(config);
          case 8:  return build_schur_elim<8>(config);
          case 9:  return build_schur_elim<9>(config);
          case 10: return build_schur_elim<10>(config);
          case 11: return build_schur_elim<11>(config);
          case 12: return build_schur_elim<12>(config);
          case 13: return build_schur_elim<13>(config);
          default:
            throw std::runtime_error("Schur elimination wrapper: unsupported block size " +
                std::to_string(block_size) + " (supported: 2..13).");
        }
      }

      // Factory registered under the name "schur_elim".
      solver_handle make_schur_elim_solver(
          const opendarts::linear_solvers::solver_config &config, int block_size)
      {
        const auto se_config =
            resolve_config<opendarts::linear_solvers::schur_elim_solver_config>(config, "schur_elim");
        return build_schur_elim_for_block_size(block_size, se_config);
      }

      // ---- FS-CPR (4-block poromechanics CPR) -----------------------------
      //
      // Two-stage poromechanics CPR: HYPRE BoomerAMG correction on the
      // displacement (U) subsystem followed by a second BoomerAMG correction
      // on the flow / pressure (P or PPSS) subsystem driven by a Schur
      // complement. Sub-preconditioners are created here and injected via
      // ``set_prec`` -- nested preconditioner spec injection is not yet
      // supported. Block sizes 4..8 are supported (ND = 3, NE = N - 3).
      //
      // Sub-prec adapter: linsolv_hypre_amg<N> intentionally does NOT inherit
      // from linsolv_iface (it predates the unified interface and wraps HYPRE
      // directly via typed csr_matrix<N>* entry points -- see comment in
      // linsolv_hypre_amg.hpp). linsolv_fs_cpr's set_prec(shared_ptr<linsolv_iface>)
      // therefore needs a thin adapter that surfaces hypre_amg through the
      // linsolv_iface_bos<N> interface. The adapter owns its wrapped solver
      // (unique_ptr -- linsolv_fs_cpr stores shared ownership of the adapter,
      // so the adapter outlives any apply). It forwards init / setup / solve
      // / refresh to the typed entry points, with the polymorphic
      // csr_matrix_base downcast already handled by linsolv_iface_bos<N>.
      //
      // refresh() forwarding is what gives FS-CPR the HYPRE value-only fast
      // path on subsequent Newton iterations: linsolv_fs_cpr calls
      // u_system_preconditioner_->refresh(U_.get()) polymorphically; the
      // adapter override below downcasts the csr_matrix_base back to
      // csr_matrix<N>* and forwards to linsolv_hypre_amg<N>::refresh, which
      // reuses the existing IJ matrix and BoomerAMG hierarchy and only
      // re-pushes the new values. Without this override the call would hit
      // linear_solver's default refresh() (full setup()), rebuilding the
      // AMG hierarchy every Newton iteration.
      template <std::uint8_t N_BLOCK_SIZE>
      class hypre_amg_adapter
          : public opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>
      {
       public:
        hypre_amg_adapter()
            : inner_(std::make_unique<opendarts::linear_solvers::linsolv_hypre_amg<N_BLOCK_SIZE>>())
        {
          // linsolv_iface_bos<N> stores a linear_solver_base* for legacy
          // callers; we are not one, mirror linsolv_fs_cpr's choice.
          this->solver = nullptr;
        }

        int set_prec(opendarts::linear_solvers::linsolv_iface *prec_in) override
        {
          return inner_->set_prec(prec_in);
        }

        // Keep the polymorphic csr_matrix_base overloads from linsolv_iface_bos<N>
        // visible alongside the typed csr_matrix<N>* overloads we provide.
        using opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::init;
        using opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::setup;

        int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_in,
            int max_iters, double tolerance) override
        {
          return inner_->init(A_in,
              static_cast<opendarts::config::index_t>(max_iters),
              static_cast<opendarts::config::mat_float>(tolerance));
        }

        int setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_in) override
        {
          return inner_->setup(A_in);
        }

        // Value-only refresh fast path. The override is on the polymorphic
        // csr_matrix_base entry point because linsolv_fs_cpr calls
        // u_system_preconditioner_->refresh(U_.get()) through the
        // linear_solver base; the U_ / P_scalar_ matrices come back to us as
        // csr_matrix_base* even though they are concretely csr_matrix<N>*.
        int refresh(opendarts::linear_solvers::csr_matrix_base *A_in) override
        {
          auto *A_typed = dynamic_cast<
              opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *>(A_in);
          if (A_typed == nullptr)
          {
            // Pathological -- caller handed us something that isn't a
            // csr_matrix<N_BLOCK_SIZE>. Fall back to a full setup() so the
            // result is still correct; this matches linear_solver's default.
            return this->setup(A_in);
          }
          inner_->refresh(A_typed);
          return 0;
        }

        int solve(opendarts::config::mat_float *B,
            opendarts::config::mat_float *X) override
        {
          return inner_->solve(B, X);
        }

        int get_n_iters() override
        {
          return static_cast<int>(inner_->get_n_iters());
        }

        opendarts::config::mat_float get_residual() override
        {
          return inner_->get_residual();
        }

       private:
        std::unique_ptr<opendarts::linear_solvers::linsolv_hypre_amg<N_BLOCK_SIZE>> inner_;
      };

      template <std::uint8_t N_BLOCK_SIZE>
      solver_handle build_fs_cpr(
          const opendarts::linear_solvers::fs_cpr_solver_config &config,
          std::uint8_t P_VAR, std::uint8_t Z_VAR,
          std::uint8_t U_VAR, std::uint8_t NC)
      {
        // Default sub-preconditioners: linsolv_hypre_amg<1> for both U and
        // PPSS stages, exposed through the linsolv_iface_bos<1> adapter above.
        // Each is created with a single V-cycle budget (preconditioner mode
        // -- tolerance is irrelevant); linsolv_fs_cpr drives init/setup/solve
        // through its private p_/u_system_preconditioner_ handles.
        // The per-stage V-cycle budgets (p_amg_max_iters / u_amg_max_iters) from
        // the spec are applied below via set_amg_sweeps -> the sub-preconditioner
        // init(max_iters) calls in linsolv_fs_cpr::build_subsystem_matrices_.
        // The default (1, 1) reproduces the previous hard-coded single sweep.
        // Default U/P sub-precs: BoomerAMG via the hypre_amg_adapter shim.
        // The bisection-diagnostic SuperLU-for-U swap that lived here while
        // we were tracking down the divergence has been reverted -- root
        // cause was a missing HYPRE_IJVectorGetValues call in
        // linsolv_hypre_amg::solve (and matching in linsolv_hypre_ilu),
        // which made the wrapper appear to return identically-zero solutions.
        auto u_prec = std::make_shared<hypre_amg_adapter<1>>();
        auto p_prec = std::make_shared<hypre_amg_adapter<1>>();

        auto solver = std::make_shared<opendarts::linear_solvers::linsolv_fs_cpr<N_BLOCK_SIZE>>(
            P_VAR, Z_VAR, U_VAR, NC);
        solver->set_force_amg_asymmetric(config.force_amg_asymmetric);
        solver->set_block_sizes(config.n_res, config.n_fracs, config.n_wells);
        solver->set_amg_sweeps(config.p_amg_max_iters, config.u_amg_max_iters);
        // 2-arg set_prec; G-prec is not used in the FS_UP path.
        solver->set_prec(p_prec, u_prec);
        return solver;
      }

      solver_handle build_fs_cpr_for_block_size(int block_size,
          const opendarts::linear_solvers::fs_cpr_solver_config &config,
          std::uint8_t P_VAR, std::uint8_t Z_VAR,
          std::uint8_t U_VAR, std::uint8_t NC)
      {
        switch (block_size)
        {
          case 4: return build_fs_cpr<4>(config, P_VAR, Z_VAR, U_VAR, NC);
          case 5: return build_fs_cpr<5>(config, P_VAR, Z_VAR, U_VAR, NC);
          case 6: return build_fs_cpr<6>(config, P_VAR, Z_VAR, U_VAR, NC);
          case 7: return build_fs_cpr<7>(config, P_VAR, Z_VAR, U_VAR, NC);
          case 8: return build_fs_cpr<8>(config, P_VAR, Z_VAR, U_VAR, NC);
          default:
            throw std::runtime_error("FS-CPR solver: unsupported block size " +
                std::to_string(block_size) + " (supported: 4..8)");
        }
      }

      // Factory registered under the name "fs_cpr".
      //
      // The variable-index info (P_VAR, Z_VAR, U_VAR, NC) is supplied
      // through the fs_cpr_solver_config overrides; any field left at -1
      // falls back to the engine_super_elastic_cpu convention default
      // derived from block_size: ND = 3, P_VAR = 0, Z_VAR = 1,
      // U_VAR = NE, NC = NE (NE = block_size - 3, no THERMAL).
      // engine_pm_cpu users override with p_var=3, u_var=0, z_var=255 and
      // nc=1.
      solver_handle make_fs_cpr_solver(
          const opendarts::linear_solvers::solver_config &config, int block_size)
      {
        const auto resolved_config =
            resolve_config<opendarts::linear_solvers::fs_cpr_solver_config>(config, "fs_cpr");
        const opendarts::linear_solvers::fs_cpr_solver_config *fs_config = &resolved_config;
        const std::uint8_t ND = 3;
        const std::uint8_t NE = static_cast<std::uint8_t>(block_size - ND);
        // Default to engine_super_elastic_cpu convention; an explicit
        // (non-negative) override on the config wins.
        auto pick = [](int override_value, std::uint8_t convention_default)
        {
          return override_value < 0
                     ? convention_default
                     : static_cast<std::uint8_t>(override_value);
        };
        const std::uint8_t P_VAR = pick(fs_config->p_var, 0);
        const std::uint8_t Z_VAR = pick(fs_config->z_var, 1);
        const std::uint8_t U_VAR = pick(fs_config->u_var, NE);
        const std::uint8_t NC    = pick(fs_config->nc, NE);
        return build_fs_cpr_for_block_size(block_size, *fs_config,
            P_VAR, Z_VAR, U_VAR, NC);
      }
    } // anonymous namespace

    void register_builtin_solvers()
    {
      // Self-guarded: repeated calls are no-ops without relying on the
      // registry's duplicate handling, so a register_solver() duplicate
      // warning always indicates a genuine name conflict.
      static bool done = false;
      if (done)
        return;
      done = true;
      register_solver("mgr", make_mgr_solver);
      register_solver("superlu", make_superlu_solver);
      register_solver("gmres", make_gmres_solver);
      register_solver("cpr", make_cpr_solver);
      register_solver("schur_elim", make_schur_elim_solver);
      register_solver("fs_cpr", make_fs_cpr_solver);
    }
  } // namespace linear_solvers
} // namespace opendarts
