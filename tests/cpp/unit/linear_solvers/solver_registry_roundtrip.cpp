// Unit test for the unified solver registry (solver_registry.hpp /
// solver_factories.cpp) and the NEW in-tree solvers it builds:
//   1. registry round-trip -- register_builtin_solvers() exposes the expected
//      names and create_linear_solver() builds each of them;
//   2. FGMRES + CPR (the CPU default stack) solves a small block-CSR system
//      to tolerance (residual asserted via the matrix's own SpMV);
//   3. SuperLU consumes a block_csr_matrix directly (the scalar_csr_adapter
//      path) and produces an exact solve;
//   4. GMRES+CPR solve_transposed (CPRA) solves A^T x = b -- verified against
//      a dense transpose assembled by probing the forward SpMV.
// Before this test the new solver line-up had no unit coverage at all
// (MR280_MASTER_PLAN.md workstream B6).

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "block_csr_matrix.hpp"
#include "solver_configs.hpp"
#include "solver_factories.hpp"
#include "solver_registry.hpp"
#include "sparsity_pattern.hpp"

using opendarts::config::index_t;
using opendarts::config::mat_float;
using opendarts::linear_solvers::block_csr_matrix;
using opendarts::linear_solvers::sparsity_pattern;

namespace
{
  constexpr int BS = 2;        // block size
  constexpr index_t NB = 4;    // block rows
  constexpr index_t N = NB * BS; // scalar unknowns

  int report(const char *name, bool ok)
  {
    std::cout << "solver_registry_roundtrip: " << name
              << (ok ? " OK" : " FAILED") << std::endl;
    return ok ? 0 : 1;
  }

  // 4x4 block tridiagonal pattern (10 nonzero blocks, diagonal present).
  std::shared_ptr<sparsity_pattern> make_tridiagonal()
  {
    const std::vector<index_t> row_ptr = {0, 2, 5, 8, 10};
    const std::vector<index_t> col_ind = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    return std::make_shared<sparsity_pattern>(NB, NB, row_ptr.data(), col_ind.data());
  }

  // Diagonally dominant, NON-symmetric block system (so the transpose solve
  // is a real test, not a no-op).
  std::shared_ptr<block_csr_matrix> make_system()
  {
    auto a = std::make_shared<block_csr_matrix>(make_tridiagonal(), BS);
    mat_float *v = a->values();
    const std::vector<index_t> row_ptr = {0, 2, 5, 8, 10};
    const std::vector<index_t> col_ind = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    for (index_t i = 0; i < NB; ++i)
    {
      for (index_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
      {
        mat_float *blk = v + static_cast<std::size_t>(k) * BS * BS;
        if (col_ind[k] == i)
        {
          blk[0] = 10.0 + i; blk[1] = 1.0;
          blk[2] = 0.5;      blk[3] = 12.0 + i;
        }
        else
        {
          const mat_float s = (col_ind[k] > i) ? -1.0 : -2.0; // non-symmetric
          blk[0] = s;   blk[1] = 0.25;
          blk[2] = 0.0; blk[3] = s;
        }
      }
    }
    return a;
  }

  // Manual block-CSR SpMV from the raw arrays. Deliberately NOT
  // csr_matrix_base::matrix_vector_product -- that deprecated shim silently
  // returns zeros for a block_csr_matrix (verified), which would make every
  // residual check here meaningless.
  void spmv(block_csr_matrix &A, const mat_float *x, mat_float *r) // accessors are non-const
  {
    const index_t *rows = A.get_rows_ptr();
    const index_t *cols = A.get_cols_ind();
    const mat_float *vals = A.get_values();
    for (index_t i = 0; i < NB; ++i)
      for (index_t k = rows[i]; k < rows[i + 1]; ++k)
      {
        const mat_float *blk = vals + static_cast<std::size_t>(k) * BS * BS;
        for (int e = 0; e < BS; ++e)
          for (int w = 0; w < BS; ++w)
            r[i * BS + e] += blk[e * BS + w] * x[cols[k] * BS + w];
      }
  }

  mat_float residual_norm(block_csr_matrix &A, const std::vector<mat_float> &x,
      const std::vector<mat_float> &b)
  {
    std::vector<mat_float> r(N, 0.0);
    spmv(A, x.data(), r.data());
    mat_float nrm = 0.0, bn = 0.0;
    for (index_t i = 0; i < N; ++i)
    {
      nrm += (r[i] - b[i]) * (r[i] - b[i]);
      bn += b[i] * b[i];
    }
    return std::sqrt(nrm) / std::max(std::sqrt(bn), mat_float(1e-30));
  }

  int test_registry_roundtrip()
  {
    const std::vector<std::string> expected = {"cpr", "fs_cpr", "gmres", "mgr", "superlu"};
    bool ok = true;
    for (const auto &name : expected)
      ok = ok && opendarts::linear_solvers::is_solver_registered(name);
    // Each name must actually build a solver from its default config.
    if (ok)
    {
      opendarts::linear_solvers::gmres_solver_config gmres_cfg;
      opendarts::linear_solvers::cpr_solver_config cpr_cfg;
      opendarts::linear_solvers::mgr_solver_config mgr_cfg;
      opendarts::linear_solvers::fs_cpr_solver_config fs_cfg;
      opendarts::linear_solvers::solver_config base_cfg;
      ok = ok && (opendarts::linear_solvers::create_linear_solver("gmres", gmres_cfg, BS) != nullptr);
      ok = ok && (opendarts::linear_solvers::create_linear_solver("cpr", cpr_cfg, BS) != nullptr);
      ok = ok && (opendarts::linear_solvers::create_linear_solver("mgr", mgr_cfg, BS) != nullptr);
      ok = ok && (opendarts::linear_solvers::create_linear_solver("superlu", base_cfg, BS) != nullptr);
      // fs_cpr is the poromechanics (4-block) preconditioner; at block size 2
      // it may either build or reject with a clean exception -- both are
      // acceptable, a crash is not.
      try
      {
        ok = ok && (opendarts::linear_solvers::create_linear_solver("fs_cpr", fs_cfg, BS) != nullptr);
      }
      catch (const std::exception &)
      {
        // clean rejection accepted
      }
    }
#ifdef WITH_GPU
    // CUDA build: the device-resident chains for host-assembled systems
    // (linsolv_host_adapter) are registered under the gpu_* names.
    ok = ok && opendarts::linear_solvers::is_solver_registered("gpu_gmres_ilu0");
    ok = ok && opendarts::linear_solvers::is_solver_registered("gpu_cusolver");
#ifdef WITH_CUDSS
    ok = ok && opendarts::linear_solvers::is_solver_registered("gpu_cudss");
#endif
#ifdef WITH_AMGX
    ok = ok && opendarts::linear_solvers::is_solver_registered("gpu_gmres_cpr_amgx");
    ok = ok && opendarts::linear_solvers::is_solver_registered("gpu_bicgstab_cpr_amgx");
#endif
    if (ok)
    {
      // Building must succeed at any supported block size without touching a
      // device (construction only allocates on init()).
      opendarts::linear_solvers::gpu_solver_config gpu_cfg;
      ok = ok && (opendarts::linear_solvers::create_linear_solver("gpu_gmres_ilu0", gpu_cfg, BS) != nullptr);
      // A plain solver_config resolves to the gpu defaults.
      opendarts::linear_solvers::solver_config base_cfg2;
      ok = ok && (opendarts::linear_solvers::create_linear_solver("gpu_gmres_ilu0", base_cfg2, BS) != nullptr);
    }
#else
    ok = ok && !opendarts::linear_solvers::is_solver_registered("gpu_gmres_ilu0");
    ok = ok && !opendarts::linear_solvers::is_solver_registered("gpu_cudss");
#endif
    // Unknown names must throw, not crash.
    if (ok)
    {
      bool threw = false;
      try
      {
        opendarts::linear_solvers::solver_config cfg;
        opendarts::linear_solvers::create_linear_solver("no_such_solver", cfg, BS);
      }
      catch (const std::exception &)
      {
        threw = true;
      }
      ok = threw;
    }
    return report("registry round-trip", ok);
  }

  int test_gmres_cpr_solve()
  {
    auto A = make_system();
    opendarts::linear_solvers::gmres_solver_config gmres_cfg;
    gmres_cfg.restart = 30;
    opendarts::linear_solvers::cpr_solver_config cpr_cfg;
    auto gmres = opendarts::linear_solvers::create_linear_solver("gmres", gmres_cfg, BS);
    auto cpr = opendarts::linear_solvers::create_linear_solver("cpr", cpr_cfg, BS);
    gmres->set_prec(cpr.get());

    const int rc_init = gmres->init(A.get(), 200, 1e-10);
    const int rc_setup = gmres->setup(A.get());

    std::vector<mat_float> b(N), x(N, 0.0);
    for (index_t i = 0; i < N; ++i)
      b[i] = 1.0 + 0.1 * i;
    const int rc_solve = gmres->solve(b.data(), x.data());
    const mat_float res = residual_norm(*A, x, b);
    const bool ok = rc_init == 0 && rc_setup == 0 && rc_solve == 0 && res < 1e-8;
    if (!ok)
      std::cout << "  [diag] init=" << rc_init << " setup=" << rc_setup
                << " solve=" << rc_solve << " n_iters=" << gmres->get_n_iters()
                << " rel_resid=" << res << std::endl;
    return report("FGMRES+CPR forward solve", ok);
  }

  int test_gmres_not_converged_code()
  {
    // Unified solve() convention: an exhausted iteration budget on a finite,
    // non-regressing iterate must report solve_result::not_converged (+1) --
    // NOT 0 (the legacy silent accept) and NOT a hard failure. A single Krylov
    // iteration at an unreachable tolerance guarantees exhaustion; GMRES's
    // per-cycle residual minimisation guarantees the iterate did not regress.
    auto A = make_system();
    opendarts::linear_solvers::gmres_solver_config gmres_cfg;
    gmres_cfg.restart = 30;
    opendarts::linear_solvers::cpr_solver_config cpr_cfg;
    auto gmres = opendarts::linear_solvers::create_linear_solver("gmres", gmres_cfg, BS);
    auto cpr = opendarts::linear_solvers::create_linear_solver("cpr", cpr_cfg, BS);
    gmres->set_prec(cpr.get());

    const int rc_init = gmres->init(A.get(), /*max_iters=*/1, /*tol=*/1e-30);
    const int rc_setup = gmres->setup(A.get());

    std::vector<mat_float> b(N), x(N, 0.0);
    for (index_t i = 0; i < N; ++i)
      b[i] = 1.0 + 0.1 * i;
    const int rc_solve = gmres->solve(b.data(), x.data());
    const mat_float res = residual_norm(*A, x, b);
    bool finite = true;
    for (index_t i = 0; i < N; ++i)
      finite = finite && std::isfinite(static_cast<double>(x[i]));
    const auto st = gmres->stats();
    const bool ok = rc_init == 0 && rc_setup == 0 &&
        rc_solve == opendarts::linear_solvers::solve_result::not_converged &&
        finite && res <= 1.0 + 1e-12 && !st.converged && st.iterations >= 1;
    if (!ok)
      std::cout << "  [diag] init=" << rc_init << " setup=" << rc_setup
                << " solve=" << rc_solve << " rel_resid=" << res
                << " stats.converged=" << st.converged
                << " stats.iters=" << st.iterations << std::endl;
    return report("FGMRES+CPR not-converged status (+1, usable iterate)", ok);
  }

  // A system whose second equation is CELL-LOCAL: the off-diagonal blocks carry
  // no entries in row 1, so unknown 1 can be Schur-eliminated without forming a
  // multi-level chain. This mirrors the intended use (cell-local mineral
  // balances) -- make_system()'s generic tridiagonal cannot be eliminated at
  // BS=2, since column 0 is the reserved pressure column.
  std::shared_ptr<block_csr_matrix> make_schur_system()
  {
    auto a = std::make_shared<block_csr_matrix>(make_tridiagonal(), BS);
    mat_float *v = a->values();
    const std::vector<index_t> row_ptr = {0, 2, 5, 8, 10};
    const std::vector<index_t> col_ind = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    for (index_t i = 0; i < NB; ++i)
    {
      for (index_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
      {
        mat_float *blk = v + static_cast<std::size_t>(k) * BS * BS;
        if (col_ind[k] == i)
        {
          blk[0] = 10.0 + i; blk[1] = 1.0;
          blk[2] = 0.5;      blk[3] = 12.0 + i;  // invertible local pivot
        }
        else
        {
          const mat_float s = (col_ind[k] > i) ? -1.0 : -2.0;
          blk[0] = s;   blk[1] = 0.25;
          blk[2] = 0.0; blk[3] = 0.0;  // row 1 is cell-local: no coupling
        }
      }
    }
    return a;
  }

  int test_schur_elim_propagates_usable_status()
  {
    // The wrapper must PROPAGATE a positive (usable) inner status AND still run
    // its back-substitution -- returning early on a positive code would hand
    // back an X whose eliminated unknowns were never written.
    // Each phase uses a FRESH wrapper: linsolv_schur_elim records its
    // elimination topology at setup and rejects a re-init against the same
    // matrix, so reusing one instance would test that guard, not this contract.
    auto A = make_schur_system();
    std::vector<mat_float> b(N);
    for (index_t i = 0; i < N; ++i)
      b[i] = 1.0 + 0.1 * i;

    auto build = [&](int max_iters, mat_float tol, std::vector<mat_float> &x, int &rc) {
      opendarts::linear_solvers::schur_elim_solver_config se_cfg;
      // Eliminate unknown 1 via its cell-local equation row 1 (column 0 is the
      // reserved pressure column and may not be eliminated).
      se_cfg.elim_rows = {1};
      se_cfg.elim_cols = {1};
      auto se = opendarts::linear_solvers::create_linear_solver("schur_elim", se_cfg, BS);
      opendarts::linear_solvers::gmres_solver_config gmres_cfg;
      gmres_cfg.restart = 30;
      // The inner solver runs on the REDUCED system, so it must be built at the
      // reduced block size BS - K (the engine's chain does the same). Building
      // it at BS makes the inner size its workspace as n_rows * BS and write
      // past the reduced vectors.
      constexpr int K_ELIM = 1;
      auto inner = opendarts::linear_solvers::create_linear_solver(
          "gmres", gmres_cfg, BS - K_ELIM);
      if (!se || !inner)
        return false;
      se->set_prec(inner.get());
      if (se->init(A.get(), max_iters, tol) != 0 || se->setup(A.get()) != 0)
        return false;
      rc = se->solve(b.data(), x.data());
      return true;
    };

    std::vector<mat_float> x_exh(N, 0.0), x_conv(N, 0.0);
    int rc_exh = 0, rc_conv = 0;
    const bool built_exh = build(/*max_iters=*/1, /*tol=*/1e-30, x_exh, rc_exh);
    const bool built_conv = build(/*max_iters=*/200, /*tol=*/1e-10, x_conv, rc_conv);
    // This target expects schur_elim to be available at BS with K=1; a factory,
    // init or setup failure is a regression, not a reason to skip.
    if (!built_exh || !built_conv)
    {
      std::cout << "  [diag] schur_elim construction/setup failed (exh=" << built_exh
                << " conv=" << built_conv << ")" << std::endl;
      return report("schur_elim wrapper propagates +1 and back-substitutes", false);
    }

    bool finite_exh = true;
    for (index_t i = 0; i < N; ++i)
      finite_exh = finite_exh && std::isfinite(static_cast<double>(x_exh[i]));
    // back-substitution writes the eliminated unknowns; an early return on the
    // positive inner status would leave every one of them at its initial 0.
    bool elim_written = false;
    for (index_t blk = 0; blk < N / BS; ++blk)
      elim_written = elim_written || (x_exh[blk * BS + 1] != 0.0);

    const mat_float res_conv = residual_norm(*A, x_conv, b);
    const bool ok =
        rc_exh == opendarts::linear_solvers::solve_result::not_converged &&
        finite_exh && elim_written && rc_conv == 0 && res_conv < 1e-8;
    if (!ok)
      std::cout << "  [diag] rc_exhausted=" << rc_exh << " (expected +1)"
                << " finite=" << finite_exh << " elim_written=" << elim_written
                << " rc_converged=" << rc_conv << " res=" << res_conv << std::endl;
    return report("schur_elim wrapper propagates +1 and back-substitutes", ok);
  }

  // NOTE: a synthetic real-HYPRE MGR *exhaustion* test is deliberately absent.
  // MGR needs the pressure/composition block roles of a real physics matrix: on
  // this 4-block harness it solves exactly in one iteration (residual ~5e-15, so
  // no iteration limit is reachable), and on synthetic stiff or large
  // tridiagonal systems its reduction degenerates and returns a non-finite
  // iterate. The iteration-limit path is therefore covered by a real-model run
  // (2ph_do with MGR at max_iterations=1) rather than fabricated here; see the
  // MR discussion. What IS covered automatically is the equivalent CPU-GMRES
  // path above, which shares the classification contract.

  int test_superlu_block_csr()
  {
    auto A = make_system();
    opendarts::linear_solvers::solver_config cfg;
    auto slu = opendarts::linear_solvers::create_linear_solver("superlu", cfg, BS);
    bool ok = slu->init(A.get(), 1, 1e-12) == 0;
    ok = ok && slu->setup(A.get()) == 0;
    std::vector<mat_float> b(N), x(N, 0.0);
    for (index_t i = 0; i < N; ++i)
      b[i] = 1.0 - 0.05 * i;
    ok = ok && slu->solve(b.data(), x.data()) == 0;
    // Direct solve: exact up to round-off.
    ok = ok && residual_norm(*A, x, b) < 1e-12;
    return report("SuperLU on block_csr_matrix (scalar_csr_adapter path)", ok);
  }

  int test_gmres_cpr_transposed()
  {
    auto A = make_system();
    // Dense A by probing the forward SpMV with unit vectors; column j of A
    // is row j of A^T.
    std::vector<mat_float> dense(static_cast<std::size_t>(N) * N, 0.0);
    for (index_t j = 0; j < N; ++j)
    {
      std::vector<mat_float> e(N, 0.0), col(N, 0.0);
      e[j] = 1.0;
      spmv(*A, e.data(), col.data()); // col = A e_j
      for (index_t i = 0; i < N; ++i)
        dense[static_cast<std::size_t>(i) * N + j] = col[i];
    }

    opendarts::linear_solvers::gmres_solver_config gmres_cfg;
    opendarts::linear_solvers::cpr_solver_config cpr_cfg;
    auto gmres = opendarts::linear_solvers::create_linear_solver("gmres", gmres_cfg, BS);
    auto cpr = opendarts::linear_solvers::create_linear_solver("cpr", cpr_cfg, BS);
    gmres->set_prec(cpr.get());
    bool ok = gmres->init(A.get(), 200, 1e-10) == 0;
    ok = ok && gmres->setup(A.get()) == 0;

    std::vector<mat_float> b(N), x(N, 0.0);
    for (index_t i = 0; i < N; ++i)
      b[i] = 0.5 + 0.2 * i;
    ok = ok && gmres->solve_transposed(b.data(), x.data()) == 0;

    // residual of A^T x - b via the dense transpose
    mat_float nrm = 0.0, bn = 0.0;
    for (index_t i = 0; i < N; ++i)
    {
      mat_float acc = 0.0;
      for (index_t j = 0; j < N; ++j)
        acc += dense[static_cast<std::size_t>(j) * N + i] * x[j]; // (A^T)_ij = A_ji
      nrm += (acc - b[i]) * (acc - b[i]);
      bn += b[i] * b[i];
    }
    ok = ok && std::sqrt(nrm / bn) < 1e-8;
    return report("FGMRES+CPRA transposed solve", ok);
  }
} // namespace

int main()
{
  // The pybind module normally does this; tests must do it themselves.
  opendarts::linear_solvers::register_builtin_solvers();

  int errors = 0;
  errors += test_registry_roundtrip();
  errors += test_gmres_cpr_solve();
  errors += test_gmres_not_converged_code();
  errors += test_schur_elim_propagates_usable_status();
  errors += test_superlu_block_csr();
  errors += test_gmres_cpr_transposed();

  if (errors == 0)
    std::cout << "solver_registry_roundtrip: all tests passed" << std::endl;
  return errors;
}
