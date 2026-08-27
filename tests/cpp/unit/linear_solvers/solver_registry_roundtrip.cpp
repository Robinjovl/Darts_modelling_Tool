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
  errors += test_superlu_block_csr();
  errors += test_gmres_cpr_transposed();

  if (errors == 0)
    std::cout << "solver_registry_roundtrip: all tests passed" << std::endl;
  return errors;
}
