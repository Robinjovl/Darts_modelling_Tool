// Numerical unit tests for linsolv_schur_elim<N, K>: exact block-local
// Schur-complement elimination of K cell-local (diagonal-block-only) equation/unknown pairs.
//
// Coverage (host path; the CUDA path shares the same detection/chain logic and
// per-cell formulas -- see the model-level GPU regression runs):
//   1. K = 1..3 with non-default row/column mappings: the eliminated solve must
//      reproduce a dense LU reference solution of the FULL system
//      (inner solver: in-tree GMRES, tight tolerance).
//   2. Well-style chains: a cell whose preferred eliminated rows are unusable
//      (control-equation zeros) with state-copy rows coupling to a neighbour --
//      including several chains on one cell -- must still solve exactly.
//   3. Singular preferred rows with an invertible alternative combination
//      (greedy fails, exhaustive fallback succeeds).
//   4. Pivot degeneration on a later setup -> setup must FAIL loudly (return
//      nonzero), not produce NaN.
//   5. Chain-topology change on a later setup (an eliminated-row off-diagonal
//      entry that was 0 at detection becomes nonzero) -> setup must FAIL.
//   6. Invalid mappings (out-of-range / duplicate / pressure column / wrong
//      length) -> constructor must throw.

#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

#include "csr_matrix.hpp"
#include "linsolv_schur_elim.hpp"
#include "linsolv_gmres.hpp"
#include "solver_configs.hpp"

using opendarts::config::index_t;
using opendarts::config::mat_float;

namespace
{
  // Dense Gaussian elimination with partial pivoting: reference solve.
  bool dense_solve(std::vector<mat_float> A, std::vector<mat_float> b,
      std::vector<mat_float> &x, int n)
  {
    x.assign(n, 0.0);
    for (int col = 0; col < n; col++)
    {
      int piv = col;
      for (int r = col + 1; r < n; r++)
        if (std::fabs(A[r * n + col]) > std::fabs(A[piv * n + col]))
          piv = r;
      if (A[piv * n + col] == 0.0)
        return false;
      if (piv != col)
      {
        for (int c = 0; c < n; c++)
          std::swap(A[col * n + c], A[piv * n + c]);
        std::swap(b[col], b[piv]);
      }
      for (int r = col + 1; r < n; r++)
      {
        const mat_float f = A[r * n + col] / A[col * n + col];
        if (f == 0.0)
          continue;
        for (int c = col; c < n; c++)
          A[r * n + c] -= f * A[col * n + c];
        b[r] -= f * b[col];
      }
    }
    for (int r = n - 1; r >= 0; r--)
    {
      mat_float s = b[r];
      for (int c = r + 1; c < n; c++)
        s -= A[r * n + c] * x[c];
      x[r] = s / A[r * n + r];
    }
    return true;
  }

  // Build a 1D chain-of-cells block system with cell-local (diagonal-block-only) eliminated rows.
  // e_rows/e_cols: the K eliminated pairs; cells in `well_cells` get zeroed
  // preferred rows (control-style) and identity state-copy rows coupled to the
  // previous cell with coefficient -1 (a one-level chain).
  template <uint8_t N>
  struct block_system
  {
    index_t n_cells;
    std::vector<index_t> rows_ptr, cols_ind, diag_ind;
    std::vector<mat_float> values; // block row-major
    std::vector<mat_float> rhs;

    opendarts::linear_solvers::csr_matrix<N> matrix;

    void to_matrix()
    {
      matrix.init(n_cells, n_cells, (index_t)cols_ind.size());
      std::copy(rows_ptr.begin(), rows_ptr.end(), matrix.rows_ptr.begin());
      std::copy(cols_ind.begin(), cols_ind.end(), matrix.cols_ind.begin());
      matrix.diag_ind.resize(n_cells);
      std::copy(diag_ind.begin(), diag_ind.end(), matrix.diag_ind.begin());
      std::copy(values.begin(), values.end(), matrix.values.begin());
    }

    // dense copy for the reference solve
    void to_dense(std::vector<mat_float> &A) const
    {
      const int n = (int)n_cells * N;
      A.assign((size_t)n * n, 0.0);
      for (index_t i = 0; i < n_cells; i++)
        for (index_t b = rows_ptr[i]; b < rows_ptr[i + 1]; b++)
        {
          const index_t j = cols_ind[b];
          for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++)
              A[((size_t)i * N + r) * n + (size_t)j * N + c] =
                  values[(size_t)b * N * N + r * N + c];
        }
    }
  };

  template <uint8_t N>
  block_system<N> make_system(index_t n_cells, const std::vector<int> &e_rows,
      const std::vector<int> &e_cols, const std::vector<index_t> &well_cells,
      unsigned seed)
  {
    block_system<N> sys;
    sys.n_cells = n_cells;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> u(-0.5, 0.5);
    const int K = (int)e_cols.size();

    auto is_erow = [&](int r) {
      for (int k = 0; k < K; k++)
        if (e_rows[k] == r)
          return true;
      return false;
    };
    auto is_well = [&](index_t i) {
      for (index_t w : well_cells)
        if (w == i)
          return true;
      return false;
    };

    sys.rows_ptr.push_back(0);
    for (index_t i = 0; i < n_cells; i++)
    {
      std::vector<index_t> nbrs;
      if (i > 0)
        nbrs.push_back(i - 1);
      nbrs.push_back(i);
      if (i + 1 < n_cells)
        nbrs.push_back(i + 1);
      for (index_t j : nbrs)
      {
        if (j == i)
          sys.diag_ind.push_back((index_t)sys.cols_ind.size());
        sys.cols_ind.push_back(j);
        std::vector<mat_float> B((size_t)N * N, 0.0);
        if (j == i)
        {
          for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++)
              B[r * N + c] = u(gen);
          for (int r = 0; r < N; r++)
            B[r * N + r] += 6.0; // diagonal dominance for a solvable reference
          // the eliminated pairing gets a strong pivot
          for (int k = 0; k < K; k++)
            B[e_rows[k] * N + e_cols[k]] += 4.0;
          if (is_well(i))
          {
            // preferred rows become control-style: zero derivative w.r.t. every
            // eliminated column (forces the fallback row search) ...
            for (int k = 0; k < K; k++)
              for (int kk = 0; kk < K; kk++)
                B[e_rows[k] * N + e_cols[kk]] = 0.0;
            // ... and K other rows become identity state-copies of e_cols
            for (int k = 0; k < K; k++)
            {
              int copy_row = -1;
              for (int r = 0; r < N && copy_row < 0; r++)
                if (!is_erow(r))
                {
                  bool taken = false;
                  for (int kk = 0; kk < k; kk++)
                    if (r == (N - 1 - kk))
                      taken = true;
                  if (!taken && r == N - 1 - k)
                    copy_row = r;
                }
              copy_row = N - 1 - k;
              for (int c = 0; c < N; c++)
                B[copy_row * N + c] = 0.0;
              B[copy_row * N + e_cols[k]] = 1.0;
            }
          }
        }
        else
        {
          // off-diagonal: eliminated rows are ZERO (cell-local) for regular
          // cells; fluid rows couple randomly
          for (int r = 0; r < N; r++)
          {
            if (is_erow(r))
              continue;
            for (int c = 0; c < N; c++)
              B[r * N + c] = 0.3 * u(gen);
          }
          if (is_well(i))
          {
            // chain: the well state-copy rows reference the neighbour's
            // eliminated unknowns with coefficient -1
            for (int k = 0; k < K; k++)
            {
              const int copy_row = N - 1 - k;
              for (int c = 0; c < N; c++)
                B[copy_row * N + c] = 0.0;
              B[copy_row * N + e_cols[k]] = -1.0;
            }
          }
        }
        sys.values.insert(sys.values.end(), B.begin(), B.end());
      }
      sys.rows_ptr.push_back((index_t)sys.cols_ind.size());
    }
    sys.rhs.resize((size_t)n_cells * N);
    for (auto &v : sys.rhs)
      v = u(gen);
    sys.to_matrix();
    return sys;
  }

  // Full pipeline: wrapper(superlu inner) solve vs dense reference.
  template <uint8_t N, uint8_t K>
  int check_exact(const std::vector<int> &e_rows, const std::vector<int> &e_cols,
      const std::vector<index_t> &well_cells, unsigned seed, const char *label,
      double tol = 1e-8)
  {
    auto sys = make_system<N>(24, e_rows, e_cols, well_cells, seed);

    opendarts::linear_solvers::linsolv_schur_elim<N, K> wrapper(
        /*on_device=*/false, e_rows, e_cols, 0.0);
    auto *inner = new opendarts::linear_solvers::linsolv_gmres<N - K>;
    {
      opendarts::linear_solvers::gmres_solver_config gc;
      gc.restart = 60; gc.tolerance = 1e-13; gc.max_iterations = 500;
      inner->reconfigure(gc);   // the factory path always reconfigures GMRES
    }
    wrapper.set_prec(inner);
    wrapper.set_inner_owned(true);
    static timer_node t_setup, t_solve;   // solvers expect engine-provided timers
    wrapper.init_timer_nodes(&t_setup, &t_solve);
    if (wrapper.init(&sys.matrix, 500, 1e-13))
    {
      printf("  [%s] FAIL: init\n", label);
      return 1;
    }
    if (wrapper.setup(&sys.matrix))
    {
      printf("  [%s] FAIL: setup\n", label);
      return 1;
    }
    std::vector<mat_float> x((size_t)sys.n_cells * N, 0.0);
    if (wrapper.solve(sys.rhs.data(), x.data()))
    {
      printf("  [%s] FAIL: solve\n", label);
      return 1;
    }
    std::vector<mat_float> A_dense, x_ref;
    sys.to_dense(A_dense);
    if (!dense_solve(A_dense, sys.rhs, x_ref, (int)sys.n_cells * N))
    {
      printf("  [%s] FAIL: dense reference singular\n", label);
      return 1;
    }
    double err = 0.0, ref = 0.0;
    for (size_t i = 0; i < x.size(); i++)
    {
      err = std::max(err, std::fabs((double)x[i] - (double)x_ref[i]));
      ref = std::max(ref, std::fabs((double)x_ref[i]));
    }
    const double rel = err / (ref > 0 ? ref : 1.0);
    if (!(rel < tol))
    {
      printf("  [%s] FAIL: max rel err %.3e\n", label, rel);
      return 1;
    }
    printf("  [%s] ok (max rel err %.3e)\n", label, rel);
    return 0;
  }
} // namespace

int main()
{
  int err = 0;

  // 1) exactness across K and NON-DEFAULT mappings, chain-free
  err += check_exact<5, 1>({0}, {1}, {}, 11, "K=1 N=5 rows[0] cols[1]");
  err += check_exact<5, 1>({2}, {3}, {}, 12, "K=1 N=5 rows[2] cols[3] (non-default)");
  err += check_exact<7, 2>({0, 1}, {1, 2}, {}, 13, "K=2 N=7 default-map");
  err += check_exact<7, 2>({1, 3}, {4, 2}, {}, 14, "K=2 N=7 shuffled map");
  err += check_exact<8, 3>({0, 1, 2}, {1, 2, 3}, {}, 15, "K=3 N=8");

  // 2) well-style chains (fallback rows + one-level dependencies), incl. K=2
  //    (two chains on one cell)
  err += check_exact<5, 1>({0}, {1}, {23}, 21, "K=1 N=5 well chain (end cell)");
  err += check_exact<5, 1>({0}, {1}, {0}, 22, "K=1 N=5 well chain (first cell)");
  err += check_exact<7, 2>({0, 1}, {1, 2}, {23}, 23, "K=2 N=7 well (2 chains on end cell)");

  // 3) singular preferred rows, invertible alternative: preferred rows give a
  //    structurally singular pivot (both rows depend only on col e_cols[0]);
  //    rows {0, 2} would work. Build manually.
  {
    std::vector<int> e_rows = {0, 1}, e_cols = {1, 2};
    auto sys = make_system<6>(10, e_rows, e_cols, {}, 31);
    for (index_t i = 0; i < sys.n_cells; i++)
    {
      // make the preferred pairing singular at every cell: zero column e_cols[1]
      // in rows 0 and 1, put its coupling into row 2 instead (row 2 is made
      // cell-local below, like a state-copy row, so the fallback can
      // select it without creating chains)
      mat_float *D = &sys.values[(size_t)sys.diag_ind[i] * 36];
      D[0 * 6 + 2] = 0.0;
      D[1 * 6 + 2] = 0.0;   // rows {0,1} now only cover column 1 -> singular
      D[2 * 6 + 2] += 5.0;  // row 2 has the strong col-2 pivot
      for (index_t b = sys.rows_ptr[i]; b < sys.rows_ptr[i + 1]; b++)
      {
        if (sys.cols_ind[b] == i)
          continue;
        for (int c = 0; c < 6; c++)
          sys.values[(size_t)b * 36 + 2 * 6 + c] = 0.0;   // row 2 cell-local
      }
    }
    sys.to_matrix();
    opendarts::linear_solvers::linsolv_schur_elim<6, 2> wrapper(false, e_rows, e_cols, 0.0);
    auto *inner = new opendarts::linear_solvers::linsolv_gmres<4>;
    {
      opendarts::linear_solvers::gmres_solver_config gc;
      gc.restart = 60; gc.tolerance = 1e-13; gc.max_iterations = 500;
      inner->reconfigure(gc);
    }
    wrapper.set_prec(inner);
    wrapper.set_inner_owned(true);
    static timer_node t_setup62, t_solve62;
    wrapper.init_timer_nodes(&t_setup62, &t_solve62);
    int rc = wrapper.init(&sys.matrix, 500, 1e-13);
    rc = rc ? rc : wrapper.setup(&sys.matrix);
    std::vector<mat_float> x((size_t)sys.n_cells * 6, 0.0);
    rc = rc ? rc : wrapper.solve(sys.rhs.data(), x.data());
    std::vector<mat_float> A_dense, x_ref;
    sys.to_dense(A_dense);
    dense_solve(A_dense, sys.rhs, x_ref, (int)sys.n_cells * 6);
    double rel = 0.0, ref = 0.0;
    for (size_t q = 0; q < x.size(); q++)
    {
      rel = std::max(rel, std::fabs((double)x[q] - (double)x_ref[q]));
      ref = std::max(ref, std::fabs((double)x_ref[q]));
    }
    rel /= (ref > 0 ? ref : 1.0);
    if (rc || !(rel < 1e-8))
    {
      printf("  [singular-preferred fallback] FAIL (rc=%d rel=%.3e)\n", rc, rel);
      err += 1;
    }
    else
      printf("  [singular-preferred fallback] ok (max rel err %.3e)\n", rel);
  }

  // 4) pivot degeneration on a later setup -> loud failure
  {
    std::vector<int> e_rows = {0}, e_cols = {1};
    auto sys = make_system<5>(8, e_rows, e_cols, {}, 41);
    opendarts::linear_solvers::linsolv_schur_elim<5, 1> wrapper(false, e_rows, e_cols, 0.0);
    auto *inner = new opendarts::linear_solvers::linsolv_gmres<4>;
    {
      opendarts::linear_solvers::gmres_solver_config gc;
      gc.restart = 60; gc.tolerance = 1e-13; gc.max_iterations = 500;
      inner->reconfigure(gc);
    }
    wrapper.set_prec(inner);
    wrapper.set_inner_owned(true);
    static timer_node t_setup51, t_solve51;
    wrapper.init_timer_nodes(&t_setup51, &t_solve51);
    int rc = wrapper.init(&sys.matrix, 500, 1e-13);
    rc = rc ? rc : wrapper.setup(&sys.matrix);
    if (rc)
    {
      printf("  [pivot degeneration] FAIL: first setup rc=%d\n", rc);
      err += 1;
    }
    else
    {
      // degenerate EVERY potential pivot for column 1 at cell 3: whole column
      // zero -> no row (preferred or fallback) can eliminate it any more
      mat_float *D = &sys.matrix.values[(size_t)sys.diag_ind[3] * 25];
      for (int r = 0; r < 5; r++)
        D[r * 5 + 1] = 0.0;
      if (wrapper.setup(&sys.matrix) == 0)
      {
        printf("  [pivot degeneration] FAIL: setup accepted a singular pivot\n");
        err += 1;
      }
      else
        printf("  [pivot degeneration] ok (setup failed loudly)\n");
    }
  }

  // 5) chain-topology change on a later setup -> loud failure
  {
    std::vector<int> e_rows = {0}, e_cols = {1};
    auto sys = make_system<5>(8, e_rows, e_cols, {}, 51);
    opendarts::linear_solvers::linsolv_schur_elim<5, 1> wrapper(false, e_rows, e_cols, 0.0);
    auto *inner = new opendarts::linear_solvers::linsolv_gmres<4>;
    {
      opendarts::linear_solvers::gmres_solver_config gc;
      gc.restart = 60; gc.tolerance = 1e-13; gc.max_iterations = 500;
      inner->reconfigure(gc);
    }
    wrapper.set_prec(inner);
    wrapper.set_inner_owned(true);
    static timer_node t_setup51, t_solve51;
    wrapper.init_timer_nodes(&t_setup51, &t_solve51);
    int rc = wrapper.init(&sys.matrix, 500, 1e-13);
    rc = rc ? rc : wrapper.setup(&sys.matrix);
    if (rc)
    {
      printf("  [topology change] FAIL: first setup rc=%d\n", rc);
      err += 1;
    }
    else
    {
      // an eliminated-row off-diagonal entry that was 0.0 at detection becomes
      // nonzero (e.g. a well control switch): must be detected, not dropped
      index_t off_blk = (index_t)-1;
      for (index_t b = sys.rows_ptr[4]; b < sys.rows_ptr[5]; b++)
        if (sys.cols_ind[b] != 4)
          off_blk = b;
      sys.matrix.values[(size_t)off_blk * 25 + 0 * 5 + 3] = 0.7;
      if (wrapper.setup(&sys.matrix) == 0)
      {
        printf("  [topology change] FAIL: setup silently dropped a new coupling\n");
        err += 1;
      }
      else
        printf("  [topology change] ok (setup failed loudly)\n");
    }
  }

  // 6) invalid mappings must throw in the constructor
  {
    auto expect_throw = [&](const char *what, std::vector<int> er, std::vector<int> ec) {
      try
      {
        opendarts::linear_solvers::linsolv_schur_elim<5, 1> w(false, er, ec, 0.0);
        (void)w;
        printf("  [ctor validation: %s] FAIL: no throw\n", what);
        return 1;
      }
      catch (const std::exception &)
      {
        printf("  [ctor validation: %s] ok\n", what);
        return 0;
      }
    };
    err += expect_throw("col out of range", {0}, {7});
    err += expect_throw("pressure column", {0}, {0});
    err += expect_throw("row out of range", {9}, {1});
    err += expect_throw("wrong length", {0, 1}, {1});
    try
    {
      opendarts::linear_solvers::linsolv_schur_elim<5, 2> w(false, {0, 1}, {1, 1}, 0.0);
      (void)w;
      printf("  [ctor validation: duplicate col] FAIL: no throw\n");
      err += 1;
    }
    catch (const std::exception &)
    {
      printf("  [ctor validation: duplicate col] ok\n");
    }
  }

  printf("schur_elim unit test: %s (%d error(s))\n", err ? "FAIL" : "PASS", err);
  return err;
}
