// Unit test for sparsity_pattern and its scalar-CSR expansion csr_expansion
// (SOLVER_REFACTORING_PLAN.md section 12.4 / 12.6).

#include <iostream>
#include <string>
#include <vector>

#include "csr_expansion.hpp"
#include "sparsity_pattern.hpp"

using opendarts::config::index_t;
using opendarts::config::mat_float;
using opendarts::linear_solvers::csr_expansion;
using opendarts::linear_solvers::sparsity_pattern;

namespace
{
  int report(const char *name, bool ok)
  {
    if (!ok)
      std::cout << "sparsity_pattern: " << name << " FAILED" << std::endl;
    return ok ? 0 : 1;
  }

  // A 4x4 block tridiagonal pattern, every row carrying its diagonal block.
  //   row 0: cols 0,1        row 1: cols 0,1,2
  //   row 2: cols 1,2,3      row 3: cols 2,3
  sparsity_pattern make_tridiagonal()
  {
    const std::vector<index_t> row_ptr = {0, 2, 5, 8, 10};
    const std::vector<index_t> col_ind = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    return sparsity_pattern(4, 4, row_ptr.data(), col_ind.data());
  }

  int test_dimensions()
  {
    sparsity_pattern p = make_tridiagonal();
    bool ok = p.n_block_rows() == 4 && p.n_block_cols() == 4 && p.n_blocks() == 10;
    return report("dimensions", ok);
  }

  int test_diag_ind()
  {
    sparsity_pattern p = make_tridiagonal();
    const index_t *d = p.diag_ind();
    const index_t expected[] = {0, 3, 6, 9};
    bool ok = true;
    for (int i = 0; i < 4 && ok; ++i)
      ok = (d[i] == expected[i]);
    return report("diagonal block indices", ok);
  }

  int test_validate_accepts_valid()
  {
    sparsity_pattern p = make_tridiagonal();
    std::string msg;
    bool ok = p.validate(&msg);
    if (!ok)
      std::cout << "  validate rejected a valid pattern: " << msg << std::endl;
    return report("validate accepts valid pattern", ok);
  }

  int test_validate_rejects_missing_diagonal()
  {
    // Row 1 has no diagonal block (no column 1).
    const std::vector<index_t> row_ptr = {0, 1, 2};
    const std::vector<index_t> col_ind = {0, 0};
    sparsity_pattern p(2, 2, row_ptr.data(), col_ind.data());
    std::string msg;
    bool ok = !p.validate(&msg); // expected to be rejected
    return report("validate rejects missing diagonal", ok);
  }

  int test_thread_partition()
  {
    sparsity_pattern p = make_tridiagonal();
    p.compute_row_thread_starts(2);
    const index_t *ts = p.row_thread_starts();
    // nnzb = 10; the split should put 5 blocks on each thread: rows [0,2) | [2,4).
    bool ok = p.n_thread_partitions() == 2 && ts[0] == 0 && ts[1] == 2 && ts[2] == 4;
    return report("thread partition balances nonzero blocks", ok);
  }

  int test_scalar_csr_expansion()
  {
    sparsity_pattern p = make_tridiagonal();
    const csr_expansion &e = p.scalar_csr(2); // block size 2

    bool ok = e.n_rows() == 8 && e.n_cols() == 8 && e.nnz() == 40 && e.block_size() == 2;

    // Block row 0 (2 blocks) -> 4 scalar entries per scalar row;
    // block row 1 (3 blocks) -> 6; block row 2 (3) -> 6; block row 3 (2) -> 4.
    const index_t *rp = e.row_ptr();
    const index_t expected_rp[] = {0, 4, 8, 14, 20, 26, 32, 36, 40};
    for (int i = 0; i < 9 && ok; ++i)
      ok = (rp[i] == expected_rp[i]);

    // Scalar row 0: blocks at block-cols 0 and 1 -> scalar cols {0,1,2,3}.
    const index_t *ci = e.col_ind();
    const index_t expected_ci0[] = {0, 1, 2, 3};
    for (int k = 0; k < 4 && ok; ++k)
      ok = ok && (ci[k] == expected_ci0[k]);

    // gather_values with bsr_values[i] = i must reproduce value_map exactly.
    std::vector<mat_float> bsr(static_cast<std::size_t>(p.n_blocks()) * 4);
    for (std::size_t i = 0; i < bsr.size(); ++i)
      bsr[i] = static_cast<mat_float>(i);
    std::vector<mat_float> csr(static_cast<std::size_t>(e.nnz()));
    e.gather_values(bsr.data(), csr.data());
    const index_t *vm = e.value_map();
    for (index_t k = 0; k < e.nnz() && ok; ++k)
      ok = (csr[k] == static_cast<mat_float>(vm[k]));

    // The expansion is cached: a second call returns the same object.
    ok = ok && (&p.scalar_csr(2) == &e);

    return report("scalar-CSR expansion", ok);
  }
} // namespace

int main()
{
  int error_output = 0;

  error_output += test_dimensions();
  error_output += test_diag_ind();
  error_output += test_validate_accepts_valid();
  error_output += test_validate_rejects_missing_diagonal();
  error_output += test_thread_partition();
  error_output += test_scalar_csr_expansion();

  if (error_output == 0)
    std::cout << "sparsity_pattern: all tests passed" << std::endl;

  return error_output;
}
