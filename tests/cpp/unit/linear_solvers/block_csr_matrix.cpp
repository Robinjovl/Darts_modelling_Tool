// Unit test for block_csr_matrix (the concrete unified matrix) and its
// compile-time-block-size view block_csr_view<N>
// (SOLVER_REFACTORING_PLAN.md section 12.4, layers 3 and 4).

#include <iostream>
#include <memory>
#include <vector>

#include "block_csr_matrix.hpp"
#include "block_csr_view.hpp"
#include "sparsity_pattern.hpp"

using opendarts::config::index_t;
using opendarts::config::mat_float;
using opendarts::linear_solvers::block_csr_matrix;
using opendarts::linear_solvers::block_csr_view;
using opendarts::linear_solvers::sparsity_pattern;

namespace
{
  int report(const char *name, bool ok)
  {
    if (!ok)
      std::cout << "block_csr_matrix: " << name << " FAILED" << std::endl;
    return ok ? 0 : 1;
  }

  // 4x4 block tridiagonal pattern (10 nonzero blocks, diagonal present).
  std::shared_ptr<sparsity_pattern> make_tridiagonal()
  {
    const std::vector<index_t> row_ptr = {0, 2, 5, 8, 10};
    const std::vector<index_t> col_ind = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    return std::make_shared<sparsity_pattern>(4, 4, row_ptr.data(), col_ind.data());
  }

  int test_construction()
  {
    block_csr_matrix a(make_tridiagonal(), 3);
    bool ok = !a.empty() && a.block_size() == 3 && a.n_block_rows() == 4
      && a.n_blocks() == 10 && a.n_rows() == 12 && a.n_values() == 10 * 9;
    // Freshly built: values must be zeroed.
    const mat_float *v = a.values();
    for (index_t i = 0; i < a.n_values() && ok; ++i)
      ok = (v[i] == 0.0);
    return report("construction + zero init", ok);
  }

  int test_empty()
  {
    block_csr_matrix a;
    return report("default construction is empty",
      a.empty() && a.block_size() == 0 && a.n_blocks() == 0 && a.n_values() == 0);
  }

  // Write blocks through the typed view, read back through the raw values.
  int test_view_block_access()
  {
    block_csr_matrix a(make_tridiagonal(), 2);
    block_csr_view<2> view(a);

    bool ok = view.n_blocks() == 10 && view.n_block_rows() == 4;

    // Fill stored block jb with the pattern 100*jb + 10*e + v.
    for (index_t jb = 0; jb < view.n_blocks(); ++jb)
      for (int e = 0; e < 2; ++e)
        for (int v = 0; v < 2; ++v)
          view.at(jb, e, v) = static_cast<mat_float>(100 * jb + 10 * e + v);

    // Read back through the raw, block-ordered values array.
    const mat_float *raw = a.values();
    for (index_t jb = 0; jb < a.n_blocks() && ok; ++jb)
      for (int e = 0; e < 2 && ok; ++e)
        for (int v = 0; v < 2 && ok; ++v)
          ok = (raw[jb * 4 + e * 2 + v] == static_cast<mat_float>(100 * jb + 10 * e + v));

    return report("typed view block access", ok);
  }

  // diag_block points at each row's diagonal block (diag_ind = {0,3,6,9}).
  int test_view_diag_block()
  {
    block_csr_matrix a(make_tridiagonal(), 2);
    block_csr_view<2> view(a);
    const index_t expected_diag[] = {0, 3, 6, 9};
    bool ok = true;
    for (index_t i = 0; i < 4; ++i)
      view.diag_block(i)[0] = static_cast<mat_float>(7 + i);
    const mat_float *raw = a.values();
    for (index_t i = 0; i < 4 && ok; ++i)
      ok = (raw[expected_diag[i] * 4] == static_cast<mat_float>(7 + i));
    return report("typed view diag_block", ok);
  }

  int test_set_zero()
  {
    block_csr_matrix a(make_tridiagonal(), 2);
    block_csr_view<2> view(a);
    view.at(0, 0, 0) = 3.14;
    a.set_zero();
    bool ok = true;
    const mat_float *v = a.values();
    for (index_t i = 0; i < a.n_values() && ok; ++i)
      ok = (v[i] == 0.0);
    return report("set_zero", ok);
  }

  // clone deep-copies the values but shares the immutable structure.
  int test_clone()
  {
    auto structure = make_tridiagonal();
    block_csr_matrix a(structure, 2);
    block_csr_view<2>(a).at(0, 0, 0) = 42.0;

    block_csr_matrix b = a.clone();
    block_csr_view<2>(a).at(0, 0, 0) = -1.0; // mutate the original

    bool ok = (b.values()[0] == 42.0)                       // clone unaffected
      && (b.structure_ptr().get() == a.structure_ptr().get()); // structure shared
    return report("clone copies values, shares structure", ok);
  }
} // namespace

int main()
{
  int error_output = 0;

  error_output += test_construction();
  error_output += test_empty();
  error_output += test_view_block_access();
  error_output += test_view_diag_block();
  error_output += test_set_zero();
  error_output += test_clone();

  if (error_output == 0)
    std::cout << "block_csr_matrix: all tests passed" << std::endl;

  return error_output;
}
