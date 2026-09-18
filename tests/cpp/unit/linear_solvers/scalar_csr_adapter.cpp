// Unit test for scalar_csr_adapter -- the block-CSR -> scalar-CSR backend
// adapter for HYPRE / Pardiso (SOLVER_REFACTORING_PLAN.md section 12.6).

#include <iostream>
#include <memory>
#include <vector>

#include "block_csr_matrix.hpp"
#include "block_csr_view.hpp"
#include "csr_expansion.hpp"
#include "scalar_csr_adapter.hpp"
#include "sparsity_pattern.hpp"

using opendarts::config::index_t;
using opendarts::config::mat_float;
using opendarts::linear_solvers::block_csr_matrix;
using opendarts::linear_solvers::block_csr_view;
using opendarts::linear_solvers::scalar_csr_adapter;
using opendarts::linear_solvers::sparsity_pattern;

namespace
{
  int report(const char *name, bool ok)
  {
    if (!ok)
      std::cout << "scalar_csr_adapter: " << name << " FAILED" << std::endl;
    return ok ? 0 : 1;
  }

  std::shared_ptr<sparsity_pattern> make_tridiagonal()
  {
    const std::vector<index_t> row_ptr = {0, 2, 5, 8, 10};
    const std::vector<index_t> col_ind = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    return std::make_shared<sparsity_pattern>(4, 4, row_ptr.data(), col_ind.data());
  }

  // Fills every block element (jb,e,v) with a distinct, predictable value.
  void fill(block_csr_matrix &a)
  {
    block_csr_view<2> view(a);
    for (index_t jb = 0; jb < a.n_blocks(); ++jb)
      for (int e = 0; e < 2; ++e)
        for (int v = 0; v < 2; ++v)
          view.at(jb, e, v) = static_cast<mat_float>(100 * jb + 10 * e + v);
  }

  int test_dimensions()
  {
    block_csr_matrix a(make_tridiagonal(), 2);
    fill(a);
    scalar_csr_adapter adapter(a);
    // 4 block rows x 2, 4 block cols x 2, 10 blocks x 2 x 2.
    bool ok = adapter.n_rows() == 8 && adapter.n_cols() == 8 && adapter.nnz() == 40;
    return report("scalar dimensions", ok);
  }

  // The adapter must reproduce exactly the csr_expansion value gather:
  // values()[k] == block_values[value_map[k]].
  int test_values_match_gather()
  {
    auto structure = make_tridiagonal();
    block_csr_matrix a(structure, 2);
    fill(a);
    scalar_csr_adapter adapter(a);

    const auto &expansion = structure->scalar_csr(2);
    const index_t *vm = expansion.value_map();
    const mat_float *block_values = a.values();
    const mat_float *scalar_values = adapter.values();

    bool ok = true;
    for (index_t k = 0; k < adapter.nnz() && ok; ++k)
      ok = (scalar_values[k] == block_values[vm[k]]);

    // Structure is borrowed straight from the cached expansion.
    ok = ok && (adapter.row_ptr() == expansion.row_ptr())
      && (adapter.col_ind() == expansion.col_ind());
    return report("values match expansion gather", ok);
  }

  // refresh() picks up a re-assembled block matrix.
  int test_refresh()
  {
    auto structure = make_tridiagonal();
    block_csr_matrix a(structure, 2);
    fill(a);
    scalar_csr_adapter adapter(a);

    // Re-assemble: scale every block value, then refresh.
    block_csr_view<2> view(a);
    for (index_t jb = 0; jb < a.n_blocks(); ++jb)
      for (int e = 0; e < 2; ++e)
        for (int v = 0; v < 2; ++v)
          view.at(jb, e, v) *= 3.0;
    adapter.refresh();

    const index_t *vm = structure->scalar_csr(2).value_map();
    const mat_float *block_values = a.values();
    const mat_float *scalar_values = adapter.values();
    bool ok = true;
    for (index_t k = 0; k < adapter.nnz() && ok; ++k)
      ok = (scalar_values[k] == block_values[vm[k]]);
    return report("refresh re-gathers values", ok);
  }
} // namespace

int main()
{
  int error_output = 0;

  error_output += test_dimensions();
  error_output += test_values_match_gather();
  error_output += test_refresh();

  if (error_output == 0)
    std::cout << "scalar_csr_adapter: all tests passed" << std::endl;

  return error_output;
}
