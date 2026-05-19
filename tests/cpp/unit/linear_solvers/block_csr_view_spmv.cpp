// Unit test for the block sparse matrix-vector products on block_csr_view<N>
// (SOLVER_REFACTORING_PLAN.md section 12.10, phase A1).
//
// The block SpMV is cross-checked against an independent scalar-CSR multiply
// built from csr_expansion + scalar_csr_adapter: the two must agree exactly,
// since block-CSR and its scalar-CSR expansion describe the same matrix.

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "block_csr_matrix.hpp"
#include "block_csr_view.hpp"
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
      std::cout << "block_csr_view spmv: " << name << " FAILED" << std::endl;
    return ok ? 0 : 1;
  }

  bool close(mat_float a, mat_float b) { return std::fabs(a - b) <= 1e-10 * (1.0 + std::fabs(b)); }

  std::shared_ptr<sparsity_pattern> make_tridiagonal()
  {
    const std::vector<index_t> row_ptr = {0, 2, 5, 8, 10};
    const std::vector<index_t> col_ind = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    return std::make_shared<sparsity_pattern>(4, 4, row_ptr.data(), col_ind.data());
  }

  // Reference scalar-CSR multiply: r = A * v.
  std::vector<mat_float> scalar_spmv(const scalar_csr_adapter &a, const std::vector<mat_float> &v)
  {
    std::vector<mat_float> r(static_cast<std::size_t>(a.n_rows()), 0.0);
    const index_t *rp = a.row_ptr();
    const index_t *ci = a.col_ind();
    const mat_float *val = a.values();
    for (index_t i = 0; i < a.n_rows(); ++i)
      for (index_t k = rp[i]; k < rp[i + 1]; ++k)
        r[i] += val[k] * v[ci[k]];
    return r;
  }

  template <uint8_t N>
  int test_spmv_against_scalar_csr()
  {
    block_csr_matrix a(make_tridiagonal(), N);
    block_csr_view<N> view(a);

    // Fill every block element with an arbitrary but reproducible value.
    for (index_t jb = 0; jb < a.n_blocks(); ++jb)
      for (int e = 0; e < static_cast<int>(N); ++e)
        for (int w = 0; w < static_cast<int>(N); ++w)
          view.at(jb, e, w) = static_cast<mat_float>(1 + (7 * jb + 3 * e + w) % 11);

    const std::size_t n = static_cast<std::size_t>(a.scalar_n_rows());
    std::vector<mat_float> v(n);
    for (std::size_t i = 0; i < n; ++i)
      v[i] = static_cast<mat_float>(1 + (i % 5)) * 0.5;

    // Block SpMV: r = A * v  (matrix_vector_product accumulates, so start at 0).
    std::vector<mat_float> r(n, 0.0);
    view.matrix_vector_product(v.data(), r.data());

    // Reference via the scalar-CSR expansion.
    scalar_csr_adapter adapter(a);
    std::vector<mat_float> r_ref = scalar_spmv(adapter, v);

    bool ok = true;
    for (std::size_t i = 0; i < n && ok; ++i)
      ok = close(r[i], r_ref[i]);

    // calc_lin_comb: r = 2*A*v + 3*v must equal 2*r_ref + 3*v.
    std::vector<mat_float> r2(n, 0.0);
    view.calc_lin_comb(2.0, 3.0, v.data(), v.data(), r2.data());
    for (std::size_t i = 0; i < n && ok; ++i)
      ok = close(r2[i], 2.0 * r_ref[i] + 3.0 * v[i]);

    return ok ? 0 : 1;
  }

  // A^T identity: <A x, y> == <x, A^T y> for arbitrary x, y.
  int test_transpose_adjoint()
  {
    block_csr_matrix a(make_tridiagonal(), 2);
    block_csr_view<2> view(a);
    for (index_t jb = 0; jb < a.n_blocks(); ++jb)
      for (int e = 0; e < 2; ++e)
        for (int w = 0; w < 2; ++w)
          view.at(jb, e, w) = static_cast<mat_float>(1 + (5 * jb + e + 2 * w) % 9);

    const std::size_t n = static_cast<std::size_t>(a.scalar_n_rows());
    std::vector<mat_float> x(n), y(n);
    for (std::size_t i = 0; i < n; ++i)
    {
      x[i] = static_cast<mat_float>(1 + (i % 4));
      y[i] = static_cast<mat_float>(1 + (i % 3)) * 0.25;
    }

    std::vector<mat_float> ax(n, 0.0), aty(n, 0.0);
    view.matrix_vector_product(x.data(), ax.data());
    view.matrix_vector_product_t(y.data(), aty.data());

    mat_float lhs = 0, rhs = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
      lhs += ax[i] * y[i];
      rhs += x[i] * aty[i];
    }
    return report("matrix_vector_product_t adjoint identity", close(lhs, rhs));
  }
} // namespace

int main()
{
  int error_output = 0;

  error_output += report("block SpMV (N=1) vs scalar CSR", test_spmv_against_scalar_csr<1>() == 0);
  error_output += report("block SpMV (N=2) vs scalar CSR", test_spmv_against_scalar_csr<2>() == 0);
  error_output += report("block SpMV (N=3) vs scalar CSR", test_spmv_against_scalar_csr<3>() == 0);
  error_output += test_transpose_adjoint();

  if (error_output == 0)
    std::cout << "block_csr_view spmv: all tests passed" << std::endl;

  return error_output;
}
