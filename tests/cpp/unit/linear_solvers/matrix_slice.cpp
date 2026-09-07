// Unit test for matrix_slice -- MatrixRange / MatrixSlice plus the helpers
// that extract a slice of a block_csr_matrix into a scalar csr_matrix<1>
// (and perform slice-aware SpMV).
//
// Layout under test: a synthetic 3-cell poromechanics-like block_csr_matrix
// with N_VARS = 4 = 1 pressure (P_VAR=0) + 3 displacement (U_VAR=1).

#include <iostream>
#include <memory>
#include <vector>

#include "block_csr_matrix.hpp"
#include "block_csr_view.hpp"
#include "csr_matrix.hpp"
#include "matrix_slice.hpp"
#include "sparsity_pattern.hpp"

using opendarts::config::index_t;
using opendarts::config::mat_float;
using opendarts::linear_solvers::block_csr_matrix;
using opendarts::linear_solvers::block_csr_view;
using opendarts::linear_solvers::csr_matrix;
using opendarts::linear_solvers::MatrixRange;
using opendarts::linear_solvers::MatrixSlice;
using opendarts::linear_solvers::sparsity_pattern;

namespace
{
  // 3-cell tridiagonal sparsity: nnzb = 2 + 3 + 2 = 7.
  // diag positions: row0 -> 0, row1 -> 2, row2 -> 5
  std::shared_ptr<sparsity_pattern> make_tridiag3()
  {
    const std::vector<index_t> row_ptr = {0, 2, 5, 7};
    const std::vector<index_t> col_ind = {0, 1, 0, 1, 2, 1, 2};
    return std::make_shared<sparsity_pattern>(3, 3, row_ptr.data(), col_ind.data());
  }

  // Predictable per-entry pattern: value(jb, e, v) = 100*jb + 10*e + v
  // (e is the intra-block row, v the intra-block column; block size 4).
  void fill_pattern(block_csr_matrix &A)
  {
    block_csr_view<4> view(A);
    for (index_t jb = 0; jb < view.n_blocks(); ++jb)
      for (int e = 0; e < 4; ++e)
        for (int v = 0; v < 4; ++v)
          view.at(jb, e, v) = static_cast<mat_float>(100 * jb + 10 * e + v);
  }

  int report(const char *name, bool ok)
  {
    if (!ok)
      std::cout << "matrix_slice: " << name << " FAILED" << std::endl;
    return ok ? 0 : 1;
  }

  // (b) MatrixSlice for the UU sub-block (displacement-displacement).
  int test_uu_slice_init()
  {
    block_csr_matrix A(make_tridiag3(), 4);
    fill_pattern(A);

    const std::uint8_t N = 4;
    const std::uint8_t U_VAR = 1;
    const std::uint8_t ND = 3;

    MatrixSlice UU;
    UU.pos = static_cast<std::uint8_t>(U_VAR * N + U_VAR); // 5
    UU.sizes[0] = ND;
    UU.sizes[1] = ND;
    UU.strides[0] = N;
    UU.strides[1] = 1;
    MatrixRange r;
    r.rows_from = 0;
    r.rows_to = 3;
    r.cols.emplace_back(0, 3);
    UU.ranges.push_back(std::move(r));

    UU.init(A);

    bool ok = (UU.is_init && UU.n_rows == 3 && UU.n_cols == 3 && UU.nnz == 7);
    return report("UU slice init counts", ok);
  }

  // (c) init_rows_cols_to_unit_matrix: scalar dest dimensions / structure.
  int test_unit_matrix_structure()
  {
    block_csr_matrix A(make_tridiag3(), 4);
    fill_pattern(A);

    const std::uint8_t N = 4;
    const std::uint8_t U_VAR = 1;
    const std::uint8_t ND = 3;

    MatrixSlice UU;
    UU.pos = static_cast<std::uint8_t>(U_VAR * N + U_VAR);
    UU.sizes[0] = ND;
    UU.sizes[1] = ND;
    UU.strides[0] = N;
    UU.strides[1] = 1;
    MatrixRange r;
    r.rows_from = 0;
    r.rows_to = 3;
    r.cols.emplace_back(0, 3);
    UU.ranges.push_back(std::move(r));
    UU.init(A);

    csr_matrix<1> dst;
    // dest dims: rows = sizes[0] * n_rows, cols = sizes[1] * n_cols,
    // scalar nnz = sizes[0] * sizes[1] * nnz.
    dst.init(static_cast<index_t>(UU.sizes[0]) * UU.n_rows,
             static_cast<index_t>(UU.sizes[1]) * UU.n_cols,
             static_cast<index_t>(UU.sizes[0]) * UU.sizes[1] * UU.nnz);
    init_rows_cols_to_unit_matrix(A, UU, dst);

    bool ok = (dst.n_rows == 9)
           && (UU.global_to_local.size() == static_cast<std::size_t>(9 * 7))
           && (UU.global_to_local_rows.size() == 9)
           && (dst.rows_ptr.back() == 9 * 7);

    // Spot-check the first row's scalar columns: row 0 covers source rows
    // 0 and 1 (block cols 0, 1) -> scalar cols {0,1,2, 3,4,5}.
    if (ok)
    {
      const index_t expected[] = {0, 1, 2, 3, 4, 5};
      for (int i = 0; i < 6 && ok; ++i)
        ok = (dst.cols_ind[i] == expected[i]);
    }

    return report("init_rows_cols_to_unit_matrix structure", ok);
  }

  // (d) extract_sub_block_to_scalar_csr: check a couple of specific gathered values.
  int test_extract_unit_values()
  {
    block_csr_matrix A(make_tridiag3(), 4);
    fill_pattern(A);

    const std::uint8_t N = 4;
    const std::uint8_t U_VAR = 1;
    const std::uint8_t ND = 3;

    MatrixSlice UU;
    UU.pos = static_cast<std::uint8_t>(U_VAR * N + U_VAR); // 5
    UU.sizes[0] = ND;
    UU.sizes[1] = ND;
    UU.strides[0] = N;
    UU.strides[1] = 1;
    MatrixRange r;
    r.rows_from = 0;
    r.rows_to = 3;
    r.cols.emplace_back(0, 3);
    UU.ranges.push_back(std::move(r));
    UU.init(A);

    csr_matrix<1> dst;
    dst.init(static_cast<index_t>(UU.sizes[0]) * UU.n_rows,
             static_cast<index_t>(UU.sizes[1]) * UU.n_cols,
             static_cast<index_t>(UU.sizes[0]) * UU.sizes[1] * UU.nnz);
    init_rows_cols_to_unit_matrix(A, UU, dst);

    std::vector<mat_float> rhs_mults(dst.n_rows, 1.0);
    extract_sub_block_to_scalar_csr(UU, A, dst, rhs_mults.data(),
                                    /*positive_diagonal=*/false,
                                    /*asymmetric_hack=*/false);

    // dst.values[0] should be A.values[0*16 + 5 + 0*4 + 0]
    //                       = value(jb=0, e=1, v=1) = 11
    bool ok = (dst.values[0] == 11.0);
    // dst.values[1] should be vals[0*16 + 5 + 0*4 + 1] = value(0,1,2) = 12
    if (ok) ok = (dst.values[1] == 12.0);
    // dst.values[2] should be vals[0*16 + 5 + 0*4 + 2] = value(0,1,3) = 13
    if (ok) ok = (dst.values[2] == 13.0);
    // dst.values[3] is the first entry coming from block j=1 (col=1):
    //   vals[1*16 + 5 + 0*4 + 0] = value(1,1,1) = 100 + 11 = 111
    if (ok) ok = (dst.values[3] == 111.0);

    return report("extract_sub_block_to_scalar_csr values", ok);
  }

  // (e) block_vector_product on the UP slice with a unit source.
  int test_block_vector_product_up()
  {
    block_csr_matrix A(make_tridiag3(), 4);
    fill_pattern(A);

    const std::uint8_t N = 4;
    const std::uint8_t P_VAR = 0;
    const std::uint8_t U_VAR = 1;
    const std::uint8_t ND = 3;

    MatrixSlice UP;
    UP.pos = static_cast<std::uint8_t>(U_VAR * N + P_VAR); // 4
    UP.sizes[0] = ND; // 3 rows of u per cell
    UP.sizes[1] = 1;  // 1 column of p per cell
    UP.strides[0] = N;
    UP.strides[1] = 1;
    MatrixRange r;
    r.rows_from = 0;
    r.rows_to = 3;
    r.cols.emplace_back(0, 3);
    UP.ranges.push_back(std::move(r));
    UP.init(A);

    // src has length sizes[1] * n_cols = 1 * 3 = 3, all ones.
    std::vector<mat_float> src(3, 1.0);
    // dst has length sizes[0] * n_rows = 3 * 3 = 9, zero-initialised.
    std::vector<mat_float> dst(9, 0.0);

    opendarts::linear_solvers::block_vector_product(A, UP, src.data(), dst.data());

    // Hand-compute expected dst[i_dest*3 + k] for k = 0..2.
    // Row i in [rows_from, rows_to) has source blocks j = row_ptr[i]..row_ptr[i+1].
    // For each j, the slice column is jx (= cols[j]); the UP block at j writes
    // dst[3*i_dest + k] += vals[j*16 + 4 + k*4 + 0] * 1.
    // value(j, e=U_VAR+k, v=P_VAR=0) = 100*j + 10*(1+k).
    //
    // Row 0: j=0 (jb=0), j=1 (jb=1) ->
    //   dst[k] = (0 + 100) + 2*(10 + 10*k) = 120 + 20*k
    //   k=0 -> 120
    //   k=1 -> 140
    //   k=2 -> 160
    // Row 1: j=2 (jb=2), j=3 (jb=3), j=4 (jb=4) ->
    //                       dst[3 + k] = sum_{jb in {2,3,4}} (100*jb + 10*(1+k))
    //                                  = 900 + 30*(1+k)
    //   k=0 -> 930
    //   k=1 -> 960
    //   k=2 -> 990
    // Row 2: j=5 (jb=5), j=6 (jb=6) -> dst[6 + k] = (500 + 600) + 20*(1+k)
    //                                             = 1100 + 20*(1+k)
    //   k=0 -> 1120
    //   k=1 -> 1140
    //   k=2 -> 1160
    const mat_float expected[9] = {
        120, 140, 160,
        930, 960, 990,
        1120, 1140, 1160};
    bool ok = true;
    for (int i = 0; i < 9 && ok; ++i)
      ok = (dst[i] == expected[i]);

    return report("block_vector_product on UP slice", ok);
  }

  // Extra: init_structural sanity (range-dense version).
  int test_init_structural()
  {
    MatrixSlice s;
    s.sizes[0] = 3;
    s.sizes[1] = 3;
    MatrixRange r;
    r.rows_from = 0;
    r.rows_to = 4;
    r.cols.emplace_back(0, 5);
    r.cols.emplace_back(7, 9);
    s.ranges.push_back(std::move(r));

    s.init_structural();

    // 4 rows, two col-intervals of width 5 and 2 -> 4 * (5 + 2) = 28 nnz,
    // n_cols = nnz / n_rows = 7.
    bool ok = (s.is_init && s.n_rows == 4 && s.nnz == 28 && s.n_cols == 7);
    return report("init_structural", ok);
  }
} // namespace

int main()
{
  int error_output = 0;

  error_output += test_init_structural();
  error_output += test_uu_slice_init();
  error_output += test_unit_matrix_structure();
  error_output += test_extract_unit_values();
  error_output += test_block_vector_product_up();

  if (error_output == 0)
    std::cout << "matrix_slice: all tests passed" << std::endl;

  return error_output;
}
