//*************************************************************************
//    Copyright (c) 2022
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

#include "matrix_slice.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>

namespace opendarts
{
  namespace linear_solvers
  {
    using opendarts::config::index_t;
    using opendarts::config::mat_float;

    // ------------------------------------------------------------------------
    // MatrixSlice::init_structural -- range-dense init (no source matrix).
    // ------------------------------------------------------------------------
    void MatrixSlice::init_structural()
    {
      is_init = true;
      nnz = 0;
      n_rows = 0;
      n_cols = 0;
      for (const auto &range : ranges)
      {
        const index_t row_span = range.rows_to - range.rows_from;
        n_rows += row_span;
        for (const auto &col : range.cols)
        {
          nnz += static_cast<std::uint64_t>(col.second - col.first)
               * static_cast<std::uint64_t>(row_span);
        }
      }
      if (n_rows > 0)
        n_cols = static_cast<index_t>(nnz / static_cast<std::uint64_t>(n_rows));
    }

    // ------------------------------------------------------------------------
    // MatrixSlice::init -- sparsity-following init from a block_csr_matrix.
    // ------------------------------------------------------------------------
    void MatrixSlice::init(const opendarts::linear_solvers::block_csr_matrix &A)
    {
      const index_t *rows = A.row_ptr();
      const index_t *cols = A.col_ind();

      is_init = true;
      nnz = 0;
      n_rows = 0;
      n_cols = 0;

      for (const auto &range : ranges)
      {
        n_rows += range.rows_to - range.rows_from;
        for (index_t i = range.rows_from; i < range.rows_to; i++)
        {
          const index_t j1 = rows[i];
          const index_t j2 = rows[i + 1];

          std::uint8_t id_range_col = 0;
          for (index_t j = j1; j < j2; j++)
          {
            const index_t col = cols[j];
            // advance through column intervals until 'col' is below the
            // current interval's upper bound, or we run out of intervals.
            if (col >= range.cols[id_range_col].second)
            {
              for (id_range_col++; id_range_col < range.cols.size(); id_range_col++)
                if (col < range.cols[id_range_col].second) break;
            }
            if (id_range_col == range.cols.size()) break;
            if (col >= range.cols[id_range_col].first) nnz++;
          }
        }
      }

      if (!ranges.empty())
      {
        for (const auto &col : ranges[0].cols)
          n_cols += col.second - col.first;
      }
    }

    // ------------------------------------------------------------------------
    // init_rows_cols_to_unit_matrix -- build the scalar-CSR structure of a
    // unit-block (size sizes[0] x sizes[1] per source block) extraction.
    //
    // Caller is responsible for pre-allocating dest via dest.init(...).
    // ------------------------------------------------------------------------
    void init_rows_cols_to_unit_matrix(
        const opendarts::linear_solvers::block_csr_matrix &src,
        MatrixSlice &s,
        opendarts::linear_solvers::csr_matrix<1> &dest)
    {
      assert(s.sizes[0] == s.sizes[1]);

      const int n_block_size = src.block_size();
      assert(n_block_size > 0);
      // The slice's intra-block (k,l) must fit inside the source block.
      // pos is a row-major intra-block offset (row_offset * N + col_offset),
      // so pos / N bounds the rows (sizes[0]) and pos % N the columns.
      assert(s.pos / n_block_size + s.sizes[0] <= n_block_size);
      assert(s.pos % n_block_size + s.sizes[1] <= n_block_size);

      const index_t *rows = src.row_ptr();
      const index_t *cols = src.col_ind();
      const index_t *diags = src.diag_ind();

      index_t *d_diags = dest.get_diag_ind();
      index_t *d_rows = dest.get_rows_ptr();
      index_t *d_cols = dest.get_cols_ind();

      d_rows[0] = 0;
      s.global_to_local.assign(
          static_cast<std::size_t>(s.sizes[0]) * s.sizes[1] * src.n_blocks(),
          -1);
      s.global_to_local_rows.assign(
          static_cast<std::size_t>(s.sizes[0]) * s.n_rows, 0);

      // The flat scalar row of a slice row is N * i + row_offset + k, so the
      // ROW half of pos is what belongs here (pos % N is the COLUMN offset).
      const std::uint8_t start_row_pos =
          static_cast<std::uint8_t>(s.pos / n_block_size);

      index_t i_dest = 0;
      index_t row_dest = 0;
      // n_block_size_sq fits in int because block sizes are <= ~16
      const int n_block_size_sq = n_block_size * n_block_size;

      for (std::size_t s_id = 0; s_id < s.ranges.size(); s_id++)
      {
        const auto &range = s.ranges[s_id];
        for (index_t i = range.rows_from; i < range.rows_to; i++, i_dest++)
        {
          const index_t j1 = rows[i];
          const index_t j2 = rows[i + 1];
          const index_t jd = diags[i];

          for (std::uint8_t k = 0; k < s.sizes[0]; k++)
          {
            row_dest = static_cast<index_t>(s.sizes[0]) * i_dest + k;

            s.global_to_local_rows[row_dest] =
                static_cast<index_t>(n_block_size) * i + start_row_pos + k;

            index_t &cur_row = d_rows[row_dest];
            index_t &next_row = d_rows[row_dest + 1];

            index_t n_missed_cols = range.cols[0].first;
            index_t j_dest = 0;
            std::uint8_t id_range_col = 0;

            for (index_t j = j1; j < j2; j++)
            {
              const index_t col = cols[j];

              if (col >= range.cols[id_range_col].second)
              {
                for (id_range_col++; id_range_col < range.cols.size(); id_range_col++)
                {
                  n_missed_cols += range.cols[id_range_col].first -
                                   range.cols[id_range_col - 1].second;
                  if (col < range.cols[id_range_col].second) break;
                }
              }
              if (id_range_col == range.cols.size()) break;

              if (col >= range.cols[id_range_col].first)
              {
                for (std::uint8_t l = 0; l < s.sizes[1]; l++)
                {
                  const index_t pos_dest = cur_row + j_dest * s.sizes[1] + l;
                  d_cols[pos_dest] = static_cast<index_t>(s.sizes[1]) *
                                         (col - n_missed_cols) + l;
                  // NOTE: global_to_local stores the offset INSIDE A's value
                  // array measured from the start of block j -- it does NOT
                  // include s.pos. The extract step adds s.pos.
                  s.global_to_local[pos_dest] =
                      j * n_block_size_sq +
                      static_cast<index_t>(k) * n_block_size + l;
                }
                if (j == jd)
                {
                  d_diags[row_dest] = cur_row + j_dest * s.sizes[1] + k;
                  assert(d_cols[d_diags[row_dest]] == row_dest);
                }
                j_dest++;
              }
            }

            next_row = cur_row + j_dest * s.sizes[1];
          }
        }
      }

      assert(i_dest == s.n_rows);
      s.global_to_local.resize(static_cast<std::size_t>(d_rows[row_dest + 1]));
      assert(static_cast<std::uint64_t>(s.sizes[0]) * s.sizes[1] * s.nnz
             == s.global_to_local.size());
    }

    // ------------------------------------------------------------------------
    // extract_sub_block_to_scalar_csr -- per-Newton value gather via the
    // cached global_to_local. Mirrors extract_block_matrix_to_unit_block.
    // ------------------------------------------------------------------------
    void extract_sub_block_to_scalar_csr(
        const MatrixSlice &s,
        const opendarts::linear_solvers::block_csr_matrix &A,
        opendarts::linear_solvers::csr_matrix<1> &dest,
        opendarts::config::mat_float *rhs_mults,
        bool positive_diagonal,
        bool asymmetric_hack)
    {
      const mat_float *vals = A.values();
      const index_t *d_rows = dest.get_rows_ptr();
      const index_t *d_diags = dest.get_diag_ind();
      mat_float *d_vals = dest.get_values();

      assert(static_cast<index_t>(s.global_to_local.size()) == d_rows[dest.n_rows]);

      for (std::size_t i = 0; i < s.global_to_local.size(); i++)
      {
        const index_t id = s.global_to_local[i];
        d_vals[i] = vals[id + s.pos];
      }

      if (positive_diagonal)
      {
        for (index_t i = 0; i < dest.n_rows; i++)
        {
          const index_t j1 = d_rows[i];
          const index_t j2 = d_rows[i + 1];
          const index_t jd = d_diags[i];
          if (d_vals[jd] < 0.0)
          {
            rhs_mults[i] = -1.0;
            for (index_t j = j1; j < j2; j++)
              d_vals[j] *= -1.0;
          }
          else
          {
            rhs_mults[i] = 1.0;
          }
        }
      }

      if (asymmetric_hack)
      {
        rhs_mults[0] *= 2.0;
        for (index_t j = d_rows[0]; j < d_rows[1]; j++)
          d_vals[j] *= 2.0;
      }
    }

    // ------------------------------------------------------------------------
    // block_vector_product -- dest += A[slice] * src.
    // ------------------------------------------------------------------------
    void block_vector_product(
        const opendarts::linear_solvers::block_csr_matrix &A,
        const MatrixSlice &s,
        const opendarts::config::mat_float *src,
        opendarts::config::mat_float *dest)
    {
      const index_t *rows = A.row_ptr();
      const index_t *cols = A.col_ind();
      const mat_float *vals = A.values();
      const int n_block_size = A.block_size();
      const int n_jac_block_vars = n_block_size * n_block_size;

      index_t i_dest = 0;
      for (std::size_t s_id = 0; s_id < s.ranges.size(); s_id++)
      {
        const auto &range = s.ranges[s_id];
        for (index_t i = range.rows_from; i < range.rows_to; i++, i_dest++)
        {
          const index_t j1 = rows[i];
          const index_t j2 = rows[i + 1];

          index_t n_missed_cols = range.cols[0].first;
          std::uint8_t id_range_col = 0;

          for (index_t j = j1; j < j2; ++j)
          {
            const index_t jx = cols[j];

            if (jx >= range.cols[id_range_col].second)
            {
              for (id_range_col++; id_range_col < range.cols.size(); id_range_col++)
              {
                n_missed_cols += range.cols[id_range_col].first -
                                 range.cols[id_range_col - 1].second;
                if (jx < range.cols[id_range_col].second) break;
              }
            }
            if (id_range_col == range.cols.size()) break;

            if (jx >= range.cols[id_range_col].first)
            {
              for (std::uint8_t k = 0; k < s.sizes[0]; k++)
                for (std::uint8_t l = 0; l < s.sizes[1]; l++)
                {
                  dest[static_cast<index_t>(s.sizes[0]) * i_dest + k] +=
                      vals[j * n_jac_block_vars + s.pos +
                           static_cast<index_t>(k) * n_block_size + l] *
                      src[static_cast<index_t>(s.sizes[1]) *
                              (jx - n_missed_cols) + l];
                }
            }
          }
        }
      }
    }

    // ========================================================================
    // Free template implementations -- defined here, instantiated below.
    // ========================================================================

    template <std::uint8_t NE>
    void extract_sub_block_to_block_csr(
        const MatrixSlice &s,
        const opendarts::linear_solvers::block_csr_matrix &A,
        opendarts::linear_solvers::csr_matrix<NE> &dest,
        opendarts::config::mat_float *rhs_mults)
    {
      // For block extraction (NE > 1) we expect a square diagonal-style slice.
      assert(s.sizes[0] == NE && s.sizes[1] == NE);

      const int n_block_size = A.block_size();
      const int n_jac_block_vars = n_block_size * n_block_size;

      const index_t *rows = A.row_ptr();
      const index_t *cols = A.col_ind();
      const mat_float *vals = A.values();

      mat_float *d_vals = dest.get_values();
      const index_t *d_cols = dest.get_cols_ind();
      (void)d_cols; // only used by an assertion in the proprietary version
      constexpr int dest_block_vars = NE * NE;

      index_t j_dest = 0;

      for (std::size_t s_id = 0; s_id < s.ranges.size(); s_id++)
      {
        const auto &range = s.ranges[s_id];
        for (index_t i = range.rows_from; i < range.rows_to; i++)
        {
          const index_t j1 = rows[i];
          const index_t j2 = rows[i + 1];

          std::uint8_t id_cols = 0;
          for (index_t j = j1; j < j2; j++)
          {
            const index_t col = cols[j];

            if (col < range.cols[id_cols].first)
              continue;

            while (id_cols < range.cols.size() &&
                   col >= range.cols[id_cols].second)
            {
              id_cols++;
            }

            if (id_cols < range.cols.size() &&
                col >= range.cols[id_cols].first)
            {
              assert(d_cols[j_dest] == col);
              for (std::uint8_t m = 0; m < NE; m++)
                for (std::uint8_t l = 0; l < NE; l++)
                {
                  d_vals[j_dest * dest_block_vars +
                         static_cast<index_t>(m) * s.sizes[1] + l] =
                      vals[j * n_jac_block_vars + s.pos +
                           static_cast<index_t>(m) * n_block_size + l];
                }
              j_dest++;
            }
          }

          for (std::uint8_t m = 0; m < NE; m++)
            rhs_mults[i * NE + m] = 1.0;
        }
      }
    }

    template <std::uint8_t NB>
    void apply_relaxation(
        opendarts::linear_solvers::csr_matrix<NB> &M,
        const opendarts::config::mat_float *r,
        opendarts::config::mat_float *mults,
        bool positive_diagonal)
    {
      const index_t n_rows = M.n_rows;
      const index_t *rows = M.get_rows_ptr();
      const index_t *diags = M.get_diag_ind();
      mat_float *vals = M.get_values();
      constexpr int block_vars = static_cast<int>(NB) * NB;

      for (index_t i = 0; i < n_rows; ++i)
      {
        const index_t j1 = rows[i];
        const index_t j2 = rows[i + 1];
        const index_t jd = diags[i];

        for (std::uint8_t k = 0; k < NB; k++)
        {
          vals[jd * block_vars + static_cast<index_t>(k) * NB + k] +=
              mults[static_cast<index_t>(NB) * i + k] *
              r[static_cast<index_t>(NB) * i + k];
        }

        if (positive_diagonal)
        {
          for (std::uint8_t k = 0; k < NB; k++)
          {
            if (vals[jd * block_vars + static_cast<index_t>(k) * NB + k] < 0.0)
            {
              mults[static_cast<index_t>(NB) * i + k] *= -1.0;
              for (index_t j = j1; j < j2; ++j)
              {
                for (std::uint8_t l = 0; l < NB; l++)
                  vals[j * block_vars + static_cast<index_t>(k) * NB + l] *= -1.0;
              }
            }
          }
        }
      }
    }

    template <std::uint8_t NE>
    void apply_ps_relaxation(
        opendarts::linear_solvers::csr_matrix<NE> &M,
        const opendarts::config::mat_float *r_p,
        const opendarts::config::mat_float *r_s,
        opendarts::config::mat_float *mults)
    {
      const index_t n_rows = M.n_rows;
      const index_t *diags = M.get_diag_ind();
      mat_float *vals = M.get_values();
      constexpr int block_vars = static_cast<int>(NE) * NE;

      for (index_t i = 0; i < n_rows; ++i)
      {
        const index_t jd = diags[i];
        vals[jd * block_vars] +=
            mults[static_cast<index_t>(NE) * i] * r_p[i];
        vals[jd * block_vars + NE] +=
            mults[static_cast<index_t>(NE) * i + 1] * r_s[i];
      }
    }

    template <std::uint8_t NB>
    void set_diag_first(opendarts::linear_solvers::csr_matrix<NB> &M)
    {
      const index_t n_rows = M.n_rows;
      index_t *rows = M.get_rows_ptr();
      index_t *cols = M.get_cols_ind();
      index_t *diag = M.get_diag_ind();
      mat_float *vals = M.get_values();
      constexpr int n_block_sq = static_cast<int>(NB) * NB;

      mat_float buf[n_block_sq];

      // Move the diagonal block to the FRONT of each row.
      // The diagonal entry was at position 'jd' (within the row's column slice).
      // After this routine, the diagonal entry sits at j1 (the start of the row).
      for (index_t i = 0; i < n_rows; i++)
      {
        const index_t j1 = rows[i];
        const index_t jd = diag[i];
        if (jd == j1) continue;

        // Save diagonal block.
        std::copy_n(vals + static_cast<std::size_t>(jd) * n_block_sq,
                    n_block_sq, buf);
        // Shift entries [j1, jd) one position to the right.
        for (index_t j = jd; j > j1; j--)
        {
          cols[j] = cols[j - 1];
          std::copy_n(vals + static_cast<std::size_t>(j - 1) * n_block_sq,
                      n_block_sq,
                      vals + static_cast<std::size_t>(j) * n_block_sq);
        }
        cols[j1] = i;
        std::copy_n(buf, n_block_sq,
                    vals + static_cast<std::size_t>(j1) * n_block_sq);
        diag[i] = j1;
      }
    }

    template <std::uint8_t NB>
    void set_diag_in_order(opendarts::linear_solvers::csr_matrix<NB> &M)
    {
      const index_t n_rows = M.n_rows;
      index_t *rows = M.get_rows_ptr();
      index_t *cols = M.get_cols_ind();
      index_t *diag = M.get_diag_ind();
      mat_float *vals = M.get_values();
      constexpr int n_block_sq = static_cast<int>(NB) * NB;

      mat_float buf[n_block_sq];

      // Restore natural (ascending) column order: the diagonal block is
      // currently at j1 (column == i); slide it back to its sorted position.
      // Mirrors the proprietary set_diag_in_order at lines ~673-704.
      for (index_t i = 0; i < n_rows; i++)
      {
        const index_t j1 = rows[i];
        const index_t j2 = rows[i + 1];

        // Save the front block (the diagonal).
        std::copy_n(vals + static_cast<std::size_t>(j1) * n_block_sq,
                    n_block_sq, buf);

        // Shift entries with cols[j] < i one position to the left.
        index_t j = j1 + 1;
        for (; j < j2 && cols[j] < i; j++)
        {
          cols[j - 1] = cols[j];
          std::copy_n(vals + static_cast<std::size_t>(j) * n_block_sq,
                      n_block_sq,
                      vals + static_cast<std::size_t>(j - 1) * n_block_sq);
        }
        cols[j - 1] = i;
        std::copy_n(buf, n_block_sq,
                    vals + static_cast<std::size_t>(j - 1) * n_block_sq);
        diag[i] = j - 1;
      }
    }

    // ========================================================================
    // Explicit instantiations for the realistic poromechanics block sizes.
    // ========================================================================
#define OD_INSTANTIATE_MATRIX_SLICE_FOR(N)                                   \
    template void extract_sub_block_to_block_csr<N>(                          \
        const MatrixSlice &,                                                  \
        const opendarts::linear_solvers::block_csr_matrix &,                  \
        opendarts::linear_solvers::csr_matrix<N> &,                           \
        opendarts::config::mat_float *);                                      \
    template void apply_relaxation<N>(                                        \
        opendarts::linear_solvers::csr_matrix<N> &,                           \
        const opendarts::config::mat_float *,                                 \
        opendarts::config::mat_float *,                                       \
        bool);                                                                \
    template void apply_ps_relaxation<N>(                                     \
        opendarts::linear_solvers::csr_matrix<N> &,                           \
        const opendarts::config::mat_float *,                                 \
        const opendarts::config::mat_float *,                                 \
        opendarts::config::mat_float *);                                      \
    template void set_diag_first<N>(                                          \
        opendarts::linear_solvers::csr_matrix<N> &);                          \
    template void set_diag_in_order<N>(                                       \
        opendarts::linear_solvers::csr_matrix<N> &);

    OD_INSTANTIATE_MATRIX_SLICE_FOR(1)
    OD_INSTANTIATE_MATRIX_SLICE_FOR(2)
    OD_INSTANTIATE_MATRIX_SLICE_FOR(3)
    OD_INSTANTIATE_MATRIX_SLICE_FOR(4)
    OD_INSTANTIATE_MATRIX_SLICE_FOR(5)

#undef OD_INSTANTIATE_MATRIX_SLICE_FOR
  } // namespace linear_solvers
} // namespace opendarts
