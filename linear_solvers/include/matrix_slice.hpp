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

//--------------------------------------------------------------------------
#ifndef OPENDARTS_LINEAR_SOLVERS_MATRIX_SLICE_HPP
#define OPENDARTS_LINEAR_SOLVERS_MATRIX_SLICE_HPP
//--------------------------------------------------------------------------

// matrix_slice -- describes a rectangular sub-block of a block_csr_matrix
// (chosen by row ranges, column intervals, and an N x N intra-block "pos"
// offset) and provides helpers to extract that sub-block into either a
// scalar csr_matrix<1> (unit-block) or a block csr_matrix<NE>, plus a
// slice-aware SpMV and a few relaxation / diagonal-reorder utilities.
//
// Ported from the proprietary FS-CPR (Full-System CPR poromechanics)
// preconditioner -- MatrixRange / MatrixSlice plus the helpers around
// extract_block_matrix*, block_vector_product, apply_(ps_)relaxation,
// set_diag_first / set_diag_in_order. See solvers/src/matrix_slice.cpp
// for the non-template implementations and explicit instantiations.

#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

#include "block_csr_matrix.hpp"
#include "csr_matrix.hpp"
#include "data_types.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** @brief One row-range of a MatrixSlice: a contiguous range of source
        rows together with a sorted, disjoint list of half-open column
        intervals [first, second) the slice keeps from each of those rows. */
    struct MatrixRange
    {
      opendarts::config::index_t rows_from = 0;
      opendarts::config::index_t rows_to = 0;
      // Sorted, disjoint, half-open column intervals [first, second).
      std::vector<std::pair<opendarts::config::index_t,
                            opendarts::config::index_t>> cols;
    };

    /** @brief A 2-D slice of a block_csr_matrix.

        - pos: starting offset inside each source N x N block (row-major).
        - sizes: { sub-block rows, sub-block cols } in DOFs.
        - strides: { N_VARS, 1 } kept for parity with the proprietary code.
        - ranges: row x col rectangles selected from the source matrix.
        - global_to_local / global_to_local_rows: caches populated by
          init_rows_cols_to_unit_matrix so that per-Newton refreshes can
          gather values without rewalking the sparsity.
        - n_rows / n_cols / nnz: dimensions of the slice (after init()). */
    class MatrixSlice
    {
    public:
      using index_t = opendarts::config::index_t;
      using mat_float = opendarts::config::mat_float;

      std::uint8_t pos = 0;            // offset in N x N where slice (0,0) starts
      std::uint8_t sizes[2] = {0, 0};  // { sub-block rows, sub-block cols } in DOFs
      std::uint8_t strides[2] = {0, 1}; // { N_VARS, 1 }; kept for compat
      std::vector<MatrixRange> ranges;

      std::vector<index_t> global_to_local;       // value-gather permutation
      std::vector<index_t> global_to_local_rows;  // scalar row in A's flat numbering

      index_t n_rows = 0;
      index_t n_cols = 0;
      std::uint64_t nnz = 0;
      bool is_init = false;

      /** Range-dense init: assumes every cell of every declared
          row x col rectangle is occupied. Mirrors proprietary
          MatrixSlice::init() (untyped overload). */
      void init_structural();

      /** Sparsity-following init: walks A's row_ptr / col_ind and counts
          only those entries that fall in the slice's column intervals.
          Mirrors proprietary MatrixSlice::init<N>(A). */
      void init(const opendarts::linear_solvers::block_csr_matrix &A);
    };

    // ------------------------------------------------------------------------
    // Free helpers (slice-aware operations on block_csr_matrix / csr_matrix<N>)
    // ------------------------------------------------------------------------

    /** Build the scalar-CSR structure for a unit-block extraction.
        Populates s.global_to_local and s.global_to_local_rows; fills dest's
        rows_ptr / cols_ind / diag_ind and sets its dimensions. The slice
        must be square (sizes[0] == sizes[1]) and the source's runtime
        block_size must match the expected N_VARS used to encode s.pos.

        Ported from proprietary init_rows_cols_to_unit_matrix. */
    void init_rows_cols_to_unit_matrix(
        const opendarts::linear_solvers::block_csr_matrix &src,
        MatrixSlice &s,
        opendarts::linear_solvers::csr_matrix<1> &dest);

    /** Per-Newton refresh: extract scalar values from A using the cached
        global_to_local. If @p positive_diagonal is true, flip any row whose
        diagonal became negative and store +/-1 in @p rhs_mults. If
        @p asymmetric_hack is true, multiply the first row's values and
        rhs_mults[0] by 2.0 (BoomerAMG asymmetry workaround).

        Ported from proprietary extract_block_matrix_to_unit_block. */
    void extract_sub_block_to_scalar_csr(
        const MatrixSlice &s,
        const opendarts::linear_solvers::block_csr_matrix &A,
        opendarts::linear_solvers::csr_matrix<1> &dest,
        opendarts::config::mat_float *rhs_mults,
        bool positive_diagonal,
        bool asymmetric_hack);

    /** NE>1 PPSS-style extraction: dest has block size NE and its sparsity
        matches A's row_ptr / col_ind verbatim. dest_block_size must equal NE.
        Ported from proprietary extract_block_matrix. */
    template <std::uint8_t NE>
    void extract_sub_block_to_block_csr(
        const MatrixSlice &s,
        const opendarts::linear_solvers::block_csr_matrix &A,
        opendarts::linear_solvers::csr_matrix<NE> &dest,
        opendarts::config::mat_float *rhs_mults);

    /** Slice-aware SpMV: dest += A[slice] * src. Caller MUST zero dest
        before calling (mirrors proprietary block_vector_product). */
    void block_vector_product(
        const opendarts::linear_solvers::block_csr_matrix &A,
        const MatrixSlice &s,
        const opendarts::config::mat_float *src,
        opendarts::config::mat_float *dest);

    /** Add r[block_size*i + k] * mults[block_size*i + k] to the diagonal
        block's (k,k) entry of every row i, for k in [0, block_size). If
        @p positive_diagonal, re-flip rows whose new diagonal turned
        negative. Ported from proprietary apply_relaxation. */
    template <std::uint8_t NB>
    void apply_relaxation(
        opendarts::linear_solvers::csr_matrix<NB> &M,
        const opendarts::config::mat_float *r,
        opendarts::config::mat_float *mults,
        bool positive_diagonal);

    /** Augment the diagonal block (0,0) with r_p[i] and the diagonal block
        entry at offset NE (= row 1, col 0 of the NE x NE diagonal block)
        with r_s[i]. positive_diagonal is unconditionally off (matches the
        proprietary version, which forces it to false). */
    template <std::uint8_t NE>
    void apply_ps_relaxation(
        opendarts::linear_solvers::csr_matrix<NE> &M,
        const opendarts::config::mat_float *r_p,
        const opendarts::config::mat_float *r_s,
        opendarts::config::mat_float *mults);

    /** For each row, move the diagonal block to the front of that row
        (column 0 of the row). The original column order can be restored
        via set_diag_in_order. Used to satisfy some HYPRE/AMG solvers'
        "diagonal first" requirement. */
    template <std::uint8_t NB>
    void set_diag_first(opendarts::linear_solvers::csr_matrix<NB> &M);

    /** Restore natural column ordering after set_diag_first. */
    template <std::uint8_t NB>
    void set_diag_in_order(opendarts::linear_solvers::csr_matrix<NB> &M);
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_MATRIX_SLICE_HPP
//--------------------------------------------------------------------------
