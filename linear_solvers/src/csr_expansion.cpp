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

#include <cassert>

#include "csr_expansion.hpp"
#include "sparsity_pattern.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    csr_expansion::csr_expansion(const sparsity_pattern &bsr, int block_size)
    {
      assert(block_size >= 1 && "csr_expansion: block size must be >= 1");

      block_size_ = block_size;
      const index_t nb = static_cast<index_t>(block_size);
      const index_t n_block_rows = bsr.n_block_rows();

      n_rows_ = n_block_rows * nb;
      n_cols_ = bsr.n_block_cols() * nb;
      nnz_ = bsr.n_blocks() * nb * nb;

      row_ptr_.resize(static_cast<std::size_t>(n_rows_) + 1);
      col_ind_.resize(static_cast<std::size_t>(nnz_));
      value_map_.resize(static_cast<std::size_t>(nnz_));

      const index_t *brow_ptr = bsr.row_ptr();
      const index_t *bcol_ind = bsr.col_ind();
      index_t *rp = row_ptr_.host_data();
      index_t *ci = col_ind_.host_data();
      index_t *vm = value_map_.host_data();

      // Each block row I expands to nb scalar rows (I*nb + e). Each nonzero
      // block jb of that row expands to nb scalar entries on scalar row e,
      // sourced from the row-major block storage values[jb*nb*nb + e*nb + v].
      index_t k = 0;
      rp[0] = 0;
      for (index_t i_block = 0; i_block < n_block_rows; ++i_block)
      {
        const index_t jb_begin = brow_ptr[i_block];
        const index_t jb_end = brow_ptr[i_block + 1];
        for (index_t e = 0; e < nb; ++e)
        {
          for (index_t jb = jb_begin; jb < jb_end; ++jb)
          {
            const index_t j_block = bcol_ind[jb];
            for (index_t v = 0; v < nb; ++v)
            {
              ci[k] = j_block * nb + v;
              vm[k] = jb * nb * nb + e * nb + v;
              ++k;
            }
          }
          rp[i_block * nb + e + 1] = k;
        }
      }
      assert(k == nnz_ && "csr_expansion: nonzero count mismatch");
    }

    void csr_expansion::gather_values(const mat_float *bsr_values, mat_float *csr_values) const
    {
      const index_t *vm = value_map_.host_data();
      for (index_t k = 0; k < nnz_; ++k)
        csr_values[k] = bsr_values[vm[k]];
    }
  } // namespace linear_solvers
} // namespace opendarts
