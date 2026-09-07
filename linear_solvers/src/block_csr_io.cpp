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

#include <fstream>

#include "block_csr_io.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    int write_block_csr_matrix(const block_csr_matrix &matrix, const std::string &filename)
    {
      std::ofstream out(filename);
      if (!out)
        return 1;

      const int nb = matrix.block_size();
      const block_csr_matrix::index_t n_block_rows = matrix.n_block_rows();
      const block_csr_matrix::index_t *row_ptr = matrix.row_ptr();
      const block_csr_matrix::index_t *col_ind = matrix.col_ind();
      const block_csr_matrix::mat_float *values = matrix.values();

      out << "# block_csr_matrix  n_block_rows=" << n_block_rows
          << " block_size=" << nb << " n_blocks=" << matrix.n_blocks() << "\n";
      out << "# block_row block_col : block values (row-major)\n";

      const int block_area = nb * nb;
      for (block_csr_matrix::index_t i = 0; i < n_block_rows; ++i)
      {
        for (block_csr_matrix::index_t jb = row_ptr[i]; jb < row_ptr[i + 1]; ++jb)
        {
          out << i << ' ' << col_ind[jb] << " :";
          const block_csr_matrix::mat_float *blk =
            values + static_cast<std::size_t>(jb) * block_area;
          for (int k = 0; k < block_area; ++k)
            out << ' ' << blk[k];
          out << '\n';
        }
      }
      return out.good() ? 0 : 1;
    }
  } // namespace linear_solvers
} // namespace opendarts
