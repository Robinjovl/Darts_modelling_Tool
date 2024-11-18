//*************************************************************************
//    Copyright (c) 2022
//    Delft University of Technology, the Netherlands
//
//    This file is part of the open Delft Advanced Research Terra Simulator (opendarts)
//
//    opendarts is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as
//    published by the Free Software Foundation, either version 3 of the
//    License, or (at your option) any later version.
//
//    DARTS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public
//    License along with DARTS. If not, see <http://www.gnu.org/licenses/>.
// *************************************************************************

#include "openDARTS/linear_solvers/csr_matrix.hpp"
#include "openDARTS/linear_solvers/data_types.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    namespace testing
    {
      // Tridiagonal matrix types
      enum class block_fill_option
      {
        index_filled_block,
        constant_filled_block,
        diagonal_block
      };

      /** Compares two files \p filename_1 and \p filename_2 . If the contents are identical
          returns 0, if not returns 1.

          @param filename_1 - The filename of the first file.
          @param filename_2 - The filename of the second file.

          @return The comparison result of the two files: (true) equal, (false) different.
      */
      bool compare_files(std::string filename_1, std::string filename_2);

      /** Generates a tridiagonal matrix with values -2, 1, 2 in the diagonals -2, 0, 2.

          @param A - The opendarts::linear_solvers::csr_matrix to populate.
          @param n - The size of the array in block sizes. Since the array is square, this will be equal
                     to the number of rows and columns. NOTE: if the matrix
                     has block size m then the matrix will have m * n rows and m * n columns.
      */
      template <uint8_t N_BLOCK_SIZE>
      void generate_tridiagonal_matrix(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> &A,
          opendarts::config::index_t n,
          opendarts::linear_solvers::testing::block_fill_option block_fill =
              opendarts::linear_solvers::testing::block_fill_option::constant_filled_block,
              bool is_Poisson = false);

    } // namespace testing
  }   // namespace linear_solvers
} // namespace opendarts
