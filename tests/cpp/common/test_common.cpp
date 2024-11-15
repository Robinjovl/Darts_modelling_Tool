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

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <string.h>
#include <string>

#include "openDARTS/linear_solvers/csr_matrix.hpp"

#include "test_common.hpp"

bool opendarts::linear_solvers::testing::compare_files(std::string filename_1, std::string filename_2)
{
  std::ifstream file_1, file_2; // the streams for the files so that we can read them

  file_1.open(filename_1.c_str(), std::ios::binary); // c_str() returns C-style string pointer
  file_2.open(filename_2.c_str(), std::ios::binary);

  // Test 1: Check if files can be openned
  if (!file_1)
  {
    std::cout << "Couldn't open the file:  " << filename_1 << std::endl;
    return false;
  }

  if (!file_2)
  {
    std::cout << "Couldn't open the file: " << filename_2 << std::endl;
    return false;
  }

  // Check if files have the same number of lines
  int n_lines_1, n_lines_2;
  n_lines_1 = 0;
  n_lines_2 = 0;
  std::string temp_line;

  // Compute number of lines of file 1
  while (!file_1.eof())
  {
    std::getline(file_1, temp_line);
    n_lines_1++;
  }

  // Compute number of lines of file 2
  while (!file_2.eof())
  {
    std::getline(file_2, temp_line);
    n_lines_2++;
  }

  file_1.clear();
  file_1.seekg(0, std::ios::beg);

  file_2.clear();
  file_2.seekg(0, std::ios::beg);

  if (n_lines_1 != n_lines_2)
  {
    std::cout << "Different number of lines in files!" << std::endl;
    std::cout << filename_1 << " has " << n_lines_1 << " lines and " << filename_2 << " has" << n_lines_2 << " lines"
              << std::endl;
    return false;
  }
  //---------- compare two files line by line ------------------//
  std::string temp_line_1, temp_line_2;
  int line_idx = 0;
  while (!file_1.eof())
  {
    line_idx++;                        // update the line number
    std::getline(file_1, temp_line_1); // read the next line of file 1
    std::getline(file_2, temp_line_2); // read the next line of file 2
    if (temp_line_1.compare(temp_line_2) != 0)
    {
      std::cout << "Line " << line_idx << ": the strings are not equal" << std::endl;
      std::cout << "   file_1:  " << temp_line_1 << std::endl;
      std::cout << "   file_2:  " << temp_line_2 << std::endl;
      return false;
    }
  }

  return true;
}

template <uint8_t N_BLOCK_SIZE>
void opendarts::linear_solvers::testing::generate_tridiagonal_matrix(
    opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> &A,
    opendarts::config::index_t n,
    opendarts::linear_solvers::testing::block_fill_option block_fill,
    bool is_Poisson)
{
  // Constructs a square tridiagonal matrix of dimension n x n with values -2, 0, 2
  // We must have that n >= 2

  // Setup some auxiliary option flags to determine how to fill the blocks
  bool is_index;
  bool is_diagonal;

  switch (block_fill)
  {
  case opendarts::linear_solvers::testing::block_fill_option::index_filled_block:
    is_index = true;
    is_diagonal = false;
    break;
  case opendarts::linear_solvers::testing::block_fill_option::constant_filled_block:
    is_index = false;
    is_diagonal = false;
    break;
  case opendarts::linear_solvers::testing::block_fill_option::diagonal_block:
    is_index = false;
    is_diagonal = true;
    break;
  default:
    throw std::invalid_argument("Invalid tridiagonal block fill option!");
    break;
  }

  // Initialize the sparse matrix by setting the number of rows, columns, the block size, number of non zero values
  opendarts::config::index_t n_rows = n;
  opendarts::config::index_t n_cols = n;
  opendarts::config::index_t n_block_size = N_BLOCK_SIZE;
  opendarts::config::index_t n_non_zeros = ((n - 4) * 3) + 8; // the matrix has n rows, each row has 3 blocks,
                                                              // except for the first two and the last two rows,
                                                              // which have only 2 blocks each
                                                              // this means we have (n - 4)*3 + 8 blocks
                                                              // this->n_non_zeros is the number of blocks

  A.init(n_rows, n_cols, n_non_zeros); // recall that n_rows and n_cols is with respect to blocks
                                       // and n_rows x n_cols matrix has in fact dimension
                                       // n_rows*n_block_size x n_cols*n_block_size

  // Populate the sparse matrix (tri-diagonal matrix, -2, 1, 2)
  opendarts::config::index_t *A_rows_ptr = A.get_rows_ptr();
  opendarts::config::index_t *A_cols_ind = A.get_cols_ind();
  opendarts::config::mat_float *A_values = A.get_values();

  A_rows_ptr[0] = 0;
  opendarts::config::index_t block_idx = 0;
  opendarts::config::mat_float
      value_temp = 0.0; // the auxiliary variable to store the temporary value to place in each nonzero value of a block
  opendarts::config::mat_float
      value_set = 0.0; // the auxiliary variable to store the value to place in each nonzero value of a block
  for (opendarts::config::index_t row_idx = 0; row_idx < n_rows; row_idx++)
  {
    // Diagonal -2 element (if is_SPD=true then the minus becomes a +)
    // When is_test = false the block is set to -2
    // When is_test = true we add more information so that we can see what is going on inside for debugging
    // These blocks will look like this, for example for a 4x4 block sparse matrix
    //     -100102  -100202   -100302  -100402
    //     -200102  -200202   -200302  -200402
    //     -300102  -300202   -300302  -300402
    //     -400102  -400202   -400302  -400402
    // That is: -(100000 * (this_block_value_row_idx + 1) + 100 * (this_block_value_col_idx + 1) + 2)
    if (row_idx - 2 >= 0)
    {
      A_cols_ind[block_idx] = row_idx - 2; // set the column index
      for (opendarts::config::index_t this_block_value_row_idx = 0; this_block_value_row_idx < n_block_size;
           this_block_value_row_idx++) // set the block data
      {
        // If we are filling the matrix in index mode use the value filling rule,
        // here we set the part for the inner block row index. Otherwise just set
        // the value to -2.
        if(is_Poisson)
        {
          value_temp = is_index ? 100000 * (this_block_value_row_idx + 1) + 1 : 1;
        }
        else
        {
          value_temp = is_index ? -100000 * (this_block_value_row_idx + 1) - 2 : -2;
        }
        for (opendarts::config::index_t this_block_value_col_idx = 0; this_block_value_col_idx < n_block_size;
             this_block_value_col_idx++)
        {
          // If we are filling the matrix in index mode, here we add the part of
          // the value associated to the inner block column index. Otherwise,
          // leave the value as it was: -2.
          value_temp += is_index ? -100 : 0;

          // If we are filling the matrix in diagonal mode, we set the off
          // diagonal block values to 0.0, otherwise we leave them as they are.
          if (is_diagonal)
          {
            if (this_block_value_row_idx != this_block_value_col_idx)
              value_set = 0.0;
            else
              value_set = value_temp;
          }
          else
            value_set = value_temp;

          // We set the value of the row and column of the block
          A_values[block_idx * n_block_size * n_block_size + this_block_value_row_idx * n_block_size +
                   this_block_value_col_idx] = value_set;
        }
      }

      block_idx++;
    }

    // Diagonal element
    // When is_test = false the block is set to 1
    // When is_test = true we add more information so that we can see what is going on inside for debugging
    // These blocks will look like this, for example for a 4x4 block sparse matrix
    //     100101  100201   100301  100401
    //     200101  200201   200301  200401
    //     300101  300201   300301  300401
    //     400101  400201   400301  400401
    // That is: (100000 * (this_block_value_row_idx + 1) + 100 * (this_block_value_col_idx + 1) + 1)
    A_cols_ind[block_idx] = row_idx; // set the column index
    for (opendarts::config::index_t this_block_value_row_idx = 0; this_block_value_row_idx < n_block_size;
         this_block_value_row_idx++) // set the block data
    {
      // If we are filling the matrix in index mode use the value filling rule,
      // here we set the part for the inner block row index. Otherwise just set
      // the value to 1.
      if(is_Poisson)
      {
        value_temp = is_index ? -100000 * (this_block_value_row_idx + 1) - 2 : -2;
      }
      else
      {
        value_temp = is_index ? 100000 * (this_block_value_row_idx + 1) + 1 : 1;
      }
      for (opendarts::config::index_t this_block_value_col_idx = 0; this_block_value_col_idx < n_block_size;
           this_block_value_col_idx++)
      {
        // If we are filling the matrix in index mode, here we add the part of
        // the value associated to the inner block column index. Otherwise,
        // leave the value as it was.
        value_temp += is_index ? 100 : 0;

        // If we are filling the matrix in diagonal mode, we set the off
        // diagonal block values to 0.0, otherwise we leave them as they are.
        if (is_diagonal)
        {
          if (this_block_value_row_idx != this_block_value_col_idx)
            value_set = 0.0;
          else
            value_set = value_temp;
        }
        else
          value_set = value_temp;

        // We set the value of the row and column of the block
        A_values[block_idx * n_block_size * n_block_size + this_block_value_row_idx * n_block_size +
                 this_block_value_col_idx] = value_set;
      }
    }

    block_idx++;

    // Diagonal 2 element
    // When is_test = false the block is set to -2
    // When is_test = true we add more information so that we can see what is going on inside for debugging
    // These blocks will look like this, for example for a 4x4 block sparse matrix
    //     100102  100202   100302  100402
    //     200102  200202   200302  200402
    //     300102  300202   300302  300402
    //     400102  400202   400302  400402
    // That is: (100000 * (this_block_value_row_idx + 1) + 100 * (this_block_value_col_idx + 1) + 2)
    if (row_idx + 2 < n_cols)
    {
      A_cols_ind[block_idx] = row_idx + 2; // set the column index
      for (opendarts::config::index_t this_block_value_row_idx = 0; this_block_value_row_idx < n_block_size;
           this_block_value_row_idx++) // set the block data
      {
        if(is_Poisson)
        {
          value_temp = is_index ? 100000 * (this_block_value_row_idx + 1) + 1 : 1;
        }
        else
        {
          value_temp = is_index ? 100000 * (this_block_value_row_idx + 1) + 2 : 2;
        }  
        for (opendarts::config::index_t this_block_value_col_idx = 0; this_block_value_col_idx < n_block_size;
             this_block_value_col_idx++)
        {
          // If we are filling the matrix in index mode, here we add the part of
          // the value associated to the inner block column index. Otherwise,
          // leave the value as it was.
          value_temp += is_index ? 100 : 0;

          // If we are filling the matrix in diagonal mode, we set the off
          // diagonal block values to 0.0, otherwise we leave them as they are.
          if (is_diagonal)
          {
            if (this_block_value_row_idx != this_block_value_col_idx)
              value_set = 0.0;
            else
              value_set = value_temp;
          }
          else
            value_set = value_temp;

          // We set the value of the row and column of the block
          A_values[block_idx * n_block_size * n_block_size + this_block_value_row_idx * n_block_size +
                   this_block_value_col_idx] = value_set;
        }
      }

      block_idx++;
    }

    A_rows_ptr[row_idx + 1] = block_idx;
  }
}

template void opendarts::linear_solvers::testing::generate_tridiagonal_matrix(
    opendarts::linear_solvers::csr_matrix<1> &A,
    opendarts::config::index_t n,
    opendarts::linear_solvers::testing::block_fill_option block_fill,
    bool is_SPD);
template void opendarts::linear_solvers::testing::generate_tridiagonal_matrix(
    opendarts::linear_solvers::csr_matrix<2> &A,
    opendarts::config::index_t n,
    opendarts::linear_solvers::testing::block_fill_option block_fill,
    bool is_SPD);
template void opendarts::linear_solvers::testing::generate_tridiagonal_matrix(
    opendarts::linear_solvers::csr_matrix<3> &A,
    opendarts::config::index_t n,
    opendarts::linear_solvers::testing::block_fill_option block_fill,
    bool is_SPD);
template void opendarts::linear_solvers::testing::generate_tridiagonal_matrix(
    opendarts::linear_solvers::csr_matrix<4> &A,
    opendarts::config::index_t n,
    opendarts::linear_solvers::testing::block_fill_option block_fill,
    bool is_SPD);
template void opendarts::linear_solvers::testing::generate_tridiagonal_matrix(
    opendarts::linear_solvers::csr_matrix<5> &A,
    opendarts::config::index_t n,
    opendarts::linear_solvers::testing::block_fill_option block_fill,
    bool is_SPD);
template void opendarts::linear_solvers::testing::generate_tridiagonal_matrix(
    opendarts::linear_solvers::csr_matrix<6> &A,
    opendarts::config::index_t n,
    opendarts::linear_solvers::testing::block_fill_option block_fill,
    bool is_SPD);
template void opendarts::linear_solvers::testing::generate_tridiagonal_matrix(
    opendarts::linear_solvers::csr_matrix<7> &A,
    opendarts::config::index_t n,
    opendarts::linear_solvers::testing::block_fill_option block_fill,
    bool is_SPD);
template void opendarts::linear_solvers::testing::generate_tridiagonal_matrix(
    opendarts::linear_solvers::csr_matrix<8> &A,
    opendarts::config::index_t n,
    opendarts::linear_solvers::testing::block_fill_option block_fill,
    bool is_SPD);
template void opendarts::linear_solvers::testing::generate_tridiagonal_matrix(
    opendarts::linear_solvers::csr_matrix<9> &A,
    opendarts::config::index_t n,
    opendarts::linear_solvers::testing::block_fill_option block_fill,
    bool is_SPD);
template void opendarts::linear_solvers::testing::generate_tridiagonal_matrix(
    opendarts::linear_solvers::csr_matrix<10> &A,
    opendarts::config::index_t n,
    opendarts::linear_solvers::testing::block_fill_option block_fill,
    bool is_SPD);
template void opendarts::linear_solvers::testing::generate_tridiagonal_matrix(
    opendarts::linear_solvers::csr_matrix<11> &A,
    opendarts::config::index_t n,
    opendarts::linear_solvers::testing::block_fill_option block_fill,
    bool is_SPD);
template void opendarts::linear_solvers::testing::generate_tridiagonal_matrix(
    opendarts::linear_solvers::csr_matrix<12> &A,
    opendarts::config::index_t n,
    opendarts::linear_solvers::testing::block_fill_option block_fill,
    bool is_SPD);
template void opendarts::linear_solvers::testing::generate_tridiagonal_matrix(
    opendarts::linear_solvers::csr_matrix<13> &A,
    opendarts::config::index_t n,
    opendarts::linear_solvers::testing::block_fill_option block_fill,
    bool is_SPD);
