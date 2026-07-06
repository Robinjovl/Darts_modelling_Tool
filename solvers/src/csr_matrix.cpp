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

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "csr_matrix.hpp"
#include "linear_solvers_data_types.hpp"
#include "omp_partition.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    // Initialization member functions
    template <uint8_t N_BLOCK_SIZE> csr_matrix<N_BLOCK_SIZE>::csr_matrix()
    {
      // Default initialization corresponds to initializing the sparse matrix
      // as a matrix of dimension 0 x 0 with block size 0 x 0 and 0 non zero values
      this->n_rows = 0;
      this->n_cols = 0;
      this->n_non_zeros = 0;
      this->n_row_size = N_BLOCK_SIZE; // keep the csr_matrix_base block-size field in sync

      this->init(n_rows, n_cols, n_non_zeros);
    }

    template <uint8_t N_BLOCK_SIZE>
    csr_matrix<N_BLOCK_SIZE>::csr_matrix(opendarts::config::index_t n_rows_input,
        opendarts::config::index_t n_cols_input,
        opendarts::config::index_t n_non_zeros_input)
    {
      this->n_row_size = N_BLOCK_SIZE; // keep the csr_matrix_base block-size field in sync
      this->init(n_rows_input, n_cols_input, n_non_zeros_input);
    }

    template <uint8_t N_BLOCK_SIZE>
    csr_matrix<N_BLOCK_SIZE>::csr_matrix(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> &csr_matrix_in)
    {
      this->n_row_size = N_BLOCK_SIZE; // keep the csr_matrix_base block-size field in sync
      this->init(csr_matrix_in);
    }

    template <uint8_t N_BLOCK_SIZE>
    int csr_matrix<N_BLOCK_SIZE>::init(opendarts::config::index_t n_rows_input,
        opendarts::config::index_t n_cols_input,
        opendarts::config::index_t n_non_zeros_input)
    {
      // Update matrix parameters
      this->n_rows = n_rows_input;
      this->n_cols = n_cols_input;
      this->n_non_zeros = n_non_zeros_input;
      this->n_total_non_zeros = this->n_non_zeros * this->b_sqr;

      if (n_rows_input == n_cols_input)
      {
        if (n_rows_input != 0)
        {
          this-> is_square = 1;
        }
        else
        {
          this-> is_square = 0;
        }
      }
      else
      {
        this->is_square = 0;
      }

      // Even-row thread partition sized to the engine's OpenMP assembly team, so
      // the matrix the engine assembles into (forward Jacobian for csr_matrix<N>,
      // adjoint dg_dx for csr_matrix<1>) is multi-threading-ready. Single range
      // when built without OpenMP. See omp_partition.hpp.
      {
        const int n_threads = opendarts::linear_solvers::omp_assembly_n_threads();
        this->row_thread_starts.resize(static_cast<std::size_t>(n_threads) + 1);
        opendarts::linear_solvers::fill_even_row_partition(
            this->row_thread_starts.data(), this->n_rows, n_threads);
      }

      // Allocate memory for storage vectors containing block csr matrix representation
      this->values.assign(this->n_total_non_zeros,
          0.0); // values need to be initialized with the total number of elements and not block elements
      this->cols_ind.assign(this->n_non_zeros, 0); // column indices point to blocks, so we use n_non_zeros
      this->rows_ptr.assign(this->n_rows + 1, 0);
      this->diag_ind.assign(this->n_rows, 0);  // diagonal indices point to blocks of the diagonal, so we use n_rows

      return 0;
    }

    // TODO: n_block_size_input makes no sense as input parameter, since this is
    //       a templated matrix with block size as the template parameter. By allowing
    //       this, one can declare the matrix with block size N and then with this
    //       function set it to M, which gives rise to all sorts of errors.
    //       Was kept for now for backwards compatibility, but as optional, so that
    //       it is removed in the (very near) future.
    template <uint8_t N_BLOCK_SIZE>
    int csr_matrix<N_BLOCK_SIZE>::init(opendarts::config::index_t n_rows_input,
        opendarts::config::index_t n_cols_input,
        opendarts::config::index_t n_non_zeros_input,
        const std::vector<opendarts::config::mat_float> &values_input,
        const std::vector<opendarts::config::index_t> &rows_ptr_input,
        const std::vector<opendarts::config::index_t> &cols_ind_input)
    {
      // First init the matrix to match the sizes of csr_matrix
      this->init(n_rows_input, n_cols_input, n_non_zeros_input);

      // Populate the matrix data with the data of the input matrix
      this->values = values_input;
      this->cols_ind = cols_ind_input;
      this->rows_ptr = rows_ptr_input;

      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int csr_matrix<N_BLOCK_SIZE>::init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> &csr_matrix_in)
    {
      // Initialize the matrix with the values of the input matrix
      this->init(csr_matrix_in.n_rows, csr_matrix_in.n_cols, csr_matrix_in.n_non_zeros, csr_matrix_in.values,
          csr_matrix_in.rows_ptr, csr_matrix_in.cols_ind);

      return 0;
    }

    // Data access Member functions
    template <uint8_t N_BLOCK_SIZE> opendarts::config::mat_float *csr_matrix<N_BLOCK_SIZE>::get_values()
    {
      return this->values.data();
    }

    template <uint8_t N_BLOCK_SIZE> opendarts::config::index_t *csr_matrix<N_BLOCK_SIZE>::get_rows_ptr()
    {
      return this->rows_ptr.data();
    }

    template <uint8_t N_BLOCK_SIZE> opendarts::config::index_t *csr_matrix<N_BLOCK_SIZE>::get_cols_ind()
    {
      return this->cols_ind.data();
    }

    template <uint8_t N_BLOCK_SIZE> opendarts::config::index_t *csr_matrix<N_BLOCK_SIZE>::get_diag_ind()
    {
      return this->diag_ind.data();
    }

    template <uint8_t N_BLOCK_SIZE> opendarts::config::index_t csr_matrix<N_BLOCK_SIZE>::get_n_non_zeros()
    {
      return this->n_non_zeros;
    }

    // Input/Output
    template <uint8_t N_BLOCK_SIZE>
    int csr_matrix<N_BLOCK_SIZE>::export_matrix_to_file(const std::string &filename,
        opendarts::linear_solvers::sparse_matrix_export_format export_format)
    {
      int error_output = 0;

      switch (export_format)
      {
      case opendarts::linear_solvers::sparse_matrix_export_format::human_readable:
        error_output = this->export_matrix_to_file_human_readable(filename);
        break;

      case opendarts::linear_solvers::sparse_matrix_export_format::csr:
        error_output = this->export_matrix_to_file_csr(filename);
        break;

      default:
        std::cout << "\nInvalid matrix export format!" << std::endl;
        error_output = 10; // invalid matrix export format
        break;
      }

      return error_output;
    }

    template <uint8_t N_BLOCK_SIZE>
    int csr_matrix<N_BLOCK_SIZE>::import_matrix_from_file(const std::string &filename,
        opendarts::linear_solvers::sparse_matrix_import_format import_format)
    {
      int error_output = 0;

      switch (import_format)
      {
      case opendarts::linear_solvers::sparse_matrix_import_format::csr:
        std::cout << "\nImporting matrix from file in csr format!" << std::endl;
        error_output = this->import_matrix_from_file_csr(filename);
        break;
      default:
        std::cout << "\nInvalid matrix import format!" << std::endl;
        error_output = 10; // invalid matrix import format
        break;
      }
      return error_output;
    }

    template <uint8_t N_BLOCK_SIZE>
    int csr_matrix<N_BLOCK_SIZE>::export_matrix_to_file_human_readable(const std::string &filename)
    {
      int error_output = 0;

      // If matrix has a block size larger than 1, first create a copy of this
      // matrix with block size 1 and save this one to file
      if (this->n_block_size_ > 1)
      {
        opendarts::linear_solvers::csr_matrix<1> this_as_nb_1;
        this->as_nb_1(this_as_nb_1);
        error_output = this_as_nb_1.export_matrix_to_file(filename,
            opendarts::linear_solvers::sparse_matrix_export_format::human_readable);
        return error_output;
      }

      // If matrix has block size 1, just go ahead and save it to file
      std::ofstream output_file;

      // Open the file for writting
      output_file.open(filename);

      // Write the header
      output_file << "N_ROWS\tN_COLS\tN_NON_ZEROS\n";
      output_file << this->n_rows << "\t" << this->n_cols << "\t" << this->n_non_zeros << "\n\n";

      // Write the column indices
      // First add a line
      for (opendarts::config::index_t col_idx = 0; col_idx < this->n_cols; col_idx++)
        output_file << "---------";
      output_file << "\n";

      // Then the indices of the columns
      output_file << "  \t" << 0 << "\t"; // first column index
      for (opendarts::config::index_t col_idx = 1; col_idx < this->n_cols; col_idx++)
        output_file << col_idx << "\t";
      output_file << "\n";

      // Then add another line
      for (opendarts::config::index_t col_idx = 0; col_idx < this->n_cols; col_idx++)
        output_file << "---------";
      output_file << "\n";

      // Now loop over the rows and show the data
      int this_value_idx;
      int this_col_idx;
      int next_col_idx;
      int n_non_zero_values;
      int n_zeros_to_add;

      for (opendarts::config::index_t row_idx = 0; row_idx < this->n_rows; row_idx++)
      {
        output_file << row_idx << "  |\t";

        n_non_zero_values = this->rows_ptr[row_idx + 1] -
                            this->rows_ptr[row_idx]; // the number of nonzero values in this row

        if (n_non_zero_values == 0)
        {
          for (opendarts::config::index_t zero_idx = 0; zero_idx < this->n_cols; zero_idx++) // add full row of zeros
            output_file << "0"
                        << "\t";
        }
        else
        {
          n_zeros_to_add = this->cols_ind[this->rows_ptr[row_idx]]; // if there are nonzero elemens in row we need to
                                                                    // add zeros until the first nonzero value
          for (opendarts::config::index_t zero_idx = 0; zero_idx < n_zeros_to_add; zero_idx++) // add zeros
            output_file << "0"
                        << "\t";
        }

        for (opendarts::config::index_t non_zero_idx = 0; non_zero_idx < n_non_zero_values; non_zero_idx++)
        {
          this_value_idx = this->rows_ptr[row_idx] + non_zero_idx;
          this_col_idx = this->cols_ind[this_value_idx];
          output_file << std::fixed << std::setprecision(2) << this->values[this_value_idx]
                      << "\t"; // print value with two floating places of precision
          if (non_zero_idx == n_non_zero_values - 1)
          {
            next_col_idx = this->n_cols;
          }
          else
          {
            next_col_idx = this->cols_ind[this_value_idx + 1];
          }
          n_zeros_to_add = (next_col_idx - this_col_idx) - 1; // add as many zeros as the columns in between the two
                                                              // nonzero values (or the end of the row)
          for (opendarts::config::index_t zero_idx = 0; zero_idx < n_zeros_to_add; zero_idx++) // add zeros
            output_file << "0"
                        << "\t";
        }
        output_file << "\n";
      }

      // Close the file
      output_file.close();

      return error_output;
    }

    template <uint8_t N_BLOCK_SIZE> int csr_matrix<N_BLOCK_SIZE>::export_matrix_to_file_csr(const std::string &filename)
    {
      // // N_ROWS	N_COLS	N_NON_ZEROS	N_BLOCK_SIZE
      // 3	3	3	2
      // // Rows indexes[1..n_rows] (with out 0)
      // 2
      // 2
      // 4
      // // END of Rows indexes
      // // Values n_non_zeros elements
      // // COLUMN	VALUE
      // // ROW 0
      // 0	 1.0  0.0  2.0  3.0
      // 2	-1.0 -3.0 -2.0  0.0
      // // ROW 1
      // // ROW 2
      // 2	 5.0  7.0  6.0  8.0
      // // END OF VALUES
      // // END OF FILE

      int error_output = 0;
      std::ofstream output_file;

      // Open the file for writting
      output_file.open(filename);

      // Write the header
      output_file << "// N_ROWS\tN_COLS\tN_NON_ZEROS\tN_BLOCK_SIZE\n";
      output_file << this->n_rows << "\t" << this->n_cols << "\t" << this->n_non_zeros << "\t" << this->n_block_size_
                  << "\n\n";

      // Write rows information
      output_file << "// Rows pointer indices [1, ..., n_rows] (with 0)";
      for (opendarts::config::index_t row_idx = 0; row_idx <= this->n_rows; row_idx++)
        output_file << "\n" << this->rows_ptr[row_idx];

      output_file << "\n"
                  << "// END Rows points indices";

      // Write values information
      output_file << "\n\n"
                  << "// Values of n_non_zero_elements";
      output_file << "\n"
                  << "// Column index\t\tBlock Values";

      opendarts::config::index_t col_idx = 0;
      // We loop over the rows
      for (opendarts::config::index_t row_idx = 0; row_idx < this->n_rows; row_idx++)
      {
        output_file << "\n"
                    << "// ROW " << row_idx;
        // Get the start and end index (the range) of the column indices of each block with non zero values in this row
        // and loop over them
        for (opendarts::config::index_t in_row_block_idx = this->rows_ptr[row_idx];
             in_row_block_idx < this->rows_ptr[row_idx + 1]; in_row_block_idx++)
        {
          // With the index of the block in this row, we can extract the index of
          // the column in the array corresponding to this block
          col_idx = this->cols_ind[in_row_block_idx]; // the column index of the block, the values are then extracted
          // At this point we know that we are processing the block at (row_idx, col_idx)
          // in the matrix
          // Since each block is n_block_size x n_block_size we need to read the
          // n_block_size^2 values associated to it
          output_file << "\n" << col_idx;
          for (opendarts::config::index_t in_block_value_idx = 0; in_block_value_idx < this->b_sqr;
               in_block_value_idx++)
          {
            output_file << "\t" << std::setprecision(16) << std::fixed << this->values[in_row_block_idx * this->b_sqr + in_block_value_idx];
          }
        }
      }

      output_file << "\n"
                  << "// END of Values\n";
      output_file << "\n"
                  << "// END of File";

      // Close the file
      output_file.close();

      return error_output;
    }

    template <uint8_t N_BLOCK_SIZE>
    int csr_matrix<N_BLOCK_SIZE>::import_matrix_from_file_csr(const std::string &filename)
    {
      // The csr file format has the following structure
      // Example for the matrix (3 x 3 with blocks of 2x2)
      //    1.0 0.0   [0] [0]   -1.0 -3.0
      //    2.0 3.0   [0] [0]   -2.0  0.0
      //    [0] [0]   [0] [0]    [0] [0]   |
      //    [0] [0]   [0] [0]    [0] [0]   | <-- Row 1 is empty
      //    [0] [0]   [0] [0]    5.0 7.0 |
      //    [0] [0]   [0] [0]    6.0 8.0 |  <-- Block on row 2 and column 2
      //                         -------

      // // N_ROWS	N_COLS	N_NON_ZEROS	N_BLOCK_SIZE
      // 3	3	3	2
      // // Rows indexes[1..n_rows] (with out 0)
      // 2
      // 2
      // 4
      // // END of Rows indexes
      // // Values n_non_zeros elements
      // // COLUMN	VALUE
      // // ROW 0
      // 0	 1.0  0.0  2.0  3.0
      // 2	-1.0 -3.0 -2.0  0.0
      // // ROW 1
      // // ROW 2
      // 2	 5.0  7.0  6.0  8.0
      // // END OF VALUES
      // // END OF FILE

      int error_output = 0;
      std::ifstream csr_file;
      std::string dummy_line;
      std::string dummy_string;

      // Open the file for writting
      csr_file.open(filename);

      // Read header line (ignore the contents)
      // // N_ROWS	N_COLS	N_NON_ZEROS	N_BLOCK_SIZE
      std::getline(csr_file, dummy_line);

      // Read matrix structure information
      opendarts::config::index_t n_rows_input, n_cols_input, n_non_zeros_input, n_block_size_input;
      csr_file >> n_rows_input >> n_cols_input >> n_non_zeros_input >> n_block_size_input;

      // Check if block size is compatible with matrix definition
      if(n_block_size_input != this->n_block_size_)
      {
        error_output = 100;
        return error_output;  // exit immediately
      }

      // Initialize the matrix with the file structure information
      this->init(n_rows_input, n_cols_input, n_non_zeros_input);

      // Read empty line and rows header line (ignore the contents)
      //
      // // N_ROWS	N_COLS	N_NON_ZEROS	N_BLOCK_SIZE
      std::getline(csr_file, dummy_line);  // finish previous line
      std::getline(csr_file, dummy_line);  // empty line
      std::getline(csr_file, dummy_line);  // rows header

      // Read rows_ptr data
      for(opendarts::config::index_t rows_ptr_idx=0; rows_ptr_idx < (n_rows_input + 1); rows_ptr_idx++)
      {
        csr_file >> this->rows_ptr[rows_ptr_idx];
      }

      // Read end of rows header
      // // END Rows points indices
      std::getline(csr_file, dummy_line);  // finish previous line
      std::getline(csr_file, dummy_line);

      // Read cols_ind header
      // [empty line]
      // // Values of n_non_zero_elements
      // // Column index		Block Values
      std::getline(csr_file, dummy_line);  // empty line
      std::getline(csr_file, dummy_line);
      std::getline(csr_file, dummy_line);

      // Read data per row and per column
      for(opendarts::config::index_t row_idx=0; row_idx < n_rows_input; row_idx++)
      {
        // Read row header
        std::getline(csr_file, dummy_line);  // ROW [row_idx] (ignore)

        // Check the start memory index of columns and how many nonzero columns exist in this row
        opendarts::config::index_t col_idx;  // the column memory index in cols_ind
        opendarts::config::index_t start_col_idx = this->rows_ptr[row_idx];  // the start memory index of column and data
        opendarts::config::index_t this_row_nnz_cols = this->rows_ptr[row_idx + 1] - this->rows_ptr[row_idx];  // check how many columns in this row

        // Read all non-zero (nnz) columns in this row
        for(opendarts::config::index_t col_nnz_idx = 0; col_nnz_idx < this_row_nnz_cols; col_nnz_idx++)
        {
          // csr_file >> dummy_string;  // read //, will be removed
          col_idx = start_col_idx + col_nnz_idx;  // compute the column memory index to know where to place
          csr_file >> this->cols_ind[col_idx];  // read the column index value for this block of values

          // Read data for the whole block
          opendarts::config::index_t block_data_mem_idx;  // the memory index of the data in the block
          for(opendarts::config::index_t block_data_idx = 0; block_data_idx < (n_block_size_input * n_block_size_input); block_data_idx++)
          {
            block_data_mem_idx = col_idx * n_block_size_input * n_block_size_input + block_data_idx;  // compute the index in memory of this data value
            csr_file >> this->values[block_data_mem_idx];  // read the value
          }
          std::getline(csr_file, dummy_line);  // end line
        }
      }

      // Close the file
      csr_file.close();

      // Return the error value capture during execution
      return error_output;
    }

    template <uint8_t N_BLOCK_SIZE>
    void csr_matrix<N_BLOCK_SIZE>::as_nb_1(opendarts::linear_solvers::csr_matrix<1> &csr_matrix_nb_1) const
    {
      if (this->n_block_size_ == 1)
      {
        // If the original matrix is already block size 1, just copy this matrix
        // to the nb_1 matrix
        csr_matrix_nb_1.init(this->n_rows, this->n_cols, this->n_non_zeros, this->values, this->rows_ptr,
            this->cols_ind);
      }
      else
      {
        // Set the sizes of the input matrix (the block size 1 matrix)
        opendarts::config::index_t n_rows_nb_1 = this->n_rows *
                                                 this->n_block_size_; // since each element of the matrix is a block,
                                                                     // then the number of actual rows is given by this
        opendarts::config::index_t n_cols_nb_1 = this->n_cols * this->n_block_size_; // same as above
        opendarts::config::index_t
            n_non_zeros_nb_1 = this->n_non_zeros *
                               this->b_sqr; // same as above, every "element" of a matrix is a block with
                                                          // n_block_size x n_block_size values

        csr_matrix_nb_1.init(n_rows_nb_1, n_cols_nb_1,
            n_non_zeros_nb_1); // initializes the memory space for the nb_1 matrix

        // Populate the nb_1 matrix with the data
        opendarts::config::index_t col_idx = 0;
        opendarts::config::index_t inner_block_value_offset = 0;
        opendarts::config::index_t row_idx_nb_1 = 0;
        opendarts::config::index_t col_idx_nb_1 = 0;

        opendarts::config::index_t value_idx_nb_1 = 0;
        csr_matrix_nb_1.rows_ptr[0] = value_idx_nb_1;

        // We loop over the rows
        for (opendarts::config::index_t row_idx = 0; row_idx < this->n_rows; row_idx++)
        {
          // Since nb_1 matrix has a block size of 1, we need to loop along the
          // same inner block row of all blocks in this row. That is:
          // -------------------------                 -------------------------
          // |                        |               |                         |
          // |  (row_idx, col_idx_1)  |               |  (row_idx, col_idx_2)   |   --> Read all values in this this
          // inner row of all the blocks |                        |               |                         |       in
          // row_idx
          // -------------------------                 -------------------------
          for (opendarts::config::index_t inner_block_row_idx = 0; inner_block_row_idx < this->n_block_size_;
               inner_block_row_idx++)
          {
            inner_block_value_offset = inner_block_row_idx *
                                       this->n_block_size_; // values for a block are stored row wise, since we read per
                                                           // row, we need this stride
            row_idx_nb_1 = row_idx * this->n_block_size_ + inner_block_row_idx; // the row index on nb_1 matrix

            // Get the start and end index (the range) of the column indices of each block with non zero values in this
            // row and loop over them to extract the row inner_block_row_idx of each of them
            for (opendarts::config::index_t in_row_block_idx = this->rows_ptr[row_idx];
                 in_row_block_idx < this->rows_ptr[row_idx + 1]; in_row_block_idx++)
            {
              // With the index of the block in this row, we can extract the index of
              // the column in the array corresponding to this block
              col_idx = this->cols_ind[in_row_block_idx]; // the column index of the block, the values are then
                                                          // extracted

              // At this point we know that we are processing the block at (row_idx, col_idx) and inner row
              // inner_block_row_idx in this matrix
              for (opendarts::config::index_t inner_block_col_idx = 0; inner_block_col_idx < this->n_block_size_;
                   inner_block_col_idx++)
              {
                col_idx_nb_1 = col_idx * this->n_block_size_ + inner_block_col_idx;
                csr_matrix_nb_1.cols_ind[value_idx_nb_1] = col_idx_nb_1;
                csr_matrix_nb_1.values[value_idx_nb_1] = this->values[in_row_block_idx * this->b_sqr +
                                                                      inner_block_value_offset + inner_block_col_idx];
                value_idx_nb_1++;
              }
              csr_matrix_nb_1.rows_ptr[row_idx_nb_1 + 1] = value_idx_nb_1;
            }
          }
        }
      }
    }

    template <uint8_t N_BLOCK_SIZE>
    void csr_matrix<N_BLOCK_SIZE>::transpose(
        opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> &csr_matrix_transpose) const
    {
      // Initialize the output transpose matrix
      csr_matrix_transpose.init(this->n_cols, this->n_rows, this->n_non_zeros);

      // For the algorithm we need to have csr_matrix_transpose.row_ptr with one extra element
      csr_matrix_transpose.rows_ptr.push_back(0);

      // Count number of nonzero values per column
      for (opendarts::config::index_t value_idx = 0; value_idx < this->n_non_zeros; ++value_idx)
      {
        // Note that we start at 2 (hence the +2). This is done for optimization.
        // The next for loop will take this count and convert it into rows_ptr proper.
        // But at this stage we will have the start and not the end of the indices.
        // rows_ptr is essentially shifted to the right.
        ++csr_matrix_transpose.rows_ptr[this->cols_ind[value_idx] + 2];
      }

      // What we need is the index of the start of the cols_ind and values for each
      // (new) row. To get that we just need to do a cumulative sum of the number of
      // elements in each (new) row
      opendarts::config::index_t rows_ptr_size = opendarts::config::static_cast_check(
          csr_matrix_transpose.rows_ptr.size()); // we need to cast to opendarts::config::index_t for safe comparison,
                                                 // we also check if there is possible overflow in the conversion
      // Continue the loop now with proper types
      for (opendarts::config::index_t row_idx = 2; row_idx < rows_ptr_size; ++row_idx)
      {
        // Incremental sum.
        // Note that, as mentioned above, we will have something like:
        //     [0 0 r_2 r_3 r_4 ... r_(n_rows-1)]
        // when usually we would like to have something like
        //     [0 r_2 r_3 r_4 r_5 ... r_(n_rows-1) n_non_zeros]
        csr_matrix_transpose.rows_ptr[row_idx] += csr_matrix_transpose.rows_ptr[row_idx - 1];
      }

      // Fill in the transpose matrix data (values and cols_ind)
      // The idea here is to go over the rows and for each row loop over the
      // columns that contain nonzero values.
      for (opendarts::config::index_t row_idx = 0; row_idx < this->n_rows; ++row_idx)
      {
        for (opendarts::config::index_t value_idx = this->rows_ptr[row_idx]; value_idx < this->rows_ptr[row_idx + 1];
             ++value_idx)
        {
          // Once we dealing with value value_idx present in row row_idx, we now the following.
          // The index of this value in the transposed matrix is given by the following line.
          // This line assumes that we go row by row. This means that if we consider
          // one column, we go along this column from top to bottom. Because of this,
          // we can increment the rows_ptr value every time we hit it. Once we finish
          // this loop, rows_ptr will go from
          //     [0 0 r_2 r_3 r_4 ... r_(n_rows-1)]
          // to
          //     [0 r_2 r_3 r_4 r_5 ... r_(n_rows-1) n_non_zeros]
          // as mentioned above. This eliminates the need to add an intermediate variable.
          opendarts::config::index_t transpose_value_idx = csr_matrix_transpose
                                                               .rows_ptr[this->cols_ind[value_idx] + 1]++;
          // With the new index of the value on the transpose matrix, we can fill
          // in the value and set the column index (which is just the current row).
          csr_matrix_transpose.cols_ind[transpose_value_idx] = row_idx;
          // Since the matrix is made of blocks, we need to transpose the block and
          // pass on the whole data of the block.
          // Loop over the rows of the block
          for (opendarts::config::index_t inner_block_row_idx = 0; inner_block_row_idx < this->n_block_size_;
               ++inner_block_row_idx)
            // Loop over the columns of the block
            for (opendarts::config::index_t inner_block_col_idx = 0; inner_block_col_idx < this->n_block_size_;
                 ++inner_block_col_idx)
              // Copy the data, transposing it
              csr_matrix_transpose.values[transpose_value_idx * this->b_sqr +
                                          inner_block_col_idx * this->n_block_size_ + inner_block_row_idx] =
                  this->values[value_idx * this->b_sqr + inner_block_row_idx * this->n_block_size_ +
                               inner_block_col_idx];
        }
      }

      // We need to remove the extra element at the end since it is not needed
      csr_matrix_transpose.rows_ptr.pop_back();
    }

    template <uint8_t N_BLOCK_SIZE>
    int csr_matrix<N_BLOCK_SIZE>::matrix_vector_product_t(opendarts::config::mat_float *v, opendarts::config::mat_float *r)
    {
      // Transposed SpMV is only defined for the scalar (block size 1) layout.
      // if constexpr discards the loop below for N_BLOCK_SIZE > 1 instead of
      // leaving it as unreachable code after a runtime return.
      if constexpr (N_BLOCK_SIZE > 1)
        return -1;
      else
      {
      opendarts::config::index_t i, j1, j2, j, cl;
      // Here we loop over the rows or matrix, which means
      // looping over the columns of the transpose
      for (i = 0; i < n_rows; ++i)
      {
        j1 = rows_ptr[i];
        j2 = rows_ptr[i + 1];

        // Here we loop over the columns of the matrix, which means
        // looping over the rows of the transpose
        for (j = j1; j < j2; ++j)
        {
          cl = cols_ind[j];
          r[cl] += v[i] * values[j];
        }
      }

      return 0;
      }
    }


    // TODO: Kept for backwards compatibility, need to check if this is to be kept or not or restructured

    template <uint8_t N_BLOCK_SIZE> int csr_matrix<N_BLOCK_SIZE>::init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *csr_matrix_in)
    {
      // Initialize the matrix with the values of the input matrix
      return this->init(csr_matrix_in->n_rows, csr_matrix_in->n_cols, csr_matrix_in->n_non_zeros, csr_matrix_in->values,
          csr_matrix_in->rows_ptr, csr_matrix_in->cols_ind);
    }

    template <uint8_t N_BLOCK_SIZE> int csr_matrix<N_BLOCK_SIZE>::init(opendarts::config::index_t n_rows_input,
        opendarts::config::index_t n_cols_input,
        opendarts::config::index_t n_block_size_input,
        opendarts::config::index_t n_non_zeros_input,
        opendarts::config::index_t *row_thread_starts_input)
    {
      if(n_block_size_input != this->n_block_size_)
        std::cout << "csr_matrix::init: You cannot initialize a sparse matrix with a different block size." << std::endl;

      (void) row_thread_starts_input;

      return this->init(n_rows_input, n_cols_input, n_non_zeros_input);
    }

    template <uint8_t N_BLOCK_SIZE> int csr_matrix<N_BLOCK_SIZE>::init_struct(opendarts::config::index_t n_rows_input,
        opendarts::config::index_t n_cols_input,
        opendarts::config::index_t n_non_zeros_input)
    {
      return this->init(n_rows_input, n_cols_input, n_non_zeros_input);
    }

    template <uint8_t N_BLOCK_SIZE> opendarts::config::index_t *csr_matrix<N_BLOCK_SIZE>::get_row_thread_starts()
    {
      // Re-derive the even-row partition when the OpenMP assembly team size
      // changed since init() built it (set_num_threads() after model init):
      // the engines call this right before opening their parallel region, so
      // a stale partition would leave tail rows unassembled (smaller team)
      // or be read out of bounds (larger team).
      const int n_threads = opendarts::linear_solvers::omp_assembly_n_threads();
      if (static_cast<int>(this->row_thread_starts.size()) != n_threads + 1)
      {
        this->row_thread_starts.resize(static_cast<std::size_t>(n_threads) + 1);
        opendarts::linear_solvers::fill_even_row_partition(
            this->row_thread_starts.data(), this->n_rows, n_threads);
      }
      return this->row_thread_starts.data();
    }

    template <uint8_t N_BLOCK_SIZE> int csr_matrix<N_BLOCK_SIZE>::build_transpose(
        opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *csr_matrix_in)
    {
      csr_matrix_in->transpose(*this);

      return 0;
    }

    template <uint8_t N_BLOCK_SIZE> int csr_matrix<N_BLOCK_SIZE>::build_transpose_struct(
        opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *csr_matrix_in)
    {
      csr_matrix_in->transpose(*this);
      // Reset the data to zero
      this->values.assign(this->n_total_non_zeros, 0.0);

      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    template <uint8_t M_BLOCK_SIZE>
    int csr_matrix<N_BLOCK_SIZE>::to_nb_1(const opendarts::linear_solvers::csr_matrix<M_BLOCK_SIZE> *csr_matrix_in)
    {
      csr_matrix_in->as_nb_1(*this);
      return 0;
    }

    // Polymorphic block-CSR -> scalar-CSR expansion. Unlike the templated
    // overload above this reads the source through the csr_matrix_base
    // accessor interface, so it works for both the legacy csr_matrix<N> and
    // the unified block_csr_matrix. The result is written into *this as a
    // scalar (block size 1) CSR matrix.
    template <uint8_t N_BLOCK_SIZE>
    int csr_matrix<N_BLOCK_SIZE>::to_nb_1(opendarts::linear_solvers::csr_matrix_base *csr_matrix_in)
    {
      const opendarts::config::index_t block_size = csr_matrix_in->n_row_size;
      const opendarts::config::index_t n_block_rows = csr_matrix_in->n_rows;
      const opendarts::config::index_t n_block_cols = csr_matrix_in->n_cols;
      const opendarts::config::index_t *src_rows = csr_matrix_in->get_rows_ptr();
      const opendarts::config::index_t *src_cols = csr_matrix_in->get_cols_ind();
      const opendarts::config::mat_float *src_vals = csr_matrix_in->get_values();
      const opendarts::config::index_t n_nnzb = src_rows[n_block_rows];

      // Scalar dimensions: every block expands to block_size x block_size scalars.
      this->init(n_block_rows * block_size, n_block_cols * block_size, n_nnzb * block_size * block_size);

      opendarts::config::index_t value_idx = 0;
      this->rows_ptr[0] = 0;
      for (opendarts::config::index_t row_idx = 0; row_idx < n_block_rows; row_idx++)
      {
        for (opendarts::config::index_t inner_row = 0; inner_row < block_size; inner_row++)
        {
          // Block values are stored row-major; this is the stride to inner_row.
          const opendarts::config::index_t inner_offset = inner_row * block_size;
          const opendarts::config::index_t row_nb_1 = row_idx * block_size + inner_row;
          for (opendarts::config::index_t bi = src_rows[row_idx]; bi < src_rows[row_idx + 1]; bi++)
          {
            const opendarts::config::index_t col_idx = src_cols[bi];
            for (opendarts::config::index_t inner_col = 0; inner_col < block_size; inner_col++)
            {
              this->cols_ind[value_idx] = col_idx * block_size + inner_col;
              this->values[value_idx] = src_vals[bi * block_size * block_size + inner_offset + inner_col];
              value_idx++;
            }
          }
          this->rows_ptr[row_nb_1 + 1] = value_idx;
        }
      }
      return 0;
    }

#ifdef WITH_GPU
    // ------------------------------------------------------------------------
    // GPU device layer (cuSPARSE block-CSR linear algebra)
    // ------------------------------------------------------------------------
    // Ported from the proprietary darts-linear-solvers csr_matrix. Compiled by
    // nvcc when WITH_GPU is set (see solvers/CMakeLists.txt). index_t is int and
    // mat_float is double, which matches the cuSPARSE legacy BSR API directly.
    //
    // CUDA 12+ deprecates the legacy block-CSR cuSPARSE routines (bsrmv,
    // bsr2csr, ...). They remain functional and there is no drop-in generic-API
    // BSR replacement, so the deprecation diagnostic is silenced for this
    // device layer; migrating to the generic cuSPARSE API is tracked separately.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

    template <uint8_t N_BLOCK_SIZE>
    int csr_matrix<N_BLOCK_SIZE>::init_device(int n_rows_input, int nnz)
    {
      cudaError_t cudaStat;
      cusparseStatus_t status;

      this->gpu_mode = 1;
      printf("CSR matrix device mode enabled\n");

      // Block-CSR structure on device.
      cudaStat = cudaMalloc((void **)&rows_ptr_d, sizeof(opendarts::config::index_t) * (n_rows_input + 1));
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't allocate device memory: %s\n", cudaGetErrorString(cudaStat));
        return -1;
      }

      cudaStat = cudaMalloc((void **)&cols_ind_d, sizeof(opendarts::config::index_t) * nnz);
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't allocate device memory for cols_ind\n");
        return -1;
      }

      cudaStat = cudaMalloc((void **)&diag_ind_d, sizeof(opendarts::config::index_t) * n_rows_input);
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't allocate device memory for diag_ind\n");
        return -1;
      }

      cudaStat = cudaMalloc((void **)&values_d, sizeof(opendarts::config::mat_float) * nnz * N_BLOCK_SIZE * N_BLOCK_SIZE);
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't allocate device memory for values\n");
        return -1;
      }

      // Device work vectors.
      cudaStat = cudaMalloc((void **)&v_d, sizeof(opendarts::config::mat_float) * n_rows_input * N_BLOCK_SIZE);
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't allocate device memory for v_d\n");
        return -1;
      }

      r_check = new opendarts::config::mat_float[n_rows_input * N_BLOCK_SIZE];

      cudaStat = cudaMalloc((void **)&r_d, sizeof(opendarts::config::mat_float) * n_rows_input * N_BLOCK_SIZE);
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't allocate device memory for r_d\n");
        return -1;
      }

      // Initialise the cuSPARSE library and the matrix descriptor.
      status = cusparseCreate(&cus_handle);
      if (status != CUSPARSE_STATUS_SUCCESS)
      {
        printf("CUSPARSE library initialization failed\n");
        return 1;
      }

      status = cusparseCreateMatDescr(&cus_descr);
      if (status != CUSPARSE_STATUS_SUCCESS)
      {
        printf("Matrix descriptor initialization failed\n");
        return 1;
      }
      cusparseSetMatType(cus_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
      cusparseSetMatIndexBase(cus_descr, CUSPARSE_INDEX_BASE_ZERO);

      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int csr_matrix<N_BLOCK_SIZE>::copy_struct_to_device()
    {
      cudaError_t cudaStat;

      cudaStat = cudaMemcpy(rows_ptr_d, this->rows_ptr.data(),
        sizeof(opendarts::config::index_t) * (this->n_rows + 1), cudaMemcpyHostToDevice);
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't copy rows_ptr to device\n");
        return -1;
      }

      cudaStat = cudaMemcpy(cols_ind_d, this->cols_ind.data(),
        sizeof(opendarts::config::index_t) * this->rows_ptr[this->n_rows], cudaMemcpyHostToDevice);
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't copy cols_ind to device\n");
        return -1;
      }

      cudaStat = cudaMemcpy(diag_ind_d, this->diag_ind.data(),
        sizeof(opendarts::config::index_t) * this->n_rows, cudaMemcpyHostToDevice);
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't copy diag_ind to device\n");
        return -1;
      }

      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int csr_matrix<N_BLOCK_SIZE>::copy_values_to_device()
    {
      cudaError_t cudaStat = cudaMemcpy(values_d, this->values.data(),
        sizeof(opendarts::config::mat_float) * this->rows_ptr[this->n_rows] * N_BLOCK_SIZE * N_BLOCK_SIZE,
        cudaMemcpyHostToDevice);
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't copy values to device\n");
        return -1;
      }
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int csr_matrix<N_BLOCK_SIZE>::copy_vector_to_device(const opendarts::config::mat_float *vector,
      opendarts::config::mat_float *vector_d)
    {
      cudaError_t cudaStat = cudaMemcpy(vector_d, vector,
        sizeof(opendarts::config::mat_float) * this->n_rows * N_BLOCK_SIZE, cudaMemcpyHostToDevice);
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't copy vector to device\n");
        return -1;
      }
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int csr_matrix<N_BLOCK_SIZE>::copy_vector_to_host(opendarts::config::mat_float *vector,
      opendarts::config::mat_float *vector_d)
    {
      cudaError_t cudaStat = cudaMemcpy(vector, vector_d,
        sizeof(opendarts::config::mat_float) * this->n_rows * N_BLOCK_SIZE, cudaMemcpyDeviceToHost);
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't copy vector to host\n");
        return -1;
      }
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int csr_matrix<N_BLOCK_SIZE>::convert_to_ELL()
    {
      cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;
      int mb = this->n_rows;
      int nb = this->n_rows;
      int blockDim = N_BLOCK_SIZE;

      cusparseStatus_t status;
      cusparseMatDescr_t descrC = 0;

      status = cusparseCreateMatDescr(&descrC);
      if (status != CUSPARSE_STATUS_SUCCESS)
      {
        printf("Matrix descriptor initialization failed\n");
        return 1;
      }
      cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
      cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);

      int m = mb * blockDim;
      int nnzb = this->rows_ptr[mb] - this->rows_ptr[0]; // number of blocks
      int nnz = nnzb * blockDim * blockDim;              // number of scalar entries
      if (!csrRowPtrC)
      {
        cudaMalloc((void **)&csrRowPtrC, sizeof(int) * (m + 1));
        cudaMalloc((void **)&csrColIndC, sizeof(int) * nnz);
        cudaMalloc((void **)&csrValC, sizeof(opendarts::config::mat_float) * nnz);
      }

      status = cusparseDbsr2csr(cus_handle, dir, mb, nb, cus_descr, values_d, rows_ptr_d, cols_ind_d,
        blockDim, descrC, csrValC, csrRowPtrC, csrColIndC);
      if (status != CUSPARSE_STATUS_SUCCESS)
      {
        printf("Conversion from BSR to CSR format failed\n");
        return 1;
      }

      cusparseDestroyMatDescr(descrC);
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int csr_matrix<N_BLOCK_SIZE>::matrix_vector_product_d(const double *v_d, double *r_d)
    {
      double alpha = 1;
      double beta = 1;
      cusparseStatus_t status = cusparseDbsrmv(cus_handle, CUSPARSE_DIRECTION_ROW,
        CUSPARSE_OPERATION_NON_TRANSPOSE, this->n_rows, this->n_rows, this->rows_ptr[this->n_rows],
        &alpha, cus_descr, values_d, rows_ptr_d, cols_ind_d, N_BLOCK_SIZE, v_d, &beta, r_d);
      if (status != CUSPARSE_STATUS_SUCCESS)
      {
        printf("cuSparse matrix-vector multiplication failed\n");
        return 1;
      }
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int csr_matrix<N_BLOCK_SIZE>::matrix_vector_product_d0(const double *v_d, double *r_d)
    {
      double alpha = 1;
      double beta = 0;
      cusparseStatus_t status = cusparseDbsrmv(cus_handle, CUSPARSE_DIRECTION_ROW,
        CUSPARSE_OPERATION_NON_TRANSPOSE, this->n_rows, this->n_rows, this->rows_ptr[this->n_rows],
        &alpha, cus_descr, values_d, rows_ptr_d, cols_ind_d, N_BLOCK_SIZE, v_d, &beta, r_d);
      if (status != CUSPARSE_STATUS_SUCCESS)
      {
        printf("cuSparse matrix-vector multiplication failed\n");
        return 1;
      }
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int csr_matrix<N_BLOCK_SIZE>::matrix_vector_product_d_ell(const double *v_d, double *r_d)
    {
      // The cuSPARSE HYB/ELL format was removed in CUDA 11. The scalar-CSR copy
      // produced by convert_to_ELL is kept, but the dedicated ELL SpMV path is
      // retired; callers fall back to the block SpMV above.
      (void)v_d;
      (void)r_d;
      printf("cuSparse ELL matrix-vector product is unavailable since CUDA 11.0\n");
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int csr_matrix<N_BLOCK_SIZE>::calc_lin_comb_d(double alpha, double *v_d, double beta, double *r_d)
    {
      cusparseStatus_t status = cusparseDbsrmv(cus_handle, CUSPARSE_DIRECTION_ROW,
        CUSPARSE_OPERATION_NON_TRANSPOSE, this->n_rows, this->n_rows, this->rows_ptr[this->n_rows],
        &alpha, cus_descr, values_d, rows_ptr_d, cols_ind_d, N_BLOCK_SIZE, v_d, &beta, r_d);
      if (status != CUSPARSE_STATUS_SUCCESS)
      {
        printf("cuSparse calc lin comb failed\n");
        return 1;
      }
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int csr_matrix<N_BLOCK_SIZE>::calc_lin_comb_d(const double alpha, const double beta,
      double *u, double *v, double *r)
    {
      cudaMemcpy(r, v, sizeof(double) * this->n_rows * N_BLOCK_SIZE, cudaMemcpyDeviceToDevice);
      cusparseStatus_t status = cusparseDbsrmv(cus_handle, CUSPARSE_DIRECTION_ROW,
        CUSPARSE_OPERATION_NON_TRANSPOSE, this->n_rows, this->n_rows, this->rows_ptr[this->n_rows],
        &alpha, cus_descr, values_d, rows_ptr_d, cols_ind_d, N_BLOCK_SIZE, u, &beta, r);
      if (status != CUSPARSE_STATUS_SUCCESS)
      {
        printf("cuSparse calc lin comb failed\n");
        return 1;
      }
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int csr_matrix<N_BLOCK_SIZE>::free_device()
    {
      if (this->gpu_mode != 1)
        return 0;

      cudaFree(rows_ptr_d);
      cudaFree(cols_ind_d);
      cudaFree(values_d);
      cudaFree(diag_ind_d);
      cudaFree(v_d);
      cudaFree(r_d);
      rows_ptr_d = cols_ind_d = diag_ind_d = nullptr;
      values_d = v_d = r_d = nullptr;

      if (csrRowPtrC)
      {
        cudaFree(csrRowPtrC);
        cudaFree(csrColIndC);
        cudaFree(csrValC);
        csrRowPtrC = csrColIndC = nullptr;
        csrValC = nullptr;
      }

      delete[] r_check;
      r_check = nullptr;

      cusparseStatus_t status = cusparseDestroyMatDescr(cus_descr);
      cus_descr = nullptr;
      if (status != CUSPARSE_STATUS_SUCCESS)
      {
        printf("Matrix descriptor destruction failed\n");
        return 1;
      }

      status = cusparseDestroy(cus_handle);
      cus_handle = nullptr;
      if (status != CUSPARSE_STATUS_SUCCESS)
      {
        printf("CUSPARSE library release of resources failed: %d\n", (int)status);
        return 1;
      }

      this->gpu_mode = 0;
      return 0;
    }
#pragma GCC diagnostic pop
#endif // WITH_GPU


    // Initialize available templates
    template class csr_matrix<1>;
    template class csr_matrix<2>;
    template class csr_matrix<3>;
    template class csr_matrix<4>;
    template class csr_matrix<5>;
    template class csr_matrix<6>;
    template class csr_matrix<7>;
    template class csr_matrix<8>;
    template class csr_matrix<9>;
    template class csr_matrix<10>;
    template class csr_matrix<11>;
    template class csr_matrix<12>;
    template class csr_matrix<13>;

    template int csr_matrix<1>::to_nb_1<1>(const csr_matrix<1>*);
    template int csr_matrix<1>::to_nb_1<2>(const csr_matrix<2>*);
    template int csr_matrix<1>::to_nb_1<3>(const csr_matrix<3>*);
    template int csr_matrix<1>::to_nb_1<4>(const csr_matrix<4>*);
    template int csr_matrix<1>::to_nb_1<5>(const csr_matrix<5>*);
    template int csr_matrix<1>::to_nb_1<6>(const csr_matrix<6>*);
    template int csr_matrix<1>::to_nb_1<7>(const csr_matrix<7>*);
    template int csr_matrix<1>::to_nb_1<8>(const csr_matrix<8>*);
    template int csr_matrix<1>::to_nb_1<9>(const csr_matrix<9>*);
    template int csr_matrix<1>::to_nb_1<10>(const csr_matrix<10>*);
    template int csr_matrix<1>::to_nb_1<11>(const csr_matrix<11>*);
    template int csr_matrix<1>::to_nb_1<12>(const csr_matrix<12>*);
    template int csr_matrix<1>::to_nb_1<13>(const csr_matrix<13>*);

  } // namespace linear_solvers
} // namespace opendarts
