#include "csr_matrix.h"
#include "openDARTS/linear_solvers/data_types.hpp"
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#include "openDARTS/linear_solvers/config.hpp"

#include <iostream>
//#include <cstdlib>
#include <string>

int main()
{
    std::cout << "Start benchmarking..." << std::endl;

    std::cout << "csr_matrix:" << std::endl;

    // Declare the sparse matrix with a block of size 1
    std::cout << "   Constructing sparse matrix...";
    csr_matrix<1> A_sparse_matrix;
    std::cout << " DONE" << std::endl;

    // Initialize the sparse matrix by setting the number of rows, columns, the block size, number of non zero values
    std::cout << "   Allocating memory for sparse matrix...";
    t_long n_rows = 10;
    t_long n_cols = 10;
    int n_block_size = 1;
    t_long n_non_zeros = 26;
    A_sparse_matrix.init(n_rows, n_cols, n_block_size, n_non_zeros);
    std::cout << " DONE" << std::endl;

    // Populate the sparse matrix (tri-diagonal matrix, -2, 0, 2)
    int *A_rows_ptr = A_sparse_matrix.get_rows_ptr();
    int *A_cols_ind = A_sparse_matrix.get_cols_ind();
    double *A_values = A_sparse_matrix.get_values();

    int block_idx = 0;
    A_rows_ptr[0] = 0;
    for (int row_idx = 0; row_idx < n_rows; row_idx++) {
        // Diagonal -2 element
        if (row_idx - 2 >= 0){
          A_cols_ind[block_idx] = row_idx - 2;  // set the column index
          for (int this_block_value_idx = 0; this_block_value_idx < n_block_size * n_block_size; this_block_value_idx++)  // set the block data
            A_values[block_idx * n_block_size * n_block_size + this_block_value_idx] = -10*row_idx;

          block_idx++;
        }

        // Diagonal element
        A_cols_ind[block_idx] = row_idx;  // set the column index
        for (int this_block_value_idx = 0; this_block_value_idx < n_block_size * n_block_size; this_block_value_idx++)  // set the block data
          A_values[block_idx * n_block_size * n_block_size + this_block_value_idx] = row_idx;

        block_idx++;

        // Diagonal 2 element
        if (row_idx + 2 < n_cols){
          A_cols_ind[block_idx] = row_idx + 2;  // set the column index
          for (int this_block_value_idx = 0; this_block_value_idx < n_block_size * n_block_size; this_block_value_idx++)  // set the block data
            A_values[block_idx * n_block_size * n_block_size + this_block_value_idx] = 10*row_idx;

          block_idx++;
        }

        A_rows_ptr[row_idx + 1] = block_idx;
    }

    // Write sparse matrix to file (human readable)
    std::cout << "   Writing sparse matrix to file (human readable)...";
    std::string filename_str = "./A_sparse_matrix.txt";
    const char *filename_char = filename_str.c_str();
    std::string zero_symbol_str = "0";
    const char *zero_symbol_char = zero_symbol_str.c_str();
    A_sparse_matrix.write_matrix_to_file (filename_char, zero_symbol_char);
    std::cout << " DONE" << std::endl;

    // Write sparse matrix to file (human readable for block matrices)
    {
      std::cout << "   Writing sparse matrix to file (human readable block matrix)...";
      std::string filename_str = "./A_sparse_matrix_block.txt";
      const char *filename_char = filename_str.c_str();
      A_sparse_matrix.write_matrix_to_file (filename_char);
      std::cout << " DONE" << std::endl;
    }

    // Write sparse matrix to file (storage format)
    {
      std::cout << "   Writing sparse matrix to file...";
      std::string filename_str = "./A_sparse_matrix_csr.txt";
      const char *filename_char = filename_str.c_str();
      std::string zero_symbol_str = "0";
      const char *zero_symbol_char = zero_symbol_str.c_str();
      A_sparse_matrix.write_matrix_to_file (filename_char);
      std::cout << " DONE" << std::endl;
    }


    // opendarts::linear_solvers_csr_matrix_storage
    // Declare the sparse matrix storage
    std::cout << "   Constructing sparse matrix with opendarts::linear_solvers::csr_matrix_storage...\n";
    opendarts::linear_solvers::csr_matrix_storage A_sparse_matrix_storage;
    std::cout << "\n   n_non_zeros:" << A_sparse_matrix_storage.n_non_zeros << std::endl;
    std::cout << "   n_rows     :" << A_sparse_matrix_storage.n_rows << std::endl;
    std::cout << "   n_cols     :" << A_sparse_matrix_storage.n_cols << std::endl;
    std::cout << "\n    DONE" << std::endl;

    // Initialize the sparse matrix by setting the number of rows, columns, the block size, number of non zero values
    std::cout << "   Allocating memory for sparse matrix...\n";
    opendarts::config::index_t n_rows_storage = 10;
    opendarts::config::index_t n_cols_storage = 10;
    opendarts::config::index_t n_block_size_storage = 1;
    opendarts::config::index_t n_non_zeros_storage = 24;
    A_sparse_matrix_storage.init(n_rows_storage, n_cols_storage, n_block_size_storage, n_non_zeros_storage);
    std::cout << "\n   n_non_zeros:" << A_sparse_matrix_storage.n_non_zeros << std::endl;
    std::cout << "   n_rows     :" << A_sparse_matrix_storage.n_rows << std::endl;
    std::cout << "   n_cols     :" << A_sparse_matrix_storage.n_cols << std::endl;
    std::cout << "\n    DONE" << std::endl;

    // Populate the sparse matrix (tri-diagonal matrix, -2, 0, 2)
    int *A_storage_rows_ptr = A_sparse_matrix_storage.get_rows_ptr();
    int *A_storage_cols_ind = A_sparse_matrix_storage.get_cols_ind();
    double *A_storage_values = A_sparse_matrix_storage.get_values();

    std::cout << "Value of A_storage_values[20] before changing:" << A_storage_values[20] << std::endl;
    A_storage_values[20] = 123.4;
    std::cout << "Value of A_storage_values[20] after changing:" << A_storage_values[20] << std::endl;
    std::cout << "Value of A_sparse_matrix_storage.values[20] after changing:" << A_sparse_matrix_storage.values[20] << std::endl;

    // Construct the sparse matrix from existing sparse matrix
    std::cout << "   Constructing sparse matrix with opendarts::linear_solvers::csr_matrix_storage...\n";
    n_rows_storage = 5;
    n_cols_storage = 4;
    n_block_size_storage = 1;
    n_non_zeros_storage = 9;

    opendarts::linear_solvers::csr_matrix_storage B_sparse_matrix_storage(n_rows_storage, n_cols_storage, n_block_size_storage, n_non_zeros_storage);
    std::cout << "\n   n_non_zeros:" << B_sparse_matrix_storage.n_non_zeros << std::endl;
    std::cout << "   n_rows     :" << B_sparse_matrix_storage.n_rows << std::endl;
    std::cout << "   n_cols     :" << B_sparse_matrix_storage.n_cols << std::endl;
    std::cout << "\n    DONE" << std::endl;

    B_sparse_matrix_storage.rows_ptr[0] = 0;
    B_sparse_matrix_storage.rows_ptr[1] = 2;
    B_sparse_matrix_storage.rows_ptr[2] = 2;
    B_sparse_matrix_storage.rows_ptr[3] = 4;
    B_sparse_matrix_storage.rows_ptr[4] = 8;
    B_sparse_matrix_storage.rows_ptr[5] = 9;

    B_sparse_matrix_storage.cols_ind[0] = 0;
    B_sparse_matrix_storage.cols_ind[1] = 3;
    B_sparse_matrix_storage.cols_ind[2] = 1;
    B_sparse_matrix_storage.cols_ind[3] = 2;
    B_sparse_matrix_storage.cols_ind[4] = 0;
    B_sparse_matrix_storage.cols_ind[5] = 1;
    B_sparse_matrix_storage.cols_ind[6] = 2;
    B_sparse_matrix_storage.cols_ind[7] = 3;
    B_sparse_matrix_storage.cols_ind[8] = 3;

    B_sparse_matrix_storage.values[0] = 1;
    B_sparse_matrix_storage.values[1] = 2;
    B_sparse_matrix_storage.values[2] = 3;
    B_sparse_matrix_storage.values[3] = 4;
    B_sparse_matrix_storage.values[4] = 5;
    B_sparse_matrix_storage.values[5] = 6;
    B_sparse_matrix_storage.values[6] = 7;
    B_sparse_matrix_storage.values[7] = 8;
    B_sparse_matrix_storage.values[8] = 9;


    std::string output_filename = "./B_sparse_matrix_storage_human_readable.txt";
    int error_output = 0;
    error_output = B_sparse_matrix_storage.write_matrix_to_file(
        output_filename,
        opendarts::linear_solvers::sparse_matrix_export_format::human_readable);



    // Initialize the sparse matrix by setting the number of rows, columns, the block size, number of non zero values
    std::cout << "   Allocating memory for C sparse matrix...";
    n_rows = 10;
    n_cols = 10;
    n_block_size = 1;
    n_non_zeros = 26;
    opendarts::linear_solvers::csr_matrix_storage C_sparse_matrix_storage;
    C_sparse_matrix_storage.init(n_rows, n_cols, n_block_size, n_non_zeros);
    std::cout << " DONE" << std::endl;

    // Populate the sparse matrix (tri-diagonal matrix, -2, 0, 2)
    int *C_rows_ptr = C_sparse_matrix_storage.get_rows_ptr();
    int *C_cols_ind = C_sparse_matrix_storage.get_cols_ind();
    double *C_values = C_sparse_matrix_storage.get_values();

    block_idx = 0;
    C_rows_ptr[0] = 0;
    for (int row_idx = 0; row_idx < n_rows; row_idx++) {
        // Diagonal -2 element
        if (row_idx - 2 >= 0){
          C_cols_ind[block_idx] = row_idx - 2;  // set the column index
          for (int this_block_value_idx = 0; this_block_value_idx < n_block_size * n_block_size; this_block_value_idx++)  // set the block data
            C_values[block_idx * n_block_size * n_block_size + this_block_value_idx] = -10*row_idx;

          block_idx++;
        }

        // Diagonal element
        C_cols_ind[block_idx] = row_idx;  // set the column index
        for (int this_block_value_idx = 0; this_block_value_idx < n_block_size * n_block_size; this_block_value_idx++)  // set the block data
          C_values[block_idx * n_block_size * n_block_size + this_block_value_idx] = row_idx;

        block_idx++;

        // Diagonal 2 element
        if (row_idx + 2 < n_cols){
          C_cols_ind[block_idx] = row_idx + 2;  // set the column index
          for (int this_block_value_idx = 0; this_block_value_idx < n_block_size * n_block_size; this_block_value_idx++)  // set the block data
            C_values[block_idx * n_block_size * n_block_size + this_block_value_idx] = 10*row_idx;

          block_idx++;
        }

        C_rows_ptr[row_idx + 1] = block_idx;
    }

    output_filename = "./C_sparse_matrix_storage_human_readable.txt";
    error_output = C_sparse_matrix_storage.write_matrix_to_file(
        output_filename,
        opendarts::linear_solvers::sparse_matrix_export_format::human_readable);

    std::cout << "\nopen_darts_linear_solvers MAJOR_VERSION: " << opendarts::linear_solvers::VERSION_MAJOR << "\n";
    std::cout << "open_darts_linear_solvers MINOR_VERSION: " << opendarts::linear_solvers::VERSION_MINOR << "\n";

    return 0;
}
