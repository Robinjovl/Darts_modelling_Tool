#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>

#include "openDARTS/config/version.hpp"
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#include "openDARTS/linear_solvers/data_types.hpp"

#include "test_common.hpp"

template <uint8_t N_BLOCK_SIZE> int test_block_size(std::string &output_filename, std::string &reference_filename);
template <uint8_t N_BLOCK_SIZE> int test_pointer(std::string &output_filename, std::string &reference_filename);
template <uint8_t N_BLOCK_SIZE> int test_to_nb_1(std::string &output_filename, std::string &reference_filename);

int main()
{
  /*
    test_03__as_nb_1
    Tests csr_matrix.as_nb_1. Generates two tridiagonal matrices, one with block
    size 1 and another with block size 4. Saves both to file, this leads to an
    implicit conversion to block size 1, testing csr_matrix.as_nb_1. The output
    is compared to references.
  */

  int error_output = 0;
  
  // Show information on the build
  std::cout << "Build date (var) : " << std::string(opendarts::config::LINSOLV_BUILD_DATE) << std::endl;
  std::cout << "Build date (func): " << opendarts::config::get_build_date() << std::endl;
  std::cout << "Build machine    : " << std::string(opendarts::config::LINSOLV_BUILD_MACHINE) << std::endl;
  std::cout << "Build git hash   : " << std::string(opendarts::config::LINSOLV_BUILD_GIT_HASH) << std::endl;

  // Test block size 1
  // The output file to save the matrix to
  std::string output_filename("test_03__human_readable_nb_1_matrix_n_block_size_1.txt");

  //  The reference ascii file
  std::string data_path_prefix = opendarts::config::get_cmake_openDARTS_source_dir() +
                                 std::string("/tests/cpp/data/tests/linear_solvers/"); // the path to the data folder
  std::string reference_base_filename(
      "test_03__human_readable_nb_1_matrix_n_block_size_1_ref.txt"); // the filename of the reference file
  std::string reference_filename = data_path_prefix +
                                   reference_base_filename; // get the full path to the reference file

  error_output = test_block_size<1>(output_filename, reference_filename);
  error_output += test_pointer<1>(output_filename, reference_filename);
  error_output += test_to_nb_1<1>(output_filename, reference_filename);

  // Test block size 4
  // The output file to save the matrix to
  output_filename.assign("test_03__human_readable_nb_1_matrix_n_block_size_4.txt");

  //  The reference ascii file
  reference_base_filename.assign(
      "test_03__human_readable_nb_1_matrix_n_block_size_4_ref.txt"); // the filename of the reference file
  reference_filename = data_path_prefix + reference_base_filename;   // get the full path to the reference file

  error_output += test_block_size<4>(output_filename, reference_filename);
  error_output += test_pointer<4>(output_filename, reference_filename);
  error_output += test_to_nb_1<4>(output_filename, reference_filename);

  return error_output;
}

template <uint8_t N_BLOCK_SIZE> int test_block_size(std::string &output_filename, std::string &reference_filename)
{
  int error_output = 0;
  bool files_are_equal = true;
  opendarts::config::index_t n = 12;

  // Generate the matrix with non 1 block size
  opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> A;
  opendarts::linear_solvers::testing::generate_tridiagonal_matrix(A,
      n); // populate the matrix, in this case a tridiagonal matrix with the values -2, 1, 2 in the -2, 0, and 2
          // diagonals

  // Save the matrix in human readable format
  // This implies a conversion to block size 1 so we test two functions in one:
  // csr_matrix.as_nb_1 and csr_matrix.export_matrix_to_file
  error_output = A.export_matrix_to_file(output_filename,
      opendarts::linear_solvers::sparse_matrix_export_format::human_readable); // save the matrix to file

  // Compare the generated output to the reference output
  files_are_equal = opendarts::linear_solvers::testing::compare_files(output_filename,
      reference_filename); // check if equal to reference

  if (files_are_equal)
  {
    error_output = 0;
  }
  else
  {
    error_output = 1;
  }

  return error_output;
}

template <uint8_t N_BLOCK_SIZE> int test_pointer(std::string &output_filename, std::string &reference_filename)
{
  int error_output = 0;
  bool files_are_equal = true;
  opendarts::config::index_t n = 12;

  // Generate the matrix with non 1 block size
  opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> A;
  opendarts::linear_solvers::testing::generate_tridiagonal_matrix(A,
      n); // populate the matrix, in this case a tridiagonal matrix with the values -2, 1, 2 in the -2, 0, and 2
          // diagonals

  // Convert the matrix to block size 1 and store it in a pointer
  // NOTE: this function using pointer will only copy data if a real conversion
  //       is needed, i.e., A.n_block_size > 1, otherwise the pointer will point
  //       to A.
  opendarts::linear_solvers::csr_matrix<1> *A_as_nb_1;

  if (N_BLOCK_SIZE > 1)
  {
    // If the system matrix A is not of block size 1 then we need to create a temporary
    // A matrix with block size 1 (A_as_nb_1).
    A_as_nb_1 = new opendarts::linear_solvers::csr_matrix<1>;
  }
  A.as_nb_1(A_as_nb_1);

  // Save the matrix in human readable format
  // This implies a conversion to block size 1 so we test two functions in one:
  // csr_matrix.as_nb_1 and csr_matrix.export_matrix_to_file
  error_output = A_as_nb_1->export_matrix_to_file(output_filename,
      opendarts::linear_solvers::sparse_matrix_export_format::human_readable); // save the matrix to file

  // Compare the generated output to the reference output
  files_are_equal = opendarts::linear_solvers::testing::compare_files(output_filename,
      reference_filename); // check if equal to reference

  if (files_are_equal)
  {
    error_output = 0;
  }
  else
  {
    error_output = 1;
  }

  if (N_BLOCK_SIZE > 1)
  {
    delete A_as_nb_1;
  }

  return error_output;
}


template <uint8_t N_BLOCK_SIZE> int test_to_nb_1(std::string &output_filename, std::string &reference_filename)
{
  int error_output = 0;
  bool files_are_equal = true;
  opendarts::config::index_t n = 12;

  // Generate the matrix with non 1 block size
  opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A = new opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE>;
  opendarts::linear_solvers::testing::generate_tridiagonal_matrix(*A, n); // populate the matrix, in this 
                                                                          // case a tridiagonal matrix with 
                                                                          // the values -2, 1, 2 in the 
                                                                          // -2, 0, and 2 diagonals

  // Convert the matrix to block size 1 and store it in a pointer
  // NOTE: this function using pointer will only copy data if a real conversion
  //       is needed, i.e., A.n_block_size > 1, otherwise the pointer will point
  //       to A.
  opendarts::linear_solvers::csr_matrix<1> A_as_nb_1;
  
  A_as_nb_1.to_nb_1(A);

  // Save the matrix in human readable format
  // This implies a conversion to block size 1 so we test two functions in one:
  // csr_matrix.as_nb_1 and csr_matrix.export_matrix_to_file
  error_output = A_as_nb_1.export_matrix_to_file(output_filename,
      opendarts::linear_solvers::sparse_matrix_export_format::human_readable); // save the matrix to file

  // Compare the generated output to the reference output
  files_are_equal = opendarts::linear_solvers::testing::compare_files(output_filename,
      reference_filename); // check if equal to reference

  if (files_are_equal)
  {
    error_output = 0;
  }
  else
  {
    error_output = 1;
  }

  delete A;

  return error_output;
}
