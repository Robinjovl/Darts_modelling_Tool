#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>

#include "openDARTS/config/version.hpp"
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#include "openDARTS/linear_solvers/data_types.hpp"

#include "test_common.hpp"

int test_csr_format_string_input();
int test_csr_format_char_input();

int main()
{
  /*
    test_02__write_matrix_to_file_csr
    Tests csr_matrix.write_matrix_to_file in csr format, therefore tests also
    csr_matrix.write_matrix_to_file_csr.
    Generates a tridiagonal matrix with block size 3 and saves it to file. The
    The output is compared to a reference result.
  */
  int error_output = 0;
  
  error_output += test_csr_format_string_input();
  error_output += test_csr_format_char_input();

  return error_output;
}


int test_csr_format_string_input()
{
  bool files_are_equal = true;
  int error_output = 0;
  opendarts::config::index_t n = 12;

  // The output file to save the matrix to
  std::string output_filename("test_02__csr_matrix.txt");

  //  The reference ascii file
  std::string data_path_prefix = opendarts::config::get_cmake_openDARTS_source_dir() +
                                 std::string("/tests/cpp/data/tests/linear_solvers/"); // the path to the data folder
  std::string reference_base_filename("test_02__csr_matrix_ref.txt");        // the filename of the reference file
  std::string reference_filename = data_path_prefix +
                                   reference_base_filename; // get the full path to the reference file

  // Generate the matrix and save it to file
  opendarts::linear_solvers::csr_matrix<3> A;
  opendarts::linear_solvers::testing::generate_tridiagonal_matrix(A,
      n); // populate the matrix, in this case a tridiagonal matrix with the values -2, 1, 2 in the -2, 0, and 2
          // diagonals

  error_output = A.export_matrix_to_file(output_filename,
      opendarts::linear_solvers::sparse_matrix_export_format::csr); // save the matrix to file

  if (error_output != 0)
  {
    // If there was an error writting the matrix to file, no need to compare the files,
    // they are necessarily different
    return error_output;
  }

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


int test_csr_format_char_input()
{
  bool files_are_equal = true;
  int error_output = 0;
  opendarts::config::index_t n = 12;

  // The output file to save the matrix to
  std::string output_filename("test_02__csr_matrix.txt");
  const char* output_filename_char = output_filename.c_str();

  //  The reference ascii file
  std::string data_path_prefix = opendarts::config::get_cmake_openDARTS_source_dir() +
                                 std::string("/tests/cpp/data/tests/linear_solvers/"); // the path to the data folder
  std::string reference_base_filename("test_02__csr_matrix_ref.txt");        // the filename of the reference file
  std::string reference_filename = data_path_prefix +
                                   reference_base_filename; // get the full path to the reference file

  // Generate the matrix and save it to file
  opendarts::linear_solvers::csr_matrix<3> A;
  opendarts::linear_solvers::testing::generate_tridiagonal_matrix(A,
      n); // populate the matrix, in this case a tridiagonal matrix with the values -2, 1, 2 in the -2, 0, and 2
          // diagonals

  error_output = A.write_matrix_to_file(output_filename_char); // save the matrix to file

  if (error_output != 0)
  {
    // If there was an error writting the matrix to file, no need to compare the files,
    // they are necessarily different
    return error_output;
  }

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
