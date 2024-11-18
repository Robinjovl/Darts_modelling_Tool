#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <stdio.h>
#include <string>
#include <vector>

#include "openDARTS/config/data_types.hpp"
#include "openDARTS/config/version.hpp"
#include "openDARTS/linear_solvers/linsolv_superlu.hpp"
#include "test_common.hpp"

// Test matrix transpose vector multiplication 
template <uint8_t N_BLOCK_SIZE>
int test_matrix_transpose_vector_multiplication(
    opendarts::config::mat_float &error_rms, 
    opendarts::config::mat_float &error_max);

// Computes the RMS (norm N2) error and max (N\infty) error between two vectors.
void compute_errors(std::vector<opendarts::config::mat_float> &solution,
    std::vector<opendarts::config::mat_float> &reference,
    opendarts::config::mat_float &error_rms,
    opendarts::config::mat_float &error_max);

int main()
{
  /*
    test_13__csr_matrix_mat_t_vec
    Tests the functionality of opendarts::linear_solvers::csr_matrix::matrix_vector_product_t.
    Constructs a tridiagonal matrix, multiplies it by a right hand size, and checks if the result 
    of the multiplication is as expected.
  */

  int error_output = 0;
  int A_t_vec_mult_output = 0;
  
  // Error settings 
  opendarts::config::mat_float error_tol = 1e-12; // the tolerance to pass the test
  opendarts::config::mat_float error_rms = 1.0; // the computed rms error
  opendarts::config::mat_float error_max = 1.0; // the computed max error
  
  // Block size 1 should work fine
  A_t_vec_mult_output = test_matrix_transpose_vector_multiplication<1>(error_rms, error_max);
        
  // Show the errors for helping debugging if needed
  std::cout << "Mat transpose vector multiplication:" << std::endl;
  std::cout << "   Error output :" << A_t_vec_mult_output << std::endl;
  std::cout << "   Error rms    :" << error_rms << std::endl;
  std::cout << "   Error max    :" << error_max << std::endl;

  // If the errors are above the tolerance flag that
  if (error_max > error_tol)
    error_output += 1;
  if (error_rms > error_tol)
    error_output += 1;
    
  // Block size N_BLOCK_SIZE > 1 should return -1
  A_t_vec_mult_output = test_matrix_transpose_vector_multiplication<4>(error_rms, error_max);
        
  // Show the errors for helping debugging if needed
  std::cout << "Mat transpose vector multiplication:" << std::endl;
  std::cout << "   Error output :" << A_t_vec_mult_output << std::endl;
  std::cout << "   Error rms    :" << error_rms << std::endl;
  std::cout << "   Error max    :" << error_max << std::endl;

  // If the errors are above the tolerance flag that
  if (A_t_vec_mult_output != -1)
    error_output += 1;
      
  return error_output;
}

template <uint8_t N_BLOCK_SIZE>
int test_matrix_transpose_vector_multiplication(
    opendarts::config::mat_float &error_rms, 
    opendarts::config::mat_float &error_max)
{
  int error_output = 0;
  
  // Required parameters (hard coded)
  opendarts::config::index_t n_rows = 12;  // the number of rows of the system to solve, not that it is block rows 
  
  // v vector that results in A v = b = [1, 1, ..., 1, 1].
  // Note that we take A to be the tridiagonal matrix, compute its transpose, and then 
  // compute the transpose product of the transpose, which is the same as computing the 
  // product. This is done because the results already exist and it is easier to compare with. 
  // NOTE: This matrix vector multiplication works only for N_BLOCK_SIZE = 1.
  //       The templating is done to check if opendarts::linear_solvers::csr_matrix::matrix_vector_product_t
  //       returns -1 (Error).
  std::vector<opendarts::config::mat_float> v{-0.071823204419889, -0.071823204419889,
      0.535911602209945, 0.535911602209945, 0.160220994475138, 0.160220994475138, 0.955801104972376, 0.955801104972376,
      0.182320441988950, 0.182320441988950, 1.364640883977901, 1.364640883977901};  // block size 1
  
  // Generate the matrix with block size 1
  opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> A;
  opendarts::linear_solvers::testing::generate_tridiagonal_matrix(A, n_rows,
      opendarts::linear_solvers::testing::block_fill_option::diagonal_block); // populate the matrix, in this case a
                                                                              // tridiagonal matrix with the values -2,
                                                                              // 1, 2 in the -2, 0, and 2 diagonals
  // Compute the transpose 
  opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> A_t;
  A.transpose(A_t);
  
  // Initialize the result reference and the result where to store the output
  std::vector<opendarts::config::mat_float> r_reference(n_rows, 1.0);
  std::vector<opendarts::config::mat_float> r(n_rows, 0.0);
  
  // Compute the matrix transpose vector product and get the result
  error_output = A_t.matrix_vector_product_t(v.data(), r.data());
  
  // Compute the errors 
  compute_errors(r, r_reference, error_rms, error_max);
  
  return error_output;
}

void compute_errors(std::vector<opendarts::config::mat_float> &solution,
    std::vector<opendarts::config::mat_float> &reference,
    opendarts::config::mat_float &error_rms,
    opendarts::config::mat_float &error_max)
{
  opendarts::config::mat_float error_temp = 0.0; // the computed error for a value of the solution vector

  // Reset the errors;
  error_rms = 0.0;
  error_max = 0.0;

  // Compute the errors in the solution
  for (size_t row_idx = 0; row_idx < solution.size(); row_idx++)
  {
    error_temp = std::abs(solution[row_idx] - reference[row_idx]);
    error_rms += std::pow(error_temp, 2);
    if (error_temp > error_max)
      error_max = error_temp;
  }
  error_rms = std::sqrt(error_rms);
}
