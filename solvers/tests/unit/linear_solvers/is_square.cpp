#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>

#include "openDARTS/config/version.hpp"
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#include "openDARTS/linear_solvers/data_types.hpp"

#include "test_common.hpp"

int test_square_matrix();
int test_rectangular_matrix();
int test_init_matrix_is_not_square();

int main()
{
  /*
    is_square
    Tests csr_matrix.is_square variable properly set. Critical in simulations.
  */
  int error_output = 0;
  
  // Test a square matrix
  error_output += test_square_matrix();
  
  // Test a rectangular matrix
  error_output += test_rectangular_matrix();
  
  // Test construction initializes as non square matrix
  error_output += test_init_matrix_is_not_square();

  return error_output;
}


int test_square_matrix()
{
  int error_output = 0;
  
  // Declare matrix
  opendarts::linear_solvers::csr_matrix<1> A;
  
  // Init matrix
  opendarts::config::index_t n_rows = 10;
  opendarts::config::index_t n_cols = 10;
  opendarts::config::index_t n_non_zeros = 5;
  A.init(n_rows, n_cols, n_non_zeros);
  
  if (A.is_square != 1)
  {
    std::cout << "Square matrix is square failed!" << std::endl;
    error_output = 1;
  }
  
  return error_output;
}

int test_rectangular_matrix()
{
  int error_output = 0;
  
  // Declare matrix
  opendarts::linear_solvers::csr_matrix<1> A;
  
  // Init matrix
  opendarts::config::index_t n_rows = 10;
  opendarts::config::index_t n_cols = 16;
  opendarts::config::index_t n_non_zeros = 5;
  A.init(n_rows, n_cols, n_non_zeros);
  
  if (A.is_square != 0)
  {
    std::cout << "Rectangular matrix is not square failed!" << std::endl;
    error_output = 1;
  }
  
  return error_output;
}

int test_init_matrix_is_not_square()
{
  int error_output = 0;
  
  // Declare matrix
  opendarts::linear_solvers::csr_matrix<1> A;
  
  if (A.is_square != 0)
  {
    std::cout << "Init matrix is not square failed!" << std::endl;
    error_output = 1;
  }
  
  return error_output;
}
