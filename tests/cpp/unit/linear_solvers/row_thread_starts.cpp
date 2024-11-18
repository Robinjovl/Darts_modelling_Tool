#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>

#include "openDARTS/config/version.hpp"
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#include "openDARTS/linear_solvers/data_types.hpp"

#include "test_common.hpp"

int main()
{
  /*
    test_07__row_thread_starts
    Tests csr_matrix.get_row_thread_starts()
    Generates a tridiagonal matrix with block size 3 and cheks if
    this->row_thread_starts is as expected.
  */

  int error_output = 0;
  opendarts::config::index_t n = 12;
  opendarts::config::index_t *row_thread_starts;

  // Generate the matrix
  opendarts::linear_solvers::csr_matrix<1> A;
  opendarts::linear_solvers::testing::generate_tridiagonal_matrix(A, n); // populate the matrix, in this case a 
                                                                         // tridiagonal matrix with the values 
                                                                         // -2, 1, 2 in the -2, 0, and 2 diagonals
  
  // Get row_thread_starts and check if it is equal to [0, n]
  row_thread_starts = A.get_row_thread_starts();
  if (row_thread_starts[0] != 0) error_output += 1;
  if (row_thread_starts[1] != n) error_output += 1;
  
  return error_output;
}
