
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

#include "openDARTS/config/version.hpp"
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#include "openDARTS/linear_solvers/data_types.hpp"
#include "openDARTS/linear_solvers/linsolv_bos_cpr.hpp"
#include "openDARTS/linear_solvers/linsolv_superlu.hpp"

#include "test_common.hpp"

template <uint8_t N_BLOCK_SIZE>
int test_member_functions();

int main()
{
  /*
    test_10__linsolv_bos_cpr
    Tests linsolv_bos_cpr shell implementation.
    These are basic checks since linsolv_bos_cpr is simply a shell implementation 
    for backwards compatibility.
  */
  
  int error_output = 0;
  
  // Check if each member function is available
  // Shell implementation, but available 
  error_output += test_member_functions<2>();  // test block size 2
  error_output += test_member_functions<3>();  // test block size 3
  error_output += test_member_functions<4>();  // test block size 4
  error_output += test_member_functions<5>();  // test block size 5
  error_output += test_member_functions<6>();  // test block size 6
  error_output += test_member_functions<7>();  // test block size 7
  error_output += test_member_functions<8>();  // test block size 8
  error_output += test_member_functions<9>();  // test block size 9
  error_output += test_member_functions<10>();  // test block size 10
  error_output += test_member_functions<11>();  // test block size 11
  error_output += test_member_functions<12>();  // test block size 12
  error_output += test_member_functions<13>();  // test block size 13
  
  return error_output;  // The test just fails if the functions are not available
                        // or some return unexpected results
}

// Function to test member functions
template <uint8_t N_BLOCK_SIZE>
int test_member_functions()
{  
  int error_output = 0;
  
  // Check if bos_cpr solver with block size N_BLOCK_SIZE is available
  opendarts::linear_solvers::linsolv_bos_cpr<N_BLOCK_SIZE> bos_cpr_solver;
  
  // Check if each member function is available
  // Shell implementation, but available 
  
  // Setup input parameters
  opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_in = new opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE>;
  opendarts::linear_solvers::linsolv_superlu<N_BLOCK_SIZE> *super_lu_solver =
      new opendarts::linear_solvers::linsolv_superlu<N_BLOCK_SIZE>;
  opendarts::linear_solvers::linsolv_iface *prec_in = super_lu_solver;
  opendarts::config::index_t max_iters = 10;  // just a random integer number, not relevant
  opendarts::config::mat_float tolerance = 0.001;  // just a random floating point number, not relevant
  opendarts::config::index_t system_size = 10;  // the size of the system to solve, not relevant
  std::vector<opendarts::config::mat_float> B(system_size, 0.0);  // just a choice to initialize with zeros
  std::vector<opendarts::config::mat_float> X(system_size, 0.0);  // just a choice to initialize with zeros
  
  if(bos_cpr_solver.solve((opendarts::linear_solvers::csr_matrix_base *) A_in, B.data(), X.data()) != 1)
    error_output += 1;
  
  if(bos_cpr_solver.setup((opendarts::linear_solvers::csr_matrix_base *) A_in) != 1)
    error_output += 1;
  
  if(bos_cpr_solver.set_prec(prec_in) != 1 )
    error_output += 1;

  if(bos_cpr_solver.init(A_in, max_iters, tolerance) != 1)
    error_output += 1;
  
  if(bos_cpr_solver.setup(A_in) != 1)
    error_output += 1;

  if(bos_cpr_solver.solve(B.data(), X.data()) != 1)
    error_output += 1;

  if(bos_cpr_solver.get_n_iters() != 0)
    error_output += 1;

  if(bos_cpr_solver.get_residual() != 0.0)
    error_output += 1;
  
  // Clean up memory
  delete A_in;
  delete super_lu_solver;
  
  // Return
  return error_output;
}
