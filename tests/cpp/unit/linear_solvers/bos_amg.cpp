
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

#include "openDARTS/config/version.hpp"
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#include "openDARTS/linear_solvers/data_types.hpp"
#include "openDARTS/linear_solvers/linsolv_bos_amg.hpp"
#include "openDARTS/linear_solvers/linsolv_superlu.hpp"

#include "test_common.hpp"

template <uint8_t N_BLOCK_SIZE>
int test_member_functions();

int main()
{
  /*
    test_08__linsolv_bos_amg
    Tests linsolv_bos_amg shell implementation.
    These are basic checks since linsolv_bos_amg is simply a shell implementation 
    for backwards compatibility.
  */
  
  int error_output = 0;
  
  // Check if each member function is available
  // Shell implementation, but available 
  error_output += test_member_functions<1>();  // test block size 1
  error_output += test_member_functions<3>();  // test block size 3
  
  return error_output;  // The test just fails if the functions are not available
                        // or some return unexpected results
}

// Function to test member functions
template <uint8_t N_BLOCK_SIZE>
int test_member_functions()
{  
  int error_output = 0;
  
  // Check if bos_amg solver with block size N_BLOCK_SIZE is available
  opendarts::linear_solvers::linsolv_bos_amg<N_BLOCK_SIZE> bos_amg_solver;
  
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
  
  bos_amg_solver.set_prec(prec_in);

  bos_amg_solver.init(A_in, max_iters, tolerance);
  
  bos_amg_solver.setup(A_in);

  bos_amg_solver.solve(B.data(), X.data());

  if(bos_amg_solver.get_n_iters() != 0)
    error_output += 1;

  if(bos_amg_solver.get_residual() != 1000.0)
    error_output += 1;
  
  // Clean up memory
  delete A_in;
  delete super_lu_solver;
  
  // Return
  return error_output;
}
