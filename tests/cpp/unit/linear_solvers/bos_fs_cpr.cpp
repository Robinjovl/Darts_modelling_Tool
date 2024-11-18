
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

#include "openDARTS/config/version.hpp"
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#include "openDARTS/linear_solvers/data_types.hpp"
#include "openDARTS/linear_solvers/linsolv_bos_fs_cpr.hpp"
#include "openDARTS/linear_solvers/linsolv_superlu.hpp"

#include "test_common.hpp"

template <uint8_t N_BLOCK_SIZE>
int test_member_functions();

int main()
{
  /*
    test_12__linsolv_bos_fs_cpr
    Tests linsolv_bos_fs_cpr shell implementation.
    These are basic checks since linsolv_bos_cpr is simply a shell implementation 
    for backwards compatibility.
  */
  
  int error_output = 0;
  
  // Check if each member function is available
  // Shell implementation, but available 
  error_output += test_member_functions<4>();  // test block size 4
  
  return error_output;  // The test just fails if the functions are not available
                        // or some return unexpected results
}

// Function to test member functions
template <uint8_t N_BLOCK_SIZE>
int test_member_functions()
{  
  int error_output = 0;
  
  // Check if bos_fs_cpr solver with block size N_BLOCK_SIZE is available
  // The initialization parameters are irrelevant since it is shell 
  // implementation only.
  opendarts::linear_solvers::linsolv_bos_fs_cpr<N_BLOCK_SIZE> bos_fs_cpr_solver(1, 2, 3, 4);  
  
  // Check if each member function is available
  // Shell implementation, but available 
  
  // Setup input parameters
  opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_in = new opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE>;
  opendarts::linear_solvers::csr_matrix<1> *D_in = new opendarts::linear_solvers::csr_matrix<1>;
  opendarts::linear_solvers::linsolv_superlu<N_BLOCK_SIZE> *super_lu_solver =
      new opendarts::linear_solvers::linsolv_superlu<N_BLOCK_SIZE>;
  opendarts::linear_solvers::linsolv_iface *prec_in = super_lu_solver;
  opendarts::config::index_t max_iters = 10;  // just a random integer number, not relevant
  opendarts::config::mat_float tolerance = 0.001;  // just a random floating point number, not relevant
  opendarts::config::index_t system_size = 10;  // the size of the system to solve, not relevant
  std::vector<opendarts::config::mat_float> B(system_size, 0.0);  // just a choice to initialize with zeros
  std::vector<opendarts::config::mat_float> X(system_size, 0.0);  // just a choice to initialize with zeros
  
  if(bos_fs_cpr_solver.solve((opendarts::linear_solvers::csr_matrix_base *) A_in, B.data(), X.data()) != 1)
    error_output += 1;
  
  if(bos_fs_cpr_solver.setup((opendarts::linear_solvers::csr_matrix_base *) A_in) != 1)
    error_output += 1;
  
  if(bos_fs_cpr_solver.set_prec(prec_in) != 1 )
    error_output += 1;
    
  if(bos_fs_cpr_solver.set_prec(prec_in, prec_in) != 1 )
    error_output += 1;
    
  if(bos_fs_cpr_solver.set_prec(prec_in, prec_in, prec_in) != 1 )
    error_output += 1;

  if(bos_fs_cpr_solver.init(A_in, max_iters, tolerance) != 1)
    error_output += 1;
  
  if(bos_fs_cpr_solver.setup(A_in) != 1)
    error_output += 1;

  if(bos_fs_cpr_solver.solve(B.data(), X.data()) != 1)
    error_output += 1;

  if(bos_fs_cpr_solver.get_n_iters() != 0)
    error_output += 1;

  if(bos_fs_cpr_solver.get_residual() != 0.0)
    error_output += 1;
    
  bos_fs_cpr_solver.set_block_sizes(1, 2, 3);  // does not return anything, just check if available
  
  bos_fs_cpr_solver.set_prec_type(opendarts::linear_solvers::Preconditioner::FS_UP);  // does not return anything,
                                                                                      // just check if available
  
  bos_fs_cpr_solver.do_update_uu();  // does not return anything, just check if available
  
  if(bos_fs_cpr_solver.set_diag_in_order(D_in) != 1)
    error_output += 1;
    
  // Clean up memory
  delete A_in;
  delete D_in;
  delete super_lu_solver;
  
  // Return
  return error_output;
}
