#include <string>
#include <vector>
#include <numeric>
#include <cmath>

#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv.h"
#include "_hypre_parcsr_mv.h"

#include "openDARTS/config/data_types.hpp"
#include "openDARTS/config/version.hpp"
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#include "openDARTS/linear_solvers/linsolv_hypre_ilu.hpp"
#include "test_common.hpp"

int test_linsolv_hypre_ilu_solver();
void check_result(int res);

int main()
{
  /*
    hypre
    Tests linsolv_hypre_ilu.
    Generates a tridiagonal matrix with block size 1 in csr_matrix format and solves
    the system using ILU.
  */
  
  int error_output = 0;
  
  // Initialize Hypre, must be done only once
  HYPRE_Init();
  std::cout << "\nHypre initialized!" << std::endl;
  
  // Run tests
  error_output += test_linsolv_hypre_ilu_solver();  // test hypre ILU solver 
  
  // Finalize Hypre, must be done at the end
  HYPRE_Finalize();
  std::cout << "\nHypre Finalized!" << std::endl;
  
  return error_output;
}

int test_linsolv_hypre_ilu_solver()
{
  // Startup parameters
  int error_output = 0;
  opendarts::config::index_t n = 120;  // size of matrix: n x n
  opendarts::config::index_t max_iters = 1000;  // maximum number of solver iterations
  opendarts::config::mat_float tolerance = 1e-12;  // solver tolerance
  bool is_SPD = true;  // generate SPD system matrix
  
  // Generate the csr matrix with block size 1 in csr_matrix format
  opendarts::linear_solvers::csr_matrix<1> A; 
  opendarts::linear_solvers::testing::generate_tridiagonal_matrix(A, n,
      opendarts::linear_solvers::testing::block_fill_option::constant_filled_block,
      is_SPD); // populate the matrix, in this case a 
              // tridiagonal matrix with the values 
              // -2, 1, 2 in the -2, 0, and 2 diagonals
  
  
  // Generate right hand side vector b
  std::vector<opendarts::config::mat_float> b(n, 1.0);
  
  // Initialize solution vector to collec the solution
  std::vector<opendarts::config::mat_float> x(n, 2.0);
  
  // Setup linsolv_hypre_ilu solver
  opendarts::linear_solvers::linsolv_hypre_ilu<1> my_hypre_ilu_solver;
  error_output += my_hypre_ilu_solver.init(&A, max_iters, tolerance);
  error_output += my_hypre_ilu_solver.setup(&A);
  error_output += my_hypre_ilu_solver.solve(b.data(), x.data());
  
  // Check the error in the solution by checking the right hand side b to 
  // the one obtained by multiplying the matrix A by the solution x
  
  const int print_level = 2;  // print level in Hypre
  opendarts::config::index_t n_rows = n;  // number of rows in vector must 
                                          // be the same as number of columns 
                                          // of system matrix
  
  opendarts::config::index_t ilower, iupper;
  ilower = 0;
  iupper = n_rows - 1;
  
  std::vector<opendarts::config::index_t> rows(n_rows);
  std::iota(rows.begin(), rows.end(), 0);
  
  // Initialize the computed right hand side
  HYPRE_IJVector b_from_x_ij;
  HYPRE_ParVector b_from_x_par;
  check_result(HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, ilower, iupper, &b_from_x_ij));
	check_result(HYPRE_IJVectorSetPrintLevel(b_from_x_ij, print_level));
	check_result(HYPRE_IJVectorSetObjectType(b_from_x_ij, HYPRE_PARCSR));
  
  check_result(HYPRE_IJVectorInitialize(b_from_x_ij));
	check_result(HYPRE_IJVectorSetValues(b_from_x_ij, n_rows, rows.data(), x.data()));
	check_result(HYPRE_IJVectorAssemble(b_from_x_ij));
	check_result(HYPRE_IJVectorGetObject(b_from_x_ij, (void **)&b_from_x_par));
  
  // Compute the right hand side
  check_result(HYPRE_ParCSRMatrixMatvec(1.0, my_hypre_ilu_solver.A_parcsr, my_hypre_ilu_solver.x_par, 0.0, b_from_x_par));
  
  // Compute the error on the right hand side
  opendarts::config::mat_float error;
  check_result(HYPRE_ParVectorAxpy(-1.0, my_hypre_ilu_solver.b_par, b_from_x_par));
  check_result(HYPRE_ParVectorInnerProd(b_from_x_par, b_from_x_par, &error));
  error = std::sqrt(error);
  
  std::cout << "\nError: " << error << std::endl;
  
  if(error >= tolerance*10)
    error_output += 1;
  
  return error_output;
}


void check_result(int res)
{
  char err_msg_char[256];
  if (res)
  {
  	HYPRE_DescribeError(res, err_msg_char);
    std::string err_msg(err_msg_char);
  	std::cout << "\n" << err_msg << std::endl;
    exit(-1);
  }
};
