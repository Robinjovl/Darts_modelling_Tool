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
#include "test_common.hpp"

int test_hypre_A_ij_construction();
int test_hypre_BoomerAMG_solver();
void check_result(int res);
void generate_hypre_tridiagonal_matrix(opendarts::config::index_t n, HYPRE_IJMatrix &A_ij, bool is_SPD = false);

int main()
{
  /*
    hypre
    Tests hypre import and linking.
    Generates a tridiagonal matrix with block size 1 in csr_matrix format and
    convert it to hypre.
  */
  
  int error_output = 0;
  
  // Initialize Hypre, must be done only once
  HYPRE_Init();
  std::cout << "\nHypre initialized!" << std::endl;
  
  // Run tests
  error_output += test_hypre_A_ij_construction();  // test hypre matrix generation
  error_output += test_hypre_BoomerAMG_solver();  // test hypre Boomer AMG solver 
  
  // Finalize Hypre, must be done at the end
  HYPRE_Finalize();
  std::cout << "\nHypre Finalized!" << std::endl;

  return error_output;
}


int test_hypre_A_ij_construction()
{
  // Startup parameters
  bool files_are_equal = true;
  int error_output = 0;
  opendarts::config::index_t n = 12;  // size of matrix: n x n
  
  // Output file: Note that Hypre adds the process number, in this case there 
  // is only one process so the filename will be appended by .00000
  std::string output_filename("hypre_ij_matrix.txt");  // the output file to save the Hypre matrix to
  
  //  The reference ascii file
  std::string data_path_prefix = opendarts::config::get_cmake_openDARTS_source_dir() +
                                 std::string("/tests/cpp/data/tests/linear_solvers/"); // the path to the data folder
  std::string reference_base_filename("hypre_ij_matrix_ref.txt");        // the filename of the reference file
  std::string reference_filename = data_path_prefix +
                                   reference_base_filename; // get the full path to the reference file

  
  // Generate Hypre tridiagonal matrix 
  HYPRE_IJMatrix A_ij;
  generate_hypre_tridiagonal_matrix(n, A_ij);
  
  // Print matrix to file
  check_result(HYPRE_IJMatrixPrint(A_ij, output_filename.data()));
  
  // Compare the generated output to the reference output
  files_are_equal = opendarts::linear_solvers::testing::compare_files(\
      output_filename + std::string(".00000"),
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

int test_hypre_BoomerAMG_solver()
{
  // Startup parameters
  int error_output = 0;
  opendarts::config::index_t n = 120;  // size of matrix: n x n
  opendarts::config::index_t max_iters = 1000;  // maximum number of solver iterations
  opendarts::config::index_t num_functions = 1;  // solver number of functions
  opendarts::config::mat_float tolerance = 1e-12;  // solver tolerance
  opendarts::config::mat_float threshold = 0.9;  // solver threshold
  
  // Generate Hypre tridiagonal matrix 
  HYPRE_IJMatrix A_ij;
  generate_hypre_tridiagonal_matrix(n, A_ij, true);
  HYPRE_ParCSRMatrix A_par;
  check_result(HYPRE_IJMatrixGetObject(A_ij, (void **)&A_par));
  
  // Generate Hypre right hand side vector b_ij
  HYPRE_IJVector b_ij;
  HYPRE_ParVector b_par;
  
  std::vector<opendarts::config::mat_float> b(n, 1.0);
  const int print_level = 2;  // print level in Hypre
  opendarts::config::index_t ilower, iupper;
  ilower = 0;
  iupper = n - 1;
  opendarts::config::index_t n_rows = n;  // number of rows in vector
  std::vector<opendarts::config::index_t> rows(n);
  std::iota(rows.begin(), rows.end(), 0);
  
  check_result(HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, ilower, iupper, &b_ij));
	check_result(HYPRE_IJVectorSetPrintLevel(b_ij, print_level));
	check_result(HYPRE_IJVectorSetObjectType(b_ij, HYPRE_PARCSR));

  check_result(HYPRE_IJVectorInitialize(b_ij));
	check_result(HYPRE_IJVectorSetValues(b_ij, n_rows, rows.data(), b.data()));
	check_result(HYPRE_IJVectorAssemble(b_ij));
	check_result(HYPRE_IJVectorGetObject(b_ij, (void **)&b_par));
  
  // Generate Hypre solution vector x_ij
  HYPRE_IJVector x_ij;
  HYPRE_ParVector x_par;
  
  std::vector<opendarts::config::mat_float> x(n, 2.0);
  
  check_result(HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, ilower, iupper, &x_ij));
	check_result(HYPRE_IJVectorSetPrintLevel(x_ij, print_level));
	check_result(HYPRE_IJVectorSetObjectType(x_ij, HYPRE_PARCSR));

  check_result(HYPRE_IJVectorInitialize(x_ij));
	check_result(HYPRE_IJVectorSetValues(x_ij, n_rows, rows.data(), x.data()));
	check_result(HYPRE_IJVectorAssemble(x_ij));
	check_result(HYPRE_IJVectorGetObject(x_ij, (void **)&x_par));
	
  // Create the Boomer AMG solver
  HYPRE_Solver solver;
	check_result(HYPRE_BoomerAMGCreate(&solver));
	check_result(HYPRE_BoomerAMGSetPrintLevel(solver, print_level));
	check_result(HYPRE_BoomerAMGSetLogging(solver, print_level));

	check_result(HYPRE_BoomerAMGSetNumFunctions(solver, num_functions));
	check_result(HYPRE_BoomerAMGSetStrongThreshold(solver, threshold));
  
	check_result(HYPRE_BoomerAMGSetMaxIter(solver, max_iters));
	check_result(HYPRE_BoomerAMGSetTol(solver, tolerance));
  
  // Solve the system
  check_result(HYPRE_BoomerAMGSetup(solver, A_par, b_par, x_par));
  check_result(HYPRE_BoomerAMGSolve(solver, A_par, b_par, x_par));
	
  // Check the error in the solution by checking the right hand side b to 
  // the one obtained by multiplying the matrix A by the solution x
  
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
  check_result(HYPRE_ParCSRMatrixMatvec(1.0, A_par, x_par, 0.0, b_from_x_par));
  
  // Compute the error on the right hand side
  opendarts::config::mat_float error;
  check_result(HYPRE_ParVectorAxpy(-1.0, b_par, b_from_x_par));
  check_result(HYPRE_ParVectorInnerProd(b_from_x_par, b_from_x_par, &error));
  error = std::sqrt(error);
  
  std::cout << "\nError: " << error << std::endl;
  
  if(error >= tolerance*10)
    error_output = 1;
    
  return error_output;
}

void generate_hypre_tridiagonal_matrix(opendarts::config::index_t n, HYPRE_IJMatrix &A_ij, bool is_SPD)
{
  const int print_level = 2;  // print level in Hypre
  
  // Generate the csr matrix with block size 1 in csr_matrix format
  opendarts::linear_solvers::csr_matrix<1> A;
  opendarts::linear_solvers::testing::generate_tridiagonal_matrix(A, n,
      opendarts::linear_solvers::testing::block_fill_option::constant_filled_block,
      is_SPD); // populate the matrix, in this case a 
              // tridiagonal matrix with the values 
              // -2, 1, 2 in the -2, 0, and 2 diagonals
    
  // Convert csr_matrix A to Hypre ij_matrix
  opendarts::config::index_t ilower, iupper;
  ilower = 0;
  iupper = A.n_rows - 1;
  
  std::vector<opendarts::config::index_t> rows(A.n_rows), n_cols(A.n_rows);
  std::iota(rows.begin(), rows.end(), 0);
  
  for (opendarts::config::index_t row_idx = 0; row_idx < A.n_rows; row_idx++)
		n_cols[row_idx] = A.rows_ptr[row_idx + 1] - A.rows_ptr[row_idx];

  check_result(HYPRE_IJMatrixCreate(hypre_MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A_ij));
	check_result(HYPRE_IJMatrixSetPrintLevel(A_ij, print_level));
	check_result(HYPRE_IJMatrixSetObjectType(A_ij, HYPRE_PARCSR));
  check_result(HYPRE_IJMatrixInitialize(A_ij));
	check_result(HYPRE_IJMatrixSetValues(A_ij, A.n_rows, n_cols.data(), rows.data(), A.get_cols_ind(), A.get_values()));
	check_result(HYPRE_IJMatrixAssemble(A_ij));
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
