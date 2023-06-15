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

// Tests the opendarts::linear_solvers::linsolv_superlu solver
template <uint8_t N_BLOCK_SIZE>
int test_linsolv_superlu(std::vector<opendarts::config::mat_float> &solution_reference, 
    opendarts::config::index_t n_rows, 
    opendarts::config::mat_float error_tol,
    bool use_iface);

// Generates a tridiagonal matrix, a right hand side, and solves the system with
// opendarts::linear_solvers::linsolv_superlu
template <uint8_t N_BLOCK_SIZE>
int get_tridiagonal_solution_linsolv_superlu(std::vector<opendarts::config::mat_float> &solution,
    double &time_setup,
    double &time_solve,
    opendarts::config::index_t n = 12,
    bool use_iface = false);

// Computes the RMS (norm N2) error and max (N\infty) error between two vectors.
void compute_errors(std::vector<opendarts::config::mat_float> &solution,
    std::vector<opendarts::config::mat_float> &reference,
    opendarts::config::mat_float &error_rms,
    opendarts::config::mat_float &error_max);

int main()
{
  /*
    test_06__SuperLU
    Tests the functionality of opendarts::linear_solvers::linsolv_superlu.
    Constructs a tridiagonal matrix and solves the linear system with a right 
    hand side.
  */

  int error_output = 0;
  
  // Reference solutions
  std::vector<opendarts::config::mat_float> solution_reference_nb_1{-0.071823204419889, -0.071823204419889,
      0.535911602209945, 0.535911602209945, 0.160220994475138, 0.160220994475138, 0.955801104972376, 0.955801104972376,
      0.182320441988950, 0.182320441988950, 1.364640883977901, 1.364640883977901};  // block size 1
  std::vector<opendarts::config::mat_float> solution_reference_nb_2{-0.071823204419889, -0.071823204419889,
      -0.071823204419889, -0.071823204419889, 0.535911602209945, 0.535911602209945, 0.535911602209945,
      0.535911602209945, 0.160220994475138, 0.160220994475138, 0.160220994475138, 0.160220994475138, 0.955801104972376,
      0.955801104972376, 0.955801104972376, 0.955801104972376, 0.182320441988950, 0.182320441988950, 0.182320441988950,
      0.182320441988950, 1.364640883977901, 1.364640883977901, 1.364640883977901, 1.364640883977901};  // block size 2
  
  // Using directly opendarts::linear_solvers::linsolv_superlu 
  
  // Solve settings
  opendarts::config::index_t n_rows = 12;  // the number of rows of the system to solve, not that it is block rows 
  opendarts::config::mat_float error_tol = 1e-12; // the tolerance to pass the test
  bool use_iface = false;  // do not use base class in the call to solve 
  
  // Generate SuperLU output
  error_output += test_linsolv_superlu<1>(solution_reference_nb_1, n_rows, error_tol, use_iface);
  error_output += test_linsolv_superlu<2>(solution_reference_nb_2, n_rows, error_tol, use_iface);
  
  // Using opendarts::linear_solvers::linsolv_iface to call solve 
  
  // Solve settings
  use_iface = true;  // use base class in the call to solve 
  
  // Generate SuperLU output
  error_output += test_linsolv_superlu<1>(solution_reference_nb_1, n_rows, error_tol, use_iface);
  error_output += test_linsolv_superlu<2>(solution_reference_nb_2, n_rows, error_tol, use_iface);

  return error_output;
}

template <uint8_t N_BLOCK_SIZE>
int test_linsolv_superlu(std::vector<opendarts::config::mat_float> &solution_reference, 
    opendarts::config::index_t n_rows, 
    opendarts::config::mat_float error_tol,
    bool use_iface)
{
  // Fixed parameter values for the test
  opendarts::config::mat_float error_max = 0.0;   // the maximum error computed
  opendarts::config::mat_float error_rms = 0.0;   // the computed root mean square error
  double time_setup, time_solve;                  // the times spent solving and setting up SuperLU solver

  // Auxiliary error check variables
  int error_output = 0;
  
  int n_block_size = solution_reference.size()/n_rows;
  
  if(solution_reference.size() % n_rows > 0)
    // Something went wrong when defining the solution vector and the size of the system
    return 1;

  // Compute the solution 
  std::vector<opendarts::config::mat_float> solution;
  error_output += get_tridiagonal_solution_linsolv_superlu<N_BLOCK_SIZE>(solution, time_setup, time_solve, n_rows, use_iface);

  // Check the errors
  compute_errors(solution, solution_reference, error_rms, error_max);

  // Show them for helping debugging if needed
  std::cout << "Block size " << n_block_size << ":" << std::endl;
  std::cout << "   Error rms :" << error_rms << std::endl;
  std::cout << "   Error max :" << error_max << std::endl;
  std::cout << "   Time setup:" << time_setup << "s" << std::endl;
  std::cout << "   Time solve:" << time_solve << "s" << std::endl;

  // If the errors are above the tolerance flag that
  if (error_max > error_tol)
    error_output += 1;
  if (error_rms > error_tol)
    error_output += 1;

  return error_output;
}

template <uint8_t N_BLOCK_SIZE>
int get_tridiagonal_solution_linsolv_superlu(std::vector<opendarts::config::mat_float> &solution,
    double &time_setup,
    double &time_solve,
    opendarts::config::index_t n,
    bool use_iface)
{
  // Auxiliary error check variables
  int error_output = 0;

  // Generate the matrix with block size 4
  opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> A;
  opendarts::linear_solvers::testing::generate_tridiagonal_matrix(A, n,
      opendarts::linear_solvers::testing::block_fill_option::diagonal_block); // populate the matrix, in this case a
                                                                              // tridiagonal matrix with the values -2,
                                                                              // 1, 2 in the -2, 0, and 2 diagonals
  
  opendarts::linear_solvers::csr_matrix_base *A_base = NULL;
  opendarts::linear_solvers::linsolv_iface *iface_solver;
  
  // Generate the solver

  // Initialise the solver
  opendarts::linear_solvers::linsolv_superlu<N_BLOCK_SIZE> superlu_solver;
  
  if(!use_iface)
  {
    error_output += superlu_solver.init(&A, 0, 0.0);
  }
  else 
  {
    A_base = &A;
    iface_solver = &superlu_solver;
    error_output += iface_solver->init(A_base, 0, 0.0);
  }

  // Initialise the timers
  opendarts::auxiliary::timer_node timer_setup, timer_setup_superLU;
  opendarts::auxiliary::timer_node timer_solve, timer_solve_superLU;

  timer_setup.node.emplace("SUPERLU", timer_setup_superLU);
  timer_solve.node.emplace("SUPERLU", timer_solve_superLU);
  
  if(!use_iface)
  {
    // Setup the solver
    superlu_solver.init_timer_nodes(&timer_setup, &timer_solve);
    error_output += superlu_solver.setup(&A);
  }
  else
  {
    // Setup the solver
    iface_solver->init_timer_nodes(&timer_setup, &timer_solve);
    error_output += iface_solver->setup(A_base);
  }

  // Solve the system

  // Setup the right hand side and initialize the solution
  std::vector<opendarts::config::mat_float> b(n * N_BLOCK_SIZE, 1.0);
  solution.assign(n * N_BLOCK_SIZE, 1.0);
  
  if(!use_iface)
  {
    error_output += superlu_solver.solve(b.data(), solution.data());
    
    // Get the time spent setting up and solving
    time_setup = superlu_solver.timer_setup->node["SUPERLU"].get_timer();
    time_solve = superlu_solver.timer_solve->node["SUPERLU"].get_timer();
  }
  else
  {
    error_output += iface_solver->solve(b.data(), solution.data());
    
    // Get the time spent setting up and solving
    time_setup = iface_solver->timer_setup->node["SUPERLU"].get_timer();
    time_solve = iface_solver->timer_solve->node["SUPERLU"].get_timer();
  }
  
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
