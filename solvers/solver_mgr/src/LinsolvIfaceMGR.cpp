//*************************************************************************
//    Copyright (c) 2025
//    MGR Linear Solver - Open-Darts Integration
//
//*************************************************************************

#include "openDARTS/linear_solvers/LinsolvIfaceMGR.hpp"
#include "mgr-linear-solver/strategies/CompositionalFlowStrategy.hpp"
#include <iostream>

namespace opendarts
{
  namespace linear_solvers
  {
    using opendarts::config::index_t;
    using opendarts::config::mat_float;

    LinsolvIfaceMGR::LinsolvIfaceMGR()
      : linsolv_iface()
      , mgr_solver_(new mgr::LinearSolver())
      , initialized_(false)
      , setup_(false)
    {
      std::cout << "[LinsolvIfaceMGR] MGR Linear Solver created" << std::endl;
    }

    LinsolvIfaceMGR::~LinsolvIfaceMGR()
    {
      std::cout << "[LinsolvIfaceMGR] MGR Linear Solver destroyed" << std::endl;
    }

    int LinsolvIfaceMGR::set_prec(linsolv_iface *prec_input)
    {
      std::cout << "[LinsolvIfaceMGR::set_prec] WARNING: MGR is its own preconditioner." << std::endl;
      std::cout << "  External preconditioner ignored." << std::endl;
      (void)prec_input;
      return 0;  // Return success (although ignored)
    }

    int LinsolvIfaceMGR::set_p_system_prec(linsolv_iface *prec_input)
    {
      std::cout << "[LinsolvIfaceMGR::set_p_system_prec] WARNING: MGR handles pressure system internally." << std::endl;
      std::cout << "  External pressure solver ignored." << std::endl;
      (void)prec_input;
      return 0;  // Return success (although ignored)
    }

    int LinsolvIfaceMGR::init(csr_matrix_base *A,
                               index_t max_iters,
                               mat_float tolerance)
    {
      std::cout << "\n[LinsolvIfaceMGR::init] Initializing MGR solver..." << std::endl;
      std::cout << "  max_iters: " << max_iters << std::endl;
      std::cout << "  tolerance: " << tolerance << std::endl;

      if (!A) {
        std::cerr << "[LinsolvIfaceMGR::init] ERROR: Matrix pointer is null!" << std::endl;
        return -1;
      }

      std::cout << "  Matrix info:" << std::endl;
      std::cout << "    n_rows: " << A->n_rows << std::endl;
      std::cout << "    n_cols: " << A->n_cols << std::endl;
      std::cout << "    n_row_size (block_size): " << A->n_row_size << std::endl;
      std::cout << "    n_non_zeros: " << A->n_non_zeros << std::endl;

      // Set matrix from csr_matrix_base
      if (!mgr_solver_->setMatrixFromCSR(
             A->n_rows,
             A->n_cols,
             A->n_row_size,
             A->n_non_zeros,
             A->get_rows_ptr(),
             A->get_cols_ind(),
             A->get_values(),
             A->get_diag_ind())) {
        std::cerr << "[LinsolvIfaceMGR::init] ERROR: Failed to set matrix!" << std::endl;
        return -2;
      }

      // Set default MGR strategy for compositional flow
      std::cout << "[LinsolvIfaceMGR::init] Setting up MGR strategy..." << std::endl;
      auto strategy = std::make_unique<mgr::strategies::CompositionalFlowStrategy>(
                       A->n_row_size,
                       A->n_rows * A->n_row_size,
                       A->n_rows);
      mgr_solver_->setStrategy(std::move(strategy));

      // Set solver parameters
      mgr::SolverParameters params;
      params.maxIter = max_iters;
      params.tolerance = tolerance;


      params.kdim = 50;
      params.useMGR = true;
      params.logLevel = 1;
      mgr_solver_->setParameters(params);

      initialized_ = true;
      setup_ = false;

      std::cout << "[LinsolvIfaceMGR::init] Initialization complete!" << std::endl;
      return 0;
    }

    int LinsolvIfaceMGR::setup(csr_matrix_base *A_input)
    {
      std::cout << "\n[LinsolvIfaceMGR::setup] Setting up MGR solver..." << std::endl;

      if (!initialized_) {
        std::cerr << "[LinsolvIfaceMGR::setup] ERROR: init() not called!" << std::endl;
        return -1;
      }

      if (A_input != nullptr) {
        std::cout << "  Matrix provided (structure assumed unchanged)" << std::endl;
        // TODO: Could update matrix values here if needed
      } else {
        std::cout << "  No matrix provided (reusing existing)" << std::endl;
      }

      // Setup MGR solver (using parameters from init)
      const auto& params = mgr_solver_->getParameters();
      int result = mgr_solver_->setup(params.maxIter, params.tolerance);

      if (result == 0) {
        setup_ = true;
        std::cout << "[LinsolvIfaceMGR::setup] Setup complete!" << std::endl;
      } else {
        std::cerr << "[LinsolvIfaceMGR::setup] ERROR: Setup failed!" << std::endl;
      }

      return result;
    }

    int LinsolvIfaceMGR::solve(mat_float *B, mat_float *X)
    {
      std::cout << "\n[LinsolvIfaceMGR::solve] Solving linear system..." << std::endl;

      if (!setup_) {
        std::cerr << "[LinsolvIfaceMGR::solve] ERROR: setup() not called!" << std::endl;
        return -1;
      }

      if (!B || !X) {
        std::cerr << "[LinsolvIfaceMGR::solve] ERROR: Null pointer!" << std::endl;
        return -1;
      }

      // Solve using MGR
      int n_iters = mgr_solver_->solve(B, X);

      std::cout << "[LinsolvIfaceMGR::solve] Solve complete!" << std::endl;
      std::cout << "  Iterations: " << n_iters << std::endl;
      std::cout << "  Final residual: " << mgr_solver_->get_residual() << std::endl;

      // Return 0 for success (open-darts convention)
      // n_iters is stored internally and can be retrieved via get_n_iters()
      return 0;
    }

    int LinsolvIfaceMGR::get_n_iters()
    {
      return mgr_solver_->get_n_iters();
    }

    mat_float LinsolvIfaceMGR::get_residual()
    {
      return mgr_solver_->get_residual();
    }

  } // namespace linear_solvers
} // namespace opendarts
