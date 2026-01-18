//*************************************************************************
//    Copyright (c) 2025
//    MGR Linear Solver - Open-Darts Integration
//
//    Integration of MGR (Multigrid Reduction) solver with open-darts
//
//    @author Xiaoming Tian
//*************************************************************************

#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_IFACE_MGR_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_IFACE_MGR_HPP

#include "openDARTS/config/data_types.hpp"
#include "openDARTS/linear_solvers/linsolv_iface.hpp"
#include "mgr-linear-solver/LinearSolver.hpp"
#include <memory>

namespace opendarts
{
  namespace linear_solvers
  {
    /**
     * @brief MGR Linear Solver implementing open-darts linsolv_iface
     *
     * This class provides MGR (Multigrid Reduction) solver with the
     * standard open-darts linear solver interface.
     *
     * Features:
     *   - Runtime adaptive block size (reads from csr_matrix_base->n_row_size)
     *   - Supports compositional flow models (3, 4, 5, ... variables)
     *   - Drop-in replacement for other open-darts linear solvers
     *
     * Usage:
     *   auto* mgr_solver = new LinsolvIfaceMGR();
     *   mgr_solver->init(Jacobian, max_iters, tolerance);
     *   mgr_solver->setup(Jacobian);
     *   mgr_solver->solve(&RHS[0], &dX[0]);
     */
    class LinsolvIfaceMGR : public linsolv_iface
    {
    public:

      /**
       * @brief Constructor
       */
      LinsolvIfaceMGR();

      /**
       * @brief Destructor
       */
      virtual ~LinsolvIfaceMGR();

      /**
       * @brief Set preconditioner (placeholder - MGR is its own preconditioner)
       */
      virtual int set_prec(linsolv_iface *prec_input) override;

      /**
       * @brief Set pressure system preconditioner (placeholder)
       */
      virtual int set_p_system_prec(linsolv_iface *prec_input) override;

      /**
       * @brief Initialize MGR solver with matrix
       */
      virtual int init(csr_matrix_base *A,
                       opendarts::config::index_t max_iters,
                       opendarts::config::mat_float tolerance) override;

      /**
       * @brief Setup MGR solver
       */
      virtual int setup(csr_matrix_base *A_input) override;

      /**
       * @brief Solve linear system
       */
      virtual int solve(opendarts::config::mat_float *B,
                        opendarts::config::mat_float *X) override;

      /**
       * @brief Get number of iterations
       */
      virtual int get_n_iters() override;

      /**
       * @brief Get final residual
       */
      virtual opendarts::config::mat_float get_residual() override;

    private:

      std::unique_ptr<mgr::LinearSolver> mgr_solver_;  ///< MGR solver instance
      bool initialized_;                                ///< Flag: init() called
      bool setup_;                                      ///< Flag: setup() called
    };

  } // namespace linear_solvers
} // namespace opendarts

#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_IFACE_MGR_HPP
