/*
 * MGR Linear Solver - Compositional Flow Strategy
 *
 * 3-level MGR reduction strategy for compositional multiphase flow
 * @author Xiaoming Tian
 */

#ifndef MGR_LINEAR_SOLVER_COMPOSITIONAL_FLOW_STRATEGY_HPP_
#define MGR_LINEAR_SOLVER_COMPOSITIONAL_FLOW_STRATEGY_HPP_

#include "MGRStrategy.hpp"

namespace mgr {
namespace strategies {

/**
 * @brief Compositional multiphase flow strategy
 *
 * DOF labeling:
 *   - 0: pressure (p)
 *   - 1,2,...,N-1: component densities (z0, z1, ...)
 *
 * 3-level MGR reduction:
 *   - Level 0: Eliminate component densities (F-points)
 *   - Level 1: Further reduction if needed
 *   - Level 2: Coarsest level contains only pressure
 *
 * Coarse grid: pressure system solved with specialized BoomerAMG
 */
class CompositionalFlowStrategy : public MGRStrategy
{
public:

  /**
   * @brief Constructor
   * @param numComponents Number of components (including pressure)
   * @param numDOF Total number of DOFs
   * @param numCells Number of cells
   */
  CompositionalFlowStrategy( int_t numComponents,
                             int_t numDOF,
                             int_t numCells );

  /**
   * @brief Setup the strategy
   */
  void setup() override;

  /**
   * @brief Get coarse solver (BoomerAMG configured for pressure)
   */
  HYPRE_Solver getCoarseSolver() const override { return m_coarseSolver; }

private:

  int_t m_numComponents;  ///< Number of components per cell
  int_t m_numDOF;         ///< Total DOFs
  int_t m_numCells;       ///< Number of cells

  HYPRE_Solver m_coarseSolver; ///< BoomerAMG solver for pressure system

  /**
   * @brief Setup BoomerAMG for pressure system
   */
  void setupPressureAMG();
};

} // namespace strategies
} // namespace mgr

#endif // MGR_LINEAR_SOLVER_COMPOSITIONAL_FLOW_STRATEGY_HPP_
