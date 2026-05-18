/*
 * MGR Linear Solver - Compositional Flow Strategy
 *
 * 3-level MGR reduction strategy for compositional multiphase flow
 * @author Xiaoming Tian
 */

#ifndef MGR_LINEAR_SOLVER_COMPOSITIONAL_FLOW_STRATEGY_HPP_
#define MGR_LINEAR_SOLVER_COMPOSITIONAL_FLOW_STRATEGY_HPP_

#include "MGRStrategy.hpp"
#include <vector>

namespace mgr {
namespace strategies {

enum class WellStrategy : int
{
  eliminateWellBlock = 0,
  keepWellPrimary = 1
};

enum class VariableRole : int
{
  pressure = 0,
  composition = 1,
  saturation = 2,
  temperature = 3,
  volumeConstraint = 4,
  wellPressure = 100,
  wellSecondary = 101,
  facility = 200,
  rockMechanics = 300,
  displacement = 301,
  stress = 302,
  other = 999
};

struct CompositionalFlowStrategyConfig
{
  WellStrategy wellStrategy;
  bool enableWellLevel;
  bool enableCompositionLevel;
  std::vector<VariableRole> reservoirVariableRoles;
  std::vector<VariableRole> wellVariableRoles;
  std::vector<MGRLevelParameters> customLevels;
  MGRLevelParameters wellLevel;
  MGRLevelParameters compositionLevel;
  MGRLevelParameters pressureLevel;
  int pressureAmgCoarsenType;
  int pressureAmgInterpType;
  int pressureAmgRelaxType;
  int pressureAmgAggNumLevels;
  int pressureAmgAggInterpType;
  int pressureAmgAggPMaxElmts;
  int pressureAmgRelaxOrder;

  CompositionalFlowStrategyConfig();
};

/**
 * @brief Compositional multiphase flow strategy
 *
 * DOF labeling:
 *   - 0: reservoir pressure (p)
 *   - 1,2,...,N-1: reservoir component variables
 *   - N, ..., 2N-1: well variables when well cells are present
 *
 * MGR reduction:
 *   - Optional first level: reduce well variables with configurable F-relaxation
 *   - Optional custom levels: reduce user-defined future physics labels
 *   - Optional intermediate level: eliminate a composition variable
 *   - Final level: reduce reservoir variables to pressure
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
   * @param numReservoirCells Number of reservoir cells at the front of the block ordering
   */
  CompositionalFlowStrategy( int_t numComponents,
                             int_t numDOF,
                             int_t numCells,
                             int_t numReservoirCells = -1,
                             CompositionalFlowStrategyConfig config = CompositionalFlowStrategyConfig() );

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
  int_t m_numReservoirCells; ///< Reservoir cells before well cells
  CompositionalFlowStrategyConfig m_config;
  std::vector<VariableRole> m_reservoirVariableRoles;
  std::vector<VariableRole> m_wellVariableRoles;

  HYPRE_Solver m_coarseSolver; ///< BoomerAMG solver for pressure system

  bool hasWellCells() const;
  bool eliminateWellBlock() const;
  bool keepWellPrimary() const;
  bool useWellEliminationLevel() const;
  bool useCompositionReductionLevel() const;
  bool keepWellPrimaryOnPressureLevel() const;
  int_t reservoirLabelCount() const;
  int_t wellLabelBase() const;
  static const char * fRelaxName( int_t fRelaxType );
  static const char * variableRoleName( VariableRole role );
  static bool isPressureRole( VariableRole role );
  static VariableRole defaultReservoirRole( int_t localVariable );
  static VariableRole defaultWellRole( int_t localVariable, VariableRole reservoirRole );

  void normalizeVariableRoles();
  void logVariableRolesAndLabels() const;
  std::vector<int_t> pressureKeepLabels() const;
  std::vector<int_t> compositionKeepLabels() const;

  static int_t computeNumLevels( int_t numComponents,
                                 int_t numCells,
                                 int_t numReservoirCells,
                                 WellStrategy wellStrategy,
                                 bool enableWellLevel,
                                 bool enableCompositionLevel,
                                 int_t numCustomLevels );

  /**
   * @brief Setup BoomerAMG for pressure system
   */
  void setupPressureAMG();

  void setupWellEliminationLevel( int_t level );
  void setupCustomReductionLevel( int_t level, int_t customLevelIndex );
  void setupCompositionReductionLevel( int_t level );
  void setupPressureReductionLevel( int_t level );
  void logLevelParameters( const MGRLevelParameters & params ) const;
};

} // namespace strategies
} // namespace mgr

#endif // MGR_LINEAR_SOLVER_COMPOSITIONAL_FLOW_STRATEGY_HPP_
