/*
 * MGR Linear Solver - MGR Strategy Base Class
 * @author Xiaoming Tian
 */

#ifndef MGR_LINEAR_SOLVER_MGR_STRATEGY_HPP_
#define MGR_LINEAR_SOLVER_MGR_STRATEGY_HPP_

#include "Types.hpp"
#include <vector>
#include <memory>
#include <functional>

// Forward declarations for HYPRE
extern "C" {
typedef struct hypre_Solver_struct *HYPRE_Solver;
}

namespace mgr {

/**
 * @brief Enumeration for F-relaxation types
 *
 * These values MUST match HYPRE's official numbering.
 * See: HYPRE_parcsr_ls.h documentation
 */
enum class FRelaxationType : int
{
  none = -1,                         ///< No F-relaxation

  // Basic methods
  jacobi = 0,                        ///< Jacobi relaxation

  // Multilevel methods
  singleVCycleSmoother = 1,           ///< Single-level V-cycle smoother
  amgVCycle = 2,                      ///< Full AMG V-cycle solver

  // Gauss-Seidel variants
  hybridGaussSeidelForward = 3,        ///< Hybrid Gauss-Seidel, forward solve
  hybridGaussSeidelBackward = 4,       ///< Hybrid Gauss-Seidel, backward solve
  hybridSymmetricGaussSeidel = 6,      ///< Hybrid symmetric Gauss-Seidel (SSOR)
  l1GaussSeidelForward = 13,          ///< l1-scaled Gauss-Seidel, forward solve
  l1GaussSeidelBackward = 14,         ///< l1-scaled Gauss-Seidel, backward solve

  // Advanced methods
  l1Jacobi = 18,                      ///< l1-scaled Jacobi
  fcfJacobi = 17,                     ///< FCF-Jacobi (color-forward)

  // Direct solvers (for F-point elimination)
  gaussianElimination = 9,            ///< Gaussian elimination (small systems)
  gaussianEliminationWPivoting = 99,  ///< Gaussian elimination with pivoting
  directInverse = 199                 ///< Direct inversion
};

/**
 * @brief Enumeration for interpolation types
 *
 * These values MUST match HYPRE's official numbering.
 */
enum class InterpolationType : int
{
  injection = 0,                     ///< Direct injection [0 I]^T
  l1Jacobi = 1,                      ///< L1-Jacobi interpolation
  jacobi = 2,                        ///< Diagonal scaling (Jacobi) - DEFAULT
  classicalModified = 3,             ///< Classical modified interpolation
  approximateInverse = 4,            ///< Approximate inverse interpolation
  blockJacobi = 12                   ///< Block Jacobi interpolation
};

/**
 * @brief Enumeration for restriction types
 *
 * These values MUST match HYPRE's official numbering.
 */
enum class RestrictionType : int
{
  injection = 0,                     ///< Direct injection [0 I] - DEFAULT
  unscaled = 1,                      ///< Unscaled (not recommended)
  jacobi = 2,                        ///< Diagonal scaling (Jacobi)
  approximateInverse = 3,            ///< Approximate inverse
  pAIRDistance1 = 4,                 ///< pAIR distance 1
  pAIRDistance2 = 5,                 ///< pAIR distance 2
  blockJacobi = 12,                  ///< Block Jacobi restriction
  cprLike = 13,                      ///< CPR-like restriction operator
  blockColLumped = 14                ///< Block column-lumped approximation
};

/**
 * @brief Enumeration for coarse grid methods
 *
 * These values MUST match HYPRE's official numbering.
 */
enum class CoarseGridMethod : int
{
  galerkin = 0,                      ///< Galerkin product: R*A*P - DEFAULT
  nonGalerkinBlockDiag = 1,          ///< Non-Galerkin, block diagonal inverse approximation
  nonGalerkinCPRDiag = 2,            ///< Non-Galerkin, CPR-like with diagonal inverse
  nonGalerkinCPRBlockDiag = 3,       ///< Non-Galerkin, CPR-like with block diagonal inverse
  nonGalerkinSparseApproxInv = 4     ///< Non-Galerkin, sparse approximate inverse
};

/**
 * @brief Enumeration for global smoother types
 *
 * These values MUST match HYPRE's official numbering.
 */
enum class GlobalSmootherType : int
{
  none = -1,                         ///< No global smoothing (default)
  blockJacobi = 0,                   ///< Block Jacobi smoother
  blockGaussSeidel = 1,              ///< Block Gauss-Seidel smoother
  jacobi = 2,                        ///< Jacobi smoother
  gaussSeidelSequential = 3,         ///< Gauss-Seidel, sequential (very slow!)
  gaussSeidelParallel = 4,           ///< Gauss-Seidel, interior parallel, boundary sequential
  hybridGaussSeidelForward = 5,       ///< Hybrid Gauss-Seidel, forward solve
  hybridGaussSeidelBackward = 6,      ///< Hybrid Gauss-Seidel, backward solve
  euclidILU = 8,                     ///< Euclid ILU smoother
  hypreILU = 16,                     ///< HYPRE ILU smoother
  l1Jacobi = 18                      ///< l1-scaled Jacobi smoother
};

/**
 * @brief MGR strategy parameters for a single level
 */
struct MGRLevelParameters
{
  std::vector<int_t> labels;              ///< DOF labels to keep at this level
  FRelaxationType fRelaxType;             ///< F-relaxation type
  int_t fRelaxIters;                      ///< Number of F-relaxation iterations
  InterpolationType interpType;           ///< Interpolation type
  RestrictionType restrictType;           ///< Restriction type
  CoarseGridMethod coarseGridMethod;      ///< Coarse grid method
  GlobalSmootherType globalSmootherType;   ///< Global smoother type
  int_t globalSmootherIters;              ///< Number of global smoother iterations

  MGRLevelParameters()
    : fRelaxType( FRelaxationType::jacobi )
    , fRelaxIters( 1 )
    , interpType( InterpolationType::jacobi )
    , restrictType( RestrictionType::injection )
    , coarseGridMethod( CoarseGridMethod::galerkin )
    , globalSmootherType( GlobalSmootherType::none )
    , globalSmootherIters( 0 )
  {}
};

/**
 * @brief Abstract base class for MGR reduction strategies
 *
 * Derived classes implement specific reduction strategies for different
 * physics problems (e.g., compositional flow, single-phase flow, etc.)
 */
class MGRStrategy
{
public:

  /**
   * @brief Constructor
   * @param numLevels Number of reduction levels
   */
  explicit MGRStrategy( int_t numLevels );

  /**
   * @brief Virtual destructor
   */
  virtual ~MGRStrategy() = default;

  /**
   * @brief Get number of reduction levels
   */
  int_t numLevels() const { return m_numLevels; }

  /**
   * @brief Get parameters for a specific level
   * @param level Level index (0 to numLevels-1)
   */
  const MGRLevelParameters & getLevelParameters( int_t level ) const
  {
    return m_levelParams[level];
  }

  /**
   * @brief Get point markers for DOFs
   *
   * Each DOF is marked with a label indicating which reduction level it belongs to.
   * For example, in compositional flow:
   *   - 0: pressure (kept until coarsest level)
   *   - 1,2,...: component densities (eliminated at various levels)
   */
  const std::vector<int_t> & getPointMarkers() const
  {
    return m_pointMarkers;
  }

  /**
   * @brief Get number of blocks (different DOF types)
   */
  int_t numBlocks() const { return m_numBlocks; }

  /**
   * @brief Setup the strategy - called after construction
   *
   * This method should populate:
   *   - m_pointMarkers: labels for each DOF
   *   - m_levelParams: parameters for each level
   */
  virtual void setup() = 0;

  /**
   * @brief Get coarse solver configuration
   *
   * @return Pointer to HYPRE solver (typically BoomerAMG)
   */
  virtual HYPRE_Solver getCoarseSolver() const = 0;

protected:

  int_t m_numLevels;                     ///< Number of reduction levels
  int_t m_numBlocks;                     ///< Number of different DOF types
  std::vector<int_t> m_pointMarkers;     ///< DOF markers
  std::vector<MGRLevelParameters> m_levelParams; ///< Parameters for each level
};

/**
 * @brief Factory function type for creating strategies
 */
using StrategyFactory = std::function<std::unique_ptr<MGRStrategy>()>;

} // namespace mgr

#endif // MGR_LINEAR_SOLVER_MGR_STRATEGY_HPP_
