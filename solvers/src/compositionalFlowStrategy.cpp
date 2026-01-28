/*
 * MGR Linear Solver - Compositional Flow Strategy Implementation
 * Method: galerkin (Full RAP)
 *
 * Galerkin Strategy
 * - Full Galerkin product: R*A*P
 * - Most accurate coarse grid operator
 * - Best convergence for small to medium problems
 */

#include "CompositionalFlowStrategy.hpp"
#include <iostream>
#include <algorithm>

// HYPRE headers
#include <_hypre_parcsr_ls.h>
#include <_hypre_utilities.h>

extern "C" {
HYPRE_Int HYPRE_Initialize(void);
HYPRE_Int HYPRE_Initialized(void);
HYPRE_Int HYPRE_ClearAllErrors(void);
}

namespace mgr {
namespace strategies {

CompositionalFlowStrategy::CompositionalFlowStrategy( int_t numComponents,
                                                      int_t numDOF,
                                                      int_t numCells )
  : MGRStrategy( 1 )  // 1-level reduction (simplified)
  , m_numComponents( numComponents )
  , m_numDOF( numDOF )
  , m_numCells( numCells )
  , m_coarseSolver( nullptr )
{
  m_numBlocks = 2;  // 2 blocks: pressure and components
}

void CompositionalFlowStrategy::setup()
{
  std::cout << "\nSetting up Compositional Flow Strategy (Galerkin)..." << std::endl;
  std::cout << "  Components: " << m_numComponents << std::endl;
  std::cout << "  DOFs: " << m_numDOF << std::endl;
  std::cout << "  Cells: " << m_numCells << std::endl;

  // Initialize point markers
  m_pointMarkers.resize( m_numDOF );

  // Label DOFs for each cell
  // Label 0 = pressure (kept in coarse grid)
  // Label 1 = components (eliminated via Schur complement)
  for( int_t cell = 0; cell < m_numCells; ++cell )
  {
    for( int_t i = 0; i < m_numComponents; ++i )
    {
      int_t global_idx = cell * m_numComponents + i;
      m_pointMarkers[global_idx] = ( i == 0 ) ? 0 : 1;  // 0=pressure (C), 1=component (F)
    }
  }

  // Level 0: Eliminate component densities (F-points)
  // Keep only label 0 (pressure - C-points)
  m_levelParams[0].labels.resize( 1 );
  m_levelParams[0].labels[0] = 0;  // Keep pressure

  // ==========================================
  // Galerkin Strategy Configuration
  // ==========================================

  // F-relaxation: None (no F-point relaxation)
  // F-points are eliminated directly without relaxation iteration
  std::cout << "  F-relaxation: none (HYPRE type -1)" << std::endl;
  m_levelParams[0].fRelaxType = FRelaxationType::none;
  m_levelParams[0].fRelaxIters = 1;

  // Interpolation: Injection (direct interpolation)
  // C-point values are directly injected to F-points during interpolation
  std::cout << "  Interpolation: injection (HYPRE type 0)" << std::endl;
  m_levelParams[0].interpType = InterpolationType::injection;

  // Restriction: Block column lumped
  // Lumps block columns when restricting from fine to coarse grid
  std::cout << "  Restriction: blockColLumped (HYPRE type 12)" << std::endl;
  m_levelParams[0].restrictType = RestrictionType::blockColLumped;

  // Coarse grid: Full Galerkin product
  // S = R * A * P where R and P are constructed from F-relaxation
  std::cout << "  Coarse grid: galerkin (HYPRE type 0)" << std::endl;
  std::cout << "    -> Full Galerkin product R*A*P (most accurate)" << std::endl;
  m_levelParams[0].coarseGridMethod = CoarseGridMethod::galerkin;

  // Global smoother: HYPRE ILU (incomplete LU factorization, ILU0)
  std::cout << "  Global smoother: hypreILU (HYPRE type 16), 2 iterations" << std::endl;
  m_levelParams[0].globalSmootherType = GlobalSmootherType::hypreILU;
  m_levelParams[0].globalSmootherIters = 2;

  // Setup coarse solver (BoomerAMG for pressure system)
  setupPressureAMG();

  std::cout << "Compositional Flow Strategy (Galerkin) setup complete" << std::endl;
}

void CompositionalFlowStrategy::setupPressureAMG()
{
  // Ensure HYPRE is initialized
  if (!HYPRE_Initialized())
  {
    HYPRE_Initialize();
  }

  // Clear any previous errors
  HYPRE_ClearAllErrors();

  // Create BoomerAMG solver for pressure system (Schur complement)
  int rc_create = HYPRE_BoomerAMGCreate( &m_coarseSolver );
  if (rc_create != 0)
  {
    std::cerr << "Error: HYPRE_BoomerAMGCreate failed with rc=" << rc_create << std::endl;
    return;
  }

  // Configure as preconditioner (not as solver)
  HYPRE_BoomerAMGSetTol( m_coarseSolver, 0.0 );
  HYPRE_BoomerAMGSetMaxIter( m_coarseSolver, 1 );
  HYPRE_BoomerAMGSetPrintLevel( m_coarseSolver, 0 );

  // Use aggressive coarsening for better scalability
  HYPRE_BoomerAMGSetAggNumLevels( m_coarseSolver, 1 );
  HYPRE_BoomerAMGSetAggPMaxElmts( m_coarseSolver, 20 );

  // Set interpolation type to multipass
  HYPRE_BoomerAMGSetAggInterpType( m_coarseSolver, 6 );  // multipass

  // For non-symmetric systems
  HYPRE_BoomerAMGSetCoarsenType( m_coarseSolver, 6 );     // PMIS coarsening
  HYPRE_BoomerAMGSetRelaxType( m_coarseSolver, 6 );       // Hybrid GS/GMRES

  // Enable C-F relaxation ordering
  HYPRE_BoomerAMGSetRelaxOrder( m_coarseSolver, 1 );

  // Set number of functions (1 = pressure only)
  HYPRE_BoomerAMGSetNumFunctions( m_coarseSolver, 1 );

  std::cout << "  Coarse solver: BoomerAMG configured for pressure system (Schur complement)" << std::endl;
}

} // namespace strategies
} // namespace mgr
