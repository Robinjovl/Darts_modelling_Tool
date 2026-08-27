/*
 * MGR Linear Solver - Compositional Flow Strategy Implementation
 * Method: galerkin (Full RAP)
 *
 * Galerkin Strategy
 * - Full Galerkin product: R*A*P
 * - Most accurate coarse grid operator
 * - Best convergence for small to medium problems
 */

#include "mgr_compositional_flow_strategy.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>

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

CompositionalFlowStrategyConfig::CompositionalFlowStrategyConfig()
  : wellStrategy( WellStrategy::eliminateWellBlock )
  , enableWellLevel( true )
  , enableCompositionLevel( true )
  , pressureAmgCoarsenType( 6 )
  , pressureAmgInterpType( 6 )
  , pressureAmgRelaxType( 6 )
  , pressureAmgAggNumLevels( 1 )
  , pressureAmgAggInterpType( 6 )
  , pressureAmgAggPMaxElmts( 20 )
  , pressureAmgRelaxOrder( 1 )
  , pressureAmgStrongThreshold( -1.0 )
  , pressureAmgTruncFactor( -1.0 )
  , pressureAmgPMaxElmts( -1 )
  , pressureAmgMaxLevels( 0 )
  , verbose( false )
{
  wellLevel.fRelaxType = FRelaxationType::directInverse;
  wellLevel.fRelaxIters = 1;
  wellLevel.interpType = InterpolationType::blockJacobi;
  wellLevel.restrictType = RestrictionType::injection;
  wellLevel.coarseGridMethod = CoarseGridMethod::galerkin;
  wellLevel.globalSmootherType = GlobalSmootherType::none;
  wellLevel.globalSmootherIters = 0;

  compositionLevel.fRelaxType = FRelaxationType::jacobi;
  compositionLevel.fRelaxIters = 1;
  compositionLevel.interpType = InterpolationType::jacobi;
  compositionLevel.restrictType = RestrictionType::injection;
  compositionLevel.coarseGridMethod = CoarseGridMethod::galerkin;
  compositionLevel.globalSmootherType = GlobalSmootherType::none;
  compositionLevel.globalSmootherIters = 0;

  pressureLevel.fRelaxType = FRelaxationType::none;
  pressureLevel.fRelaxIters = 0;
  pressureLevel.interpType = InterpolationType::injection;
  pressureLevel.restrictType = RestrictionType::blockColLumped;
  pressureLevel.coarseGridMethod = CoarseGridMethod::galerkin;
  pressureLevel.globalSmootherType = GlobalSmootherType::hypreILU;
  pressureLevel.globalSmootherIters = 1;
}

CompositionalFlowStrategy::CompositionalFlowStrategy( int_t numComponents,
                                                      int_t numDOF,
                                                      int_t numCells,
                                                      int_t numReservoirCells,
                                                      CompositionalFlowStrategyConfig config )
  : MGRStrategy( computeNumLevels( numComponents,
                                   numCells,
                                   numReservoirCells,
                                   config.wellStrategy,
                                   config.enableWellLevel,
                                   config.enableCompositionLevel,
                                   static_cast<int_t>( config.customLevels.size() ) ) )
  , m_numComponents( numComponents )
  , m_numDOF( numDOF )
  , m_numCells( numCells )
  , m_numReservoirCells( numReservoirCells >= 0 ? std::min( numReservoirCells, numCells ) : numCells )
  , m_config( config )
  , m_reservoirVariableRoles( config.reservoirVariableRoles )
  , m_wellVariableRoles( config.wellVariableRoles )
  , m_coarseSolver( nullptr )
{
  normalizeVariableRoles();

  m_numBlocks = m_numComponents;
  if( hasWellCells() )
  {
    if( useWellEliminationLevel() )
    {
      m_numBlocks = 2 * m_numComponents;
    }
    else if( keepWellPrimaryOnPressureLevel() )
    {
      m_numBlocks = m_numComponents + 1;
    }
  }
}

CompositionalFlowStrategy::~CompositionalFlowStrategy()
{
  if ( m_coarseSolver != nullptr )
  {
    HYPRE_BoomerAMGDestroy( m_coarseSolver );
    m_coarseSolver = nullptr;
  }
}

void CompositionalFlowStrategy::setup()
{
  // Diagnostics gated behind config.verbose: previously ~25 unconditional
  // stdout lines were re-emitted on every strategy rebuild (per adjoint
  // solve / configuration change).
  if( m_config.verbose )
  {
    std::cout << "\nSetting up Compositional Flow Strategy (physical-role-aware MGR)..." << std::endl;
    std::cout << "  Components: " << m_numComponents << std::endl;
    std::cout << "  DOFs: " << m_numDOF << std::endl;
    std::cout << "  Cells: " << m_numCells << std::endl;
    std::cout << "  Reservoir cells: " << m_numReservoirCells << std::endl;
    std::cout << "  Well cells: " << (m_numCells - m_numReservoirCells) << std::endl;
    std::cout << "  Marker labels: " << m_numBlocks << std::endl;
    std::cout << "  Reduction levels: " << m_numLevels << std::endl;
    std::cout << "  Well level: " << (m_config.enableWellLevel ? "enabled" : "disabled") << std::endl;
    std::cout << "  Custom levels: " << m_config.customLevels.size() << std::endl;
    std::cout << "  Composition level: " << (m_config.enableCompositionLevel ? "enabled" : "disabled") << std::endl;
    logVariableRolesAndLabels();
    if( hasWellCells() )
    {
      if( useWellEliminationLevel() )
      {
        std::cout << "  Well strategy: eliminate well block" << std::endl;
        std::cout << "  Well marker labels: " << wellLabelBase()
                  << ".." << (wellLabelBase() + m_numComponents - 1) << std::endl;
      }
      else if( keepWellPrimaryOnPressureLevel() )
      {
        std::cout << "  Well strategy: keep well primary on coarse grid" << std::endl;
        std::cout << "  Well primary marker label: " << wellLabelBase() << std::endl;
      }
      else
      {
        std::cout << "  Well strategy: fold well variables into reservoir labels" << std::endl;
      }
    }
  }

  // Initialize point markers
  m_pointMarkers.resize( m_numDOF );

  // Label DOFs for each reservoir cell.
  for( int_t cell = 0; cell < m_numReservoirCells; ++cell )
  {
    for( int_t i = 0; i < m_numComponents; ++i )
    {
      int_t global_idx = cell * m_numComponents + i;
      m_pointMarkers[global_idx] = i;
    }
  }

  for( int_t cell = m_numReservoirCells; cell < m_numCells; ++cell )
  {
    for( int_t i = 0; i < m_numComponents; ++i )
    {
      int_t global_idx = cell * m_numComponents + i;
      if( useWellEliminationLevel() )
      {
        // Give every well DOF its own label so the first MGR level can reduce
        // the whole well block.
        m_pointMarkers[global_idx] = wellLabelBase() + i;
      }
      else
      {
        // Without a dedicated well-elimination level, keep well pressure on the
        // pressure coarse grid and fold secondary well variables with reservoir
        // secondary labels.
        m_pointMarkers[global_idx] =
            (keepWellPrimaryOnPressureLevel() && isPressureRole( m_wellVariableRoles[i] )) ? wellLabelBase() : i;
      }
    }
  }

  int_t level = 0;
  if( useWellEliminationLevel() )
  {
    setupWellEliminationLevel( level++ );
  }

  for( int_t customLevel = 0; customLevel < static_cast<int_t>( m_config.customLevels.size() ); ++customLevel )
  {
    setupCustomReductionLevel( level++, customLevel );
  }

  if( useCompositionReductionLevel() )
  {
    setupCompositionReductionLevel( level++ );
  }

  if( level < m_numLevels )
  {
    setupPressureReductionLevel( level++ );
  }

  setupPressureAMG();

  std::cout << "Compositional Flow Strategy (physical-role-aware MGR) setup complete" << std::endl;
}

bool CompositionalFlowStrategy::hasWellCells() const
{
  return m_numReservoirCells < m_numCells;
}

bool CompositionalFlowStrategy::eliminateWellBlock() const
{
  return m_config.wellStrategy == WellStrategy::eliminateWellBlock;
}

bool CompositionalFlowStrategy::keepWellPrimary() const
{
  return m_config.wellStrategy == WellStrategy::keepWellPrimary;
}

bool CompositionalFlowStrategy::useWellEliminationLevel() const
{
  return hasWellCells() && m_config.enableWellLevel && eliminateWellBlock();
}

bool CompositionalFlowStrategy::useCompositionReductionLevel() const
{
  return m_config.enableCompositionLevel && m_numComponents > 2;
}

bool CompositionalFlowStrategy::keepWellPrimaryOnPressureLevel() const
{
  return hasWellCells() && m_numComponents > 1 && !useWellEliminationLevel();
}

int_t CompositionalFlowStrategy::reservoirLabelCount() const
{
  return m_numComponents;
}

int_t CompositionalFlowStrategy::wellLabelBase() const
{
  return reservoirLabelCount();
}

const char * CompositionalFlowStrategy::fRelaxName( int_t fRelaxType )
{
  switch( fRelaxType )
  {
    case static_cast<int_t>( FRelaxationType::none ):
      return "none";
    case static_cast<int_t>( FRelaxationType::weightedJacobi ):
      return "weightedJacobi";
    case static_cast<int_t>( FRelaxationType::singleVCycleSmoother ):
      return "singleVCycle";
    case static_cast<int_t>( FRelaxationType::amgVCycle ):
      return "amg";
    case static_cast<int_t>( FRelaxationType::hybridGaussSeidelForward ):
      return "hybridGaussSeidelForward";
    case static_cast<int_t>( FRelaxationType::hybridGaussSeidelBackward ):
      return "hybridGaussSeidelBackward";
    case static_cast<int_t>( FRelaxationType::hybridChaoticGaussSeidel ):
      return "hybridChaoticGaussSeidel";
    case static_cast<int_t>( FRelaxationType::hybridSymmetricGaussSeidel ):
      return "hybridSymmetricGaussSeidel";
    case static_cast<int_t>( FRelaxationType::jacobi ):
      return "jacobi";
    case static_cast<int_t>( FRelaxationType::l1HybridSymmetricGaussSeidel ):
      return "l1HybridSymmetricGaussSeidel";
    case static_cast<int_t>( FRelaxationType::l1GaussSeidelForward ):
      return "l1GaussSeidelForward";
    case static_cast<int_t>( FRelaxationType::l1GaussSeidelBackward ):
      return "l1GaussSeidelBackward";
    case static_cast<int_t>( FRelaxationType::fcfJacobi ):
      return "fcfJacobi";
    case static_cast<int_t>( FRelaxationType::l1Jacobi ):
      return "l1Jacobi";
    case static_cast<int_t>( FRelaxationType::sparseDirectSolver ):
      return "sparseDirectSolver";
    case static_cast<int_t>( FRelaxationType::ilu ):
      return "ilu";
    case static_cast<int_t>( FRelaxationType::gaussianElimination ):
      return "gaussianElimination";
    case static_cast<int_t>( FRelaxationType::gaussianEliminationWPivoting ):
      return "gaussianEliminationWPivoting";
    case static_cast<int_t>( FRelaxationType::directInverse ):
      return "directInverse";
    default:
      return "custom";
  }
}

const char * CompositionalFlowStrategy::variableRoleName( VariableRole role )
{
  switch( role )
  {
    case VariableRole::pressure:
      return "reservoir pressure";
    case VariableRole::composition:
      return "composition";
    case VariableRole::saturation:
      return "saturation";
    case VariableRole::temperature:
      return "temperature";
    case VariableRole::volumeConstraint:
      return "volume constraint";
    case VariableRole::wellPressure:
      return "well pressure";
    case VariableRole::wellSecondary:
      return "well secondary";
    case VariableRole::facility:
      return "facility";
    case VariableRole::rockMechanics:
      return "rock mechanics";
    case VariableRole::displacement:
      return "displacement";
    case VariableRole::stress:
      return "stress";
    case VariableRole::other:
      return "other";
    default:
      return "custom";
  }
}

bool CompositionalFlowStrategy::isPressureRole( VariableRole role )
{
  return role == VariableRole::pressure || role == VariableRole::wellPressure;
}

VariableRole CompositionalFlowStrategy::defaultReservoirRole( int_t localVariable )
{
  return localVariable == 0 ? VariableRole::pressure : VariableRole::composition;
}

VariableRole CompositionalFlowStrategy::defaultWellRole( int_t localVariable, VariableRole reservoirRole )
{
  if( localVariable == 0 || isPressureRole( reservoirRole ) )
  {
    return VariableRole::wellPressure;
  }
  return VariableRole::wellSecondary;
}

void CompositionalFlowStrategy::normalizeVariableRoles()
{
  std::vector<VariableRole> reservoirRoles( m_numComponents );
  for( int_t i = 0; i < m_numComponents; ++i )
  {
    reservoirRoles[i] = i < static_cast<int_t>( m_reservoirVariableRoles.size() )
                            ? m_reservoirVariableRoles[i]
                            : defaultReservoirRole( i );
  }

  const bool hasReservoirPressure =
      std::any_of( reservoirRoles.begin(), reservoirRoles.end(),
                   []( VariableRole role ) { return CompositionalFlowStrategy::isPressureRole( role ); } );
  if( !hasReservoirPressure && m_numComponents > 0 )
  {
    std::cerr << "[MGR] Warning: no reservoir pressure variable role was configured; "
              << "falling back to local variable 0 as pressure." << std::endl;
    reservoirRoles[0] = VariableRole::pressure;
  }
  m_reservoirVariableRoles.swap( reservoirRoles );

  std::vector<VariableRole> wellRoles( m_numComponents );
  for( int_t i = 0; i < m_numComponents; ++i )
  {
    wellRoles[i] = i < static_cast<int_t>( m_wellVariableRoles.size() )
                       ? m_wellVariableRoles[i]
                       : defaultWellRole( i, m_reservoirVariableRoles[i] );
  }

  const bool hasWellPressure =
      std::any_of( wellRoles.begin(), wellRoles.end(),
                   []( VariableRole role ) { return CompositionalFlowStrategy::isPressureRole( role ); } );
  if( !hasWellPressure && m_numComponents > 0 )
  {
    std::cerr << "[MGR] Warning: no well pressure variable role was configured; "
              << "falling back to local well variable 0 as pressure." << std::endl;
    wellRoles[0] = VariableRole::wellPressure;
  }
  m_wellVariableRoles.swap( wellRoles );
}

void CompositionalFlowStrategy::logVariableRolesAndLabels() const
{
  std::cout << "  Variable roles and marker labels:" << std::endl;
  for( int_t i = 0; i < m_numComponents; ++i )
  {
    std::cout << "    reservoir local var " << i << " -> "
              << variableRoleName( m_reservoirVariableRoles[i] )
              << " -> label " << i << std::endl;
  }

  if( !hasWellCells() )
  {
    return;
  }

  for( int_t i = 0; i < m_numComponents; ++i )
  {
    int_t label = i;
    if( useWellEliminationLevel() )
    {
      label = wellLabelBase() + i;
    }
    else if( keepWellPrimaryOnPressureLevel() && isPressureRole( m_wellVariableRoles[i] ) )
    {
      label = wellLabelBase();
    }

    std::cout << "    well local var " << i << " -> "
              << variableRoleName( m_wellVariableRoles[i] )
              << " -> label " << label << std::endl;
  }
}

std::vector<int_t> CompositionalFlowStrategy::pressureKeepLabels() const
{
  std::vector<int_t> labels;
  for( int_t i = 0; i < m_numComponents; ++i )
  {
    if( isPressureRole( m_reservoirVariableRoles[i] ) )
    {
      labels.push_back( i );
    }
  }

  if( labels.empty() && m_numComponents > 0 )
  {
    labels.push_back( 0 );
  }

  if( keepWellPrimaryOnPressureLevel() )
  {
    labels.push_back( wellLabelBase() );
  }
  return labels;
}

std::vector<int_t> CompositionalFlowStrategy::compositionKeepLabels() const
{
  int_t eliminatedLabel = m_numComponents - 1;
  for( int_t offset = 0; offset < m_numComponents; ++offset )
  {
    const int_t i = m_numComponents - 1 - offset;
    if( !isPressureRole( m_reservoirVariableRoles[i] ) )
    {
      eliminatedLabel = i;
      break;
    }
  }

  std::vector<int_t> labels;
  labels.reserve( std::max<int_t>( 0, m_numComponents - 1 ) );
  for( int_t i = 0; i < m_numComponents; ++i )
  {
    if( i != eliminatedLabel )
    {
      labels.push_back( i );
    }
  }
  return labels;
}

int_t CompositionalFlowStrategy::computeNumLevels( int_t numComponents,
                                                   int_t numCells,
                                                   int_t numReservoirCells,
                                                   WellStrategy wellStrategy,
                                                   bool enableWellLevel,
                                                   bool enableCompositionLevel,
                                                   int_t numCustomLevels )
{
  int_t levels = 1;  // pressure reduction
  const bool hasWells = numReservoirCells >= 0 && numReservoirCells < numCells;
  const bool eliminateWells =
      enableWellLevel && wellStrategy == WellStrategy::eliminateWellBlock;

  if( hasWells && eliminateWells )
  {
    ++levels;  // well elimination
  }
  levels += std::max<int_t>( 0, numCustomLevels );
  if( enableCompositionLevel && numComponents > 2 )
  {
    ++levels;  // composition-variable reservoir reduction
  }
  if( hasWells && eliminateWells && numComponents == 1 )
  {
    --levels;  // single-phase reservoir-with-wells needs only well elimination
  }
  return levels;
}

void CompositionalFlowStrategy::setupWellEliminationLevel( int_t level )
{
  auto & params = m_levelParams[level];
  params = m_config.wellLevel;
  params.labels.resize( reservoirLabelCount() );
  std::iota( params.labels.begin(), params.labels.end(), 0 );

  std::cout << "  Level " << level << ": reduce well block" << std::endl;
  std::cout << "    keep reservoir labels: 0.." << (reservoirLabelCount() - 1) << std::endl;
  logLevelParameters( params );
}

void CompositionalFlowStrategy::setupCustomReductionLevel( int_t level, int_t customLevelIndex )
{
  auto & params = m_levelParams[level];
  params = m_config.customLevels[customLevelIndex];

  std::cout << "  Level " << level << ": custom reduction level " << customLevelIndex << std::endl;
  std::cout << "    keep labels:";
  for( const auto label : params.labels )
  {
    std::cout << " " << label;
  }
  std::cout << std::endl;
  logLevelParameters( params );
}

void CompositionalFlowStrategy::setupCompositionReductionLevel( int_t level )
{
  auto & params = m_levelParams[level];
  params = m_config.compositionLevel;
  params.labels = compositionKeepLabels();

  std::cout << "  Level " << level << ": eliminate one reservoir composition variable" << std::endl;
  std::cout << "    keep reservoir labels:";
  for( const auto label : params.labels )
  {
    std::cout << " " << label;
  }
  std::cout << std::endl;
  logLevelParameters( params );
}

void CompositionalFlowStrategy::setupPressureReductionLevel( int_t level )
{
  auto & params = m_levelParams[level];
  params = m_config.pressureLevel;
  params.labels = pressureKeepLabels();

  std::cout << "  Level " << level << ": reduce reservoir variables to pressure" << std::endl;
  std::cout << "    keep pressure labels:";
  for( const auto label : params.labels )
  {
    std::cout << " " << label;
  }
  std::cout << std::endl;
  logLevelParameters( params );
}

void CompositionalFlowStrategy::logLevelParameters( const MGRLevelParameters & params ) const
{
  std::cout << "    F-relaxation: " << fRelaxName( static_cast<int_t>( params.fRelaxType ) )
            << " (HYPRE type " << static_cast<int_t>( params.fRelaxType ) << "), "
            << params.fRelaxIters << " sweep(s)" << std::endl;
  std::cout << "    Interpolation: HYPRE type "
            << static_cast<int_t>( params.interpType ) << std::endl;
  std::cout << "    Restriction: HYPRE type "
            << static_cast<int_t>( params.restrictType ) << std::endl;
  std::cout << "    Coarse grid: HYPRE type "
            << static_cast<int_t>( params.coarseGridMethod ) << std::endl;
  std::cout << "    Global smoother: HYPRE type "
            << static_cast<int_t>( params.globalSmootherType ) << ", "
            << params.globalSmootherIters << " iteration(s)" << std::endl;
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

  // Re-entrant setup: release the previous hierarchy first (it was leaked
  // once per strategy rebuild before).
  if ( m_coarseSolver != nullptr )
  {
    HYPRE_BoomerAMGDestroy( m_coarseSolver );
    m_coarseSolver = nullptr;
  }

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

  HYPRE_BoomerAMGSetCoarsenType( m_coarseSolver, m_config.pressureAmgCoarsenType );
  HYPRE_BoomerAMGSetInterpType( m_coarseSolver, m_config.pressureAmgInterpType );
  HYPRE_BoomerAMGSetRelaxType( m_coarseSolver, m_config.pressureAmgRelaxType );
  HYPRE_BoomerAMGSetAggNumLevels( m_coarseSolver, m_config.pressureAmgAggNumLevels );
  HYPRE_BoomerAMGSetAggPMaxElmts( m_coarseSolver, m_config.pressureAmgAggPMaxElmts );
  HYPRE_BoomerAMGSetAggInterpType( m_coarseSolver, m_config.pressureAmgAggInterpType );
  HYPRE_BoomerAMGSetRelaxOrder( m_coarseSolver, m_config.pressureAmgRelaxOrder );
  if( m_config.pressureAmgStrongThreshold >= 0.0 )
  {
    HYPRE_BoomerAMGSetStrongThreshold( m_coarseSolver, m_config.pressureAmgStrongThreshold );
  }
  if( m_config.pressureAmgTruncFactor >= 0.0 )
  {
    HYPRE_BoomerAMGSetTruncFactor( m_coarseSolver, m_config.pressureAmgTruncFactor );
  }
  if( m_config.pressureAmgPMaxElmts >= 0 )
  {
    HYPRE_BoomerAMGSetPMaxElmts( m_coarseSolver, m_config.pressureAmgPMaxElmts );
  }
  if( m_config.pressureAmgMaxLevels > 0 )
  {
    HYPRE_BoomerAMGSetMaxLevels( m_coarseSolver, m_config.pressureAmgMaxLevels );
  }

  std::cout << "  Coarse solver: BoomerAMG configured for pressure system (Schur complement)" << std::endl;
  std::cout << "    AMG coarsen/interp/relax: "
            << m_config.pressureAmgCoarsenType << "/"
            << m_config.pressureAmgInterpType << "/"
            << m_config.pressureAmgRelaxType << std::endl;
  std::cout << "    AMG aggressive levels/interp/pmax/order: "
            << m_config.pressureAmgAggNumLevels << "/"
            << m_config.pressureAmgAggInterpType << "/"
            << m_config.pressureAmgAggPMaxElmts << "/"
            << m_config.pressureAmgRelaxOrder << std::endl;
  std::cout << "    AMG strength/trunc/pmax/max_levels: "
            << m_config.pressureAmgStrongThreshold << "/"
            << m_config.pressureAmgTruncFactor << "/"
            << m_config.pressureAmgPMaxElmts << "/"
            << m_config.pressureAmgMaxLevels << std::endl;
}

} // namespace strategies
} // namespace mgr
