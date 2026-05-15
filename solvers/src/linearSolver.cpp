/*
 * MGR Linear Solver - Linear Solver Implementation
 */

#include "LinearSolver.hpp"
#include "OpendartsJacobian.hpp"
#include "CompositionalFlowStrategy.hpp"
#include <iostream>
#include <chrono>
#include <cmath>
#include <limits>
#include <algorithm>
#include <string>

// HYPRE headers
#include <_hypre_parcsr_ls.h>
#include <HYPRE_parcsr_ls.h>
#include <_hypre_IJ_mv.h>
#include <HYPRE_IJ_mv.h>
#include <HYPRE_utilities.h>

namespace mgr {
namespace {
HYPRE_Int MGRDummySetup(HYPRE_Solver,
                        HYPRE_ParCSRMatrix,
                        HYPRE_ParVector,
                        HYPRE_ParVector)
{
  return 0;
}

std::string describeHypreError( HYPRE_Int rc )
{
  if( rc == 0 )
  {
    return "OK";
  }

  char description[512] = {};
  HYPRE_DescribeError( rc, description );

  if( description[0] == '\0' )
  {
    return "<no HYPRE error description available>";
  }

  return std::string( description );
}

void logMGRSetupContext( const char * stage,
                         int_t block_size,
                         int_t num_levels,
                         int_t num_points,
                         int_t global_rows,
                         int_t matrix_rows,
                         int_t matrix_cols,
                         int_t num_nonzero_blocks,
                         const SolverParameters & params )
{
  std::cerr << "[MGR] Context (" << stage << "): "
            << "block_size=" << block_size
            << ", num_levels=" << num_levels
            << ", num_points=" << num_points
            << ", global_rows=" << global_rows
            << ", matrix_rows=" << matrix_rows
            << ", matrix_cols=" << matrix_cols
            << ", nnz_blocks=" << num_nonzero_blocks
            << ", tol=" << params.tolerance
            << ", maxIter=" << params.maxIter
            << ", kdim=" << params.kdim
            << ", logLevel=" << params.logLevel
            << std::endl;
}

void logHypreFailure( const char * stage,
                      HYPRE_Int rc,
                      int_t block_size,
                      int_t num_levels,
                      int_t num_points,
                      int_t global_rows,
                      int_t matrix_rows,
                      int_t matrix_cols,
                      int_t num_nonzero_blocks,
                      const SolverParameters & params )
{
  std::cerr << "[MGR] " << stage << " failed with rc=" << rc
            << " (" << describeHypreError( rc ) << ")" << std::endl;

  const HYPRE_Int hypre_error = HYPRE_GetError();
  if( hypre_error != 0 && hypre_error != rc )
  {
    std::cerr << "[MGR] HYPRE_GetError()=" << hypre_error
              << " (" << describeHypreError( hypre_error ) << ")" << std::endl;
  }

  logMGRSetupContext( stage,
                      block_size,
                      num_levels,
                      num_points,
                      global_rows,
                      matrix_rows,
                      matrix_cols,
                      num_nonzero_blocks,
                      params );
}
} // namespace

LinearSolver::LinearSolver()
  : m_hasInitialGuess( false )
  , m_ijMatrix( nullptr )
  , m_ijRHS( nullptr )
  , m_ijSol( nullptr )
  , m_parMatrix( nullptr )
  , m_parRHS( nullptr )
  , m_parSol( nullptr )
  , m_matrixLoaded( false )
  , m_matrixAssembled( false )
{
  // Set default parameters
  m_params.maxIter = 100;
  m_params.tolerance = 1e-6;
  m_params.kdim = 30;
  m_params.useMGR = true;
  m_params.logLevel = 1;
  m_params.usePhysicsScaling = true;
}

LinearSolver::~LinearSolver()
{
  cleanup();
}

void LinearSolver::setStrategy( std::unique_ptr<MGRStrategy> strategy )
{
  m_strategy = std::move( strategy );
}

void LinearSolver::setMGRBlockSize( int_t block_size )
{
  m_mgrBlockSize = block_size;
}

void LinearSolver::setInitialGuess( const std::vector<real_type> & initialGuess )
{
  if( initialGuess.size() != static_cast<size_t>( m_matrix.global_num_rows ) )
  {
    std::cerr << "Error: Initial guess size mismatch! Expected: "
              << m_matrix.global_num_rows << ", Got: "
              << initialGuess.size() << std::endl;
    return;
  }
  m_initialGuess = initialGuess;
  m_hasInitialGuess = true;
}

void LinearSolver::clearInitialGuess()
{
  m_initialGuess.clear();
  m_hasInitialGuess = false;
}

bool LinearSolver::createHYPREMatrix()
{
  int_t num_rows = m_matrix.global_num_rows;
  const bool apply_scaling = m_params.usePhysicsScaling &&
                             ( m_scaling.size() == static_cast<size_t>( num_rows ) );
  int_t num_cols = m_matrix.global_num_cols;
  int_t num_cells = m_matrix.num_rows;
  int_t block_size = m_matrix.block_size;
  // Create IJ matrix
  HYPRE_IJMatrixCreate( MPI_COMM_WORLD, 0, num_rows - 1, 0, num_cols - 1, &m_ijMatrix );
  HYPRE_IJMatrixSetObjectType( m_ijMatrix, HYPRE_PARCSR );
  HYPRE_IJMatrixInitialize( m_ijMatrix );

  // Fill matrix (row-wise)
  int_t total_nonzeros = 0;

  for( int_t cell = 0; cell < num_cells; ++cell )
  {
    for( int_t i = 0; i < block_size; ++i )
    {
      bigint_t global_row = cell * block_size + i;
      real_type row_scale = apply_scaling ? m_scaling[global_row] : 1.0;

      std::vector<bigint_t> cols;
      std::vector<real_type> vals;

      for( int_t block_idx = m_matrix.row_ptr[cell];
           block_idx < m_matrix.row_ptr[cell + 1];
           ++block_idx )
      {
        int_t col_cell = m_matrix.col_ind[block_idx];
        int_t block_start = block_idx * block_size * block_size;

        for( int_t j = 0; j < block_size; ++j )
        {
          bigint_t global_col = col_cell * block_size + j;
          real_type val = m_matrix.values[block_start + i * block_size + j];
          if( apply_scaling )
          {
            val *= row_scale * m_scaling[global_col];
          }

          // Add ALL values from CSR file (no filtering)
          cols.push_back( global_col );
          vals.push_back( val );
          total_nonzeros++;
        }
      }

      // Set values for this row
      int_t ncols = cols.size();
      if( ncols > 0 )
      {
        HYPRE_IJMatrixSetValues( m_ijMatrix, 1, &ncols,
                                 &global_row, cols.data(), vals.data() );
      }
    }
  }

  // Assemble matrix
  HYPRE_IJMatrixAssemble( m_ijMatrix );
  HYPRE_IJMatrixGetObject( m_ijMatrix, (void**)&m_parMatrix );


  return true;
}

bool LinearSolver::createHYPREVectors()
{
  int_t num_rows = m_matrix.global_num_rows;
  const bool apply_scaling = m_params.usePhysicsScaling &&
                             ( m_scaling.size() == static_cast<size_t>( num_rows ) );

  // Create RHS vector
  HYPRE_IJVectorCreate( MPI_COMM_WORLD, 0, num_rows - 1, &m_ijRHS );
  HYPRE_IJVectorSetObjectType( m_ijRHS, HYPRE_PARCSR );
  HYPRE_IJVectorInitialize( m_ijRHS );

  // Create solution vector
  HYPRE_IJVectorCreate( MPI_COMM_WORLD, 0, num_rows - 1, &m_ijSol );
  HYPRE_IJVectorSetObjectType( m_ijSol, HYPRE_PARCSR );
  HYPRE_IJVectorInitialize( m_ijSol );

  // Fill RHS
  std::vector<bigint_t> rows( num_rows );
  for( int_t i = 0; i < num_rows; ++i )
  {
    rows[i] = i;
  }

  std::vector<real_type> rhs_scaled;
  std::vector<real_type> rhs_fallback;
  const real_type * rhs_values = nullptr;
  if( m_rhs.size() == static_cast<size_t>( num_rows ) )
  {
    rhs_values = m_rhs.data();
  }
  else
  {
    rhs_fallback.assign( num_rows, 0.0 );
    rhs_values = rhs_fallback.data();
  }
  if( apply_scaling )
  {
    rhs_scaled.resize( num_rows );
    for( int_t i = 0; i < num_rows; ++i )
    {
      rhs_scaled[i] = rhs_values[i] * m_scaling[i];
    }
    rhs_values = rhs_scaled.data();
  }
  HYPRE_IJVectorSetValues( m_ijRHS, num_rows, rows.data(), rhs_values );

  // Initialize solution (use initial guess if available, otherwise zero)
  // Use persistent m_solution member to ensure data lifetime through Assemble
  if( m_hasInitialGuess )
  {
    m_solution = m_initialGuess;
    if( apply_scaling )
    {
      for( int_t i = 0; i < num_rows; ++i )
      {
        if( m_scaling[i] != 0.0 )
        {
          m_solution[i] /= m_scaling[i];
        }
      }
    }
  }
  else
  {
    m_solution.assign( num_rows, 0.0 );
  }
  HYPRE_IJVectorSetValues( m_ijSol, num_rows, rows.data(), m_solution.data() );

  // Assemble vectors
  HYPRE_IJVectorAssemble( m_ijRHS );
  HYPRE_IJVectorAssemble( m_ijSol );

  HYPRE_IJVectorGetObject( m_ijRHS, (void**)&m_parRHS );
  HYPRE_IJVectorGetObject( m_ijSol, (void**)&m_parSol );

  return true;
}

void LinearSolver::computePhysicsScaling()
{
  m_scaling.clear();

  if( !m_params.usePhysicsScaling )
  {
    return;
  }

  const int_t block_size = m_matrix.block_size;
  const int_t num_rows = m_matrix.global_num_rows;

  if( block_size <= 0 || num_rows <= 0 )
  {
    return;
  }

  const std::vector<int_t>* point_markers = nullptr;
  if( m_strategy )
  {
    if( m_strategy->getPointMarkers().empty() )
    {
      m_strategy->setup();
    }
    if( m_strategy->getPointMarkers().size() == static_cast<size_t>( num_rows ) )
    {
      point_markers = &m_strategy->getPointMarkers();
    }
  }

  int_t num_labels = block_size;
  if( point_markers && !point_markers->empty() )
  {
    int_t max_label = *std::max_element( point_markers->begin(), point_markers->end() );
    if( max_label >= 0 )
    {
      num_labels = std::max( num_labels, max_label + 1 );
    }
  }

  std::vector<real_type> weights( num_labels, 0.0 );

  // Compute local squared Frobenius norms of diagonal blocks per component
  for( int_t cell = 0; cell < m_matrix.num_rows; ++cell )
  {
    for( int_t block_idx = m_matrix.row_ptr[cell];
         block_idx < m_matrix.row_ptr[cell + 1];
         ++block_idx )
    {
      int_t col_cell = m_matrix.col_ind[block_idx];
      int_t block_start = block_idx * block_size * block_size;
      for( int_t i = 0; i < block_size; ++i )
      {
        const int_t global_row = cell * block_size + i;
        const int_t row_label = point_markers ? (*point_markers)[global_row] : i;
        for( int_t j = 0; j < block_size; ++j )
        {
          const int_t global_col = col_cell * block_size + j;
          const int_t col_label = point_markers ? (*point_markers)[global_col] : j;
          if( row_label == col_label && row_label >= 0 && row_label < num_labels )
          {
            real_type val = m_matrix.values[block_start + i * block_size + j];
            weights[row_label] += val * val;
          }
        }
      }
    }
  }

  // Compute scaling weights: w = sqrt(1 / sqrt(sum(val^2)))
  for( int_t c = 0; c < num_labels; ++c )
  {
    if( weights[c] > 0.0 )
    {
      weights[c] = std::sqrt( 1.0 / std::sqrt( weights[c] ) );
    }
    else
    {
      weights[c] = 1.0;
    }
  }

  // Populate scaling vector
  m_scaling.resize( num_rows );
  for( int_t cell = 0; cell < m_matrix.num_rows; ++cell )
  {
    for( int_t i = 0; i < block_size; ++i )
    {
      int_t global_row = cell * block_size + i;
      int_t label = point_markers ? (*point_markers)[global_row] : i;
      if( label >= 0 && label < num_labels )
      {
        m_scaling[global_row] = weights[label];
      }
      else
      {
        m_scaling[global_row] = 1.0;
      }
    }
  }
}

SolverResults LinearSolver::solve()
{
  if( !m_matrixLoaded || !m_matrixAssembled )
  {
    std::cerr << "Error: Matrix not loaded or assembled" << std::endl;
    SolverResults results;
    results.converged = false;
    return results;
  }

  // Choose solver based on parameters
  if( m_params.useMGR && m_strategy )
  {
    return solveGMRES_MGR();
  }
  else
  {
    return solveGMRES_AMG();
  }
}

HYPRE_Solver LinearSolver::setupMGRPreconditioner()
{
  if( !m_strategy )
  {
    std::cerr << "Error: MGR strategy not set" << std::endl;
    return nullptr;
  }

  // Setup strategy if not already done
  if( m_strategy->getPointMarkers().empty() )
  {
    m_strategy->setup();
  }

  HYPRE_Solver mgr_precond;
  HYPRE_MGRCreate( &mgr_precond );

  // Set as preconditioner
  HYPRE_MGRSetTol( mgr_precond, 0.0 );
  HYPRE_MGRSetMaxIter( mgr_precond, 1 );
  // Match GEOS behavior: suppress MGR log bit 0x2 and shift log level by 1.
  int_t mgr_log_level = std::max<int_t>( m_params.logLevel - 1, 0 );
  mgr_log_level &= ~static_cast<int_t>( 0x2 );
  HYPRE_MGRSetPrintLevel( mgr_precond, static_cast<HYPRE_Int>( mgr_log_level ) );

  // Set point markers and reduction strategy
  const auto & point_markers = m_strategy->getPointMarkers();
  int_t num_points = static_cast<int_t>( point_markers.size() );
  int_t num_levels = m_strategy->numLevels();
  // Always use m_matrix.block_size - m_mgrBlockSize can get corrupted due to memory layout issues
  int_t block_size = m_matrix.block_size;
  int_t marker_block_size = std::max( block_size, m_strategy->numBlocks() );
  const int_t global_rows = m_matrix.global_num_rows;
  const int_t matrix_rows = m_matrix.num_rows;
  const int_t matrix_cols = m_matrix.num_cols;
  const int_t num_nonzero_blocks = m_matrix.num_nonzero_blocks;

  if( num_points <= 0 )
  {
    std::cerr << "Error: MGR point markers are empty" << std::endl;
    logMGRSetupContext( "empty-point-markers",
                        block_size,
                        num_levels,
                        num_points,
                        global_rows,
                        matrix_rows,
                        matrix_cols,
                        num_nonzero_blocks,
                        m_params );
    return nullptr;
  }

  if( global_rows > 0 && num_points != global_rows )
  {
    std::cerr << "Error: MGR point markers size (" << num_points
              << ") does not match matrix size (" << global_rows
              << ")" << std::endl;
    logMGRSetupContext( "point-marker-size-mismatch",
                        block_size,
                        num_levels,
                        num_points,
                        global_rows,
                        matrix_rows,
                        matrix_cols,
                        num_nonzero_blocks,
                        m_params );
    return nullptr;
  }

  if( block_size <= 0 )
  {
    std::cerr << "Error: Invalid MGR block size (" << block_size << ")" << std::endl;
    logMGRSetupContext( "invalid-block-size",
                        block_size,
                        num_levels,
                        num_points,
                        global_rows,
                        matrix_rows,
                        matrix_cols,
                        num_nonzero_blocks,
                        m_params );
    return nullptr;
  }

  if( num_points % block_size != 0 )
  {
    std::cerr << "Error: MGR block size (" << block_size
              << ") does not divide global DOFs (" << num_points << ")" << std::endl;
    logMGRSetupContext( "block-size-does-not-divide-dofs",
                        block_size,
                        num_levels,
                        num_points,
                        global_rows,
                        matrix_rows,
                        matrix_cols,
                        num_nonzero_blocks,
                        m_params );
    return nullptr;
  }

  std::vector<int_t> num_labels( num_levels );
  std::vector<int_t*> label_ptrs( num_levels );

  for( int_t i = 0; i < num_levels; ++i )
  {
    const auto & level_params = m_strategy->getLevelParameters( i );
    num_labels[i] = level_params.labels.size();

    // Warning: we need a non-const pointer for HYPRE
    label_ptrs[i] = const_cast<int_t*>( level_params.labels.data() );
  }

  HYPRE_ClearAllErrors();
  HYPRE_Int rc = HYPRE_MGRSetCpointsByPointMarkerArray( mgr_precond,
                                                        marker_block_size,
                                                        num_levels,
                                                        num_labels.data(),
                                                        label_ptrs.data(),
                                                        const_cast<int_t*>( point_markers.data() ) );
  if( rc != 0 )
  {
    logHypreFailure( "HYPRE_MGRSetCpointsByPointMarkerArray",
                     rc,
                     block_size,
                     num_levels,
                     num_points,
                     global_rows,
                     matrix_rows,
                     matrix_cols,
                     num_nonzero_blocks,
                     m_params );
    HYPRE_MGRDestroy( mgr_precond );
    return nullptr;
  }

  // Set level-wise parameters
  std::vector<int_t> f_relax_types( num_levels );
  std::vector<int_t> f_relax_iters( num_levels );
  std::vector<int_t> interp_types( num_levels );
  std::vector<int_t> restrict_types( num_levels );
  std::vector<int_t> coarse_methods( num_levels );
  std::vector<int_t> smooth_types( num_levels );
  std::vector<int_t> smooth_iters( num_levels );

  for( int_t i = 0; i < num_levels; ++i )
  {
    const auto & params = m_strategy->getLevelParameters( i );
    f_relax_types[i] = static_cast<int_t>( params.fRelaxType );
    f_relax_iters[i] = params.fRelaxIters;
    interp_types[i] = static_cast<int_t>( params.interpType );
    restrict_types[i] = static_cast<int_t>( params.restrictType );
    coarse_methods[i] = static_cast<int_t>( params.coarseGridMethod );
    smooth_types[i] = static_cast<int_t>( params.globalSmootherType );
    smooth_iters[i] = params.globalSmootherIters;

    // Match GEOS behavior: if no relaxation/smoothing, force 0 iterations
    if( params.fRelaxType == FRelaxationType::none )
    {
      f_relax_iters[i] = 0;
    }
    if( params.globalSmootherType == GlobalSmootherType::none )
    {
      smooth_iters[i] = 0;
    }
  }

  HYPRE_ClearAllErrors();
  rc = HYPRE_MGRSetLevelFRelaxType( mgr_precond, f_relax_types.data() );
  if( rc == 0 ) rc = HYPRE_MGRSetLevelNumRelaxSweeps( mgr_precond, f_relax_iters.data() );
  if( rc == 0 ) rc = HYPRE_MGRSetLevelInterpType( mgr_precond, interp_types.data() );
  if( rc == 0 ) rc = HYPRE_MGRSetLevelRestrictType( mgr_precond, restrict_types.data() );
  if( rc == 0 ) rc = HYPRE_MGRSetCoarseGridMethod( mgr_precond, coarse_methods.data() );
  if( rc == 0 ) rc = HYPRE_MGRSetLevelSmoothType( mgr_precond, smooth_types.data() );
  if( rc == 0 ) rc = HYPRE_MGRSetLevelSmoothIters( mgr_precond, smooth_iters.data() );
  if( rc != 0 )
  {
    logHypreFailure( "HYPRE_MGRSetLevel*",
                     rc,
                     block_size,
                     num_levels,
                     num_points,
                     global_rows,
                     matrix_rows,
                     matrix_cols,
                     num_nonzero_blocks,
                     m_params );
    HYPRE_MGRDestroy( mgr_precond );
    return nullptr;
  }

  // Match GEOS defaults for coarse grid truncation and non-Galerkin settings
  HYPRE_ClearAllErrors();
  rc = HYPRE_MGRSetTruncateCoarseGridThreshold( mgr_precond, 1.0e-20 );
#if defined(HYPRE_RELEASE_NUMBER) && (HYPRE_RELEASE_NUMBER >= 23300)
  if( rc == 0 ) rc = HYPRE_MGRSetNonGalerkinMaxElmts( mgr_precond, 1 );
#endif
  if( rc != 0 )
  {
    logHypreFailure( "HYPRE_MGRSetTruncateCoarseGridThreshold/NonGalerkin",
                     rc,
                     block_size,
                     num_levels,
                     num_points,
                     global_rows,
                     matrix_rows,
                     matrix_cols,
                     num_nonzero_blocks,
                     m_params );
    HYPRE_MGRDestroy( mgr_precond );
    return nullptr;
  }

  // Set non-C-points to F-points
  HYPRE_ClearAllErrors();
  rc = HYPRE_MGRSetNonCpointsToFpoints( mgr_precond, 1 );
  if( rc != 0 )
  {
    logHypreFailure( "HYPRE_MGRSetNonCpointsToFpoints",
                     rc,
                     block_size,
                     num_levels,
                     num_points,
                     global_rows,
                     matrix_rows,
                     matrix_cols,
                     num_nonzero_blocks,
                     m_params );
    HYPRE_MGRDestroy( mgr_precond );
    return nullptr;
  }

  // Set coarse solver
  HYPRE_Solver coarse_solver = m_strategy->getCoarseSolver();
  HYPRE_ClearAllErrors();
  rc = HYPRE_MGRSetCoarseSolver( mgr_precond,
                                 HYPRE_BoomerAMGSolve,
                                 HYPRE_BoomerAMGSetup,
                                 coarse_solver );
  if( rc != 0 )
  {
    logHypreFailure( "HYPRE_MGRSetCoarseSolver",
                     rc,
                     block_size,
                     num_levels,
                     num_points,
                     global_rows,
                     matrix_rows,
                     matrix_cols,
                     num_nonzero_blocks,
                     m_params );
    HYPRE_MGRDestroy( mgr_precond );
    return nullptr;
  }

  return mgr_precond;
}

HYPRE_Solver LinearSolver::setupAMGPreconditioner()
{
  HYPRE_Solver amg_precond;
  HYPRE_BoomerAMGCreate( &amg_precond );

  // Configure as preconditioner
  HYPRE_BoomerAMGSetTol( amg_precond, 0.0 );
  HYPRE_BoomerAMGSetMaxIter( amg_precond, 1 );
  HYPRE_BoomerAMGSetPrintLevel( amg_precond, 0 );

  // For non-symmetric systems
  HYPRE_BoomerAMGSetCoarsenType( amg_precond, 6 );     // PMIS
  HYPRE_BoomerAMGSetInterpType( amg_precond, 6 );      // Direct interpolation
  HYPRE_BoomerAMGSetRelaxType( amg_precond, 6 );       // Hybrid GS/GMRES

  return amg_precond;
}

SolverResults LinearSolver::solveGMRES_MGR()
{

  SolverResults results;

  // Create GMRES solver
  HYPRE_Solver gmres_solver;
  HYPRE_ParCSRGMRESCreate( MPI_COMM_WORLD, &gmres_solver );

  HYPRE_ParCSRGMRESSetMaxIter( gmres_solver, m_params.maxIter );
  HYPRE_ParCSRGMRESSetTol( gmres_solver, m_params.tolerance );
  HYPRE_ParCSRGMRESSetKDim( gmres_solver, m_params.kdim );
  HYPRE_ParCSRGMRESSetPrintLevel( gmres_solver, m_params.logLevel );
  HYPRE_ParCSRGMRESSetLogging( gmres_solver, 1 );

  // Setup MGR preconditioner
  auto setup_start = std::chrono::high_resolution_clock::now();
  HYPRE_Solver mgr_precond = setupMGRPreconditioner();
  if( !mgr_precond )
  {
    logMGRSetupContext( "setupMGRPreconditioner-returned-null",
                        m_matrix.block_size,
                        m_strategy ? m_strategy->numLevels() : 0,
                        m_strategy ? static_cast<int_t>( m_strategy->getPointMarkers().size() ) : 0,
                        m_matrix.global_num_rows,
                        m_matrix.num_rows,
                        m_matrix.num_cols,
                        m_matrix.num_nonzero_blocks,
                        m_params );
    std::cerr << "Error: MGR preconditioner setup failed" << std::endl;
    results.converged = false;
    results.finalResidual = std::numeric_limits<real_type>::infinity();
    results.iterations = 0;
    return results;
  }

  HYPRE_ClearAllErrors();
  HYPRE_Int mgr_setup_rc = HYPRE_MGRSetup( mgr_precond, m_parMatrix, m_parRHS, m_parSol );
  if( mgr_setup_rc != 0 )
  {
    logHypreFailure( "HYPRE_MGRSetup",
                     mgr_setup_rc,
                     m_matrix.block_size,
                     m_strategy ? m_strategy->numLevels() : 0,
                     m_strategy ? static_cast<int_t>( m_strategy->getPointMarkers().size() ) : 0,
                     m_matrix.global_num_rows,
                     m_matrix.num_rows,
                     m_matrix.num_cols,
                     m_matrix.num_nonzero_blocks,
                     m_params );
    std::cerr << "Error: MGR preconditioner setup failed" << std::endl;
    results.converged = false;
    results.finalResidual = std::numeric_limits<real_type>::infinity();
    results.iterations = 0;
    return results;
  }

  // Set MGR as preconditioner for GMRES
  HYPRE_ParCSRGMRESSetPrecond( gmres_solver,
                                HYPRE_MGRSolve,
                                MGRDummySetup,
                                mgr_precond );

  auto setup_end = std::chrono::high_resolution_clock::now();
  results.setupTime = std::chrono::duration<double>( setup_end - setup_start ).count();
  m_setupTime = results.setupTime;  // Save to member for getSetupTime()

  // Setup and solve
  auto solve_start = std::chrono::high_resolution_clock::now();

  HYPRE_ParCSRGMRESSetup( gmres_solver, m_parMatrix, m_parRHS, m_parSol );
  HYPRE_ParCSRGMRESSolve( gmres_solver, m_parMatrix, m_parRHS, m_parSol );

  auto solve_end = std::chrono::high_resolution_clock::now();
  results.solveTime = std::chrono::duration<double>( solve_end - solve_start ).count();
  m_solveTime = results.solveTime;  // Save to member for getSolveTime()

  // Get statistics
  int_t num_iterations;
  real_type final_res_norm;
  HYPRE_GMRESGetNumIterations( gmres_solver, &num_iterations );
  HYPRE_GMRESGetFinalRelativeResidualNorm( gmres_solver, &final_res_norm );

  results.iterations = num_iterations;
  results.finalResidual = final_res_norm;

  // Convergence check: consider converged if residual is very small, even if slightly above tolerance
  // This handles cases where the solver reaches machine precision
  const real_type machine_epsilon = std::numeric_limits<real_type>::epsilon() * 100.0; // ~1e-14
  if( final_res_norm < m_params.tolerance )
  {
    results.converged = true;
  }
  else if( final_res_norm < machine_epsilon )
  {
    // Residual is at machine precision level, consider it converged
    results.converged = true;
    std::cout << "[MGR] Warning: Residual (" << final_res_norm << ") is above tolerance ("
              << m_params.tolerance << ") but at machine precision. Considering converged.\n";
  }
  else
  {
    results.converged = false;
  }

  // Extract solution
  int_t num_rows = m_matrix.global_num_rows;
  m_solution.resize( num_rows );

  std::vector<bigint_t> rows( num_rows );
  for( int_t i = 0; i < num_rows; ++i )
  {
    rows[i] = i;
  }

  HYPRE_IJVectorGetValues( m_ijSol, num_rows, rows.data(), m_solution.data() );

  // Unscale solution if physics-based scaling was applied
  const bool apply_scaling = m_params.usePhysicsScaling &&
                             ( m_scaling.size() == m_solution.size() );
  if( apply_scaling )
  {
    for( size_t i = 0; i < m_solution.size(); ++i )
    {
      m_solution[i] *= m_scaling[i];
    }
  }

  // Calculate relative error if reference is available
  if( !m_reference.empty() )
  {
    real_type max_error = 0.0;
    real_type max_ref = 0.0;

    for( size_t i = 0; i < m_solution.size(); ++i )
    {
      real_type error = std::abs( m_solution[i] - m_reference[i] );
      max_error = std::max( max_error, error );
      max_ref = std::max( max_ref, std::abs( m_reference[i] ) );
    }

    results.relError = ( max_ref > 0.0 ) ? ( max_error / max_ref * 100.0 ) : 0.0;
  }

  // Cleanup
  HYPRE_MGRDestroy( mgr_precond );
  HYPRE_ParCSRGMRESDestroy( gmres_solver );

  return results;
}

SolverResults LinearSolver::solveFlexGMRES_MGR()
{

  SolverResults results;

  // Create FlexGMRES solver
  HYPRE_Solver gmres_solver;
  HYPRE_ParCSRFlexGMRESCreate( MPI_COMM_WORLD, &gmres_solver );

  HYPRE_ParCSRFlexGMRESSetMaxIter( gmres_solver, m_params.maxIter );
  HYPRE_ParCSRFlexGMRESSetTol( gmres_solver, m_params.tolerance );
  HYPRE_ParCSRFlexGMRESSetKDim( gmres_solver, m_params.kdim );
  HYPRE_ParCSRFlexGMRESSetPrintLevel( gmres_solver, m_params.logLevel );
  HYPRE_ParCSRFlexGMRESSetLogging( gmres_solver, 1 );

  // Setup MGR preconditioner
  auto setup_start = std::chrono::high_resolution_clock::now();
  HYPRE_Solver mgr_precond = setupMGRPreconditioner();
  if( !mgr_precond )
  {
    logMGRSetupContext( "setupMGRPreconditioner-returned-null",
                        m_matrix.block_size,
                        m_strategy ? m_strategy->numLevels() : 0,
                        m_strategy ? static_cast<int_t>( m_strategy->getPointMarkers().size() ) : 0,
                        m_matrix.global_num_rows,
                        m_matrix.num_rows,
                        m_matrix.num_cols,
                        m_matrix.num_nonzero_blocks,
                        m_params );
    std::cerr << "Error: MGR preconditioner setup failed" << std::endl;
    results.converged = false;
    results.finalResidual = std::numeric_limits<real_type>::infinity();
    results.iterations = 0;
    return results;
  }

  HYPRE_ClearAllErrors();
  HYPRE_Int mgr_setup_rc = HYPRE_MGRSetup( mgr_precond, m_parMatrix, m_parRHS, m_parSol );
  if( mgr_setup_rc != 0 )
  {
    logHypreFailure( "HYPRE_MGRSetup",
                     mgr_setup_rc,
                     m_matrix.block_size,
                     m_strategy ? m_strategy->numLevels() : 0,
                     m_strategy ? static_cast<int_t>( m_strategy->getPointMarkers().size() ) : 0,
                     m_matrix.global_num_rows,
                     m_matrix.num_rows,
                     m_matrix.num_cols,
                     m_matrix.num_nonzero_blocks,
                     m_params );
    std::cerr << "Error: MGR preconditioner setup failed" << std::endl;
    results.converged = false;
    results.finalResidual = std::numeric_limits<real_type>::infinity();
    results.iterations = 0;
    return results;
  }

  // Set MGR as preconditioner for FlexGMRES
  HYPRE_ParCSRFlexGMRESSetPrecond( gmres_solver,
                                  HYPRE_MGRSolve,
                                  MGRDummySetup,
                                  mgr_precond );

  auto setup_end = std::chrono::high_resolution_clock::now();
  results.setupTime = std::chrono::duration<double>( setup_end - setup_start ).count();
  m_setupTime = results.setupTime;  // Save to member for getSetupTime()

  // Setup and solve
  auto solve_start = std::chrono::high_resolution_clock::now();

  HYPRE_ParCSRFlexGMRESSetup( gmres_solver, m_parMatrix, m_parRHS, m_parSol );
  HYPRE_ParCSRFlexGMRESSolve( gmres_solver, m_parMatrix, m_parRHS, m_parSol );

  auto solve_end = std::chrono::high_resolution_clock::now();
  results.solveTime = std::chrono::duration<double>( solve_end - solve_start ).count();
  m_solveTime = results.solveTime;  // Save to member for getSolveTime()

  // Get statistics
  int_t num_iterations;
  real_type final_res_norm;
  HYPRE_ParCSRFlexGMRESGetNumIterations( gmres_solver, &num_iterations );
  HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm( gmres_solver, &final_res_norm );

  results.iterations = num_iterations;
  results.finalResidual = final_res_norm;

  // Convergence check: consider converged if residual is very small, even if slightly above tolerance
  // This handles cases where the solver reaches machine precision
  const real_type machine_epsilon = std::numeric_limits<real_type>::epsilon() * 100.0; // ~1e-14
  if( final_res_norm < m_params.tolerance )
  {
    results.converged = true;
  }
  else if( final_res_norm < machine_epsilon )
  {
    // Residual is at machine precision level, consider it converged
    results.converged = true;
    std::cout << "[MGR] Warning: Residual (" << final_res_norm << ") is above tolerance ("
              << m_params.tolerance << ") but at machine precision. Considering converged.\n";
  }
  else
  {
    results.converged = false;
  }

  // Extract solution
  int_t num_rows = m_matrix.global_num_rows;
  m_solution.resize( num_rows );

  std::vector<bigint_t> rows( num_rows );
  for( int_t i = 0; i < num_rows; ++i )
  {
    rows[i] = i;
  }

  HYPRE_IJVectorGetValues( m_ijSol, num_rows, rows.data(), m_solution.data() );

  // Unscale solution if physics-based scaling was applied
  const bool apply_scaling = m_params.usePhysicsScaling &&
                             ( m_scaling.size() == m_solution.size() );
  if( apply_scaling )
  {
    for( size_t i = 0; i < m_solution.size(); ++i )
    {
      m_solution[i] *= m_scaling[i];
    }
  }

  // Calculate relative error if reference is available
  if( !m_reference.empty() )
  {
    real_type max_error = 0.0;
    real_type max_ref = 0.0;

    for( size_t i = 0; i < m_solution.size(); ++i )
    {
      real_type error = std::abs( m_solution[i] - m_reference[i] );
      max_error = std::max( max_error, error );
      max_ref = std::max( max_ref, std::abs( m_reference[i] ) );
    }

    results.relError = ( max_ref > 0.0 ) ? ( max_error / max_ref * 100.0 ) : 0.0;
  }

  // Cleanup
  HYPRE_MGRDestroy( mgr_precond );
  HYPRE_ParCSRFlexGMRESDestroy( gmres_solver );

  return results;
}

SolverResults LinearSolver::solveGMRES_AMG()
{

  SolverResults results;

  // Create GMRES solver
  HYPRE_Solver gmres_solver;
  HYPRE_ParCSRGMRESCreate( MPI_COMM_WORLD, &gmres_solver );

  HYPRE_ParCSRGMRESSetMaxIter( gmres_solver, m_params.maxIter );
  HYPRE_ParCSRGMRESSetTol( gmres_solver, m_params.tolerance );
  HYPRE_ParCSRGMRESSetKDim( gmres_solver, m_params.kdim );
  HYPRE_ParCSRGMRESSetPrintLevel( gmres_solver, m_params.logLevel );

  // Setup AMG preconditioner
  auto setup_start = std::chrono::high_resolution_clock::now();
  HYPRE_Solver amg_precond = setupAMGPreconditioner();

  HYPRE_ParCSRGMRESSetPrecond( gmres_solver,
                                HYPRE_BoomerAMGSolve,
                                HYPRE_BoomerAMGSetup,
                                amg_precond );

  auto setup_end = std::chrono::high_resolution_clock::now();
  results.setupTime = std::chrono::duration<double>( setup_end - setup_start ).count();
  m_setupTime = results.setupTime;  // Save to member for getSetupTime()

  // Setup and solve
  auto solve_start = std::chrono::high_resolution_clock::now();

  HYPRE_ParCSRGMRESSetup( gmres_solver, m_parMatrix, m_parRHS, m_parSol );
  HYPRE_ParCSRGMRESSolve( gmres_solver, m_parMatrix, m_parRHS, m_parSol );

  auto solve_end = std::chrono::high_resolution_clock::now();
  results.solveTime = std::chrono::duration<double>( solve_end - solve_start ).count();
  m_solveTime = results.solveTime;  // Save to member for getSolveTime()

  // Get statistics
  int_t num_iterations;
  real_type final_res_norm;
  HYPRE_GMRESGetNumIterations( gmres_solver, &num_iterations );
  HYPRE_GMRESGetFinalRelativeResidualNorm( gmres_solver, &final_res_norm );

  results.iterations = num_iterations;
  results.finalResidual = final_res_norm;

  // Convergence check: consider converged if residual is very small, even if slightly above tolerance
  // This handles cases where the solver reaches machine precision
  const real_type machine_epsilon = std::numeric_limits<real_type>::epsilon() * 100.0; // ~1e-14
  if( final_res_norm < m_params.tolerance )
  {
    results.converged = true;
  }
  else if( final_res_norm < machine_epsilon )
  {
    // Residual is at machine precision level, consider it converged
    results.converged = true;
    std::cout << "[MGR] Warning: Residual (" << final_res_norm << ") is above tolerance ("
              << m_params.tolerance << ") but at machine precision. Considering converged.\n";
  }
  else
  {
    results.converged = false;
  }

  // Extract solution
  int_t num_rows = m_matrix.global_num_rows;
  m_solution.resize( num_rows );

  std::vector<bigint_t> rows( num_rows );
  for( int_t i = 0; i < num_rows; ++i )
  {
    rows[i] = i;
  }

  HYPRE_IJVectorGetValues( m_ijSol, num_rows, rows.data(), m_solution.data() );

  // Unscale solution if physics-based scaling was applied
  const bool apply_scaling = m_params.usePhysicsScaling &&
                             ( m_scaling.size() == m_solution.size() );
  if( apply_scaling )
  {
    for( size_t i = 0; i < m_solution.size(); ++i )
    {
      m_solution[i] *= m_scaling[i];
    }
  }

  // Calculate relative error if reference is available
  if( !m_reference.empty() )
  {
    real_type max_error = 0.0;
    real_type max_ref = 0.0;

    for( size_t i = 0; i < m_solution.size(); ++i )
    {
      real_type error = std::abs( m_solution[i] - m_reference[i] );
      max_error = std::max( max_error, error );
      max_ref = std::max( max_ref, std::abs( m_reference[i] ) );
    }

    results.relError = ( max_ref > 0.0 ) ? ( max_error / max_ref * 100.0 ) : 0.0;
  }

  // Cleanup
  HYPRE_BoomerAMGDestroy( amg_precond );
  HYPRE_ParCSRGMRESDestroy( gmres_solver );

  return results;
}

SolverResults LinearSolver::solveFlexGMRES_AMG()
{

  SolverResults results;

  // Create FlexGMRES solver
  HYPRE_Solver gmres_solver;
  HYPRE_ParCSRFlexGMRESCreate( MPI_COMM_WORLD, &gmres_solver );

  HYPRE_ParCSRFlexGMRESSetMaxIter( gmres_solver, m_params.maxIter );
  HYPRE_ParCSRFlexGMRESSetTol( gmres_solver, m_params.tolerance );
  HYPRE_ParCSRFlexGMRESSetKDim( gmres_solver, m_params.kdim );
  HYPRE_ParCSRFlexGMRESSetPrintLevel( gmres_solver, m_params.logLevel );

  // Setup AMG preconditioner
  auto setup_start = std::chrono::high_resolution_clock::now();
  HYPRE_Solver amg_precond = setupAMGPreconditioner();

  HYPRE_ParCSRFlexGMRESSetPrecond( gmres_solver,
                                  HYPRE_BoomerAMGSolve,
                                  HYPRE_BoomerAMGSetup,
                                  amg_precond );

  auto setup_end = std::chrono::high_resolution_clock::now();
  results.setupTime = std::chrono::duration<double>( setup_end - setup_start ).count();
  m_setupTime = results.setupTime;  // Save to member for getSetupTime()

  // Setup and solve
  auto solve_start = std::chrono::high_resolution_clock::now();

  HYPRE_ParCSRFlexGMRESSetup( gmres_solver, m_parMatrix, m_parRHS, m_parSol );
  HYPRE_ParCSRFlexGMRESSolve( gmres_solver, m_parMatrix, m_parRHS, m_parSol );

  auto solve_end = std::chrono::high_resolution_clock::now();
  results.solveTime = std::chrono::duration<double>( solve_end - solve_start ).count();
  m_solveTime = results.solveTime;  // Save to member for getSolveTime()

  // Get statistics
  int_t num_iterations;
  real_type final_res_norm;
  HYPRE_ParCSRFlexGMRESGetNumIterations( gmres_solver, &num_iterations );
  HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm( gmres_solver, &final_res_norm );

  results.iterations = num_iterations;
  results.finalResidual = final_res_norm;

  // Convergence check: consider converged if residual is very small, even if slightly above tolerance
  // This handles cases where the solver reaches machine precision
  const real_type machine_epsilon = std::numeric_limits<real_type>::epsilon() * 100.0; // ~1e-14
  if( final_res_norm < m_params.tolerance )
  {
    results.converged = true;
  }
  else if( final_res_norm < machine_epsilon )
  {
    // Residual is at machine precision level, consider it converged
    results.converged = true;
    std::cout << "[MGR] Warning: Residual (" << final_res_norm << ") is above tolerance ("
              << m_params.tolerance << ") but at machine precision. Considering converged.\n";
  }
  else
  {
    results.converged = false;
  }

  // Extract solution
  int_t num_rows = m_matrix.global_num_rows;
  m_solution.resize( num_rows );

  std::vector<bigint_t> rows( num_rows );
  for( int_t i = 0; i < num_rows; ++i )
  {
    rows[i] = i;
  }

  HYPRE_IJVectorGetValues( m_ijSol, num_rows, rows.data(), m_solution.data() );

  // Unscale solution if physics-based scaling was applied
  const bool apply_scaling = m_params.usePhysicsScaling &&
                             ( m_scaling.size() == m_solution.size() );
  if( apply_scaling )
  {
    for( size_t i = 0; i < m_solution.size(); ++i )
    {
      m_solution[i] *= m_scaling[i];
    }
  }

  // Calculate relative error if reference is available
  if( !m_reference.empty() )
  {
    real_type max_error = 0.0;
    real_type max_ref = 0.0;

    for( size_t i = 0; i < m_solution.size(); ++i )
    {
      real_type error = std::abs( m_solution[i] - m_reference[i] );
      max_error = std::max( max_error, error );
      max_ref = std::max( max_ref, std::abs( m_reference[i] ) );
    }

    results.relError = ( max_ref > 0.0 ) ? ( max_error / max_ref * 100.0 ) : 0.0;
  }

  // Cleanup
  HYPRE_BoomerAMGDestroy( amg_precond );
  HYPRE_ParCSRFlexGMRESDestroy( gmres_solver );

  return results;
}

void LinearSolver::cleanup()
{
  if( m_ijMatrix )
  {
    HYPRE_IJMatrixDestroy( m_ijMatrix );
    m_ijMatrix = nullptr;
  }

  if( m_ijRHS )
  {
    HYPRE_IJVectorDestroy( m_ijRHS );
    m_ijRHS = nullptr;
  }

  if( m_ijSol )
  {
    HYPRE_IJVectorDestroy( m_ijSol );
    m_ijSol = nullptr;
  }

  m_parMatrix = nullptr;
  m_parRHS = nullptr;
  m_parSol = nullptr;
  m_matrixLoaded = false;
  m_matrixAssembled = false;
}

// ============================================================================
// open-darts Integration Interface Implementation
// ============================================================================

int LinearSolver::init(void* jacobian, index_t max_iters, mat_float tolerance)
{

  // Type conversion: void* -> OpendartsJacobian*
  // Note: This expects the jacobian to follow OpendartsJacobian memory layout
  OpendartsJacobian* jac = static_cast<OpendartsJacobian*>(jacobian);

  if (!jac) {
    std::cerr << "[LinearSolver::init] ERROR: Jacobian pointer is null!" << std::endl;
    return -1;
  }

  // Call existing setMatrixFromCSR method
  if (!setMatrixFromCSR(
         jac->n_rows,
         jac->n_cols,
         jac->n_row_size,
         jac->get_n_non_zeros(),
         jac->get_rows_ptr(),
         jac->get_cols_ind(),
         jac->get_values(),
         jac->get_diag_ind())) {
    std::cerr << "[LinearSolver::init] ERROR: Failed to set matrix from Jacobian!" << std::endl;
    return -2;
  }

  // Set default MGR strategy if not already set
  if (!m_strategy) {
    auto strategy = std::make_unique<strategies::CompositionalFlowStrategy>(
                     jac->n_row_size,
                     jac->get_global_dof(),
                     jac->n_rows);
    setStrategy(std::move(strategy));
  }

  // Save parameters for later use in setup()
  m_initMaxIters = max_iters;
  m_initTolerance = tolerance;

  return 0;  // Success
}

int LinearSolver::setup(void* jacobian)
{
  (void)jacobian;
  return setup(m_initMaxIters, m_initTolerance);
}

bool LinearSolver::setMatrixFromCSR( int_t num_rows,
                                      int_t num_cols,
                                      int_t block_size,
                                      int_t num_nonzero_blocks,
                                      const int_t * row_ptr,
                                      const int_t * col_ind,
                                      const double * values,
                                      const int_t * diag_ind )
{
  // Check for valid pointers
  if( !row_ptr || !col_ind || !values )
  {
    std::cerr << "Error: Null pointer passed to setMatrixFromCSR" << std::endl;
    return false;
  }

  // Copy data into BlockCSRMatrix structure
  m_matrix.num_rows = num_rows;
  m_matrix.num_cols = num_cols;
  m_matrix.block_size = block_size;
  m_matrix.num_nonzero_blocks = num_nonzero_blocks;
  m_matrix.global_num_rows = num_rows * block_size;
  m_matrix.global_num_cols = num_cols * block_size;

  // Copy row_ptr (size = num_rows + 1)
  m_matrix.row_ptr.resize( num_rows + 1 );
  std::copy( row_ptr, row_ptr + num_rows + 1, m_matrix.row_ptr.begin() );

  // Copy col_ind (size = num_nonzero_blocks)
  m_matrix.col_ind.resize( num_nonzero_blocks );
  std::copy( col_ind, col_ind + num_nonzero_blocks, m_matrix.col_ind.begin() );

  // Copy values (size = num_nonzero_blocks × block_size²)
  int_t values_size = num_nonzero_blocks * block_size * block_size;
  m_matrix.values.resize( values_size );
  std::copy( values, values + values_size, m_matrix.values.begin() );

  // Copy diag_ind if provided
  if( diag_ind )
  {
    m_matrix.diag_ind.resize( num_rows );
    std::copy( diag_ind, diag_ind + num_rows, m_matrix.diag_ind.begin() );
  }
  else
  {
    m_matrix.diag_ind.clear();
  }

  // New matrix values invalidate any previously assembled HYPRE objects.
  m_matrixLoaded = false;
  m_matrixAssembled = false;

  return true;
}

bool LinearSolver::setMatrixFromVector( int_t num_rows,
                                         int_t num_cols,
                                         int_t block_size,
                                         int_t num_nonzero_blocks,
                                         const std::vector< int_t > & row_ptr,
                                         const std::vector< int_t > & col_ind,
                                         const std::vector< double > & values,
                                         const std::vector< int_t > & diag_ind )
{
  // Validate input sizes
  if( static_cast<int_t>( row_ptr.size() ) < num_rows + 1 )
  {
    std::cerr << "Error: row_ptr size too small" << std::endl;
    return false;
  }

  if( static_cast<int_t>( col_ind.size() ) < num_nonzero_blocks )
  {
    std::cerr << "Error: col_ind size too small" << std::endl;
    return false;
  }

  int_t expected_values_size = num_nonzero_blocks * block_size * block_size;
  if( static_cast<int_t>( values.size() ) < expected_values_size )
  {
    std::cerr << "Error: values size too small. Expected: "
              << expected_values_size << ", Got: " << values.size() << std::endl;
    return false;
  }

  // Copy data into BlockCSRMatrix structure
  m_matrix.num_rows = num_rows;
  m_matrix.num_cols = num_cols;
  m_matrix.block_size = block_size;
  m_matrix.num_nonzero_blocks = num_nonzero_blocks;
  m_matrix.global_num_rows = num_rows * block_size;
  m_matrix.global_num_cols = num_cols * block_size;

  // Copy vectors
  m_matrix.row_ptr = row_ptr;
  m_matrix.col_ind = col_ind;
  m_matrix.values = values;

  // Copy diag_ind if provided
  if( !diag_ind.empty() )
  {
    if( static_cast<int_t>( diag_ind.size() ) < num_rows )
    {
      std::cerr << "Error: diag_ind size too small" << std::endl;
      return false;
    }
    m_matrix.diag_ind = diag_ind;
  }
  else
  {
    m_matrix.diag_ind.clear();
  }

  // New matrix values invalidate any previously assembled HYPRE objects.
  m_matrixLoaded = false;
  m_matrixAssembled = false;

  return true;
}

int_t LinearSolver::setup( int_t max_iters, double tolerance )
{
  // Update solver parameters
  m_params.maxIter = max_iters;
  m_params.tolerance = tolerance;

  // Rebuild from the latest matrix contents on every setup().
  cleanup();

  // Compute physics-based scaling if enabled
  if( m_params.usePhysicsScaling )
  {
    computePhysicsScaling();
  }

  // Create HYPRE matrix from BlockCSR
  if( !createHYPREMatrix() )
  {
    std::cerr << "Error: Failed to create HYPRE matrix in setup()" << std::endl;
    return -1;
  }

  // Create HYPRE vectors (but don't fill them yet)
  if( !createHYPREVectors() )
  {
    std::cerr << "Error: Failed to create HYPRE vectors in setup()" << std::endl;
    return -1;
  }

  m_matrixLoaded = true;
  m_matrixAssembled = true;

  return 0;
}

int LinearSolver::solve(mat_float* B, mat_float* X)
{
  if( !m_matrixLoaded || !m_matrixAssembled )
  {
    std::cerr << "Error: Matrix not set up. Call setup() first." << std::endl;
    return -1;
  }

  if( !B || !X )
  {
    std::cerr << "Error: Null pointer passed to solve()" << std::endl;
    return -1;
  }

  int_t num_rows = m_matrix.global_num_rows;
  const bool apply_scaling = m_params.usePhysicsScaling &&
                             ( m_scaling.size() == static_cast<size_t>( num_rows ) );

  // Update RHS vector in HYPRE
  std::vector<bigint_t> rows( num_rows );
  for( int_t i = 0; i < num_rows; ++i )
  {
    rows[i] = i;
  }

  std::vector<real_type> rhs_scaled;
  const real_type * rhs_values = B;
  if( apply_scaling )
  {
    rhs_scaled.resize( num_rows );
    for( int_t i = 0; i < num_rows; ++i )
    {
      rhs_scaled[i] = B[i] * m_scaling[i];
    }
    rhs_values = rhs_scaled.data();
  }
  HYPRE_IJVectorSetValues( m_ijRHS, num_rows, rows.data(), rhs_values );
  HYPRE_IJVectorAssemble( m_ijRHS );

  // Initialize solution vector to zero
  std::vector<real_type> zeros( num_rows, 0.0 );
  HYPRE_IJVectorSetValues( m_ijSol, num_rows, rows.data(), zeros.data() );
  HYPRE_IJVectorAssemble( m_ijSol );

  // Choose solver based on parameters
  const bool use_flex = ( m_params.krylovType == KrylovType::flexgmres );
  if( m_params.useMGR && m_strategy )
  {
    m_lastResults = use_flex ? solveFlexGMRES_MGR() : solveGMRES_MGR();
  }
  else
  {
    m_lastResults = use_flex ? solveFlexGMRES_AMG() : solveGMRES_AMG();
  }

  // Extract solution to X array
  HYPRE_IJVectorGetValues( m_ijSol, num_rows, rows.data(), X );
  if( apply_scaling )
  {
    for( int_t i = 0; i < num_rows; ++i )
    {
      X[i] *= m_scaling[i];
    }
  }

  // Return number of iterations (or negative error code)
  if( m_lastResults.converged )
  {
    return m_lastResults.iterations;
  }
  else
  {
    return -m_lastResults.iterations; // Negative to indicate non-convergence
  }
}

int_t LinearSolver::get_n_iters() const
{
  return m_lastResults.iterations;
}

double LinearSolver::get_residual() const
{
  return m_lastResults.finalResidual;
}

int LinearSolver::set_prec(void* prec)
{
  (void)prec;
  return 0;  // Return success (although ignored)
}

int LinearSolver::set_p_system_prec(void* prec)
{
  (void)prec;
  return 0;  // Return success (although ignored)
}

} // namespace mgr
