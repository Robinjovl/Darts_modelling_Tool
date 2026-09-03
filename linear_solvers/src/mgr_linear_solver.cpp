/*
 * MGR Linear Solver - Linear Solver Implementation
 */

#include "mgr_linear_solver.hpp"
#include "OpendartsJacobian.hpp"
#include "mgr_compositional_flow_strategy.hpp"
#include "timer_node.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include "linear_solver.hpp"
#include <limits>
#include <algorithm>
#include <string>
#include <sstream>
#include <iomanip>
#include <utility>

// HYPRE headers
#include <_hypre_parcsr_ls.h>
#include <_hypre_parcsr_mv.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_parcsr_mv.h>
#include <_hypre_IJ_mv.h>
#include <HYPRE_IJ_mv.h>
#include <HYPRE_utilities.h>

namespace mgr {
namespace {
class ScopedTimer
{
public:
  explicit ScopedTimer(::timer_node* timer)
    : timer_(timer)
  {
    if( timer_ )
    {
      timer_->start();
    }
  }

  ~ScopedTimer()
  {
    if( timer_ )
    {
      timer_->stop();
    }
  }

  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
  ::timer_node* timer_;
};

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

bool isHypreConvergenceError( HYPRE_Int rc )
{
  return ( rc & HYPRE_ERROR_CONV ) != 0;
}

real_type vectorL2Norm( const real_type * values, int_t size )
{
  if( values == nullptr || size <= 0 )
  {
    return 0.0;
  }

  real_type sum = 0.0;
  for( int_t i = 0; i < size; ++i )
  {
    sum += values[i] * values[i];
  }
  return std::sqrt( sum );
}

real_type safeRatio( real_type numerator, real_type denominator )
{
  const real_type floor = std::numeric_limits<real_type>::min();
  return numerator / std::max( denominator, floor );
}

const char * boolLabel( bool value )
{
  return value ? "true" : "false";
}

const char * krylovLabel( KrylovType value )
{
  switch( value )
  {
    case KrylovType::gmres: return "gmres";
    case KrylovType::flexgmres: return "flexgmres";
  }
  return "unknown";
}

const char * scalingLabel( ScalingType value )
{
  switch( value )
  {
    case ScalingType::none: return "none";
    case ScalingType::physics: return "physics";
    case ScalingType::rowColOneNorm: return "row_col_one_norm";
    case ScalingType::diagonal: return "diagonal";
  }
  return "unknown";
}

const char * compositeModeLabel( CompositePreconditionerMode value )
{
  switch( value )
  {
    case CompositePreconditionerMode::mgrOnly: return "mgr_only";
    case CompositePreconditionerMode::mgrThenLocal: return "mgr_then_local";
    case CompositePreconditionerMode::localOnly: return "local_only";
  }
  return "unknown";
}

const char * localPreconditionerLabel( LocalPreconditionerType value )
{
  switch( value )
  {
    case LocalPreconditionerType::none: return "none";
    case LocalPreconditionerType::blockJacobi: return "block_jacobi";
    case LocalPreconditionerType::blockILU0: return "block_ilu0";
    case LocalPreconditionerType::blockILU1: return "block_ilu1";
  }
  return "unknown";
}

const char * localFallbackLabel( LocalFallbackStrategy value )
{
  switch( value )
  {
    case LocalFallbackStrategy::identity: return "identity";
    case LocalFallbackStrategy::shiftedDense: return "shifted_dense";
    case LocalFallbackStrategy::boundedDiagonal: return "bounded_diagonal";
    case LocalFallbackStrategy::shiftedDenseThenDiagonal: return "shifted_dense_then_diagonal";
  }
  return "unknown";
}

const char * bcsrCPRReductionLabel( BCSRCPRReductionType value )
{
  switch( value )
  {
    case BCSRCPRReductionType::pressureRow: return "pressure_row";
    case BCSRCPRReductionType::trueIMPES: return "true_impes";
    case BCSRCPRReductionType::trueIMPESWellElimination: return "true_impes_well_elimination";
  }
  return "unknown";
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

bool solveDenseLinearSystem( std::vector<real_type> matrix,
                             std::vector<real_type> rhs,
                             int_t n,
                             real_type pivot_tolerance,
                             std::vector<real_type> & solution )
{
  solution.assign( n, 0.0 );
  if( n <= 0 ||
      static_cast<int_t>( matrix.size() ) < n * n ||
      static_cast<int_t>( rhs.size() ) < n )
  {
    return false;
  }

  for( int_t col = 0; col < n; ++col )
  {
    int_t pivot_row = col;
    real_type pivot_abs = std::abs( matrix[col * n + col] );
    for( int_t row = col + 1; row < n; ++row )
    {
      const real_type candidate = std::abs( matrix[row * n + col] );
      if( candidate > pivot_abs )
      {
        pivot_abs = candidate;
        pivot_row = row;
      }
    }
    if( pivot_abs <= pivot_tolerance || !std::isfinite( pivot_abs ) )
    {
      return false;
    }
    if( pivot_row != col )
    {
      for( int_t j = col; j < n; ++j )
      {
        std::swap( matrix[col * n + j], matrix[pivot_row * n + j] );
      }
      std::swap( rhs[col], rhs[pivot_row] );
    }

    const real_type pivot = matrix[col * n + col];
    for( int_t row = col + 1; row < n; ++row )
    {
      const real_type factor = matrix[row * n + col] / pivot;
      matrix[row * n + col] = 0.0;
      for( int_t j = col + 1; j < n; ++j )
      {
        matrix[row * n + j] -= factor * matrix[col * n + j];
      }
      rhs[row] -= factor * rhs[col];
    }
  }

  for( int_t row = n - 1; row >= 0; --row )
  {
    real_type value = rhs[row];
    for( int_t col = row + 1; col < n; ++col )
    {
      value -= matrix[row * n + col] * solution[col];
    }
    const real_type pivot = matrix[row * n + row];
    if( std::abs( pivot ) <= pivot_tolerance || !std::isfinite( pivot ) )
    {
      return false;
    }
    solution[row] = value / pivot;
    if( !std::isfinite( solution[row] ) )
    {
      return false;
    }
    if( row == 0 )
    {
      break;
    }
  }
  return true;
}
} // namespace

enum class BlockInverseFailureReason
{
  none,
  nonFiniteInput,
  zeroNorm,
  smallPivot,
  smallDeterminant,
  nonFiniteInverse
};

class BlockLocalPreconditioner
{
public:
  bool setup( const BlockCSRMatrix & matrix,
              LocalPreconditionerType type,
              const std::vector<real_type> & row_scaling,
              const std::vector<real_type> & col_scaling,
              bool apply_scaling,
              real_type pivot_shift,
              LocalFallbackStrategy fallback_strategy,
              real_type fallback_diagonal_tolerance,
              real_type fallback_shift_max,
              real_type fallback_shift_growth,
              int_t reservoir_block_count )
  {
    clear();
    if( type == LocalPreconditionerType::none )
    {
      return false;
    }
    if( matrix.num_rows <= 0 || matrix.num_cols <= 0 ||
        matrix.block_size <= 0 || matrix.num_rows != matrix.num_cols )
    {
      return false;
    }

    m_type = type;
    m_numRows = matrix.num_rows;
    m_blockSize = matrix.block_size;
    m_blockSizeSquared = m_blockSize * m_blockSize;
    m_numReservoirRows = reservoir_block_count > 0
                         ? std::min<int_t>( reservoir_block_count, m_numRows )
                         : m_numRows;
    m_pivotShift = std::max<real_type>( pivot_shift, 0.0 );
    m_fallbackStrategy = fallback_strategy;
    m_fallbackDiagonalTolerance = std::max<real_type>( fallback_diagonal_tolerance, 0.0 );
    m_fallbackShiftMax = std::max<real_type>( fallback_shift_max, 0.0 );
    m_fallbackShiftGrowth = std::max<real_type>( fallback_shift_growth, 1.0 );
    m_originalRowPtr = matrix.row_ptr;
    m_originalColInd = matrix.col_ind;

    if( static_cast<int_t>( m_originalRowPtr.size() ) < m_numRows + 1 ||
        static_cast<int_t>( m_originalColInd.size() ) < matrix.num_nonzero_blocks ||
        static_cast<int_t>( matrix.values.size() ) < matrix.num_nonzero_blocks * m_blockSizeSquared )
    {
      clear();
      return false;
    }

    m_originalValues.resize( matrix.num_nonzero_blocks * m_blockSizeSquared );
    for( int_t row = 0; row < m_numRows; ++row )
    {
      for( int_t block = m_originalRowPtr[row]; block < m_originalRowPtr[row + 1]; ++block )
      {
        const int_t col = m_originalColInd[block];
        const int_t block_offset = block * m_blockSizeSquared;
        for( int_t r = 0; r < m_blockSize; ++r )
        {
          const int_t scalar_row = row * m_blockSize + r;
          const real_type row_scale =
              apply_scaling && scalar_row < static_cast<int_t>( row_scaling.size() )
              ? row_scaling[scalar_row] : 1.0;
          for( int_t c = 0; c < m_blockSize; ++c )
          {
            const int_t scalar_col = col * m_blockSize + c;
            const real_type col_scale =
                apply_scaling && scalar_col < static_cast<int_t>( col_scaling.size() )
                ? col_scaling[scalar_col] : 1.0;
            m_originalValues[block_offset + r * m_blockSize + c] =
                matrix.values[block_offset + r * m_blockSize + c] * row_scale * col_scale;
          }
        }
      }
    }

    if( m_type == LocalPreconditionerType::blockILU1 )
    {
      if( !buildBlockILU1Pattern() )
      {
        clear();
        return false;
      }
      m_luValues.assign( static_cast<size_t>( m_colInd.size() ) * m_blockSizeSquared,
                         0.0 );
      for( int_t row = 0; row < m_numRows; ++row )
      {
        for( int_t block = m_originalRowPtr[row]; block < m_originalRowPtr[row + 1]; ++block )
        {
          const int_t lu_block = findBlock( row, m_originalColInd[block] );
          if( lu_block < 0 )
          {
            clear();
            return false;
          }
          std::copy( &m_originalValues[block * m_blockSizeSquared],
                     &m_originalValues[( block + 1 ) * m_blockSizeSquared],
                     &m_luValues[lu_block * m_blockSizeSquared] );
        }
      }
    }
    else
    {
      m_rowPtr = m_originalRowPtr;
      m_colInd = m_originalColInd;
      m_fillLevel.assign( m_colInd.size(), 0 );
      m_diagInd.assign( m_numRows, -1 );
      for( int_t row = 0; row < m_numRows; ++row )
      {
        int_t diag = -1;
        if( static_cast<int_t>( matrix.diag_ind.size() ) > row )
        {
          const int_t candidate = matrix.diag_ind[row];
          if( candidate >= m_rowPtr[row] && candidate < m_rowPtr[row + 1] &&
              m_colInd[candidate] == row )
          {
            diag = candidate;
          }
        }
        if( diag < 0 )
        {
          diag = findBlock( row, row );
        }
        if( diag < 0 )
        {
          clear();
          return false;
        }
        m_diagInd[row] = diag;
      }
      m_luValues = m_originalValues;
    }

    m_diagInverse.assign( m_numRows * m_blockSizeSquared, 0.0 );
    m_blockWork.assign( m_blockSizeSquared, 0.0 );
    m_backwardBlock.assign( m_blockSize, 0.0 );
    if( m_type == LocalPreconditionerType::blockJacobi )
    {
      factorBlockJacobi();
    }
    else
    {
      factorBlockILU();
    }

    m_forwardWork.assign( m_numRows * m_blockSize, 0.0 );
    m_ready = true;
    return true;
  }

  void clear()
  {
    m_type = LocalPreconditionerType::none;
    m_numRows = 0;
    m_numReservoirRows = 0;
    m_blockSize = 0;
    m_blockSizeSquared = 0;
    m_pivotShift = 0.0;
    m_fallbackStrategy = LocalFallbackStrategy::identity;
    m_fallbackDiagonalTolerance = 1.0e-4;
    m_fallbackShiftMax = 1.0e-4;
    m_fallbackShiftGrowth = 100.0;
    m_failedPivots = 0;
    m_shiftedDenseFallbackPivots = 0;
    m_diagonalFallbackPivots = 0;
    m_reservoirFallbackPivots = 0;
    m_wellFallbackPivots = 0;
    m_primaryFailureNonFinite = 0;
    m_primaryFailureZeroNorm = 0;
    m_primaryFailureSmallPivot = 0;
    m_primaryFailureSmallDeterminant = 0;
    m_primaryFailureNonFiniteInverse = 0;
    m_boundedDiagonalInverseVariables = 0;
    m_boundedDiagonalIdentityVariables = 0;
    m_shiftedDenseShiftValues.clear();
    m_shiftedDenseShiftCounts.clear();
    m_ready = false;
    m_rowPtr.clear();
    m_colInd.clear();
    m_fillLevel.clear();
    m_diagInd.clear();
    m_originalRowPtr.clear();
    m_originalColInd.clear();
    m_originalValues.clear();
    m_luValues.clear();
    m_diagInverse.clear();
    m_forwardWork.clear();
    m_blockWork.clear();
    m_backwardBlock.clear();
  }

  bool ready() const
  {
    return m_ready;
  }

  int_t failedPivots() const
  {
    return m_failedPivots;
  }

  int_t shiftedDenseFallbackPivots() const
  {
    return m_shiftedDenseFallbackPivots;
  }

  int_t diagonalFallbackPivots() const
  {
    return m_diagonalFallbackPivots;
  }

  int_t totalFallbackPivots() const
  {
    return m_failedPivots + m_shiftedDenseFallbackPivots + m_diagonalFallbackPivots;
  }

  real_type fallbackRatio() const
  {
    return m_numRows > 0 ? static_cast<real_type>( totalFallbackPivots() ) /
                           static_cast<real_type>( m_numRows ) : 0.0;
  }

  int_t reservoirFallbackPivots() const
  {
    return m_reservoirFallbackPivots;
  }

  int_t wellFallbackPivots() const
  {
    return m_wellFallbackPivots;
  }

  int_t primaryFailureNonFinite() const
  {
    return m_primaryFailureNonFinite;
  }

  int_t primaryFailureZeroNorm() const
  {
    return m_primaryFailureZeroNorm;
  }

  int_t primaryFailureSmallPivot() const
  {
    return m_primaryFailureSmallPivot;
  }

  int_t primaryFailureSmallDeterminant() const
  {
    return m_primaryFailureSmallDeterminant;
  }

  int_t primaryFailureNonFiniteInverse() const
  {
    return m_primaryFailureNonFiniteInverse;
  }

  int_t boundedDiagonalInverseVariables() const
  {
    return m_boundedDiagonalInverseVariables;
  }

  int_t boundedDiagonalIdentityVariables() const
  {
    return m_boundedDiagonalIdentityVariables;
  }

  std::string shiftedDenseShiftSummary() const
  {
    if( m_shiftedDenseShiftValues.empty() )
    {
      return "none";
    }
    std::ostringstream out;
    out << "{";
    for( size_t i = 0; i < m_shiftedDenseShiftValues.size(); ++i )
    {
      if( i > 0 )
      {
        out << ", ";
      }
      out << std::scientific << std::setprecision( 3 )
          << m_shiftedDenseShiftValues[i] << ":" << m_shiftedDenseShiftCounts[i];
    }
    out << "}";
    return out.str();
  }

  const char * name() const
  {
    if( m_type == LocalPreconditionerType::blockJacobi )
    {
      return "block Jacobi";
    }
    if( m_type == LocalPreconditionerType::blockILU1 )
    {
      return "block ILU(1)";
    }
    return "block ILU(0)";
  }

  void matvec( const real_type * x, real_type * y ) const
  {
    const int_t n = m_numRows * m_blockSize;
    std::fill( y, y + n, 0.0 );
    for( int_t row = 0; row < m_numRows; ++row )
    {
      real_type * y_block = y + row * m_blockSize;
      for( int_t p = m_originalRowPtr[row]; p < m_originalRowPtr[row + 1]; ++p )
      {
        const real_type * block = &m_originalValues[p * m_blockSizeSquared];
        const real_type * x_block = x + m_originalColInd[p] * m_blockSize;
        for( int_t r = 0; r < m_blockSize; ++r )
        {
          real_type sum = 0.0;
          for( int_t c = 0; c < m_blockSize; ++c )
          {
            sum += block[r * m_blockSize + c] * x_block[c];
          }
          y_block[r] += sum;
        }
      }
    }
  }

  void apply( const real_type * rhs, real_type * x )
  {
    if( !m_ready )
    {
      return;
    }

    for( int_t row = 0; row < m_numRows; ++row )
    {
      real_type * y = &m_forwardWork[row * m_blockSize];
      const real_type * rhs_block = rhs + row * m_blockSize;
      for( int_t r = 0; r < m_blockSize; ++r )
      {
        y[r] = rhs_block[r];
      }

      if( usesILUFactorization() )
      {
        for( int_t p = m_rowPtr[row]; p < m_rowPtr[row + 1]; ++p )
        {
          const int_t col = m_colInd[p];
          if( col >= row )
          {
            continue;
          }
          subtractBlockMatvec( &m_luValues[p * m_blockSizeSquared],
                               &m_forwardWork[col * m_blockSize],
                               y );
        }
      }
    }

    std::fill( x, x + m_numRows * m_blockSize, 0.0 );
    for( int_t row = m_numRows - 1; row >= 0; --row )
    {
      for( int_t r = 0; r < m_blockSize; ++r )
      {
        m_backwardBlock[r] = m_forwardWork[row * m_blockSize + r];
      }

      if( usesILUFactorization() )
      {
        for( int_t p = m_rowPtr[row]; p < m_rowPtr[row + 1]; ++p )
        {
          const int_t col = m_colInd[p];
          if( col <= row )
          {
            continue;
          }
          subtractBlockMatvec( &m_luValues[p * m_blockSizeSquared],
                               x + col * m_blockSize,
                               m_backwardBlock.data() );
        }
      }

      multiplyBlockVector( &m_diagInverse[row * m_blockSizeSquared],
                           m_backwardBlock.data(),
                           x + row * m_blockSize );

      if( row == 0 )
      {
        break;
      }
    }
  }

private:
  bool usesILUFactorization() const
  {
    return m_type == LocalPreconditionerType::blockILU0 ||
           m_type == LocalPreconditionerType::blockILU1;
  }

  int_t findBlock( int_t row, int_t col ) const
  {
    for( int_t p = m_rowPtr[row]; p < m_rowPtr[row + 1]; ++p )
    {
      if( m_colInd[p] == col )
      {
        return p;
      }
    }
    return -1;
  }

  bool buildBlockILU1Pattern()
  {
    m_rowPtr.assign( m_numRows + 1, 0 );
    m_colInd.clear();
    m_fillLevel.clear();
    m_diagInd.assign( m_numRows, -1 );

    std::vector<std::pair<int_t, int_t>> candidates;
    candidates.reserve( 64 );
    for( int_t row = 0; row < m_numRows; ++row )
    {
      candidates.clear();
      for( int_t block = m_originalRowPtr[row]; block < m_originalRowPtr[row + 1]; ++block )
      {
        candidates.emplace_back( m_originalColInd[block], 0 );
      }

      for( int_t block = m_originalRowPtr[row]; block < m_originalRowPtr[row + 1]; ++block )
      {
        const int_t pivot_row = m_originalColInd[block];
        if( pivot_row >= row )
        {
          continue;
        }
        for( int_t pivot_block = m_originalRowPtr[pivot_row];
             pivot_block < m_originalRowPtr[pivot_row + 1]; ++pivot_block )
        {
          const int_t col = m_originalColInd[pivot_block];
          if( col > pivot_row )
          {
            candidates.emplace_back( col, 1 );
          }
        }
      }

      std::sort( candidates.begin(), candidates.end(),
                 []( const auto & lhs, const auto & rhs )
                 {
                   if( lhs.first != rhs.first )
                   {
                     return lhs.first < rhs.first;
                   }
                   return lhs.second < rhs.second;
                 } );

      m_rowPtr[row] = static_cast<int_t>( m_colInd.size() );
      int_t previous_col = -1;
      int_t previous_level = 0;
      bool have_previous = false;
      for( const auto & candidate : candidates )
      {
        const int_t col = candidate.first;
        const int_t level = candidate.second;
        if( col < 0 || col >= m_numRows )
        {
          continue;
        }
        if( !have_previous )
        {
          previous_col = col;
          previous_level = level;
          have_previous = true;
          continue;
        }
        if( col == previous_col )
        {
          previous_level = std::min( previous_level, level );
          continue;
        }
        appendPatternBlock( row, previous_col, previous_level );
        previous_col = col;
        previous_level = level;
      }
      if( have_previous )
      {
        appendPatternBlock( row, previous_col, previous_level );
      }
      m_rowPtr[row + 1] = static_cast<int_t>( m_colInd.size() );
      if( m_diagInd[row] < 0 )
      {
        return false;
      }
    }
    return true;
  }

  void appendPatternBlock( int_t row, int_t col, int_t level )
  {
    const int_t pos = static_cast<int_t>( m_colInd.size() );
    m_colInd.push_back( col );
    m_fillLevel.push_back( level );
    if( col == row )
    {
      m_diagInd[row] = pos;
    }
  }

  void factorBlockJacobi()
  {
    for( int_t row = 0; row < m_numRows; ++row )
    {
      invertDiagonalBlock( row );
    }
  }

  void factorBlockILU()
  {
    const int_t max_fill_level =
        m_type == LocalPreconditionerType::blockILU1 ? 1 : 0;
    std::vector<int_t> lower_blocks;
    for( int_t row = 0; row < m_numRows; ++row )
    {
      lower_blocks.clear();
      for( int_t p = m_rowPtr[row]; p < m_rowPtr[row + 1]; ++p )
      {
        if( m_colInd[p] < row )
        {
          lower_blocks.push_back( p );
        }
      }
      std::sort( lower_blocks.begin(), lower_blocks.end(),
                 [this]( int_t lhs, int_t rhs )
                 {
                   return m_colInd[lhs] < m_colInd[rhs];
                 } );

      for( int_t lower_pos : lower_blocks )
      {
        const int_t pivot_row = m_colInd[lower_pos];
        rightMultiplyBlockInPlace( &m_luValues[lower_pos * m_blockSizeSquared],
                                   &m_diagInverse[pivot_row * m_blockSizeSquared] );

        for( int_t upper_pos = m_rowPtr[pivot_row]; upper_pos < m_rowPtr[pivot_row + 1]; ++upper_pos )
        {
          const int_t col = m_colInd[upper_pos];
          if( col <= pivot_row )
          {
            continue;
          }
          if( m_type == LocalPreconditionerType::blockILU1 &&
              m_fillLevel[lower_pos] + m_fillLevel[upper_pos] + 1 >
                  max_fill_level )
          {
            continue;
          }
          const int_t row_pos = findBlock( row, col );
          if( row_pos < 0 )
          {
            continue;
          }
          subtractBlockProduct( &m_luValues[lower_pos * m_blockSizeSquared],
                                &m_luValues[upper_pos * m_blockSizeSquared],
                                &m_luValues[row_pos * m_blockSizeSquared] );
        }
      }

      invertDiagonalBlock( row );
    }
  }

  void invertDiagonalBlock( int_t row )
  {
    const int_t diag = m_diagInd[row];
    const real_type * block = &m_luValues[diag * m_blockSizeSquared];
    real_type * inverse = &m_diagInverse[row * m_blockSizeSquared];
    BlockInverseFailureReason primary_failure = BlockInverseFailureReason::none;
    if( invertBlock( block, inverse, m_pivotShift, &primary_failure ) )
    {
      return;
    }
    recordPrimaryFailure( primary_failure );

    if( m_fallbackStrategy == LocalFallbackStrategy::shiftedDense ||
        m_fallbackStrategy == LocalFallbackStrategy::shiftedDenseThenDiagonal )
    {
      real_type accepted_shift = 0.0;
      if( invertBlockWithShiftFallback( block, inverse, &accepted_shift ) )
      {
        ++m_shiftedDenseFallbackPivots;
        recordShiftedDenseShift( accepted_shift );
        recordFallbackLocation( row );
        return;
      }
    }

    if( m_fallbackStrategy == LocalFallbackStrategy::boundedDiagonal ||
        m_fallbackStrategy == LocalFallbackStrategy::shiftedDenseThenDiagonal )
    {
      int_t used_diagonal_variables = 0;
      if( invertBlockDiagonal( block, inverse, &used_diagonal_variables ) )
      {
        ++m_diagonalFallbackPivots;
        m_boundedDiagonalInverseVariables += used_diagonal_variables;
        m_boundedDiagonalIdentityVariables += m_blockSize - used_diagonal_variables;
        recordFallbackLocation( row );
        return;
      }
    }

    ++m_failedPivots;
    recordFallbackLocation( row );
    std::fill( inverse, inverse + m_blockSizeSquared, 0.0 );
    for( int_t i = 0; i < m_blockSize; ++i )
    {
      inverse[i * m_blockSize + i] = 1.0;
    }
  }

  bool invertBlockWithShiftFallback( const real_type * block,
                                     real_type * inverse,
                                     real_type * accepted_shift ) const
  {
    if( m_fallbackShiftMax <= 0.0 )
    {
      return false;
    }

    real_type shift = std::max<real_type>( m_pivotShift * m_fallbackShiftGrowth, 1.0e-10 );
    if( shift > m_fallbackShiftMax )
    {
      shift = m_fallbackShiftMax;
    }

    while( shift <= m_fallbackShiftMax )
    {
      if( invertBlock( block, inverse, shift ) )
      {
        if( accepted_shift )
        {
          *accepted_shift = shift;
        }
        return true;
      }
      if( m_fallbackShiftGrowth <= 1.0 || shift == m_fallbackShiftMax )
      {
        break;
      }
      shift = std::min<real_type>( shift * m_fallbackShiftGrowth, m_fallbackShiftMax );
    }
    return false;
  }

  bool invertBlock( const real_type * block,
                    real_type * inverse,
                    real_type relative_shift,
                    BlockInverseFailureReason * failure_reason = nullptr ) const
  {
    if( failure_reason )
    {
      *failure_reason = BlockInverseFailureReason::none;
    }
    real_type norm = 0.0;
    for( int_t i = 0; i < m_blockSizeSquared; ++i )
    {
      if( !std::isfinite( block[i] ) )
      {
        if( failure_reason )
        {
          *failure_reason = BlockInverseFailureReason::nonFiniteInput;
        }
        return false;
      }
      norm = std::max( norm, std::abs( block[i] ) );
    }
    const bool zero_norm = norm == 0.0;
    const real_type shift = relative_shift * std::max<real_type>( norm, 1.0 );
    const real_type pivot_tol = std::numeric_limits<real_type>::epsilon() *
                                std::max<real_type>( norm + std::abs( shift ), 1.0 ) * 100.0;

    if( m_blockSize == 1 )
    {
      const real_type pivot = block[0] + shift;
      if( std::abs( pivot ) <= pivot_tol || !std::isfinite( pivot ) )
      {
        if( failure_reason )
        {
          *failure_reason = !std::isfinite( pivot )
                            ? BlockInverseFailureReason::nonFiniteInput
                            : ( zero_norm ? BlockInverseFailureReason::zeroNorm
                                          : BlockInverseFailureReason::smallPivot );
        }
        return false;
      }
      inverse[0] = 1.0 / pivot;
      if( !std::isfinite( inverse[0] ) )
      {
        if( failure_reason )
        {
          *failure_reason = BlockInverseFailureReason::nonFiniteInverse;
        }
        return false;
      }
      return true;
    }

    if( m_blockSize == 2 )
    {
      const real_type a = block[0] + shift;
      const real_type b = block[1];
      const real_type c = block[2];
      const real_type d = block[3] + shift;
      const real_type det = a * d - b * c;
      if( std::abs( det ) <= pivot_tol || !std::isfinite( det ) )
      {
        if( failure_reason )
        {
          *failure_reason = !std::isfinite( det )
                            ? BlockInverseFailureReason::nonFiniteInput
                            : ( zero_norm ? BlockInverseFailureReason::zeroNorm
                                          : BlockInverseFailureReason::smallDeterminant );
        }
        return false;
      }
      const real_type inv_det = 1.0 / det;
      inverse[0] = d * inv_det;
      inverse[1] = -b * inv_det;
      inverse[2] = -c * inv_det;
      inverse[3] = a * inv_det;
      const bool finite_inverse =
          std::isfinite( inverse[0] ) && std::isfinite( inverse[1] ) &&
          std::isfinite( inverse[2] ) && std::isfinite( inverse[3] );
      if( !finite_inverse && failure_reason )
      {
        *failure_reason = BlockInverseFailureReason::nonFiniteInverse;
      }
      return finite_inverse;
    }

    const int_t width = 2 * m_blockSize;
    std::vector<real_type> aug( m_blockSize * width, 0.0 );
    for( int_t r = 0; r < m_blockSize; ++r )
    {
      for( int_t c = 0; c < m_blockSize; ++c )
      {
        aug[r * width + c] = block[r * m_blockSize + c] + ( r == c ? shift : 0.0 );
      }
      aug[r * width + m_blockSize + r] = 1.0;
    }

    for( int_t col = 0; col < m_blockSize; ++col )
    {
      int_t pivot_row = col;
      real_type pivot_abs = std::abs( aug[col * width + col] );
      for( int_t row = col + 1; row < m_blockSize; ++row )
      {
        const real_type candidate = std::abs( aug[row * width + col] );
        if( candidate > pivot_abs )
        {
          pivot_abs = candidate;
          pivot_row = row;
        }
      }
      if( pivot_abs <= pivot_tol || !std::isfinite( pivot_abs ) )
      {
        if( failure_reason )
        {
          *failure_reason = !std::isfinite( pivot_abs )
                            ? BlockInverseFailureReason::nonFiniteInput
                            : ( zero_norm ? BlockInverseFailureReason::zeroNorm
                                          : BlockInverseFailureReason::smallPivot );
        }
        return false;
      }
      if( pivot_row != col )
      {
        for( int_t j = 0; j < width; ++j )
        {
          std::swap( aug[col * width + j], aug[pivot_row * width + j] );
        }
      }

      const real_type pivot = aug[col * width + col];
      for( int_t j = 0; j < width; ++j )
      {
        aug[col * width + j] /= pivot;
      }
      for( int_t row = 0; row < m_blockSize; ++row )
      {
        if( row == col )
        {
          continue;
        }
        const real_type factor = aug[row * width + col];
        if( factor == 0.0 )
        {
          continue;
        }
        for( int_t j = 0; j < width; ++j )
        {
          aug[row * width + j] -= factor * aug[col * width + j];
        }
      }
    }

    for( int_t r = 0; r < m_blockSize; ++r )
    {
      for( int_t c = 0; c < m_blockSize; ++c )
      {
        inverse[r * m_blockSize + c] = aug[r * width + m_blockSize + c];
        if( !std::isfinite( inverse[r * m_blockSize + c] ) )
        {
          if( failure_reason )
          {
            *failure_reason = BlockInverseFailureReason::nonFiniteInverse;
          }
          return false;
        }
      }
    }
    return true;
  }

  bool invertBlockDiagonal( const real_type * block,
                            real_type * inverse,
                            int_t * used_diagonal_variables ) const
  {
    real_type norm = 0.0;
    for( int_t i = 0; i < m_blockSizeSquared; ++i )
    {
      if( !std::isfinite( block[i] ) )
      {
        return false;
      }
      norm = std::max( norm, std::abs( block[i] ) );
    }

    const real_type scale = std::max<real_type>( norm, 1.0 );
    const real_type diag_tol = m_fallbackDiagonalTolerance * scale;
    const real_type shift = m_pivotShift * scale;
    bool used_diagonal_inverse = false;
    int_t used_diagonal_count = 0;
    std::fill( inverse, inverse + m_blockSizeSquared, 0.0 );
    for( int_t i = 0; i < m_blockSize; ++i )
    {
      const real_type diagonal = block[i * m_blockSize + i] + shift;
      if( std::abs( diagonal ) > diag_tol && std::isfinite( diagonal ) )
      {
        inverse[i * m_blockSize + i] = 1.0 / diagonal;
        if( !std::isfinite( inverse[i * m_blockSize + i] ) )
        {
          return false;
        }
        used_diagonal_inverse = true;
        ++used_diagonal_count;
      }
      else
      {
        inverse[i * m_blockSize + i] = 1.0;
      }
    }
    if( used_diagonal_variables )
    {
      *used_diagonal_variables = used_diagonal_count;
    }
    return used_diagonal_inverse;
  }

  void recordShiftedDenseShift( real_type accepted_shift )
  {
    const real_type tolerance =
        std::numeric_limits<real_type>::epsilon() *
        std::max<real_type>( std::abs( accepted_shift ), 1.0 ) * 100.0;
    for( size_t i = 0; i < m_shiftedDenseShiftValues.size(); ++i )
    {
      if( std::abs( m_shiftedDenseShiftValues[i] - accepted_shift ) <= tolerance )
      {
        ++m_shiftedDenseShiftCounts[i];
        return;
      }
    }
    m_shiftedDenseShiftValues.push_back( accepted_shift );
    m_shiftedDenseShiftCounts.push_back( 1 );
  }

  void recordFallbackLocation( int_t row )
  {
    if( row < m_numReservoirRows )
    {
      ++m_reservoirFallbackPivots;
    }
    else
    {
      ++m_wellFallbackPivots;
    }
  }

  void recordPrimaryFailure( BlockInverseFailureReason reason )
  {
    switch( reason )
    {
      case BlockInverseFailureReason::nonFiniteInput:
        ++m_primaryFailureNonFinite;
        break;
      case BlockInverseFailureReason::zeroNorm:
        ++m_primaryFailureZeroNorm;
        break;
      case BlockInverseFailureReason::smallPivot:
        ++m_primaryFailureSmallPivot;
        break;
      case BlockInverseFailureReason::smallDeterminant:
        ++m_primaryFailureSmallDeterminant;
        break;
      case BlockInverseFailureReason::nonFiniteInverse:
        ++m_primaryFailureNonFiniteInverse;
        break;
      case BlockInverseFailureReason::none:
      default:
        break;
    }
  }

  void rightMultiplyBlockInPlace( real_type * block, const real_type * right ) const
  {
    std::fill( m_blockWork.begin(), m_blockWork.end(), 0.0 );
    for( int_t r = 0; r < m_blockSize; ++r )
    {
      for( int_t c = 0; c < m_blockSize; ++c )
      {
        real_type sum = 0.0;
        for( int_t k = 0; k < m_blockSize; ++k )
        {
          sum += block[r * m_blockSize + k] * right[k * m_blockSize + c];
        }
        m_blockWork[r * m_blockSize + c] = sum;
      }
    }
    std::copy( m_blockWork.begin(), m_blockWork.end(), block );
  }

  void subtractBlockProduct( const real_type * left,
                             const real_type * right,
                             real_type * target ) const
  {
    for( int_t r = 0; r < m_blockSize; ++r )
    {
      for( int_t c = 0; c < m_blockSize; ++c )
      {
        real_type sum = 0.0;
        for( int_t k = 0; k < m_blockSize; ++k )
        {
          sum += left[r * m_blockSize + k] * right[k * m_blockSize + c];
        }
        target[r * m_blockSize + c] -= sum;
      }
    }
  }

  void multiplyBlockVector( const real_type * block,
                            const real_type * vector,
                            real_type * result ) const
  {
    for( int_t r = 0; r < m_blockSize; ++r )
    {
      real_type sum = 0.0;
      for( int_t c = 0; c < m_blockSize; ++c )
      {
        sum += block[r * m_blockSize + c] * vector[c];
      }
      result[r] = sum;
    }
  }

  void subtractBlockMatvec( const real_type * block,
                            const real_type * vector,
                            real_type * target ) const
  {
    for( int_t r = 0; r < m_blockSize; ++r )
    {
      real_type sum = 0.0;
      for( int_t c = 0; c < m_blockSize; ++c )
      {
        sum += block[r * m_blockSize + c] * vector[c];
      }
      target[r] -= sum;
    }
  }

  LocalPreconditionerType m_type = LocalPreconditionerType::none;
  int_t m_numRows = 0;
  int_t m_numReservoirRows = 0;
  int_t m_blockSize = 0;
  int_t m_blockSizeSquared = 0;
  real_type m_pivotShift = 0.0;
  LocalFallbackStrategy m_fallbackStrategy = LocalFallbackStrategy::identity;
  real_type m_fallbackDiagonalTolerance = 1.0e-4;
  real_type m_fallbackShiftMax = 1.0e-4;
  real_type m_fallbackShiftGrowth = 100.0;
  int_t m_failedPivots = 0;
  int_t m_shiftedDenseFallbackPivots = 0;
  int_t m_diagonalFallbackPivots = 0;
  int_t m_reservoirFallbackPivots = 0;
  int_t m_wellFallbackPivots = 0;
  int_t m_primaryFailureNonFinite = 0;
  int_t m_primaryFailureZeroNorm = 0;
  int_t m_primaryFailureSmallPivot = 0;
  int_t m_primaryFailureSmallDeterminant = 0;
  int_t m_primaryFailureNonFiniteInverse = 0;
  int_t m_boundedDiagonalInverseVariables = 0;
  int_t m_boundedDiagonalIdentityVariables = 0;
  std::vector<real_type> m_shiftedDenseShiftValues;
  std::vector<int_t> m_shiftedDenseShiftCounts;
  bool m_ready = false;
  std::vector<int_t> m_rowPtr;
  std::vector<int_t> m_colInd;
  std::vector<int_t> m_fillLevel;
  std::vector<int_t> m_diagInd;
  std::vector<int_t> m_originalRowPtr;
  std::vector<int_t> m_originalColInd;
  std::vector<real_type> m_originalValues;
  std::vector<real_type> m_luValues;
  std::vector<real_type> m_diagInverse;
  std::vector<real_type> m_forwardWork;
  mutable std::vector<real_type> m_blockWork;
  std::vector<real_type> m_backwardBlock;
};

LinearSolver::LinearSolver()
  : m_hasInitialGuess( false )
  , m_ijMatrix( nullptr )
  , m_ijRHS( nullptr )
  , m_ijSol( nullptr )
  , m_parMatrix( nullptr )
  , m_parRHS( nullptr )
  , m_parSol( nullptr )
  , m_cprPressureIJMatrix( nullptr )
  , m_cprPressureIJRHS( nullptr )
  , m_cprPressureIJSol( nullptr )
  , m_cprPressureParMatrix( nullptr )
  , m_cprPressureParRHS( nullptr )
  , m_cprPressureParSol( nullptr )
  , m_cprPressureAMG( nullptr )
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
  m_params.scalingType = ScalingType::physics;
  m_params.compositeMode = CompositePreconditionerMode::mgrOnly;
  m_params.localPreconditioner = LocalPreconditionerType::none;
  m_params.localPivotShift = 1.0e-12;
  m_params.localFallbackStrategy = LocalFallbackStrategy::identity;
  m_params.localFallbackDiagonalTolerance = 1.0e-4;
  m_params.localFallbackShiftMax = 1.0e-4;
  m_params.localFallbackShiftGrowth = 100.0;
  m_params.localCorrectionAlpha = 1.0;
  m_params.localCorrectionAdaptiveFallbackThreshold = -1.0;
  m_params.localCorrectionAdaptiveAlpha = 0.0;
  m_params.localCorrectionAdaptiveFallbackThresholdHigh = -1.0;
  m_params.localCorrectionAdaptiveAlphaHigh = 0.0;
  m_params.localCorrectionQualityGate = false;
  m_params.localCorrectionQualityMinAlpha = 0.0;
  m_params.useBCSRCPR = false;
  m_params.bcsrCPRReduction = BCSRCPRReductionType::trueIMPES;
  m_params.bcsrCPRPressureVariable = 0;
  m_params.bcsrCPRWeightMax = 1.0e6;
  m_params.bcsrCPRReuseAMGHierarchy = false;
  m_params.bcsrCPRAMGRebuildInterval = 1;
  m_params.bcsrCPRAdaptiveAMGRebuild = false;
  m_params.bcsrCPRAdaptiveLIThreshold = 80;
  m_params.bcsrCPRAdaptiveLIGrowthFactor = 2.0;
  m_params.bcsrCPRAdaptiveMinReuseSetups = 1;
  m_params.bcsrCPRAdaptiveMaxReuseSetups = 0;
  m_params.bcsrCPRAdaptivePressureOvershootThreshold = -1.0;
  m_params.bcsrCPRAdaptiveFinalProxyThreshold = -1.0;
  m_params.bcsrCPRAdaptiveFallbackThreshold = -1.0;
  m_params.pressureAMGMaxIter = 1;
  m_params.pressureAMGTolerance = 0.0;
  m_params.pressureAMGCoarsenType = 6;
  m_params.pressureAMGInterpType = 6;
  m_params.pressureAMGRelaxType = 6;
  m_params.pressureAMGAggNumLevels = 1;
  m_params.pressureAMGAggInterpType = 6;
  m_params.pressureAMGAggPMaxElmts = 20;
  m_params.pressureAMGRelaxOrder = 1;
  m_params.pressureAMGStrongThreshold = -1.0;
  m_params.pressureAMGTruncFactor = -1.0;
  m_params.pressureAMGPMaxElmts = -1;
  m_params.pressureAMGMaxLevels = 0;
  m_params.bcsrCPRPressureCorrectionAlpha = 1.0;
  m_params.bcsrCPRPressureCorrectionGuardThreshold = -1.0;
  m_params.bcsrCPRPressureCorrectionGuardMinAlpha = 0.0;
  m_params.bcsrCPRTransposeApply = false;
  m_params.bcsrCPRForwardSource = false;
  m_params.bcsrCPRDiagnostics = false;
  m_params.bcsrCPRDiagnosticApplyInterval = 0;
  m_params.bcsrCPRDiagnosticMatrixInterval = 0;
}

LinearSolver::~LinearSolver()
{
  cleanup();
}

void LinearSolver::init_timer_nodes(::timer_node* timer_setup, ::timer_node* timer_solve)
{
  m_timerSetup = timer_setup;
  m_timerSolve = timer_solve;
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

bool LinearSolver::scalingActive(int_t num_rows) const
{
  return m_params.scalingType != ScalingType::none &&
         m_rowScaling.size() == static_cast<size_t>( num_rows ) &&
         m_colScaling.size() == static_cast<size_t>( m_matrix.global_num_cols );
}

real_type LinearSolver::computeScaledVectorNorm(const real_type* values, int_t num_rows) const
{
  if( !values || num_rows <= 0 )
  {
    return 0.0;
  }

  const bool apply_scaling = scalingActive( num_rows );
  real_type sum = 0.0;
  for( int_t i = 0; i < num_rows; ++i )
  {
    real_type value = values[i];
    if( apply_scaling )
    {
      value *= m_rowScaling[i];
    }
    sum += value * value;
  }
  return std::sqrt( sum );
}

real_type LinearSolver::computeInitialResidualNorm(const real_type* rhs,
                                                   const real_type* initial_guess,
                                                   int_t num_rows) const
{
  if( !rhs || num_rows <= 0 )
  {
    return 0.0;
  }

  if( !initial_guess )
  {
    return computeScaledVectorNorm( rhs, num_rows );
  }

  const int_t block_size = m_matrix.block_size;
  if( block_size <= 0 || m_matrix.num_rows <= 0 )
  {
    return computeScaledVectorNorm( rhs, num_rows );
  }

  const bool apply_scaling = scalingActive( num_rows );
  real_type sum = 0.0;
  for( int_t cell = 0; cell < m_matrix.num_rows; ++cell )
  {
    for( int_t i = 0; i < block_size; ++i )
    {
      const int_t global_row = cell * block_size + i;
      real_type ax = 0.0;
      for( int_t block_idx = m_matrix.row_ptr[cell];
           block_idx < m_matrix.row_ptr[cell + 1];
           ++block_idx )
      {
        const int_t col_cell = m_matrix.col_ind[block_idx];
        const int_t block_start = block_idx * block_size * block_size;
        for( int_t j = 0; j < block_size; ++j )
        {
          const int_t global_col = col_cell * block_size + j;
          ax += m_matrix.values[block_start + i * block_size + j] * initial_guess[global_col];
        }
      }

      real_type residual = rhs[global_row] - ax;
      if( apply_scaling )
      {
        residual *= m_rowScaling[global_row];
      }
      sum += residual * residual;
    }
  }

  return std::sqrt( sum );
}

void LinearSolver::updateResidualNormalization(const real_type* rhs,
                                               const real_type* initial_guess,
                                               int_t num_rows)
{
  m_lastRhsNorm = computeScaledVectorNorm( rhs, num_rows );
  m_lastInitialResidualNorm = computeInitialResidualNorm( rhs, initial_guess, num_rows );
  m_lastResidualDenominator =
      ( m_lastRhsNorm > 0.0 ) ? m_lastRhsNorm : m_lastInitialResidualNorm;
}

real_type LinearSolver::normalizeFinalResidual(real_type hypre_final_residual) const
{
  if( m_lastRhsNorm > 0.0 )
  {
    return hypre_final_residual;
  }

  if( m_lastResidualDenominator > 0.0 )
  {
    return hypre_final_residual / m_lastResidualDenominator;
  }

  return hypre_final_residual;
}

bool LinearSolver::recordKrylovOutcome( SolverResults& results,
                                        HYPRE_Int setup_rc, HYPRE_Int solve_rc,
                                        HYPRE_Int iters_rc, HYPRE_Int resid_rc,
                                        int_t num_iterations, real_type final_res_norm,
                                        const char* variant )
{
  // HYPRE_ERROR_CONV is NOT a malfunction: hypre_error(HYPRE_ERROR_CONV) is
  // raised deliberately when GMRES/FlexGMRES stops at max_iter without reaching
  // the tolerance (krylov/gmres.c, krylov/flexgmres.c). The statistics getters
  // still write valid values -- they merely return the same latched flag. So
  // mask that bit out and judge only the remaining ones; treating it as a
  // backend error would discard a perfectly good iterate and make ordinary
  // exhaustion unreachable for the +1 ("usable") status.
  const HYPRE_Int hypre_err = HYPRE_GetError();
  const HYPRE_Int conv_bit = static_cast< HYPRE_Int >( HYPRE_ERROR_CONV );
  const HYPRE_Int real_err = ( setup_rc | solve_rc | iters_rc | resid_rc | hypre_err )
                             & ~conv_bit;
  // Consume the latched state either way, so it cannot leak into the next solve.
  HYPRE_ClearAllErrors();

  if( real_err )
  {
    std::cerr << "[MGR] " << variant
              << ": HYPRE reported an error (setup=" << setup_rc
              << ", solve=" << solve_rc << ", iters=" << iters_rc
              << ", resid=" << resid_rc << ", hypre_error=" << hypre_err
              << ", non-convergence bits=" << real_err
              << "); solver statistics are not trustworthy" << std::endl;
    results.iterations = 0;
    results.finalResidual = std::numeric_limits< real_type >::infinity();
    results.converged = false;
    results.stopReason = SolverStopReason::backendError;
    return false;
  }

  if( !std::isfinite( static_cast< double >( final_res_norm ) ) )
  {
    std::cerr << "[MGR] " << variant << ": non-finite final residual" << std::endl;
    results.iterations = num_iterations;
    results.finalResidual = std::numeric_limits< real_type >::infinity();
    results.converged = false;
    results.stopReason = SolverStopReason::breakdown;
    return false;
  }

  results.iterations = num_iterations;
  setConvergenceFromResidual( results, final_res_norm );
  // Statistics are trustworthy; leave stopReason == unclassified so the wrapper
  // assigns converged / iterationLimit / breakdown from them.
  results.stopReason = SolverStopReason::unclassified;
  return true;
}

void LinearSolver::setConvergenceFromResidual(SolverResults& results,
                                              real_type hypre_final_residual) const
{
  const real_type reported_residual = normalizeFinalResidual( hypre_final_residual );
  results.finalResidual = reported_residual;

  const real_type machine_epsilon = std::numeric_limits<real_type>::epsilon() * 100.0;
  if( reported_residual < m_params.tolerance )
  {
    results.converged = true;
  }
  else if( reported_residual < machine_epsilon )
  {
    results.converged = true;
    std::cout << "[MGR] Warning: Residual (" << reported_residual << ") is above tolerance ("
              << m_params.tolerance << ") but at machine precision. Considering converged.\n";
  }
  else
  {
    results.converged = false;
  }
}

::timer_node* LinearSolver::setupTimerNode(const std::string& name) const
{
  if( !m_timerSetup )
  {
    return nullptr;
  }
  return &m_timerSetup->node["MGR"].node[name];
}

::timer_node* LinearSolver::solveTimerNode(const std::string& krylov_name,
                                           const std::string& name) const
{
  if( !m_timerSolve )
  {
    return nullptr;
  }
  return &m_timerSolve->node[krylov_name].node[name];
}

void LinearSolver::clearCompositeWorkVectors()
{
  m_compositeResidual.clear();
  m_compositeAx.clear();
  m_compositeCorrection.clear();
}

void LinearSolver::clearHYPRESystemObjects()
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
  m_hypreSystemDirectUpdateReady = false;
  m_hypreSystemParCSRDiagDataIndex.clear();
  m_matrixLoaded = false;
  m_matrixAssembled = false;
  m_activeMGRPrecond = nullptr;
  m_activeKrylovName.clear();
}

bool LinearSolver::blockLocalPreconditionerReady() const
{
  return m_blockLocalPreconditioner && m_blockLocalPreconditioner->ready();
}

void LinearSolver::logMGRConfigurationOnce(const char* stage)
{
  if( m_mgrConfigurationLogged )
  {
    return;
  }
  m_mgrConfigurationLogged = true;

  std::ostream & config_log = std::cout;
  config_log << "[MGR] Effective MGR/BCSR CPR configuration"
            << " (" << ( stage ? stage : "setup" ) << "):" << std::endl;
  config_log << "[MGR]   solver: use_mgr=" << boolLabel( m_params.useMGR )
            << ", krylov=" << krylovLabel( m_params.krylovType )
            << ", max_iter=" << m_params.maxIter
            << ", tolerance=" << m_params.tolerance
            << ", kdim=" << m_params.kdim
            << ", log_level=" << m_params.logLevel
            << "." << std::endl;
  config_log << "[MGR]   matrix: block_rows=" << m_matrix.num_rows
            << ", block_cols=" << m_matrix.num_cols
            << ", block_size=" << m_matrix.block_size
            << ", scalar_rows=" << m_matrix.global_num_rows
            << ", nnz_blocks=" << m_matrix.num_nonzero_blocks
            << ", reservoir_blocks=" << m_params.localReservoirBlockCount
            << ", cpr_pressure_rows=" << m_cprPressureRows
            << "." << std::endl;
  config_log << "[MGR]   scaling: enabled="
            << boolLabel( m_params.usePhysicsScaling )
            << ", type=" << scalingLabel( m_params.scalingType )
            << "(" << static_cast<int_t>( m_params.scalingType ) << ")"
            << "." << std::endl;
  config_log << "[MGR]   composite/local: mode="
            << compositeModeLabel( m_params.compositeMode )
            << "(" << static_cast<int_t>( m_params.compositeMode ) << ")"
            << ", local_solver="
            << localPreconditionerLabel( m_params.localPreconditioner )
            << "(" << static_cast<int_t>( m_params.localPreconditioner ) << ")"
            << ", pivot_shift=" << m_params.localPivotShift
            << ", fallback=" << localFallbackLabel( m_params.localFallbackStrategy )
            << "(" << static_cast<int_t>( m_params.localFallbackStrategy ) << ")"
            << ", diagonal_tolerance="
            << m_params.localFallbackDiagonalTolerance
            << ", shifted_max=" << m_params.localFallbackShiftMax
            << ", shifted_growth=" << m_params.localFallbackShiftGrowth
            << "." << std::endl;
  config_log << "[MGR]   local_correction: alpha="
            << m_params.localCorrectionAlpha
            << ", adaptive_fallback_threshold="
            << m_params.localCorrectionAdaptiveFallbackThreshold
            << ", adaptive_alpha=" << m_params.localCorrectionAdaptiveAlpha
            << ", adaptive_fallback_threshold_high="
            << m_params.localCorrectionAdaptiveFallbackThresholdHigh
            << ", adaptive_alpha_high="
            << m_params.localCorrectionAdaptiveAlphaHigh
            << ", quality_gate="
            << boolLabel( m_params.localCorrectionQualityGate )
            << ", quality_min_alpha="
            << m_params.localCorrectionQualityMinAlpha
            << "." << std::endl;
  config_log << "[MGR]   bcsr_cpr: enabled="
            << boolLabel( m_params.useBCSRCPR )
            << ", reduction="
            << bcsrCPRReductionLabel( m_params.bcsrCPRReduction )
            << "(" << static_cast<int_t>( m_params.bcsrCPRReduction ) << ")"
            << ", pressure_variable=" << m_params.bcsrCPRPressureVariable
            << ", weight_max=" << m_params.bcsrCPRWeightMax
            << "." << std::endl;
  config_log << "[MGR]   bcsr_cpr_reuse: reuse_amg="
            << boolLabel( m_params.bcsrCPRReuseAMGHierarchy )
            << ", rebuild_interval=" << m_params.bcsrCPRAMGRebuildInterval
            << ", adaptive_rebuild="
            << boolLabel( m_params.bcsrCPRAdaptiveAMGRebuild )
            << ", li_threshold=" << m_params.bcsrCPRAdaptiveLIThreshold
            << ", li_growth_factor="
            << m_params.bcsrCPRAdaptiveLIGrowthFactor
            << ", min_reuse_setups="
            << m_params.bcsrCPRAdaptiveMinReuseSetups
            << ", max_reuse_setups="
            << m_params.bcsrCPRAdaptiveMaxReuseSetups
            << "." << std::endl;
  config_log << "[MGR]   bcsr_cpr_quality: pressure_overshoot_threshold="
            << m_params.bcsrCPRAdaptivePressureOvershootThreshold
            << ", final_proxy_threshold="
            << m_params.bcsrCPRAdaptiveFinalProxyThreshold
            << ", fallback_threshold="
            << m_params.bcsrCPRAdaptiveFallbackThreshold
            << "." << std::endl;
  config_log << "[MGR]   pressure_amg: coarsen_type="
            << m_params.pressureAMGCoarsenType
            << ", interp_type=" << m_params.pressureAMGInterpType
            << ", relax_type=" << m_params.pressureAMGRelaxType
            << ", agg_num_levels=" << m_params.pressureAMGAggNumLevels
            << ", agg_interp_type=" << m_params.pressureAMGAggInterpType
            << ", agg_pmax_elmts=" << m_params.pressureAMGAggPMaxElmts
            << ", relax_order=" << m_params.pressureAMGRelaxOrder
            << ", strong_threshold=" << m_params.pressureAMGStrongThreshold
            << ", trunc_factor=" << m_params.pressureAMGTruncFactor
            << ", pmax_elmts=" << m_params.pressureAMGPMaxElmts
            << ", max_levels=" << m_params.pressureAMGMaxLevels
            << ", max_iter=" << m_params.pressureAMGMaxIter
            << ", tolerance=" << m_params.pressureAMGTolerance
            << "." << std::endl;
  config_log << "[MGR]   pressure_correction: alpha="
            << m_params.bcsrCPRPressureCorrectionAlpha
            << ", guard_threshold="
            << m_params.bcsrCPRPressureCorrectionGuardThreshold
            << ", guard_min_alpha="
            << m_params.bcsrCPRPressureCorrectionGuardMinAlpha
            << ", transpose_apply="
            << boolLabel( m_params.bcsrCPRTransposeApply )
            << ", forward_source="
            << boolLabel( m_params.bcsrCPRForwardSource )
            << "." << std::endl;
  config_log << "[MGR]   diagnostics: enabled="
            << boolLabel( m_params.bcsrCPRDiagnostics )
            << ", apply_interval="
            << m_params.bcsrCPRDiagnosticApplyInterval
            << ", matrix_interval="
            << m_params.bcsrCPRDiagnosticMatrixInterval
            << "." << std::endl;
}

bool LinearSolver::bcsrCPRPreconditionerReady() const
{
  return m_bcsrCPRReady && m_cprPressureAMG && m_cprPressureParMatrix &&
         m_cprPressureParRHS && m_cprPressureParSol && blockLocalPreconditionerReady();
}

void LinearSolver::clearBCSRCPRPreconditioner()
{
  const bool had_state = m_cprPressureAMG || m_cprPressureIJMatrix ||
                         m_cprPressureIJRHS || m_cprPressureIJSol ||
                         m_bcsrCPRReady || m_cprPressureRows > 0;
  if( had_state )
  {
    ++m_cprClearCount;
  }

  if( m_cprPressureAMG )
  {
    HYPRE_BoomerAMGDestroy( m_cprPressureAMG );
    m_cprPressureAMG = nullptr;
  }
  if( m_cprPressureIJMatrix )
  {
    HYPRE_IJMatrixDestroy( m_cprPressureIJMatrix );
    m_cprPressureIJMatrix = nullptr;
  }
  if( m_cprPressureIJRHS )
  {
    HYPRE_IJVectorDestroy( m_cprPressureIJRHS );
    m_cprPressureIJRHS = nullptr;
  }
  if( m_cprPressureIJSol )
  {
    HYPRE_IJVectorDestroy( m_cprPressureIJSol );
    m_cprPressureIJSol = nullptr;
  }
  m_cprPressureParMatrix = nullptr;
  m_cprPressureParRHS = nullptr;
  m_cprPressureParSol = nullptr;
  m_bcsrCPRReady = false;
  m_cprPressureRows = 0;
  m_cprPressureSetupCount = 0;
  m_cprPressurePatternReady = false;
  m_cprPressureMatrixAssembled = false;
  m_cprPressureVectorsReady = false;
  m_cprPressureAMGSetupDone = false;
  m_cprSetupsSinceAMGSetup = 0;
  m_cprLastLinearIterations = -1;
  m_cprLastAMGSetupLinearIterations = -1;
  m_cprLastLinearConverged = true;
  m_cprAMGSetupForCurrentSolve = false;
  m_cprAdaptiveQualityRebuildRequested = false;
  m_cprLastPressureOvershootRel = 0.0;
  m_cprLastFinalProxyRel = 0.0;
  m_cprLastFallbackRatio = 0.0;
  m_cprLastAMGRebuildReason.clear();
  m_cprPressureDirectUpdateReady = false;
  m_cprApplyCount = 0;
  m_cprPressureMatrixDiagnosticCount = 0;
  m_cprWeightTrueIMPESRows = 0;
  m_cprWeightFallbackRows = 0;
  m_cprWeightMissingDiagRows = 0;
  m_cprWeightSolveFailureRows = 0;
  m_cprWeightNonFiniteRows = 0;
  m_cprWeightLimitedRows = 0;
  m_cprWeightMaxAbs = 0.0;
  m_cprPressureParCSRDiagDataIndex.clear();
  m_cprPressureWeights.clear();
  m_cprPressureRowIndices.clear();
  m_cprPressureRowNCols.clear();
  m_cprPressureRowOffsets.clear();
  m_cprPressureCols.clear();
  m_cprPressureValues.clear();
  m_cprPressureRHSValues.clear();
  m_cprPressureSolution.clear();
  m_cprPressureCorrection.clear();
  m_cprResidual.clear();
  m_cprAx.clear();
  m_cprLocalCorrection.clear();
}

bool LinearSolver::bcsrCPRSourceMatrixCompatible() const
{
  if( !m_bcsrCPRSourceMatrixReady )
  {
    return false;
  }
  if( m_bcsrCPRSourceMatrix.num_rows != m_matrix.num_rows ||
      m_bcsrCPRSourceMatrix.num_cols != m_matrix.num_cols ||
      m_bcsrCPRSourceMatrix.block_size != m_matrix.block_size ||
      m_bcsrCPRSourceMatrix.global_num_rows != m_matrix.global_num_rows ||
      m_bcsrCPRSourceMatrix.global_num_cols != m_matrix.global_num_cols )
  {
    return false;
  }
  if( m_bcsrCPRSourceMatrix.num_rows < m_cprPressureRows ||
      static_cast<int_t>( m_bcsrCPRSourceMatrix.row_ptr.size() ) !=
          m_bcsrCPRSourceMatrix.num_rows + 1 ||
      static_cast<int_t>( m_bcsrCPRSourceMatrix.col_ind.size() ) !=
          m_bcsrCPRSourceMatrix.num_nonzero_blocks )
  {
    return false;
  }
  const int_t expected_values =
      m_bcsrCPRSourceMatrix.num_nonzero_blocks *
      m_bcsrCPRSourceMatrix.block_size *
      m_bcsrCPRSourceMatrix.block_size;
  return static_cast<int_t>( m_bcsrCPRSourceMatrix.values.size() ) ==
         expected_values;
}

void LinearSolver::transposeBCSRCPRPressureMatrix()
{
  const int_t rows = m_cprPressureRows;
  if( rows <= 0 || static_cast<int_t>( m_cprPressureRowOffsets.size() ) != rows + 1 )
  {
    return;
  }

  const int_t nnz = m_cprPressureRowOffsets[rows];
  if( nnz <= 0 ||
      static_cast<int_t>( m_cprPressureCols.size() ) != nnz ||
      static_cast<int_t>( m_cprPressureValues.size() ) != nnz )
  {
    return;
  }

  std::vector<int_t> transpose_counts( rows, 0 );
  for( int_t k = 0; k < nnz; ++k )
  {
    const bigint_t col_big = m_cprPressureCols[k];
    if( col_big >= 0 && col_big < rows )
    {
      ++transpose_counts[static_cast<int_t>( col_big )];
    }
  }

  std::vector<int_t> transpose_offsets( rows + 1, 0 );
  for( int_t row = 0; row < rows; ++row )
  {
    transpose_offsets[row + 1] = transpose_offsets[row] + transpose_counts[row];
  }

  std::vector<bigint_t> transpose_cols( nnz, 0 );
  std::vector<real_type> transpose_values( nnz, 0.0 );
  std::vector<int_t> next = transpose_offsets;
  for( int_t row = 0; row < rows; ++row )
  {
    for( int_t k = m_cprPressureRowOffsets[row];
         k < m_cprPressureRowOffsets[row + 1];
         ++k )
    {
      const bigint_t col_big = m_cprPressureCols[k];
      if( col_big < 0 || col_big >= rows )
      {
        continue;
      }
      const int_t col = static_cast<int_t>( col_big );
      const int_t out = next[col]++;
      transpose_cols[out] = row;
      transpose_values[out] = m_cprPressureValues[k];
    }
  }

  for( int_t row = 0; row < rows; ++row )
  {
    const int_t begin = transpose_offsets[row];
    const int_t end = transpose_offsets[row + 1];
    std::vector<int_t> order( end - begin );
    for( int_t i = 0; i < end - begin; ++i )
    {
      order[i] = begin + i;
    }
    std::sort( order.begin(), order.end(),
               [&]( int_t lhs, int_t rhs )
               {
                 return transpose_cols[lhs] < transpose_cols[rhs];
               } );
    std::vector<bigint_t> sorted_cols( order.size(), 0 );
    std::vector<real_type> sorted_values( order.size(), 0.0 );
    for( int_t i = 0; i < static_cast<int_t>( order.size() ); ++i )
    {
      sorted_cols[i] = transpose_cols[order[i]];
      sorted_values[i] = transpose_values[order[i]];
    }
    for( int_t i = 0; i < static_cast<int_t>( order.size() ); ++i )
    {
      transpose_cols[begin + i] = sorted_cols[i];
      transpose_values[begin + i] = sorted_values[i];
    }
  }

  m_cprPressureRowNCols = transpose_counts;
  m_cprPressureRowOffsets = std::move( transpose_offsets );
  m_cprPressureCols = std::move( transpose_cols );
  m_cprPressureValues = std::move( transpose_values );
  m_cprPressureDirectUpdateReady = false;
  m_cprPressureParCSRDiagDataIndex.clear();
}

void LinearSolver::recordBCSRCPRLinearIterations(int_t iterations, bool converged)
{
  m_cprLastLinearIterations = iterations;
  m_cprLastLinearConverged = converged;
  if( m_cprAMGSetupForCurrentSolve )
  {
    m_cprLastAMGSetupLinearIterations = iterations;
  }
}

void LinearSolver::computeBCSRCPRPressureWeights()
{
  const int_t block_size = m_matrix.block_size;
  const int_t pressure_var =
      std::clamp<int_t>( m_params.bcsrCPRPressureVariable, 0, block_size - 1 );
  const bool apply_scaling = scalingActive( m_matrix.global_num_rows );
  const real_type weight_max = std::max<real_type>( m_params.bcsrCPRWeightMax, 1.0 );

  m_cprPressureWeights.assign( m_cprPressureRows * block_size, 0.0 );
  m_cprWeightTrueIMPESRows = 0;
  m_cprWeightFallbackRows = 0;
  m_cprWeightMissingDiagRows = 0;
  m_cprWeightSolveFailureRows = 0;
  m_cprWeightNonFiniteRows = 0;
  m_cprWeightLimitedRows = 0;
  m_cprWeightMaxAbs = 0.0;

  for( int_t row = 0; row < m_cprPressureRows; ++row )
  {
    real_type * weights = &m_cprPressureWeights[row * block_size];
    weights[pressure_var] = 1.0;

    const bool use_true_impes_weights =
        m_params.bcsrCPRReduction == BCSRCPRReductionType::trueIMPES ||
        m_params.bcsrCPRReduction == BCSRCPRReductionType::trueIMPESWellElimination;
    if( !use_true_impes_weights ||
        block_size <= 1 )
    {
      continue;
    }

    int_t diag = -1;
    if( static_cast<int_t>( m_matrix.diag_ind.size() ) > row )
    {
      const int_t candidate = m_matrix.diag_ind[row];
      if( candidate >= m_matrix.row_ptr[row] &&
          candidate < m_matrix.row_ptr[row + 1] &&
          m_matrix.col_ind[candidate] == row )
      {
        diag = candidate;
      }
    }
    if( diag < 0 )
    {
      for( int_t p = m_matrix.row_ptr[row]; p < m_matrix.row_ptr[row + 1]; ++p )
      {
        if( m_matrix.col_ind[p] == row )
        {
          diag = p;
          break;
        }
      }
    }
    if( diag < 0 )
    {
      ++m_cprWeightFallbackRows;
      ++m_cprWeightMissingDiagRows;
      continue;
    }

    std::vector<int_t> f_vars;
    f_vars.reserve( block_size - 1 );
    for( int_t v = 0; v < block_size; ++v )
    {
      if( v != pressure_var )
      {
        f_vars.push_back( v );
      }
    }
    const int_t n_f = static_cast<int_t>( f_vars.size() );
    std::vector<real_type> matrix_ff_t( n_f * n_f, 0.0 );
    std::vector<real_type> rhs( n_f, 0.0 );

    real_type norm = 0.0;
    const int_t block_offset = diag * block_size * block_size;

    for( int_t a = 0; a < n_f; ++a )
    {
      const int_t f_col = f_vars[a];
      const int_t scalar_row = row * block_size + pressure_var;
      const int_t scalar_col = row * block_size + f_col;
      const real_type row_scale =
          apply_scaling && scalar_row < static_cast<int_t>( m_rowScaling.size() )
          ? m_rowScaling[scalar_row] : 1.0;
      const real_type col_scale =
          apply_scaling && scalar_col < static_cast<int_t>( m_colScaling.size() )
          ? m_colScaling[scalar_col] : 1.0;
      rhs[a] = -m_matrix.values[block_offset + pressure_var * block_size + f_col] *
               row_scale * col_scale;
      // Track the RHS magnitude as part of the dense-system norm: the pivot
      // tolerance must reflect the scale of the matrix the dense solver
      // actually sees, NOT the full A_pp+A_pf+A_fp+A_ff diagonal block.
      // (Including A_pp inflated the norm by orders of magnitude for
      // compositional physics, which made any well-conditioned f-block
      // pivot ~1 look "too small" and dense_solve failed on every row.)
      norm = std::max( norm, std::abs( rhs[a] ) );
      for( int_t b = 0; b < n_f; ++b )
      {
        const int_t f_row = f_vars[b];
        const int_t ff_scalar_row = row * block_size + f_row;
        const int_t ff_scalar_col = row * block_size + f_col;
        const real_type ff_row_scale =
            apply_scaling && ff_scalar_row < static_cast<int_t>( m_rowScaling.size() )
            ? m_rowScaling[ff_scalar_row] : 1.0;
        const real_type ff_col_scale =
            apply_scaling && ff_scalar_col < static_cast<int_t>( m_colScaling.size() )
            ? m_colScaling[ff_scalar_col] : 1.0;
        const real_type ff_value =
            m_matrix.values[block_offset + f_row * block_size + f_col] *
            ff_row_scale * ff_col_scale;
        matrix_ff_t[a * n_f + b] = ff_value;
        norm = std::max( norm, std::abs( ff_value ) );
      }
    }

    std::vector<real_type> solution;
    const real_type pivot_tolerance =
        std::numeric_limits<real_type>::epsilon() * std::max<real_type>( norm, 1.0 ) * 100.0;
    const bool ok = solveDenseLinearSystem( matrix_ff_t,
                                            rhs,
                                            n_f,
                                            pivot_tolerance,
                                            solution );
    if( !ok )
    {
      ++m_cprWeightFallbackRows;
      ++m_cprWeightSolveFailureRows;
      continue;
    }

    bool accept = true;
    bool nonfinite_weight = false;
    bool limited_weight = false;
    for( int_t a = 0; a < n_f; ++a )
    {
      const real_type abs_weight = std::abs( solution[a] );
      if( std::isfinite( abs_weight ) )
      {
        m_cprWeightMaxAbs = std::max( m_cprWeightMaxAbs, abs_weight );
      }
      if( !std::isfinite( solution[a] ) )
      {
        nonfinite_weight = true;
        accept = false;
      }
      else if( abs_weight > weight_max )
      {
        limited_weight = true;
        accept = false;
      }
    }
    if( !accept )
    {
      ++m_cprWeightFallbackRows;
      if( nonfinite_weight )
      {
        ++m_cprWeightNonFiniteRows;
      }
      if( limited_weight )
      {
        ++m_cprWeightLimitedRows;
      }
      continue;
    }

    for( int_t a = 0; a < n_f; ++a )
    {
      weights[f_vars[a]] = solution[a];
    }
    ++m_cprWeightTrueIMPESRows;
  }

  if( m_params.bcsrCPRDiagnostics )
  {
    std::cout << "[MGR] BCSR CPR pressure weight diagnostics: rows="
              << m_cprPressureRows
              << ", true_impes=" << m_cprWeightTrueIMPESRows
              << ", fallback=" << m_cprWeightFallbackRows
              << ", missing_diag=" << m_cprWeightMissingDiagRows
              << ", dense_solve_failure=" << m_cprWeightSolveFailureRows
              << ", nonfinite_weight=" << m_cprWeightNonFiniteRows
              << ", weight_limited=" << m_cprWeightLimitedRows
              << ", max_abs_weight=" << m_cprWeightMaxAbs
              << "." << std::endl;
  }
  else if( m_params.logLevel >= 1 )
  {
    std::cout << "[MGR] BCSR CPR pressure weights: rows=" << m_cprPressureRows
              << ", true_impes=" << m_cprWeightTrueIMPESRows
              << ", pressure_row_fallback=" << m_cprWeightFallbackRows
              << "." << std::endl;
  }
}

bool LinearSolver::bcsrCPRUsesWellElimination() const
{
  return m_params.bcsrCPRReduction ==
         BCSRCPRReductionType::trueIMPESWellElimination;
}

int_t LinearSolver::findBCSRBlock(int_t row, int_t col) const
{
  if( row < 0 || row >= m_matrix.num_rows )
  {
    return -1;
  }
  for( int_t block = m_matrix.row_ptr[row]; block < m_matrix.row_ptr[row + 1]; ++block )
  {
    if( m_matrix.col_ind[block] == col )
    {
      return block;
    }
  }
  return -1;
}

int_t LinearSolver::findBCSRCPRPressureColumnPosition(int_t row, int_t col) const
{
  if( row < 0 || row >= m_cprPressureRows )
  {
    return -1;
  }
  for( int_t pos = m_cprPressureRowOffsets[row];
       pos < m_cprPressureRowOffsets[row + 1]; ++pos )
  {
    if( m_cprPressureCols[pos] == col )
    {
      return pos;
    }
  }
  return -1;
}

void LinearSolver::addBCSRCPRPressureValue(int_t row, int_t col, real_type value)
{
  const int_t pos = findBCSRCPRPressureColumnPosition( row, col );
  if( pos >= 0 )
  {
    m_cprPressureValues[pos] += value;
  }
  else
  {
    ++m_cprWellEliminationMissingPattern;
  }
}

real_type LinearSolver::computeBCSRCPRProjectedPressureValue(
    int_t row,
    int_t block,
    int_t pressure_var,
    bool apply_scaling) const
{
  const int_t block_size = m_matrix.block_size;
  const int_t col_cell = m_matrix.col_ind[block];
  const int_t block_offset = block * block_size * block_size;
  const real_type * weights = &m_cprPressureWeights[row * block_size];
  real_type value = 0.0;
  for( int_t r = 0; r < block_size; ++r )
  {
    const int_t scalar_row = row * block_size + r;
    const int_t scalar_col = col_cell * block_size + pressure_var;
    const real_type row_scale =
        apply_scaling && scalar_row < static_cast<int_t>( m_rowScaling.size() )
        ? m_rowScaling[scalar_row] : 1.0;
    const real_type col_scale =
        apply_scaling && scalar_col < static_cast<int_t>( m_colScaling.size() )
        ? m_colScaling[scalar_col] : 1.0;
    value += weights[r] *
             m_matrix.values[block_offset + r * block_size + pressure_var] *
             row_scale * col_scale;
  }
  return value;
}

bool LinearSolver::computeBCSRCPRScaledBlockInverse(
    int_t row,
    int_t block,
    std::vector<real_type> & inverse) const
{
  const int_t block_size = m_matrix.block_size;
  const int_t col_cell = m_matrix.col_ind[block];
  const int_t block_offset = block * block_size * block_size;
  const bool apply_scaling = scalingActive( m_matrix.global_num_rows );

  std::vector<real_type> matrix( block_size * block_size, 0.0 );
  real_type norm = 0.0;
  for( int_t r = 0; r < block_size; ++r )
  {
    const int_t scalar_row = row * block_size + r;
    const real_type row_scale =
        apply_scaling && scalar_row < static_cast<int_t>( m_rowScaling.size() )
        ? m_rowScaling[scalar_row] : 1.0;
    for( int_t c = 0; c < block_size; ++c )
    {
      const int_t scalar_col = col_cell * block_size + c;
      const real_type col_scale =
          apply_scaling && scalar_col < static_cast<int_t>( m_colScaling.size() )
          ? m_colScaling[scalar_col] : 1.0;
      const real_type value =
          m_matrix.values[block_offset + r * block_size + c] * row_scale * col_scale;
      matrix[r * block_size + c] = value;
      norm = std::max( norm, std::abs( value ) );
    }
  }

  const real_type pivot_tolerance =
      std::numeric_limits<real_type>::epsilon() * std::max<real_type>( norm, 1.0 ) * 100.0;
  inverse.assign( block_size * block_size, 0.0 );
  std::vector<real_type> rhs( block_size, 0.0 );
  std::vector<real_type> solution;
  for( int_t col = 0; col < block_size; ++col )
  {
    std::fill( rhs.begin(), rhs.end(), 0.0 );
    rhs[col] = 1.0;
    if( !solveDenseLinearSystem( matrix,
                                 rhs,
                                 block_size,
                                 pivot_tolerance,
                                 solution ) )
    {
      inverse.clear();
      return false;
    }
    for( int_t row_local = 0; row_local < block_size; ++row_local )
    {
      inverse[row_local * block_size + col] = solution[row_local];
    }
  }
  return true;
}

void LinearSolver::buildBCSRCPRPressurePattern()
{
  m_cprPressurePatternReady = false;
  m_cprPressureRowIndices.resize( m_cprPressureRows );
  m_cprPressureRowNCols.assign( m_cprPressureRows, 0 );
  m_cprPressureRowOffsets.assign( m_cprPressureRows + 1, 0 );
  const bool use_well_elimination = bcsrCPRUsesWellElimination();
  std::vector<int_t> row_cols;

  for( int_t row = 0; row < m_cprPressureRows; ++row )
  {
    m_cprPressureRowIndices[row] = row;
    row_cols.clear();
    for( int_t block = m_matrix.row_ptr[row]; block < m_matrix.row_ptr[row + 1]; ++block )
    {
      const int_t col_cell = m_matrix.col_ind[block];
      if( col_cell >= 0 && col_cell < m_cprPressureRows )
      {
        row_cols.push_back( col_cell );
      }
      else if( use_well_elimination && col_cell >= m_cprPressureRows &&
               col_cell < m_matrix.num_rows )
      {
        for( int_t well_block = m_matrix.row_ptr[col_cell];
             well_block < m_matrix.row_ptr[col_cell + 1]; ++well_block )
        {
          const int_t reservoir_col = m_matrix.col_ind[well_block];
          if( reservoir_col >= 0 && reservoir_col < m_cprPressureRows )
          {
            row_cols.push_back( reservoir_col );
          }
        }
      }
    }
    std::sort( row_cols.begin(), row_cols.end() );
    row_cols.erase( std::unique( row_cols.begin(), row_cols.end() ), row_cols.end() );
    if( row_cols.empty() )
    {
      row_cols.push_back( row );
    }
    const int_t count = static_cast<int_t>( row_cols.size() );
    m_cprPressureRowNCols[row] = count;
    m_cprPressureRowOffsets[row + 1] = m_cprPressureRowOffsets[row] + count;
  }

  const int_t nnz = m_cprPressureRowOffsets[m_cprPressureRows];
  m_cprPressureCols.assign( nnz, 0 );
  m_cprPressureValues.assign( nnz, 0.0 );

  for( int_t row = 0; row < m_cprPressureRows; ++row )
  {
    row_cols.clear();
    for( int_t block = m_matrix.row_ptr[row]; block < m_matrix.row_ptr[row + 1]; ++block )
    {
      const int_t col_cell = m_matrix.col_ind[block];
      if( col_cell >= 0 && col_cell < m_cprPressureRows )
      {
        row_cols.push_back( col_cell );
      }
      else if( use_well_elimination && col_cell >= m_cprPressureRows &&
               col_cell < m_matrix.num_rows )
      {
        for( int_t well_block = m_matrix.row_ptr[col_cell];
             well_block < m_matrix.row_ptr[col_cell + 1]; ++well_block )
        {
          const int_t reservoir_col = m_matrix.col_ind[well_block];
          if( reservoir_col >= 0 && reservoir_col < m_cprPressureRows )
          {
            row_cols.push_back( reservoir_col );
          }
        }
      }
    }
    std::sort( row_cols.begin(), row_cols.end() );
    row_cols.erase( std::unique( row_cols.begin(), row_cols.end() ), row_cols.end() );
    if( row_cols.empty() )
    {
      row_cols.push_back( row );
    }
    int_t out = m_cprPressureRowOffsets[row];
    for( int_t col : row_cols )
    {
      m_cprPressureCols[out++] = col;
    }
  }

  m_cprPressurePatternReady = true;
}

void LinearSolver::fillBCSRCPRPressureMatrixValues()
{
  const int_t block_size = m_matrix.block_size;
  const int_t pressure_var =
      std::clamp<int_t>( m_params.bcsrCPRPressureVariable, 0, block_size - 1 );
  const bool apply_scaling = scalingActive( m_matrix.global_num_rows );
  const bool use_well_elimination = bcsrCPRUsesWellElimination();

  std::fill( m_cprPressureValues.begin(), m_cprPressureValues.end(), 0.0 );
  m_cprWellEliminationLinks = 0;
  m_cprWellEliminationContributions = 0;
  m_cprWellEliminationMissingDiag = 0;
  m_cprWellEliminationInverseFailure = 0;
  m_cprWellEliminationMissingPattern = 0;
  std::vector<char> row_has_value( m_cprPressureRows, 0 );

  for( int_t row = 0; row < m_cprPressureRows; ++row )
  {
    for( int_t block = m_matrix.row_ptr[row]; block < m_matrix.row_ptr[row + 1]; ++block )
    {
      const int_t col_cell = m_matrix.col_ind[block];
      if( col_cell < 0 || col_cell >= m_cprPressureRows )
      {
        continue;
      }

      const real_type value =
          computeBCSRCPRProjectedPressureValue( row, block, pressure_var, apply_scaling );
      addBCSRCPRPressureValue( row, col_cell, value );
      row_has_value[row] = 1;
    }
  }

  if( use_well_elimination )
  {
    const int_t num_well_blocks =
        std::max<int_t>( m_matrix.num_rows - m_cprPressureRows, 0 );
    std::vector<char> well_inverse_status( num_well_blocks, 0 );
    std::vector<real_type> well_inverses(
        static_cast<size_t>( num_well_blocks ) * block_size * block_size, 0.0 );
    std::vector<real_type> row_vec( block_size, 0.0 );
    std::vector<real_type> tmp_vec( block_size, 0.0 );

    auto get_well_inverse = [&]( int_t well_cell ) -> const real_type *
    {
      const int_t well_index = well_cell - m_cprPressureRows;
      if( well_index < 0 || well_index >= num_well_blocks )
      {
        return nullptr;
      }
      if( well_inverse_status[well_index] == 0 )
      {
        const int_t diag = findBCSRBlock( well_cell, well_cell );
        if( diag < 0 )
        {
          ++m_cprWellEliminationMissingDiag;
          well_inverse_status[well_index] = 2;
        }
        else
        {
          std::vector<real_type> inverse;
          if( computeBCSRCPRScaledBlockInverse( well_cell, diag, inverse ) )
          {
            std::copy( inverse.begin(),
                       inverse.end(),
                       well_inverses.begin() +
                           static_cast<size_t>( well_index ) * block_size * block_size );
            well_inverse_status[well_index] = 1;
          }
          else
          {
            ++m_cprWellEliminationInverseFailure;
            well_inverse_status[well_index] = 2;
          }
        }
      }
      if( well_inverse_status[well_index] != 1 )
      {
        return nullptr;
      }
      return well_inverses.data() +
             static_cast<size_t>( well_index ) * block_size * block_size;
    };

    for( int_t row = 0; row < m_cprPressureRows; ++row )
    {
      const real_type * weights = &m_cprPressureWeights[row * block_size];
      for( int_t block = m_matrix.row_ptr[row];
           block < m_matrix.row_ptr[row + 1]; ++block )
      {
        const int_t well_cell = m_matrix.col_ind[block];
        if( well_cell < m_cprPressureRows || well_cell >= m_matrix.num_rows )
        {
          continue;
        }
        ++m_cprWellEliminationLinks;
        const real_type * well_inverse = get_well_inverse( well_cell );
        if( well_inverse == nullptr )
        {
          continue;
        }

        std::fill( row_vec.begin(), row_vec.end(), 0.0 );
        const int_t row_well_offset = block * block_size * block_size;
        for( int_t r = 0; r < block_size; ++r )
        {
          const int_t scalar_row = row * block_size + r;
          const real_type row_scale =
              apply_scaling && scalar_row < static_cast<int_t>( m_rowScaling.size() )
              ? m_rowScaling[scalar_row] : 1.0;
          for( int_t c = 0; c < block_size; ++c )
          {
            const int_t scalar_col = well_cell * block_size + c;
            const real_type col_scale =
                apply_scaling && scalar_col < static_cast<int_t>( m_colScaling.size() )
                ? m_colScaling[scalar_col] : 1.0;
            row_vec[c] += weights[r] *
                          m_matrix.values[row_well_offset + r * block_size + c] *
                          row_scale * col_scale;
          }
        }

        std::fill( tmp_vec.begin(), tmp_vec.end(), 0.0 );
        for( int_t c = 0; c < block_size; ++c )
        {
          for( int_t k = 0; k < block_size; ++k )
          {
            tmp_vec[k] += row_vec[c] * well_inverse[c * block_size + k];
          }
        }

        for( int_t well_block = m_matrix.row_ptr[well_cell];
             well_block < m_matrix.row_ptr[well_cell + 1]; ++well_block )
        {
          const int_t reservoir_col = m_matrix.col_ind[well_block];
          if( reservoir_col < 0 || reservoir_col >= m_cprPressureRows )
          {
            continue;
          }
          const int_t well_res_offset = well_block * block_size * block_size;
          real_type schur_value = 0.0;
          for( int_t k = 0; k < block_size; ++k )
          {
            const int_t scalar_row = well_cell * block_size + k;
            const int_t scalar_col = reservoir_col * block_size + pressure_var;
            const real_type row_scale =
                apply_scaling && scalar_row < static_cast<int_t>( m_rowScaling.size() )
                ? m_rowScaling[scalar_row] : 1.0;
            const real_type col_scale =
                apply_scaling && scalar_col < static_cast<int_t>( m_colScaling.size() )
                ? m_colScaling[scalar_col] : 1.0;
            schur_value += tmp_vec[k] *
                           m_matrix.values[well_res_offset + k * block_size + pressure_var] *
                           row_scale * col_scale;
          }
          if( std::isfinite( schur_value ) )
          {
            addBCSRCPRPressureValue( row, reservoir_col, -schur_value );
            row_has_value[row] = 1;
            ++m_cprWellEliminationContributions;
          }
        }
      }
    }
  }

  for( int_t row = 0; row < m_cprPressureRows; ++row )
  {
    if( row_has_value[row] )
    {
      continue;
    }
    int_t pos = findBCSRCPRPressureColumnPosition( row, row );
    if( pos < 0 && row + 1 < static_cast<int_t>( m_cprPressureRowOffsets.size() ) &&
        m_cprPressureRowOffsets[row] < m_cprPressureRowOffsets[row + 1] )
    {
      pos = m_cprPressureRowOffsets[row];
    }
    if( pos >= 0 )
    {
      m_cprPressureValues[pos] = 1.0;
    }
  }

  if( use_well_elimination &&
      ( m_params.bcsrCPRDiagnostics || m_params.logLevel >= 1 ) )
  {
    std::cout << "[MGR] BCSR CPR well elimination diagnostics: links="
              << m_cprWellEliminationLinks
              << ", schur_contributions=" << m_cprWellEliminationContributions
              << ", missing_diag=" << m_cprWellEliminationMissingDiag
              << ", inverse_failure=" << m_cprWellEliminationInverseFailure
              << ", missing_pattern=" << m_cprWellEliminationMissingPattern
              << "." << std::endl;
  }

  if( m_params.bcsrCPRDiagnostics &&
      m_params.bcsrCPRDiagnosticMatrixInterval > 0 )
  {
    ++m_cprPressureMatrixDiagnosticCount;
    if( m_cprPressureMatrixDiagnosticCount <= 3 ||
        ( m_cprPressureMatrixDiagnosticCount %
          m_params.bcsrCPRDiagnosticMatrixInterval ) == 0 )
    {
      logBCSRCPRPressureMatrixDiagnostics();
    }
  }
}

void LinearSolver::logBCSRCPRPressureMatrixDiagnostics() const
{
  const int_t rows = m_cprPressureRows;
  const int_t nnz = rows > 0 && !m_cprPressureRowOffsets.empty()
                    ? m_cprPressureRowOffsets[rows] : 0;

  int_t missing_diag_rows = 0;
  int_t near_zero_diag_rows = 0;
  int_t weak_diag_rows = 0;
  int_t positive_diag_rows = 0;
  int_t negative_diag_rows = 0;
  int_t diagnosed_diag_rows = 0;
  int_t positive_offdiag = 0;
  int_t negative_offdiag = 0;
  int_t nonfinite_values = 0;
  real_type min_diag_abs = std::numeric_limits<real_type>::infinity();
  real_type max_diag_abs = 0.0;
  real_type min_dominance = std::numeric_limits<real_type>::infinity();
  real_type max_dominance = 0.0;
  real_type sum_dominance = 0.0;
  real_type max_row_sum_ratio = 0.0;
  real_type sum_row_sum_ratio = 0.0;

  for( int_t row = 0; row < rows; ++row )
  {
    bool has_diag = false;
    real_type diag = 0.0;
    real_type offdiag_abs_sum = 0.0;
    real_type row_abs_sum = 0.0;
    real_type row_sum = 0.0;

    for( int_t p = m_cprPressureRowOffsets[row];
         p < m_cprPressureRowOffsets[row + 1]; ++p )
    {
      const real_type value = m_cprPressureValues[p];
      if( !std::isfinite( value ) )
      {
        ++nonfinite_values;
        continue;
      }

      const real_type abs_value = std::abs( value );
      row_abs_sum += abs_value;
      row_sum += value;

      if( m_cprPressureCols[p] == row )
      {
        has_diag = true;
        diag += value;
      }
      else
      {
        offdiag_abs_sum += abs_value;
        if( value > 0.0 )
        {
          ++positive_offdiag;
        }
        else if( value < 0.0 )
        {
          ++negative_offdiag;
        }
      }
    }

    if( !has_diag )
    {
      ++missing_diag_rows;
      continue;
    }
    ++diagnosed_diag_rows;

    if( diag > 0.0 )
    {
      ++positive_diag_rows;
    }
    else if( diag < 0.0 )
    {
      ++negative_diag_rows;
    }

    const real_type diag_abs = std::abs( diag );
    const real_type scale = std::max<real_type>( row_abs_sum, 1.0 );
    const real_type near_zero_tol =
        std::numeric_limits<real_type>::epsilon() * scale * 100.0;
    if( diag_abs <= near_zero_tol )
    {
      ++near_zero_diag_rows;
    }
    if( diag_abs < offdiag_abs_sum )
    {
      ++weak_diag_rows;
    }

    min_diag_abs = std::min( min_diag_abs, diag_abs );
    max_diag_abs = std::max( max_diag_abs, diag_abs );

    const real_type dominance =
        diag_abs / std::max( offdiag_abs_sum, std::numeric_limits<real_type>::min() );
    min_dominance = std::min( min_dominance, dominance );
    max_dominance = std::max( max_dominance, dominance );
    sum_dominance += dominance;

    const real_type row_sum_ratio =
        std::abs( row_sum ) / std::max( row_abs_sum, std::numeric_limits<real_type>::min() );
    max_row_sum_ratio = std::max( max_row_sum_ratio, row_sum_ratio );
    sum_row_sum_ratio += row_sum_ratio;
  }

  if( diagnosed_diag_rows <= 0 )
  {
    min_diag_abs = 0.0;
    min_dominance = 0.0;
  }

  std::ostringstream out;
  out << std::scientific << std::setprecision( 3 )
      << "[MGR] BCSR CPR pressure matrix diagnostics: call="
      << m_cprPressureMatrixDiagnosticCount
      << ", rows=" << rows
      << ", nnz=" << nnz
      << ", diag_abs(min=" << min_diag_abs
      << ", max=" << max_diag_abs
      << "), diag_sign(pos=" << positive_diag_rows
      << ", neg=" << negative_diag_rows
      << "), missing_diag=" << missing_diag_rows
      << ", near_zero_diag=" << near_zero_diag_rows
      << ", weak_diag=" << weak_diag_rows
      << ", dominance(min=" << min_dominance
      << ", avg=" << ( diagnosed_diag_rows > 0
                        ? sum_dominance / diagnosed_diag_rows : 0.0 )
      << ", max=" << max_dominance
      << "), offdiag_sign(pos=" << positive_offdiag
      << ", neg=" << negative_offdiag
      << "), row_sum_ratio(avg="
      << ( diagnosed_diag_rows > 0
           ? sum_row_sum_ratio / diagnosed_diag_rows : 0.0 )
      << ", max=" << max_row_sum_ratio
      << "), nonfinite_values=" << nonfinite_values
      << ".";
  std::cout << out.str() << std::endl;
}

void LinearSolver::logBCSRCPRAMGHierarchyDiagnostics() const
{
  if( !m_params.bcsrCPRDiagnostics || !m_cprPressureAMG || m_cprPressureRows <= 0 )
  {
    return;
  }

  std::vector<HYPRE_Int> cgrid( m_cprPressureRows, 0 );
  HYPRE_ClearAllErrors();
  HYPRE_Int rc =
      HYPRE_BoomerAMGGetGridHierarchy( m_cprPressureAMG, cgrid.data() );
  bool have_cgrid = true;
  if( rc != 0 )
  {
    std::cerr << "[MGR] Warning: failed to get BCSR CPR pressure AMG hierarchy, rc="
              << rc << " (" << describeHypreError( rc ) << ")." << std::endl;
    HYPRE_ClearAllErrors();
    have_cgrid = false;
  }

  auto * amg_data = reinterpret_cast<hypre_ParAMGData *>( m_cprPressureAMG );
  hypre_ParCSRMatrix ** a_array = hypre_ParAMGDataAArray( amg_data );
  const HYPRE_Int num_levels = hypre_ParAMGDataNumLevels( amg_data );
  if( num_levels <= 0 || a_array == nullptr )
  {
    std::cerr << "[MGR] Warning: failed to inspect BCSR CPR pressure AMG hierarchy internals."
              << std::endl;
    return;
  }

  std::vector<HYPRE_BigInt> level_rows( static_cast<size_t>( num_levels ), 0 );
  std::vector<HYPRE_BigInt> level_nnz( static_cast<size_t>( num_levels ), 0 );
  HYPRE_BigInt total_grid_rows = 0;
  real_type total_nnz = 0.0;
  for( HYPRE_Int level = 0; level < num_levels; ++level )
  {
    hypre_ParCSRMatrix * matrix = a_array[level];
    if( matrix == nullptr )
    {
      continue;
    }

    level_rows[level] =
        static_cast<HYPRE_BigInt>( hypre_ParCSRMatrixNumRows( matrix ) );
    hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag( matrix );
    hypre_CSRMatrix * offd = hypre_ParCSRMatrixOffd( matrix );
    const HYPRE_BigInt diag_nnz =
        diag != nullptr
        ? static_cast<HYPRE_BigInt>( hypre_CSRMatrixNumNonzeros( diag ) )
        : 0;
    const HYPRE_BigInt offd_nnz =
        offd != nullptr
        ? static_cast<HYPRE_BigInt>( hypre_CSRMatrixNumNonzeros( offd ) )
        : 0;
    level_nnz[level] = diag_nnz + offd_nnz;
    real_type d_nnz =
        static_cast<real_type>( hypre_ParCSRMatrixDNumNonzeros( matrix ) );
    if( !std::isfinite( d_nnz ) || d_nnz <= 0.0 )
    {
      d_nnz = static_cast<real_type>( level_nnz[level] );
    }
    total_grid_rows += level_rows[level];
    total_nnz += d_nnz;
  }

  const HYPRE_BigInt fine_nnz =
      !level_nnz.empty() && level_nnz[0] > 0
      ? level_nnz[0]
      : ( m_cprPressureRows > 0 && !m_cprPressureRowOffsets.empty()
          ? static_cast<HYPRE_BigInt>( m_cprPressureRowOffsets[m_cprPressureRows] )
          : 0 );
  const real_type operator_complexity =
      fine_nnz > 0 ? total_nnz / static_cast<real_type>( fine_nnz )
                   : 0.0;
  const real_type grid_complexity =
      m_cprPressureRows > 0 ? static_cast<real_type>( total_grid_rows ) /
                                  static_cast<real_type>( m_cprPressureRows )
                            : 0.0;

  std::ostringstream levels;
  levels << "[";
  for( size_t i = 0; i < level_rows.size(); ++i )
  {
    if( i > 0 )
    {
      levels << ",";
    }
    levels << level_rows[i];
  }
  levels << "]";

  std::ostringstream nnz_levels;
  nnz_levels << "[";
  for( size_t i = 0; i < level_nnz.size(); ++i )
  {
    if( i > 0 )
    {
      nnz_levels << ",";
    }
    nnz_levels << level_nnz[i];
  }
  nnz_levels << "]";

  std::ostringstream coarsening_ratios;
  coarsening_ratios << "[";
  for( size_t i = 1; i < level_rows.size(); ++i )
  {
    if( i > 1 )
    {
      coarsening_ratios << ",";
    }
    const real_type ratio =
        level_rows[i - 1] > 0
        ? static_cast<real_type>( level_rows[i] ) /
              static_cast<real_type>( level_rows[i - 1] )
        : 0.0;
    coarsening_ratios << ratio;
  }
  coarsening_ratios << "]";

  std::ostringstream cf_ratios;
  cf_ratios << "[";
  for( size_t i = 0; i + 1 < level_rows.size(); ++i )
  {
    if( i > 0 )
    {
      cf_ratios << ",";
    }
    const int_t coarse_rows = level_rows[i + 1];
    const int_t fine_only_rows = std::max<int_t>( level_rows[i] - coarse_rows, 0 );
    const real_type ratio =
        fine_only_rows > 0
        ? static_cast<real_type>( coarse_rows ) /
              static_cast<real_type>( fine_only_rows )
        : 0.0;
    cf_ratios << ratio;
  }
  cf_ratios << "]";

  std::ostringstream cgrid_counts;
  if( have_cgrid )
  {
    HYPRE_Int max_cgrid_level = 0;
    for( HYPRE_Int level : cgrid )
    {
      max_cgrid_level = std::max( max_cgrid_level, level );
    }
    std::vector<int_t> cgrid_level_rows( static_cast<size_t>( max_cgrid_level + 1 ), 0 );
    for( HYPRE_Int last_level : cgrid )
    {
      const int_t capped_last =
          std::clamp<int_t>( static_cast<int_t>( last_level ),
                             0,
                             static_cast<int_t>( cgrid_level_rows.size() ) - 1 );
      for( int_t level = 0; level <= capped_last; ++level )
      {
        ++cgrid_level_rows[level];
      }
    }
    cgrid_counts << "[";
    for( size_t i = 0; i < cgrid_level_rows.size(); ++i )
    {
      if( i > 0 )
      {
        cgrid_counts << ",";
      }
      cgrid_counts << cgrid_level_rows[i];
    }
    cgrid_counts << "]";
  }
  else
  {
    cgrid_counts << "[]";
  }

  const int_t coarsest_rows = level_rows.empty() ? 0 : level_rows.back();
  std::cout << "[MGR] BCSR CPR pressure AMG hierarchy: setup_call="
            << m_cprPressureSetupCount
            << ", amg_setups=" << m_cprPressureAMGSetupCount
            << ", reason=" << m_cprLastAMGRebuildReason
            << ", levels=" << level_rows.size()
            << ", fine_rows=" << m_cprPressureRows
            << ", coarsest_rows=" << coarsest_rows
            << ", fine_nnz=" << fine_nnz
            << ", total_level_nnz=" << total_nnz
            << ", operator_complexity=" << operator_complexity
            << ", grid_complexity=" << grid_complexity
            << ", level_rows=" << levels.str()
            << ", level_nnz=" << nnz_levels.str()
            << ", cgrid_level_rows=" << cgrid_counts.str()
            << ", coarsening_ratios=" << coarsening_ratios.str()
            << ", cf_ratios=" << cf_ratios.str()
            << "." << std::endl;
}

bool LinearSolver::prepareBCSRCPRPressureDirectUpdate()
{
  m_cprPressureDirectUpdateReady = false;
  m_cprPressureParCSRDiagDataIndex.clear();

  if( !m_cprPressureParMatrix || m_cprPressureRows <= 0 ||
      m_cprPressureRowOffsets.empty() || m_cprPressureCols.empty() )
  {
    return false;
  }

  hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag( m_cprPressureParMatrix );
  hypre_CSRMatrix * offd = hypre_ParCSRMatrixOffd( m_cprPressureParMatrix );
  if( !diag || !hypre_CSRMatrixI( diag ) || !hypre_CSRMatrixJ( diag ) ||
      !hypre_CSRMatrixData( diag ) )
  {
    return false;
  }

  const HYPRE_Int offd_nnz =
      offd ? hypre_CSRMatrixNumNonzeros( offd ) : 0;
  const int_t expected_nnz = m_cprPressureRowOffsets[m_cprPressureRows];
  if( offd_nnz != 0 ||
      static_cast<int_t>( hypre_CSRMatrixNumRows( diag ) ) != m_cprPressureRows ||
      static_cast<int_t>( hypre_CSRMatrixNumNonzeros( diag ) ) != expected_nnz )
  {
    return false;
  }

  const HYPRE_Int * diag_i = hypre_CSRMatrixI( diag );
  const HYPRE_Int * diag_j = hypre_CSRMatrixJ( diag );
  const HYPRE_BigInt first_col = hypre_ParCSRMatrixFirstColDiag( m_cprPressureParMatrix );

  m_cprPressureParCSRDiagDataIndex.assign( expected_nnz, -1 );
  for( int_t row = 0; row < m_cprPressureRows; ++row )
  {
    const int_t expected_begin = m_cprPressureRowOffsets[row];
    const int_t expected_end = m_cprPressureRowOffsets[row + 1];
    const HYPRE_Int diag_begin = diag_i[row];
    const HYPRE_Int diag_end = diag_i[row + 1];

    if( diag_end < diag_begin ||
        static_cast<int_t>( diag_end - diag_begin ) != expected_end - expected_begin )
    {
      m_cprPressureParCSRDiagDataIndex.clear();
      return false;
    }

    for( int_t k = expected_begin; k < expected_end; ++k )
    {
      const HYPRE_BigInt global_col =
          static_cast<HYPRE_BigInt>( m_cprPressureCols[k] );
      const HYPRE_BigInt local_col_big = global_col - first_col;
      if( local_col_big < 0 ||
          local_col_big > static_cast<HYPRE_BigInt>( std::numeric_limits<HYPRE_Int>::max() ) )
      {
        m_cprPressureParCSRDiagDataIndex.clear();
        return false;
      }
      const HYPRE_Int local_col = static_cast<HYPRE_Int>( local_col_big );

      int_t matched = -1;
      for( HYPRE_Int p = diag_begin; p < diag_end; ++p )
      {
        if( diag_j[p] == local_col )
        {
          matched = static_cast<int_t>( p );
          break;
        }
      }
      if( matched < 0 )
      {
        m_cprPressureParCSRDiagDataIndex.clear();
        return false;
      }
      m_cprPressureParCSRDiagDataIndex[k] = matched;
    }
  }

  m_cprPressureDirectUpdateReady =
      static_cast<int_t>( m_cprPressureParCSRDiagDataIndex.size() ) == expected_nnz;
  return m_cprPressureDirectUpdateReady;
}

bool LinearSolver::updateBCSRCPRPressureMatrixDirect()
{
  if( !m_cprPressureDirectUpdateReady && !prepareBCSRCPRPressureDirectUpdate() )
  {
    return false;
  }

  hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag( m_cprPressureParMatrix );
  if( !diag || !hypre_CSRMatrixData( diag ) ||
      m_cprPressureParCSRDiagDataIndex.size() != m_cprPressureValues.size() )
  {
    m_cprPressureDirectUpdateReady = false;
    return false;
  }

  HYPRE_Complex * diag_data = hypre_CSRMatrixData( diag );
  for( std::size_t i = 0; i < m_cprPressureValues.size(); ++i )
  {
    const int_t data_index = m_cprPressureParCSRDiagDataIndex[i];
    if( data_index < 0 )
    {
      m_cprPressureDirectUpdateReady = false;
      return false;
    }
    diag_data[data_index] = static_cast<HYPRE_Complex>( m_cprPressureValues[i] );
  }

  ++m_cprPressureMatrixDirectUpdateCount;
  ++m_cprPressureMatrixUpdateCount;
  return true;
}

bool LinearSolver::createBCSRCPRPressureMatrix(bool transpose_values)
{
  if( m_cprPressureRows <= 0 )
  {
    return false;
  }

  ScopedTimer timer( setupTimerNode( "BCSR CPR pressure matrix" ) );

  if( !m_cprPressurePatternReady )
  {
    buildBCSRCPRPressurePattern();
  }
  fillBCSRCPRPressureMatrixValues();
  if( transpose_values )
  {
    transposeBCSRCPRPressureMatrix();
  }

  if( !m_cprPressureIJMatrix )
  {
    HYPRE_IJMatrixCreate( MPI_COMM_WORLD,
                          0,
                          m_cprPressureRows - 1,
                          0,
                          m_cprPressureRows - 1,
                          &m_cprPressureIJMatrix );
    HYPRE_IJMatrixSetObjectType( m_cprPressureIJMatrix, HYPRE_PARCSR );
    HYPRE_IJMatrixInitialize( m_cprPressureIJMatrix );
    ++m_cprPressureMatrixCreateCount;
  }

  const bool updating_existing_matrix = m_cprPressureMatrixAssembled;
  if( updating_existing_matrix && updateBCSRCPRPressureMatrixDirect() )
  {
    return m_cprPressureParMatrix != nullptr;
  }

  HYPRE_ClearAllErrors();
  HYPRE_Int rc = HYPRE_IJMatrixSetValues( m_cprPressureIJMatrix,
                                          m_cprPressureRows,
                                          m_cprPressureRowNCols.data(),
                                          m_cprPressureRowIndices.data(),
                                          m_cprPressureCols.data(),
                                          m_cprPressureValues.data() );
  ++m_cprPressureMatrixSetValuesCount;
  if( rc != 0 )
  {
    if( updating_existing_matrix )
    {
      if( m_params.logLevel >= 1 )
      {
        std::cerr << "[MGR] Warning: HYPRE rejected value update on assembled "
                  << "BCSR CPR pressure matrix, rc=" << rc << " ("
                  << describeHypreError( rc ) << "); recreating pressure matrix."
                  << std::endl;
      }
      if( m_cprPressureAMG )
      {
        HYPRE_BoomerAMGDestroy( m_cprPressureAMG );
        m_cprPressureAMG = nullptr;
      }
      HYPRE_IJMatrixDestroy( m_cprPressureIJMatrix );
      m_cprPressureIJMatrix = nullptr;
      m_cprPressureParMatrix = nullptr;
      m_cprPressureMatrixAssembled = false;
      m_cprPressureAMGSetupDone = false;
      m_cprPressureDirectUpdateReady = false;
      m_cprPressureParCSRDiagDataIndex.clear();

      HYPRE_IJMatrixCreate( MPI_COMM_WORLD,
                            0,
                            m_cprPressureRows - 1,
                            0,
                            m_cprPressureRows - 1,
                            &m_cprPressureIJMatrix );
      HYPRE_IJMatrixSetObjectType( m_cprPressureIJMatrix, HYPRE_PARCSR );
      HYPRE_IJMatrixInitialize( m_cprPressureIJMatrix );
      ++m_cprPressureMatrixCreateCount;
      HYPRE_ClearAllErrors();
      rc = HYPRE_IJMatrixSetValues( m_cprPressureIJMatrix,
                                    m_cprPressureRows,
                                    m_cprPressureRowNCols.data(),
                                    m_cprPressureRowIndices.data(),
                                    m_cprPressureCols.data(),
                                    m_cprPressureValues.data() );
      ++m_cprPressureMatrixSetValuesCount;
    }
    if( rc != 0 )
    {
      std::cerr << "[MGR] Error: failed to set BCSR CPR pressure matrix values, rc="
                << rc << " (" << describeHypreError( rc ) << ")." << std::endl;
      return false;
    }
  }
  else if( updating_existing_matrix )
  {
    ++m_cprPressureMatrixUpdateCount;
  }

  if( !m_cprPressureMatrixAssembled )
  {
    HYPRE_IJMatrixAssemble( m_cprPressureIJMatrix );
    HYPRE_IJMatrixGetObject( m_cprPressureIJMatrix,
                             reinterpret_cast<void **>( &m_cprPressureParMatrix ) );
    m_cprPressureMatrixAssembled = m_cprPressureParMatrix != nullptr;
    if( m_cprPressureMatrixAssembled )
    {
      prepareBCSRCPRPressureDirectUpdate();
    }
    ++m_cprPressureMatrixAssembleCount;
  }
  return m_cprPressureParMatrix != nullptr;
}

bool LinearSolver::createBCSRCPRPressureVectors()
{
  if( m_cprPressureRows <= 0 )
  {
    return false;
  }

  ScopedTimer timer( setupTimerNode( "BCSR CPR pressure vectors" ) );

  if( m_cprPressureVectorsReady )
  {
    ++m_cprPressureVectorReuseCount;
    return m_cprPressureParRHS != nullptr && m_cprPressureParSol != nullptr;
  }
  if( m_cprPressureRowIndices.empty() )
  {
    m_cprPressureRowIndices.resize( m_cprPressureRows );
    for( int_t i = 0; i < m_cprPressureRows; ++i )
    {
      m_cprPressureRowIndices[i] = i;
    }
  }
  m_cprPressureRHSValues.assign( m_cprPressureRows, 0.0 );
  m_cprPressureSolution.assign( m_cprPressureRows, 0.0 );

  HYPRE_IJVectorCreate( MPI_COMM_WORLD,
                        0,
                        m_cprPressureRows - 1,
                        &m_cprPressureIJRHS );
  HYPRE_IJVectorSetObjectType( m_cprPressureIJRHS, HYPRE_PARCSR );
  HYPRE_IJVectorInitialize( m_cprPressureIJRHS );

  HYPRE_IJVectorCreate( MPI_COMM_WORLD,
                        0,
                        m_cprPressureRows - 1,
                        &m_cprPressureIJSol );
  HYPRE_IJVectorSetObjectType( m_cprPressureIJSol, HYPRE_PARCSR );
  HYPRE_IJVectorInitialize( m_cprPressureIJSol );

  HYPRE_IJVectorSetValues( m_cprPressureIJRHS,
                           m_cprPressureRows,
                           m_cprPressureRowIndices.data(),
                           m_cprPressureRHSValues.data() );
  HYPRE_IJVectorSetValues( m_cprPressureIJSol,
                           m_cprPressureRows,
                           m_cprPressureRowIndices.data(),
                           m_cprPressureSolution.data() );
  HYPRE_IJVectorAssemble( m_cprPressureIJRHS );
  HYPRE_IJVectorAssemble( m_cprPressureIJSol );
  HYPRE_IJVectorGetObject( m_cprPressureIJRHS,
                           reinterpret_cast<void **>( &m_cprPressureParRHS ) );
  HYPRE_IJVectorGetObject( m_cprPressureIJSol,
                           reinterpret_cast<void **>( &m_cprPressureParSol ) );
  m_cprPressureVectorsReady =
      m_cprPressureParRHS != nullptr && m_cprPressureParSol != nullptr;
  if( m_cprPressureVectorsReady )
  {
    ++m_cprPressureVectorCreateCount;
  }
  return m_cprPressureVectorsReady;
}

bool LinearSolver::setupBCSRCPRPreconditioner()
{
  if( !m_params.useBCSRCPR )
  {
    clearBCSRCPRPreconditioner();
    return true;
  }
  if( !blockLocalPreconditionerReady() )
  {
    std::cerr << "[MGR] Error: BCSR CPR requires a ready full-system BCSR local "
              << "preconditioner." << std::endl;
    return false;
  }

  const int_t reservoir_rows =
      m_params.localReservoirBlockCount > 0
      ? std::min<int_t>( m_params.localReservoirBlockCount, m_matrix.num_rows )
      : m_matrix.num_rows;
  if( reservoir_rows <= 0 )
  {
    std::cerr << "[MGR] Error: BCSR CPR pressure system has no reservoir rows."
              << std::endl;
    return false;
  }

  const bool preserve_cpr_objects =
      m_params.bcsrCPRReuseAMGHierarchy && !m_matrixStructureChanged;
  if( m_cprPressureRows != reservoir_rows || !m_cprPressureIJMatrix ||
      !m_cprPressureIJRHS || !m_cprPressureIJSol || !preserve_cpr_objects )
  {
    ++m_cprPressureStructureResetCount;
    if( m_params.logLevel >= 2 )
    {
      std::cout << "[MGR] BCSR CPR structure reset: reservoir_rows="
                << reservoir_rows
                << ", previous_rows=" << m_cprPressureRows
                << ", reuse_amg=" << m_params.bcsrCPRReuseAMGHierarchy
                << ", matrix_structure_changed=" << m_matrixStructureChanged
                << "." << std::endl;
    }
    clearBCSRCPRPreconditioner();
  }
  else
  {
    ++m_cprPressureStructureReuseCount;
  }
  m_cprPressureRows = reservoir_rows;
  logMGRConfigurationOnce( "BCSR CPR setup" );

  const int_t setup_count_before = m_cprPressureSetupCount;
  const int_t matrix_create_before = m_cprPressureMatrixCreateCount;
  const int_t matrix_set_before = m_cprPressureMatrixSetValuesCount;
  const int_t matrix_assemble_before = m_cprPressureMatrixAssembleCount;
  const int_t matrix_update_before = m_cprPressureMatrixUpdateCount;
  const int_t matrix_direct_update_before = m_cprPressureMatrixDirectUpdateCount;
  const int_t vector_create_before = m_cprPressureVectorCreateCount;
  const int_t vector_reuse_before = m_cprPressureVectorReuseCount;
  const int_t amg_create_before = m_cprPressureAMGCreateCount;
  const int_t amg_setup_before = m_cprPressureAMGSetupCount;
  const int_t amg_reuse_before = m_cprPressureAMGReuseCount;

  const bool source_requested = m_params.bcsrCPRForwardSource;
  const bool source_usable = source_requested && bcsrCPRSourceMatrixCompatible();
  if( source_requested && !source_usable && m_params.logLevel >= 1 )
  {
    std::cerr << "[MGR] Warning: BCSR CPR forward source requested but no "
              << "compatible source matrix is available; using the active matrix."
              << std::endl;
  }

  bool pressure_matrix_ready = false;
  if( source_usable )
  {
    std::swap( m_matrix, m_bcsrCPRSourceMatrix );
    computeBCSRCPRPressureWeights();
    m_cprPressurePatternReady = false;
    pressure_matrix_ready =
        createBCSRCPRPressureMatrix( m_bcsrCPRSourcePressureTranspose );
    std::swap( m_matrix, m_bcsrCPRSourceMatrix );
  }
  else
  {
    computeBCSRCPRPressureWeights();
    pressure_matrix_ready = createBCSRCPRPressureMatrix();
  }

  if( !pressure_matrix_ready || !createBCSRCPRPressureVectors() )
  {
    std::cerr << "[MGR] Error: failed to create BCSR CPR pressure matrix/vectors."
              << std::endl;
    clearBCSRCPRPreconditioner();
    return false;
  }

  if( !m_cprPressureAMG )
  {
    m_cprPressureAMG = setupAMGPreconditioner();
    if( m_cprPressureAMG )
    {
      ++m_cprPressureAMGCreateCount;
    }
    if( !m_cprPressureAMG )
    {
      clearBCSRCPRPreconditioner();
      return false;
    }
  }

  const int_t reuse_age_before = m_cprSetupsSinceAMGSetup;
  const int_t min_reuse_setups =
      std::max<int_t>( m_params.bcsrCPRAdaptiveMinReuseSetups, 0 );
  const int_t max_reuse_setups = m_params.bcsrCPRAdaptiveMaxReuseSetups;
  bool setup_amg = false;
  std::ostringstream rebuild_reason;

  if( !m_cprPressureAMGSetupDone )
  {
    setup_amg = true;
    rebuild_reason << "initial";
  }
  else if( !m_params.bcsrCPRReuseAMGHierarchy )
  {
    setup_amg = true;
    rebuild_reason << "reuse_disabled";
  }
  else if( m_params.bcsrCPRAdaptiveAMGRebuild )
  {
    const bool min_reuse_satisfied = reuse_age_before >= min_reuse_setups;
    if( max_reuse_setups > 0 && reuse_age_before >= max_reuse_setups )
    {
      setup_amg = true;
      rebuild_reason << "adaptive_max_age";
    }
    else if( min_reuse_satisfied && m_cprLastLinearIterations >= 0 &&
             !m_cprLastLinearConverged )
    {
      setup_amg = true;
      rebuild_reason << "adaptive_not_converged";
    }
    else if( min_reuse_satisfied && m_cprAdaptiveQualityRebuildRequested )
    {
      setup_amg = true;
      rebuild_reason << "adaptive_quality"
                     << "(pressure_rel=" << m_cprLastPressureOvershootRel
                     << ", final_rel=" << m_cprLastFinalProxyRel
                     << ", fallback=" << m_cprLastFallbackRatio << ")";
    }
    else if( min_reuse_satisfied && m_cprLastLinearIterations >= 0 &&
             m_params.bcsrCPRAdaptiveLIThreshold > 0 &&
             m_cprLastLinearIterations >= m_params.bcsrCPRAdaptiveLIThreshold )
    {
      setup_amg = true;
      rebuild_reason << "adaptive_li_threshold";
    }
    else if( min_reuse_satisfied && m_cprLastLinearIterations >= 0 &&
             m_cprLastAMGSetupLinearIterations > 0 &&
             m_params.bcsrCPRAdaptiveLIGrowthFactor > 1.0 )
    {
      const real_type growth_limit =
          std::ceil( m_params.bcsrCPRAdaptiveLIGrowthFactor *
                     static_cast<real_type>( m_cprLastAMGSetupLinearIterations ) );
      if( static_cast<real_type>( m_cprLastLinearIterations ) >= growth_limit )
      {
        setup_amg = true;
        rebuild_reason << "adaptive_li_growth";
      }
    }

    if( !setup_amg )
    {
      rebuild_reason << ( min_reuse_satisfied ? "adaptive_reuse"
                                               : "adaptive_min_age" );
    }
  }
  else if( m_params.bcsrCPRAMGRebuildInterval > 0 &&
           ( m_cprPressureSetupCount % m_params.bcsrCPRAMGRebuildInterval ) == 0 )
  {
    setup_amg = true;
    rebuild_reason << "fixed_interval";
  }
  else
  {
    rebuild_reason << "fixed_reuse";
  }

  if( setup_amg )
  {
    ScopedTimer timer( setupTimerNode( "BCSR CPR AMG setup" ) );
    HYPRE_ClearAllErrors();
    const HYPRE_Int rc = HYPRE_BoomerAMGSetup( m_cprPressureAMG,
                                               m_cprPressureParMatrix,
                                               m_cprPressureParRHS,
                                               m_cprPressureParSol );
    if( rc != 0 )
    {
      std::cerr << "[MGR] Error: BCSR CPR pressure AMG setup failed with rc="
                << rc << " (" << describeHypreError( rc ) << ")." << std::endl;
      clearBCSRCPRPreconditioner();
      return false;
    }
    m_cprPressureAMGSetupDone = true;
    m_cprAMGSetupForCurrentSolve = true;
    m_cprSetupsSinceAMGSetup = 0;
    m_cprAdaptiveQualityRebuildRequested = false;
    m_cprLastPressureOvershootRel = 0.0;
    m_cprLastFinalProxyRel = 0.0;
    m_cprLastFallbackRatio = 0.0;
    ++m_cprPressureAMGSetupCount;
  }
  else
  {
    ScopedTimer timer( setupTimerNode( "BCSR CPR AMG setup reused" ) );
    m_cprAMGSetupForCurrentSolve = false;
    ++m_cprSetupsSinceAMGSetup;
    ++m_cprPressureAMGReuseCount;
  }
  ++m_cprPressureSetupCount;
  m_cprLastAMGRebuildReason = rebuild_reason.str();
  if( setup_amg )
  {
    logBCSRCPRAMGHierarchyDiagnostics();
  }

  if( m_params.logLevel >= 1 )
  {
    std::cout << "[MGR] BCSR CPR reuse diagnostics: setup_call="
              << m_cprPressureSetupCount
              << ", local_setup_index=" << ( setup_count_before + 1 )
              << ", structure="
              << ( preserve_cpr_objects ? "reused" : "reset" )
              << ", matrix(created="
              << ( m_cprPressureMatrixCreateCount - matrix_create_before )
              << ", set_values="
              << ( m_cprPressureMatrixSetValuesCount - matrix_set_before )
              << ", value_updates="
              << ( m_cprPressureMatrixUpdateCount - matrix_update_before )
              << ", direct_updates="
              << ( m_cprPressureMatrixDirectUpdateCount - matrix_direct_update_before )
              << ", assembled="
              << ( m_cprPressureMatrixAssembleCount - matrix_assemble_before )
              << "), vectors(created="
              << ( m_cprPressureVectorCreateCount - vector_create_before )
              << ", reused="
              << ( m_cprPressureVectorReuseCount - vector_reuse_before )
              << "), amg(created="
              << ( m_cprPressureAMGCreateCount - amg_create_before )
              << ", setup="
              << ( m_cprPressureAMGSetupCount - amg_setup_before )
              << ", reused="
              << ( m_cprPressureAMGReuseCount - amg_reuse_before )
              << "), adaptive(enabled="
              << ( m_params.bcsrCPRAdaptiveAMGRebuild ? 1 : 0 )
              << ", reason=" << m_cprLastAMGRebuildReason
              << ", age_before=" << reuse_age_before
              << ", age_after=" << m_cprSetupsSinceAMGSetup
              << ", last_li=" << m_cprLastLinearIterations
              << ", last_converged=" << ( m_cprLastLinearConverged ? 1 : 0 )
              << ", last_amg_li=" << m_cprLastAMGSetupLinearIterations
              << ", quality_rebuild_pending="
              << ( m_cprAdaptiveQualityRebuildRequested ? 1 : 0 )
              << ", last_pressure_rel=" << m_cprLastPressureOvershootRel
              << ", last_final_rel=" << m_cprLastFinalProxyRel
              << ", last_fallback=" << m_cprLastFallbackRatio
              << ", threshold=" << m_params.bcsrCPRAdaptiveLIThreshold
              << ", growth=" << m_params.bcsrCPRAdaptiveLIGrowthFactor
              << ", min_age=" << min_reuse_setups
              << ", max_age=" << max_reuse_setups
              << "), totals(clears=" << m_cprClearCount
              << ", structure_resets=" << m_cprPressureStructureResetCount
              << ", structure_reuses=" << m_cprPressureStructureReuseCount
              << ")." << std::endl;
  }

  const int_t n = m_matrix.global_num_rows;
  if( static_cast<int_t>( m_cprPressureRHSValues.size() ) != m_cprPressureRows )
  {
    m_cprPressureRHSValues.assign( m_cprPressureRows, 0.0 );
  }
  if( static_cast<int_t>( m_cprPressureSolution.size() ) != m_cprPressureRows )
  {
    m_cprPressureSolution.assign( m_cprPressureRows, 0.0 );
  }
  if( static_cast<int_t>( m_cprPressureCorrection.size() ) != n )
  {
    m_cprPressureCorrection.assign( n, 0.0 );
  }
  else
  {
    std::fill( m_cprPressureCorrection.begin(), m_cprPressureCorrection.end(), 0.0 );
  }
  if( static_cast<int_t>( m_cprResidual.size() ) != n )
  {
    m_cprResidual.assign( n, 0.0 );
  }
  if( static_cast<int_t>( m_cprAx.size() ) != n )
  {
    m_cprAx.assign( n, 0.0 );
  }
  if( static_cast<int_t>( m_cprLocalCorrection.size() ) != n )
  {
    m_cprLocalCorrection.assign( n, 0.0 );
  }
  m_bcsrCPRReady = true;
  return true;
}

void LinearSolver::setupBlockLocalPreconditioner()
{
  if( ( !m_params.useBCSRCPR &&
        m_params.compositeMode == CompositePreconditionerMode::mgrOnly ) ||
      m_params.localPreconditioner == LocalPreconditionerType::none )
  {
    if( m_blockLocalPreconditioner )
    {
      m_blockLocalPreconditioner->clear();
    }
    clearCompositeWorkVectors();
    return;
  }

  if( !m_blockLocalPreconditioner )
  {
    m_blockLocalPreconditioner = std::make_unique<BlockLocalPreconditioner>();
  }

  const char * timer_name =
      m_params.localPreconditioner == LocalPreconditionerType::blockJacobi
      ? "block Jacobi setup"
      : ( m_params.localPreconditioner == LocalPreconditionerType::blockILU1
          ? "block ILU(1) setup" : "block ILU(0) setup" );
  ScopedTimer timer( setupTimerNode( timer_name ) );

  const bool apply_scaling = scalingActive( m_matrix.global_num_rows );
  const bool ok = m_blockLocalPreconditioner->setup( m_matrix,
                                                     m_params.localPreconditioner,
                                                     m_rowScaling,
                                                     m_colScaling,
                                                     apply_scaling,
                                                     m_params.localPivotShift,
                                                     m_params.localFallbackStrategy,
                                                     m_params.localFallbackDiagonalTolerance,
                                                     m_params.localFallbackShiftMax,
                                                     m_params.localFallbackShiftGrowth,
                                                     m_params.localReservoirBlockCount );
  if( !ok )
  {
    std::cerr << "[MGR] Warning: full-system BCSR local correction setup failed; "
              << "falling back to MGR-only preconditioning." << std::endl;
    m_params.compositeMode = CompositePreconditionerMode::mgrOnly;
    m_params.localPreconditioner = LocalPreconditionerType::none;
    clearCompositeWorkVectors();
    return;
  }

  const int_t shifted_fallback_pivots = m_blockLocalPreconditioner->shiftedDenseFallbackPivots();
  if( shifted_fallback_pivots > 0 )
  {
    std::cout << "[MGR] Info: " << m_blockLocalPreconditioner->name()
              << " used shifted dense fallback for " << shifted_fallback_pivots
              << " diagonal block(s), shifts="
              << m_blockLocalPreconditioner->shiftedDenseShiftSummary()
              << "." << std::endl;
  }

  const int_t diagonal_fallback_pivots = m_blockLocalPreconditioner->diagonalFallbackPivots();
  if( diagonal_fallback_pivots > 0 )
  {
    std::cout << "[MGR] Info: " << m_blockLocalPreconditioner->name()
              << " used bounded diagonal fallback for " << diagonal_fallback_pivots
              << " diagonal block(s), variables(diagonal_inverse="
              << m_blockLocalPreconditioner->boundedDiagonalInverseVariables()
              << ", identity="
              << m_blockLocalPreconditioner->boundedDiagonalIdentityVariables()
              << ")." << std::endl;
  }

  const int_t failed_pivots = m_blockLocalPreconditioner->failedPivots();
  if( failed_pivots > 0 )
  {
    std::cerr << "[MGR] Warning: " << m_blockLocalPreconditioner->name()
              << " used identity fallback for " << failed_pivots
              << " diagonal block(s)." << std::endl;
  }

  const int_t total_fallback_pivots = m_blockLocalPreconditioner->totalFallbackPivots();
  if( total_fallback_pivots > 0 )
  {
    std::cout << "[MGR] Info: " << m_blockLocalPreconditioner->name()
              << " fallback diagnostics: total=" << total_fallback_pivots
              << ", ratio=" << m_blockLocalPreconditioner->fallbackRatio()
              << ", reservoir=" << m_blockLocalPreconditioner->reservoirFallbackPivots()
              << ", well=" << m_blockLocalPreconditioner->wellFallbackPivots()
              << ", primary_failures(nonfinite="
              << m_blockLocalPreconditioner->primaryFailureNonFinite()
              << ", zero_norm="
              << m_blockLocalPreconditioner->primaryFailureZeroNorm()
              << ", small_pivot="
              << m_blockLocalPreconditioner->primaryFailureSmallPivot()
              << ", small_det="
              << m_blockLocalPreconditioner->primaryFailureSmallDeterminant()
              << ", nonfinite_inverse="
              << m_blockLocalPreconditioner->primaryFailureNonFiniteInverse()
              << ")." << std::endl;
  }

  const int_t n = m_matrix.global_num_rows;
  m_compositeResidual.assign( n, 0.0 );
  m_compositeAx.assign( n, 0.0 );
  m_compositeCorrection.assign( n, 0.0 );
}

int LinearSolver::compositePreconditionerSetup(HYPRE_Solver,
                                               HYPRE_ParCSRMatrix,
                                               HYPRE_ParVector,
                                               HYPRE_ParVector)
{
  return 0;
}

int LinearSolver::compositePreconditionerSolve(HYPRE_Solver solver,
                                               HYPRE_ParCSRMatrix A,
                                               HYPRE_ParVector b,
                                               HYPRE_ParVector x)
{
  LinearSolver * self = reinterpret_cast<LinearSolver *>( solver );
  if( !self )
  {
    return 1;
  }
  return self->applyCompositePreconditioner( A, b, x );
}

int LinearSolver::bcsrCPRPreconditionerSetup(HYPRE_Solver,
                                             HYPRE_ParCSRMatrix,
                                             HYPRE_ParVector,
                                             HYPRE_ParVector)
{
  return 0;
}

int LinearSolver::bcsrCPRPreconditionerSolve(HYPRE_Solver solver,
                                             HYPRE_ParCSRMatrix A,
                                             HYPRE_ParVector b,
                                             HYPRE_ParVector x)
{
  LinearSolver * self = reinterpret_cast<LinearSolver *>( solver );
  if( !self )
  {
    return 1;
  }
  if( self->m_params.bcsrCPRTransposeApply )
  {
    return self->applyBCSRCPRTransposePreconditioner( A, b, x );
  }
  return self->applyBCSRCPRPreconditioner( A, b, x );
}

int LinearSolver::applyBCSRCPRPreconditioner(HYPRE_ParCSRMatrix,
                                             HYPRE_ParVector b,
                                             HYPRE_ParVector x)
{
  if( !bcsrCPRPreconditionerReady() )
  {
    return 1;
  }

  ::timer_node * cpr_timer =
      m_activeKrylovName.empty() ? nullptr : solveTimerNode( m_activeKrylovName, "BCSR_CPR" );
  ScopedTimer total_timer( cpr_timer );

  hypre_Vector * b_local = hypre_ParVectorLocalVector( b );
  hypre_Vector * x_local = hypre_ParVectorLocalVector( x );
  if( !b_local || !x_local )
  {
    return 1;
  }

  const int_t local_size = static_cast<int_t>( hypre_VectorSize( b_local ) );
  if( local_size != m_matrix.global_num_rows ||
      static_cast<int_t>( hypre_VectorSize( x_local ) ) != local_size )
  {
    return 1;
  }

  real_type * b_data = hypre_VectorData( b_local );
  real_type * x_data = hypre_VectorData( x_local );
  if( !b_data || !x_data )
  {
    return 1;
  }

  const int_t block_size = m_matrix.block_size;
  const int_t pressure_var =
      std::clamp<int_t>( m_params.bcsrCPRPressureVariable, 0, block_size - 1 );
  ++m_cprApplyCount;
  const bool log_apply_diagnostics =
      m_params.bcsrCPRDiagnostics &&
      m_params.bcsrCPRDiagnosticApplyInterval > 0 &&
      ( m_cprApplyCount <= 3 ||
        ( m_cprApplyCount % m_params.bcsrCPRDiagnosticApplyInterval ) == 0 );
  const bool pressure_guard_enabled =
      m_params.bcsrCPRPressureCorrectionGuardThreshold > 0.0;
  const bool adaptive_pressure_signal_enabled =
      m_params.bcsrCPRAdaptivePressureOvershootThreshold > 0.0;
  const bool adaptive_final_signal_enabled =
      m_params.bcsrCPRAdaptiveFinalProxyThreshold > 0.0;
  const bool adaptive_fallback_signal_enabled =
      m_params.bcsrCPRAdaptiveFallbackThreshold > 0.0;
  const bool need_pressure_norms =
      log_apply_diagnostics || pressure_guard_enabled || adaptive_pressure_signal_enabled;
  const bool need_final_proxy =
      log_apply_diagnostics || adaptive_final_signal_enabled;
  const real_type input_norm =
      need_pressure_norms ? vectorL2Norm( b_data, local_size ) : 0.0;
  real_type pressure_rhs_norm = 0.0;
  real_type pressure_correction_norm = 0.0;
  real_type pressure_residual_norm = 0.0;
  real_type local_correction_norm = 0.0;
  real_type correction_norm = 0.0;
  real_type final_residual_norm = 0.0;
  real_type pressure_alpha =
      std::clamp<real_type>( m_params.bcsrCPRPressureCorrectionAlpha, 0.0, 1.0 );
  real_type pressure_guard_raw_rel = 0.0;
  bool pressure_guard_triggered = false;

  hypre_Vector * pressure_rhs_local =
      hypre_ParVectorLocalVector( m_cprPressureParRHS );
  hypre_Vector * pressure_sol_local =
      hypre_ParVectorLocalVector( m_cprPressureParSol );
  if( !pressure_rhs_local || !pressure_sol_local ||
      static_cast<int_t>( hypre_VectorSize( pressure_rhs_local ) ) != m_cprPressureRows ||
      static_cast<int_t>( hypre_VectorSize( pressure_sol_local ) ) != m_cprPressureRows )
  {
    return 1;
  }
  real_type * pressure_rhs_data = hypre_VectorData( pressure_rhs_local );
  real_type * pressure_sol_data = hypre_VectorData( pressure_sol_local );
  if( !pressure_rhs_data || !pressure_sol_data )
  {
    return 1;
  }

  {
    ScopedTimer timer( cpr_timer ? &cpr_timer->node["pressure RHS"] : nullptr );
    for( int_t row = 0; row < m_cprPressureRows; ++row )
    {
      const real_type * weights = &m_cprPressureWeights[row * block_size];
      const real_type * rhs_block = b_data + row * block_size;
      real_type value = 0.0;
      for( int_t r = 0; r < block_size; ++r )
      {
        value += weights[r] * rhs_block[r];
      }
      pressure_rhs_data[row] = value;
      pressure_sol_data[row] = 0.0;
    }
    if( log_apply_diagnostics )
    {
      pressure_rhs_norm = vectorL2Norm( pressure_rhs_data, m_cprPressureRows );
    }
  }

  {
    ScopedTimer timer( cpr_timer ? &cpr_timer->node["AMG pressure solve"] : nullptr );
    HYPRE_ClearAllErrors();
    const HYPRE_Int rc = HYPRE_BoomerAMGSolve( m_cprPressureAMG,
                                               m_cprPressureParMatrix,
                                               m_cprPressureParRHS,
                                               m_cprPressureParSol );
    if( rc != 0 )
    {
      if( isHypreConvergenceError( rc ) )
      {
        if( m_params.logLevel >= 2 )
        {
          std::cerr << "[MGR] Warning: BCSR CPR pressure AMG reached its "
                    << "inner iteration limit; using the current correction, rc="
                    << rc << " (" << describeHypreError( rc ) << ")." << std::endl;
        }
        HYPRE_ClearAllErrors();
      }
      else
      {
        return rc;
      }
    }
  }

  {
    ScopedTimer timer( cpr_timer ? &cpr_timer->node["pressure injection"] : nullptr );
    for( int_t row = 0; row < m_cprPressureRows; ++row )
    {
      m_cprPressureCorrection[row * block_size + pressure_var] =
          pressure_sol_data[row];
    }
  }

  {
    ScopedTimer timer( cpr_timer ? &cpr_timer->node["BCSR residual"] : nullptr );
    m_blockLocalPreconditioner->matvec( m_cprPressureCorrection.data(),
                                        m_cprAx.data() );
    auto update_pressure_residual = [&]( real_type alpha )
    {
      for( int_t i = 0; i < local_size; ++i )
      {
        m_cprResidual[i] = b_data[i] - alpha * m_cprAx[i];
      }
    };

    update_pressure_residual( pressure_alpha );
    if( need_pressure_norms )
    {
      pressure_residual_norm = vectorL2Norm( m_cprResidual.data(), local_size );
    }

    if( pressure_guard_enabled || adaptive_pressure_signal_enabled )
    {
      pressure_guard_raw_rel = safeRatio( pressure_residual_norm, input_norm );
      if( adaptive_pressure_signal_enabled &&
          std::isfinite( pressure_guard_raw_rel ) )
      {
        m_cprLastPressureOvershootRel =
            std::max( m_cprLastPressureOvershootRel, pressure_guard_raw_rel );
        if( pressure_guard_raw_rel >
            m_params.bcsrCPRAdaptivePressureOvershootThreshold )
        {
          m_cprAdaptiveQualityRebuildRequested = true;
        }
      }
      if( pressure_guard_enabled &&
          std::isfinite( pressure_guard_raw_rel ) &&
          pressure_guard_raw_rel >
              m_params.bcsrCPRPressureCorrectionGuardThreshold )
      {
        const real_type requested_alpha =
            pressure_alpha *
            m_params.bcsrCPRPressureCorrectionGuardThreshold /
            pressure_guard_raw_rel;
        const real_type min_alpha = std::clamp<real_type>(
            m_params.bcsrCPRPressureCorrectionGuardMinAlpha, 0.0, 1.0 );
        const real_type guarded_alpha =
            std::clamp<real_type>( std::max( requested_alpha, min_alpha ),
                                   0.0, pressure_alpha );
        pressure_guard_triggered = guarded_alpha < pressure_alpha;
        pressure_alpha = guarded_alpha;
        update_pressure_residual( pressure_alpha );
        if( need_pressure_norms )
        {
          pressure_residual_norm =
              vectorL2Norm( m_cprResidual.data(), local_size );
        }
      }
    }
  }

  if( pressure_alpha != 1.0 )
  {
    for( int_t i = 0; i < local_size; ++i )
    {
      m_cprPressureCorrection[i] *= pressure_alpha;
    }
  }
  if( log_apply_diagnostics )
  {
    pressure_correction_norm =
        vectorL2Norm( m_cprPressureCorrection.data(), local_size );
  }

  real_type local_alpha = std::max<real_type>( m_params.localCorrectionAlpha, 0.0 );
  real_type fallback_ratio = -1.0;
  real_type local_quality_raw_alpha = 0.0;
  real_type local_quality_after_rel = 0.0;
  bool local_quality_enabled = false;
  bool local_quality_valid = false;
  if( blockLocalPreconditionerReady() )
  {
    fallback_ratio = m_blockLocalPreconditioner->fallbackRatio();
    m_cprLastFallbackRatio = std::max( m_cprLastFallbackRatio, fallback_ratio );
    if( adaptive_fallback_signal_enabled &&
        fallback_ratio > m_params.bcsrCPRAdaptiveFallbackThreshold )
    {
      m_cprAdaptiveQualityRebuildRequested = true;
    }
    if( m_params.localCorrectionAdaptiveFallbackThresholdHigh >= 0.0 &&
        fallback_ratio >= m_params.localCorrectionAdaptiveFallbackThresholdHigh )
    {
      local_alpha = std::max<real_type>( m_params.localCorrectionAdaptiveAlphaHigh, 0.0 );
    }
    else if( m_params.localCorrectionAdaptiveFallbackThreshold >= 0.0 &&
             fallback_ratio >= m_params.localCorrectionAdaptiveFallbackThreshold )
    {
      local_alpha = std::max<real_type>( m_params.localCorrectionAdaptiveAlpha, 0.0 );
    }
  }

  if( local_alpha != 0.0 )
  {
    ScopedTimer timer( cpr_timer ? &cpr_timer->node["block local solve"] : nullptr );
    m_blockLocalPreconditioner->apply( m_cprResidual.data(),
                                       m_cprLocalCorrection.data() );
    if( log_apply_diagnostics )
    {
      local_correction_norm =
          vectorL2Norm( m_cprLocalCorrection.data(), local_size );
    }

    local_quality_enabled = m_params.localCorrectionQualityGate;
    if( local_quality_enabled )
    {
      ScopedTimer quality_timer(
          cpr_timer ? &cpr_timer->node["block local quality"] : nullptr );
      m_blockLocalPreconditioner->matvec( m_cprLocalCorrection.data(),
                                          m_cprAx.data() );
      real_type dot_residual_correction = 0.0;
      real_type correction_image_sq = 0.0;
      real_type pressure_residual_sq = 0.0;
      for( int_t i = 0; i < local_size; ++i )
      {
        dot_residual_correction += m_cprResidual[i] * m_cprAx[i];
        correction_image_sq += m_cprAx[i] * m_cprAx[i];
        pressure_residual_sq += m_cprResidual[i] * m_cprResidual[i];
      }

      if( correction_image_sq > std::numeric_limits<real_type>::epsilon() &&
          std::isfinite( dot_residual_correction ) &&
          std::isfinite( correction_image_sq ) )
      {
        local_quality_raw_alpha =
            dot_residual_correction / correction_image_sq;
        real_type gated_alpha = std::clamp<real_type>( local_quality_raw_alpha,
                                                       0.0, local_alpha );
        const real_type min_alpha = std::min<real_type>(
            std::clamp<real_type>( m_params.localCorrectionQualityMinAlpha,
                                   0.0, 1.0 ),
            local_alpha );
        if( gated_alpha > 0.0 && gated_alpha < min_alpha )
        {
          gated_alpha = min_alpha;
        }
        local_alpha = gated_alpha;
        local_quality_valid = true;

        if( log_apply_diagnostics )
        {
          real_type gated_residual_sq = 0.0;
          for( int_t i = 0; i < local_size; ++i )
          {
            const real_type residual =
                m_cprResidual[i] - local_alpha * m_cprAx[i];
            gated_residual_sq += residual * residual;
          }
          local_quality_after_rel =
              safeRatio( std::sqrt( gated_residual_sq ),
                         std::sqrt( pressure_residual_sq ) );
        }
      }
      else
      {
        local_alpha = 0.0;
      }
    }
  }
  else
  {
    std::fill( m_cprLocalCorrection.begin(), m_cprLocalCorrection.end(), 0.0 );
  }

  {
    ScopedTimer timer( cpr_timer ? &cpr_timer->node["combine"] : nullptr );
    for( int_t i = 0; i < local_size; ++i )
    {
      x_data[i] = m_cprPressureCorrection[i] + local_alpha * m_cprLocalCorrection[i];
    }
  }

  if( need_final_proxy )
  {
    if( log_apply_diagnostics )
    {
      correction_norm = vectorL2Norm( x_data, local_size );
    }
    {
      ScopedTimer timer( cpr_timer ? &cpr_timer->node["diagnostic residual"] : nullptr );
      m_blockLocalPreconditioner->matvec( x_data, m_cprAx.data() );
    }
    real_type final_residual_sq = 0.0;
    for( int_t i = 0; i < local_size; ++i )
    {
      const real_type residual = b_data[i] - m_cprAx[i];
      final_residual_sq += residual * residual;
    }
    final_residual_norm = std::sqrt( final_residual_sq );
    const real_type final_proxy_rel = safeRatio( final_residual_norm, input_norm );
    m_cprLastFinalProxyRel = std::max( m_cprLastFinalProxyRel, final_proxy_rel );
    if( adaptive_final_signal_enabled && std::isfinite( final_proxy_rel ) &&
        final_proxy_rel > m_params.bcsrCPRAdaptiveFinalProxyThreshold )
    {
      m_cprAdaptiveQualityRebuildRequested = true;
    }
  }

  if( log_apply_diagnostics )
  {
    std::ostringstream out;
    out << std::scientific << std::setprecision( 3 )
        << "[MGR] BCSR CPR stage diagnostics: apply=" << m_cprApplyCount
        << ", input_norm=" << input_norm
        << ", pressure_rhs_norm=" << pressure_rhs_norm
        << ", pressure_correction_norm=" << pressure_correction_norm
        << ", after_pressure_norm=" << pressure_residual_norm
        << ", after_pressure_rel=" << safeRatio( pressure_residual_norm, input_norm )
        << ", pressure_alpha=" << pressure_alpha
        << ", pressure_guard_triggered=" << ( pressure_guard_triggered ? 1 : 0 )
        << ", pressure_guard_raw_rel=" << pressure_guard_raw_rel
        << ", adaptive_rebuild_pending="
        << ( m_cprAdaptiveQualityRebuildRequested ? 1 : 0 )
        << ", local_alpha=" << local_alpha
        << ", local_quality_enabled=" << ( local_quality_enabled ? 1 : 0 )
        << ", local_quality_valid=" << ( local_quality_valid ? 1 : 0 )
        << ", local_quality_raw_alpha=" << local_quality_raw_alpha
        << ", local_quality_after_rel=" << local_quality_after_rel
        << ", local_fallback_ratio=" << fallback_ratio
        << ", local_correction_norm=" << local_correction_norm
        << ", correction_norm=" << correction_norm
        << ", final_proxy_norm=" << final_residual_norm
        << ", final_proxy_rel=" << safeRatio( final_residual_norm, input_norm )
        << ".";
    std::cout << out.str() << std::endl;
  }

  return 0;
}

int LinearSolver::applyBCSRCPRTransposePreconditioner(HYPRE_ParCSRMatrix,
                                                      HYPRE_ParVector b,
                                                      HYPRE_ParVector x)
{
  if( !bcsrCPRPreconditionerReady() )
  {
    return 1;
  }

  ::timer_node * cpr_timer =
      m_activeKrylovName.empty() ? nullptr : solveTimerNode( m_activeKrylovName, "BCSR_CPR_T" );
  ScopedTimer total_timer( cpr_timer );

  hypre_Vector * b_local = hypre_ParVectorLocalVector( b );
  hypre_Vector * x_local = hypre_ParVectorLocalVector( x );
  if( !b_local || !x_local )
  {
    return 1;
  }

  const int_t local_size = static_cast<int_t>( hypre_VectorSize( b_local ) );
  if( local_size != m_matrix.global_num_rows ||
      static_cast<int_t>( hypre_VectorSize( x_local ) ) != local_size )
  {
    return 1;
  }

  real_type * b_data = hypre_VectorData( b_local );
  real_type * x_data = hypre_VectorData( x_local );
  if( !b_data || !x_data )
  {
    return 1;
  }

  const int_t block_size = m_matrix.block_size;
  const int_t pressure_var =
      std::clamp<int_t>( m_params.bcsrCPRPressureVariable, 0, block_size - 1 );
  ++m_cprApplyCount;
  const bool log_apply_diagnostics =
      m_params.bcsrCPRDiagnostics &&
      m_params.bcsrCPRDiagnosticApplyInterval > 0 &&
      ( m_cprApplyCount <= 3 ||
        ( m_cprApplyCount % m_params.bcsrCPRDiagnosticApplyInterval ) == 0 );
  const bool pressure_guard_enabled =
      m_params.bcsrCPRPressureCorrectionGuardThreshold > 0.0;
  const bool adaptive_pressure_signal_enabled =
      m_params.bcsrCPRAdaptivePressureOvershootThreshold > 0.0;
  const bool adaptive_final_signal_enabled =
      m_params.bcsrCPRAdaptiveFinalProxyThreshold > 0.0;
  const bool adaptive_fallback_signal_enabled =
      m_params.bcsrCPRAdaptiveFallbackThreshold > 0.0;
  const bool need_pressure_norms =
      log_apply_diagnostics || pressure_guard_enabled || adaptive_pressure_signal_enabled;
  const bool need_final_proxy =
      log_apply_diagnostics || adaptive_final_signal_enabled;
  const real_type input_norm =
      ( need_pressure_norms || need_final_proxy ) ? vectorL2Norm( b_data, local_size ) : 0.0;
  real_type after_local_norm = 0.0;
  real_type pressure_rhs_norm = 0.0;
  real_type pressure_correction_norm = 0.0;
  real_type pressure_residual_norm = 0.0;
  real_type local_correction_norm = 0.0;
  real_type correction_norm = 0.0;
  real_type final_residual_norm = 0.0;
  real_type pressure_alpha =
      std::clamp<real_type>( m_params.bcsrCPRPressureCorrectionAlpha, 0.0, 1.0 );
  real_type pressure_guard_raw_rel = 0.0;
  bool pressure_guard_triggered = false;

  hypre_Vector * pressure_rhs_local =
      hypre_ParVectorLocalVector( m_cprPressureParRHS );
  hypre_Vector * pressure_sol_local =
      hypre_ParVectorLocalVector( m_cprPressureParSol );
  if( !pressure_rhs_local || !pressure_sol_local ||
      static_cast<int_t>( hypre_VectorSize( pressure_rhs_local ) ) != m_cprPressureRows ||
      static_cast<int_t>( hypre_VectorSize( pressure_sol_local ) ) != m_cprPressureRows )
  {
    return 1;
  }
  real_type * pressure_rhs_data = hypre_VectorData( pressure_rhs_local );
  real_type * pressure_sol_data = hypre_VectorData( pressure_sol_local );
  if( !pressure_rhs_data || !pressure_sol_data )
  {
    return 1;
  }

  real_type local_alpha = std::max<real_type>( m_params.localCorrectionAlpha, 0.0 );
  real_type fallback_ratio = -1.0;
  if( blockLocalPreconditionerReady() )
  {
    fallback_ratio = m_blockLocalPreconditioner->fallbackRatio();
    m_cprLastFallbackRatio = std::max( m_cprLastFallbackRatio, fallback_ratio );
    if( adaptive_fallback_signal_enabled &&
        fallback_ratio > m_params.bcsrCPRAdaptiveFallbackThreshold )
    {
      m_cprAdaptiveQualityRebuildRequested = true;
    }
    if( m_params.localCorrectionAdaptiveFallbackThresholdHigh >= 0.0 &&
        fallback_ratio >= m_params.localCorrectionAdaptiveFallbackThresholdHigh )
    {
      local_alpha = std::max<real_type>( m_params.localCorrectionAdaptiveAlphaHigh, 0.0 );
    }
    else if( m_params.localCorrectionAdaptiveFallbackThreshold >= 0.0 &&
             fallback_ratio >= m_params.localCorrectionAdaptiveFallbackThreshold )
    {
      local_alpha = std::max<real_type>( m_params.localCorrectionAdaptiveAlpha, 0.0 );
    }
  }

  {
    ScopedTimer timer( cpr_timer ? &cpr_timer->node["block local solve"] : nullptr );
    if( local_alpha != 0.0 )
    {
      m_blockLocalPreconditioner->apply( b_data, m_cprLocalCorrection.data() );
      if( local_alpha != 1.0 )
      {
        for( int_t i = 0; i < local_size; ++i )
        {
          m_cprLocalCorrection[i] *= local_alpha;
        }
      }
    }
    else
    {
      std::fill( m_cprLocalCorrection.begin(), m_cprLocalCorrection.end(), 0.0 );
    }
    if( log_apply_diagnostics )
    {
      local_correction_norm =
          vectorL2Norm( m_cprLocalCorrection.data(), local_size );
    }
  }

  {
    ScopedTimer timer( cpr_timer ? &cpr_timer->node["BCSR residual"] : nullptr );
    m_blockLocalPreconditioner->matvec( m_cprLocalCorrection.data(),
                                        m_cprAx.data() );
    for( int_t i = 0; i < local_size; ++i )
    {
      m_cprResidual[i] = b_data[i] - m_cprAx[i];
    }
    if( need_pressure_norms )
    {
      after_local_norm = vectorL2Norm( m_cprResidual.data(), local_size );
    }
  }

  {
    ScopedTimer timer( cpr_timer ? &cpr_timer->node["pressure RHS"] : nullptr );
    for( int_t row = 0; row < m_cprPressureRows; ++row )
    {
      pressure_rhs_data[row] =
          m_cprResidual[row * block_size + pressure_var];
      pressure_sol_data[row] = 0.0;
    }
    if( log_apply_diagnostics )
    {
      pressure_rhs_norm = vectorL2Norm( pressure_rhs_data, m_cprPressureRows );
    }
  }

  {
    ScopedTimer timer( cpr_timer ? &cpr_timer->node["AMG pressure solve"] : nullptr );
    HYPRE_ClearAllErrors();
    const HYPRE_Int rc = HYPRE_BoomerAMGSolve( m_cprPressureAMG,
                                               m_cprPressureParMatrix,
                                               m_cprPressureParRHS,
                                               m_cprPressureParSol );
    if( rc != 0 )
    {
      if( isHypreConvergenceError( rc ) )
      {
        if( m_params.logLevel >= 2 )
        {
          std::cerr << "[MGR] Warning: BCSR CPR pressure AMG reached its "
                    << "inner iteration limit in transpose apply; using the "
                    << "current correction, rc=" << rc << " ("
                    << describeHypreError( rc ) << ")." << std::endl;
        }
        HYPRE_ClearAllErrors();
      }
      else
      {
        return rc;
      }
    }
  }

  auto inject_transpose_pressure = [&]()
  {
    std::fill( m_cprPressureCorrection.begin(), m_cprPressureCorrection.end(), 0.0 );
    for( int_t row = 0; row < m_cprPressureRows; ++row )
    {
      const real_type pressure_value = pressure_sol_data[row];
      const real_type * weights = &m_cprPressureWeights[row * block_size];
      real_type * correction_block = &m_cprPressureCorrection[row * block_size];
      for( int_t r = 0; r < block_size; ++r )
      {
        correction_block[r] += weights[r] * pressure_value;
      }
    }
  };

  {
    ScopedTimer timer( cpr_timer ? &cpr_timer->node["pressure transpose injection"] : nullptr );
    inject_transpose_pressure();
  }

  {
    ScopedTimer timer( cpr_timer ? &cpr_timer->node["pressure guard"] : nullptr );
    m_blockLocalPreconditioner->matvec( m_cprPressureCorrection.data(),
                                        m_cprAx.data() );
    auto update_pressure_residual = [&]( real_type alpha )
    {
      for( int_t i = 0; i < local_size; ++i )
      {
        m_cprAx[i] *= alpha;
        m_cprResidual[i] -= m_cprAx[i];
      }
    };

    std::vector<real_type> residual_before_pressure;
    if( pressure_guard_enabled || adaptive_pressure_signal_enabled )
    {
      residual_before_pressure = m_cprResidual;
    }

    update_pressure_residual( pressure_alpha );
    if( need_pressure_norms )
    {
      pressure_residual_norm = vectorL2Norm( m_cprResidual.data(), local_size );
    }

    if( pressure_guard_enabled || adaptive_pressure_signal_enabled )
    {
      const real_type pressure_reference_norm =
          after_local_norm > 0.0 ? after_local_norm : input_norm;
      pressure_guard_raw_rel =
          safeRatio( pressure_residual_norm, pressure_reference_norm );
      if( adaptive_pressure_signal_enabled &&
          std::isfinite( pressure_guard_raw_rel ) )
      {
        m_cprLastPressureOvershootRel =
            std::max( m_cprLastPressureOvershootRel, pressure_guard_raw_rel );
        if( pressure_guard_raw_rel >
            m_params.bcsrCPRAdaptivePressureOvershootThreshold )
        {
          m_cprAdaptiveQualityRebuildRequested = true;
        }
      }
      if( pressure_guard_enabled &&
          std::isfinite( pressure_guard_raw_rel ) &&
          pressure_guard_raw_rel >
              m_params.bcsrCPRPressureCorrectionGuardThreshold )
      {
        const real_type requested_alpha =
            pressure_alpha *
            m_params.bcsrCPRPressureCorrectionGuardThreshold /
            pressure_guard_raw_rel;
        const real_type min_alpha = std::clamp<real_type>(
            m_params.bcsrCPRPressureCorrectionGuardMinAlpha, 0.0, 1.0 );
        const real_type guarded_alpha =
            std::clamp<real_type>( std::max( requested_alpha, min_alpha ),
                                   0.0, pressure_alpha );
        pressure_guard_triggered = guarded_alpha < pressure_alpha;
        if( pressure_guard_triggered )
        {
          pressure_alpha = guarded_alpha;
          inject_transpose_pressure();
          for( int_t i = 0; i < local_size; ++i )
          {
            m_cprPressureCorrection[i] *= pressure_alpha;
          }
          m_blockLocalPreconditioner->matvec( m_cprPressureCorrection.data(),
                                              m_cprAx.data() );
          std::copy( residual_before_pressure.begin(),
                     residual_before_pressure.end(),
                     m_cprResidual.begin() );
          for( int_t i = 0; i < local_size; ++i )
          {
            m_cprResidual[i] -= m_cprAx[i];
          }
          if( need_pressure_norms )
          {
            pressure_residual_norm =
                vectorL2Norm( m_cprResidual.data(), local_size );
          }
        }
      }
      if( !pressure_guard_triggered && pressure_alpha != 1.0 )
      {
        for( int_t i = 0; i < local_size; ++i )
        {
          m_cprPressureCorrection[i] *= pressure_alpha;
        }
      }
    }
    else if( pressure_alpha != 1.0 )
    {
      for( int_t i = 0; i < local_size; ++i )
      {
        m_cprPressureCorrection[i] *= pressure_alpha;
      }
    }
  }

  if( log_apply_diagnostics )
  {
    pressure_correction_norm =
        vectorL2Norm( m_cprPressureCorrection.data(), local_size );
  }

  {
    ScopedTimer timer( cpr_timer ? &cpr_timer->node["combine"] : nullptr );
    for( int_t i = 0; i < local_size; ++i )
    {
      x_data[i] = m_cprLocalCorrection[i] + m_cprPressureCorrection[i];
    }
  }

  if( need_final_proxy )
  {
    if( log_apply_diagnostics )
    {
      correction_norm = vectorL2Norm( x_data, local_size );
    }
    {
      ScopedTimer timer( cpr_timer ? &cpr_timer->node["diagnostic residual"] : nullptr );
      m_blockLocalPreconditioner->matvec( x_data, m_cprAx.data() );
    }
    real_type final_residual_sq = 0.0;
    for( int_t i = 0; i < local_size; ++i )
    {
      const real_type residual = b_data[i] - m_cprAx[i];
      final_residual_sq += residual * residual;
    }
    final_residual_norm = std::sqrt( final_residual_sq );
    const real_type final_proxy_rel = safeRatio( final_residual_norm, input_norm );
    m_cprLastFinalProxyRel = std::max( m_cprLastFinalProxyRel, final_proxy_rel );
    if( adaptive_final_signal_enabled && std::isfinite( final_proxy_rel ) &&
        final_proxy_rel > m_params.bcsrCPRAdaptiveFinalProxyThreshold )
    {
      m_cprAdaptiveQualityRebuildRequested = true;
    }
  }

  if( log_apply_diagnostics )
  {
    std::ostringstream out;
    out << std::scientific << std::setprecision( 3 )
        << "[MGR] BCSR CPR transpose stage diagnostics: apply=" << m_cprApplyCount
        << ", input_norm=" << input_norm
        << ", local_correction_norm=" << local_correction_norm
        << ", after_local_norm=" << after_local_norm
        << ", after_local_rel=" << safeRatio( after_local_norm, input_norm )
        << ", pressure_rhs_norm=" << pressure_rhs_norm
        << ", pressure_correction_norm=" << pressure_correction_norm
        << ", after_pressure_norm=" << pressure_residual_norm
        << ", after_pressure_rel=" << safeRatio( pressure_residual_norm, input_norm )
        << ", pressure_alpha=" << pressure_alpha
        << ", pressure_guard_triggered=" << ( pressure_guard_triggered ? 1 : 0 )
        << ", pressure_guard_raw_rel=" << pressure_guard_raw_rel
        << ", adaptive_rebuild_pending="
        << ( m_cprAdaptiveQualityRebuildRequested ? 1 : 0 )
        << ", local_alpha=" << local_alpha
        << ", local_fallback_ratio=" << fallback_ratio
        << ", correction_norm=" << correction_norm
        << ", final_proxy_norm=" << final_residual_norm
        << ", final_proxy_rel=" << safeRatio( final_residual_norm, input_norm )
        << ".";
    std::cout << out.str() << std::endl;
  }

  return 0;
}

int LinearSolver::applyCompositePreconditioner(HYPRE_ParCSRMatrix A,
                                               HYPRE_ParVector b,
                                               HYPRE_ParVector x)
{
  if( m_params.compositeMode == CompositePreconditionerMode::mgrOnly ||
      !blockLocalPreconditionerReady() )
  {
    return HYPRE_MGRSolve( m_activeMGRPrecond, A, b, x );
  }

  ::timer_node * mgr_timer =
      m_activeKrylovName.empty() ? nullptr : solveTimerNode( m_activeKrylovName, "MGR" );
  ScopedTimer total_timer( mgr_timer );

  hypre_Vector * b_local = hypre_ParVectorLocalVector( b );
  hypre_Vector * x_local = hypre_ParVectorLocalVector( x );
  if( !b_local || !x_local )
  {
    return HYPRE_MGRSolve( m_activeMGRPrecond, A, b, x );
  }

  const int_t local_size = static_cast<int_t>( hypre_VectorSize( b_local ) );
  if( local_size != m_matrix.global_num_rows ||
      static_cast<int_t>( hypre_VectorSize( x_local ) ) != local_size )
  {
    return HYPRE_MGRSolve( m_activeMGRPrecond, A, b, x );
  }

  real_type * b_data = hypre_VectorData( b_local );
  real_type * x_data = hypre_VectorData( x_local );
  if( !b_data || !x_data )
  {
    return HYPRE_MGRSolve( m_activeMGRPrecond, A, b, x );
  }

  real_type local_alpha = std::max<real_type>( m_params.localCorrectionAlpha, 0.0 );
  if( blockLocalPreconditionerReady() )
  {
    const real_type fallback_ratio = m_blockLocalPreconditioner->fallbackRatio();
    if( m_params.localCorrectionAdaptiveFallbackThresholdHigh >= 0.0 &&
        fallback_ratio >= m_params.localCorrectionAdaptiveFallbackThresholdHigh )
    {
      local_alpha = std::max<real_type>( m_params.localCorrectionAdaptiveAlphaHigh, 0.0 );
    }
    else if( m_params.localCorrectionAdaptiveFallbackThreshold >= 0.0 &&
             fallback_ratio >= m_params.localCorrectionAdaptiveFallbackThreshold )
    {
      local_alpha = std::max<real_type>( m_params.localCorrectionAdaptiveAlpha, 0.0 );
    }
  }

  if( m_params.compositeMode == CompositePreconditionerMode::localOnly )
  {
    if( local_alpha == 0.0 )
    {
      HYPRE_ParVectorSetConstantValues( x, 0.0 );
      return 0;
    }
    {
      ScopedTimer local_timer( mgr_timer ? &mgr_timer->node["block local solve"] : nullptr );
      m_blockLocalPreconditioner->apply( b_data, x_data );
    }
    if( local_alpha != 1.0 )
    {
      for( int_t i = 0; i < local_size; ++i )
      {
        x_data[i] *= local_alpha;
      }
    }
    return 0;
  }

  HYPRE_ParVectorSetConstantValues( x, 0.0 );
  int rc = 0;
  {
    ScopedTimer mgr_solve_timer( mgr_timer ? &mgr_timer->node["HYPRE_MGRSolve"] : nullptr );
    rc = HYPRE_MGRSolve( m_activeMGRPrecond, A, b, x );
  }
  if( rc != 0 )
  {
    return rc;
  }
  if( local_alpha == 0.0 )
  {
    return 0;
  }

  {
    ScopedTimer residual_timer( mgr_timer ? &mgr_timer->node["BCSR residual"] : nullptr );
    m_blockLocalPreconditioner->matvec( x_data, m_compositeAx.data() );
    for( int_t i = 0; i < local_size; ++i )
    {
      m_compositeResidual[i] = b_data[i] - m_compositeAx[i];
    }
  }

  {
    {
      ScopedTimer local_timer( mgr_timer ? &mgr_timer->node["block local solve"] : nullptr );
      m_blockLocalPreconditioner->apply( m_compositeResidual.data(),
                                         m_compositeCorrection.data() );
    }

    for( int_t i = 0; i < local_size; ++i )
    {
      x_data[i] += local_alpha * m_compositeCorrection[i];
    }
  }
  return 0;
}

bool LinearSolver::prepareHYPRESystemDirectUpdate()
{
  m_hypreSystemDirectUpdateReady = false;
  m_hypreSystemParCSRDiagDataIndex.clear();

  if( !m_parMatrix || m_matrix.global_num_rows <= 0 ||
      m_matrix.block_size <= 0 || m_matrix.num_rows <= 0 )
  {
    return false;
  }

  hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag( m_parMatrix );
  hypre_CSRMatrix * offd = hypre_ParCSRMatrixOffd( m_parMatrix );
  if( !diag || !hypre_CSRMatrixI( diag ) || !hypre_CSRMatrixJ( diag ) ||
      !hypre_CSRMatrixData( diag ) )
  {
    return false;
  }

  if( offd && hypre_CSRMatrixNumNonzeros( offd ) != 0 )
  {
    return false;
  }

  const int_t num_rows = m_matrix.global_num_rows;
  const int_t block_size = m_matrix.block_size;
  const int_t expected_nnz =
      m_matrix.num_nonzero_blocks * block_size * block_size;

  if( hypre_CSRMatrixNumRows( diag ) != num_rows ||
      hypre_CSRMatrixNumNonzeros( diag ) != expected_nnz )
  {
    return false;
  }

  const HYPRE_Int * diag_i = hypre_CSRMatrixI( diag );
  const HYPRE_Int * diag_j = hypre_CSRMatrixJ( diag );
  const HYPRE_BigInt first_col = hypre_ParCSRMatrixFirstColDiag( m_parMatrix );

  m_hypreSystemParCSRDiagDataIndex.assign( expected_nnz, -1 );

  for( int_t cell = 0; cell < m_matrix.num_rows; ++cell )
  {
    const int_t row_block_start = m_matrix.row_ptr[cell];
    const int_t row_block_end = m_matrix.row_ptr[cell + 1];
    const int_t expected_row_ncols =
        ( row_block_end - row_block_start ) * block_size;

    for( int_t i = 0; i < block_size; ++i )
    {
      const int_t scalar_row = cell * block_size + i;
      const HYPRE_Int row_begin = diag_i[scalar_row];
      const HYPRE_Int row_end = diag_i[scalar_row + 1];
      if( row_end - row_begin != expected_row_ncols )
      {
        m_hypreSystemParCSRDiagDataIndex.clear();
        return false;
      }

      for( int_t block_idx = row_block_start;
           block_idx < row_block_end;
           ++block_idx )
      {
        const int_t col_cell = m_matrix.col_ind[block_idx];
        for( int_t j = 0; j < block_size; ++j )
        {
          const HYPRE_BigInt global_col =
              static_cast<HYPRE_BigInt>( col_cell * block_size + j );
          const HYPRE_BigInt local_col_big = global_col - first_col;
          if( local_col_big < 0 ||
              local_col_big >
                  static_cast<HYPRE_BigInt>(
                      std::numeric_limits<HYPRE_Int>::max() ) )
          {
            m_hypreSystemParCSRDiagDataIndex.clear();
            return false;
          }

          const HYPRE_Int local_col =
              static_cast<HYPRE_Int>( local_col_big );
          int_t matched = -1;
          for( HYPRE_Int p = row_begin; p < row_end; ++p )
          {
            if( diag_j[p] == local_col )
            {
              matched = p;
              break;
            }
          }

          if( matched < 0 )
          {
            m_hypreSystemParCSRDiagDataIndex.clear();
            return false;
          }

          const int_t value_index =
              block_idx * block_size * block_size + i * block_size + j;
          m_hypreSystemParCSRDiagDataIndex[value_index] = matched;
        }
      }
    }
  }

  m_hypreSystemDirectUpdateReady =
      static_cast<int_t>( m_hypreSystemParCSRDiagDataIndex.size() ) ==
      expected_nnz;
  return m_hypreSystemDirectUpdateReady;
}

bool LinearSolver::updateHYPRESystemMatrixDirect()
{
  if( !m_hypreSystemDirectUpdateReady || !m_parMatrix ||
      m_hypreSystemParCSRDiagDataIndex.size() != m_matrix.values.size() )
  {
    return false;
  }

  hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag( m_parMatrix );
  if( !diag || !hypre_CSRMatrixData( diag ) )
  {
    return false;
  }

  HYPRE_Complex * diag_data = hypre_CSRMatrixData( diag );
  const int_t block_size = m_matrix.block_size;
  const bool apply_scaling = scalingActive( m_matrix.global_num_rows );

  for( int_t cell = 0; cell < m_matrix.num_rows; ++cell )
  {
    for( int_t block_idx = m_matrix.row_ptr[cell];
         block_idx < m_matrix.row_ptr[cell + 1];
         ++block_idx )
    {
      const int_t col_cell = m_matrix.col_ind[block_idx];
      const int_t block_start = block_idx * block_size * block_size;
      for( int_t i = 0; i < block_size; ++i )
      {
        const int_t global_row = cell * block_size + i;
        const real_type row_scale =
            apply_scaling ? m_rowScaling[global_row] : 1.0;
        for( int_t j = 0; j < block_size; ++j )
        {
          const int_t value_index = block_start + i * block_size + j;
          const int_t data_index =
              m_hypreSystemParCSRDiagDataIndex[value_index];
          if( data_index < 0 )
          {
            return false;
          }

          real_type value = m_matrix.values[value_index];
          if( apply_scaling )
          {
            const int_t global_col = col_cell * block_size + j;
            value *= row_scale * m_colScaling[global_col];
          }
          diag_data[data_index] = value;
        }
      }
    }
  }

  ++m_hypreSystemMatrixDirectUpdateCount;
  return true;
}

bool LinearSolver::createHYPREMatrix()
{
  ScopedTimer timer( setupTimerNode( "HYPRE IJ matrix" ) );

  int_t num_rows = m_matrix.global_num_rows;
  int_t num_cols = m_matrix.global_num_cols;
  const bool apply_scaling = scalingActive( num_rows );
  int_t num_cells = m_matrix.num_rows;
  int_t block_size = m_matrix.block_size;

  if( m_ijMatrix && m_parMatrix && !m_matrixStructureChanged &&
      updateHYPRESystemMatrixDirect() )
  {
    if( m_params.logLevel >= 1 )
    {
      std::cout << "[MGR] HYPRE system matrix direct ParCSR value update used."
                << std::endl;
    }
    return true;
  }

  if( m_ijMatrix )
  {
    HYPRE_IJMatrixDestroy( m_ijMatrix );
    m_ijMatrix = nullptr;
    m_parMatrix = nullptr;
    m_hypreSystemDirectUpdateReady = false;
    m_hypreSystemParCSRDiagDataIndex.clear();
  }

  // Create IJ matrix
  HYPRE_IJMatrixCreate( MPI_COMM_WORLD, 0, num_rows - 1, 0, num_cols - 1, &m_ijMatrix );
  HYPRE_IJMatrixSetObjectType( m_ijMatrix, HYPRE_PARCSR );
  HYPRE_IJMatrixInitialize( m_ijMatrix );
  ++m_hypreSystemMatrixCreateCount;

  // Fill matrix (row-wise)
  int_t total_nonzeros = 0;

  for( int_t cell = 0; cell < num_cells; ++cell )
  {
    for( int_t i = 0; i < block_size; ++i )
    {
      bigint_t global_row = cell * block_size + i;
      real_type row_scale = apply_scaling ? m_rowScaling[global_row] : 1.0;

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
            val *= row_scale * m_colScaling[global_col];
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
        ++m_hypreSystemMatrixSetValuesCount;
      }
    }
  }

  // Assemble matrix
  HYPRE_IJMatrixAssemble( m_ijMatrix );
  ++m_hypreSystemMatrixAssembleCount;
  HYPRE_IJMatrixGetObject( m_ijMatrix, (void**)&m_parMatrix );
  prepareHYPRESystemDirectUpdate();


  return true;
}

bool LinearSolver::createHYPREVectors()
{
  ScopedTimer timer( setupTimerNode( "HYPRE vectors" ) );

  int_t num_rows = m_matrix.global_num_rows;
  const bool apply_scaling = scalingActive( num_rows );

  if( m_ijRHS && m_ijSol && m_parRHS && m_parSol )
  {
    return true;
  }

  if( m_ijRHS )
  {
    HYPRE_IJVectorDestroy( m_ijRHS );
    m_ijRHS = nullptr;
    m_parRHS = nullptr;
  }
  if( m_ijSol )
  {
    HYPRE_IJVectorDestroy( m_ijSol );
    m_ijSol = nullptr;
    m_parSol = nullptr;
  }

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

  const real_type* initial_guess = m_hasInitialGuess ? m_initialGuess.data() : nullptr;
  updateResidualNormalization( rhs_values, initial_guess, num_rows );

  if( apply_scaling )
  {
    rhs_scaled.resize( num_rows );
    for( int_t i = 0; i < num_rows; ++i )
    {
      rhs_scaled[i] = rhs_values[i] * m_rowScaling[i];
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
        if( m_colScaling[i] != 0.0 )
        {
          m_solution[i] /= m_colScaling[i];
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

void LinearSolver::computeScaling()
{
  m_scaling.clear();
  m_rowScaling.clear();
  m_colScaling.clear();

  if( !m_params.usePhysicsScaling || m_params.scalingType == ScalingType::none )
  {
    return;
  }

  switch( m_params.scalingType )
  {
    case ScalingType::physics:
      computePhysicsScaling();
      break;
    case ScalingType::rowColOneNorm:
      computeRowColOneNormScaling();
      break;
    case ScalingType::diagonal:
      computeDiagonalScaling();
      break;
    case ScalingType::none:
    default:
      break;
  }
}

void LinearSolver::computePhysicsScaling()
{
  ScopedTimer timer( setupTimerNode( "physics scaling" ) );

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
  m_rowScaling.resize( num_rows );
  m_colScaling.resize( m_matrix.global_num_cols, 1.0 );
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
      m_rowScaling[global_row] = m_scaling[global_row];
      m_colScaling[global_row] = m_scaling[global_row];
    }
  }
}

void LinearSolver::computeRowColOneNormScaling()
{
  ScopedTimer timer( setupTimerNode( "row-col one-norm scaling" ) );

  const int_t block_size = m_matrix.block_size;
  const int_t num_rows = m_matrix.global_num_rows;
  const int_t num_cols = m_matrix.global_num_cols;

  if( block_size <= 0 || num_rows <= 0 || num_cols <= 0 )
  {
    return;
  }

  std::vector<real_type> row_norms( num_rows, 0.0 );
  std::vector<real_type> col_norms( num_cols, 0.0 );

  for( int_t cell = 0; cell < m_matrix.num_rows; ++cell )
  {
    for( int_t block_idx = m_matrix.row_ptr[cell];
         block_idx < m_matrix.row_ptr[cell + 1];
         ++block_idx )
    {
      const int_t col_cell = m_matrix.col_ind[block_idx];
      const int_t block_start = block_idx * block_size * block_size;
      for( int_t i = 0; i < block_size; ++i )
      {
        const int_t global_row = cell * block_size + i;
        for( int_t j = 0; j < block_size; ++j )
        {
          const real_type val = std::abs( m_matrix.values[block_start + i * block_size + j] );
          row_norms[global_row] += val;
        }
      }
    }
  }

  const real_type min_norm = std::numeric_limits<real_type>::min();
  m_rowScaling.resize( num_rows, 1.0 );
  for( int_t i = 0; i < num_rows; ++i )
  {
    if( row_norms[i] > min_norm && std::isfinite( row_norms[i] ) )
    {
      m_rowScaling[i] = 1.0 / row_norms[i];
    }
  }

  for( int_t cell = 0; cell < m_matrix.num_rows; ++cell )
  {
    for( int_t block_idx = m_matrix.row_ptr[cell];
         block_idx < m_matrix.row_ptr[cell + 1];
         ++block_idx )
    {
      const int_t col_cell = m_matrix.col_ind[block_idx];
      const int_t block_start = block_idx * block_size * block_size;
      for( int_t i = 0; i < block_size; ++i )
      {
        const int_t global_row = cell * block_size + i;
        for( int_t j = 0; j < block_size; ++j )
        {
          const int_t global_col = col_cell * block_size + j;
          const real_type val = std::abs( m_matrix.values[block_start + i * block_size + j] );
          col_norms[global_col] += val * m_rowScaling[global_row];
        }
      }
    }
  }

  m_colScaling.resize( num_cols, 1.0 );
  for( int_t i = 0; i < num_cols; ++i )
  {
    if( col_norms[i] > min_norm && std::isfinite( col_norms[i] ) )
    {
      m_colScaling[i] = 1.0 / col_norms[i];
    }
  }
  m_scaling = m_colScaling;
}

void LinearSolver::computeDiagonalScaling()
{
  ScopedTimer timer( setupTimerNode( "diagonal scaling" ) );

  const int_t block_size = m_matrix.block_size;
  const int_t num_rows = m_matrix.global_num_rows;
  const int_t num_cols = m_matrix.global_num_cols;

  if( block_size <= 0 || num_rows <= 0 || num_cols <= 0 )
  {
    return;
  }

  std::vector<real_type> diag_abs( num_rows, 0.0 );
  for( int_t cell = 0; cell < m_matrix.num_rows; ++cell )
  {
    for( int_t block_idx = m_matrix.row_ptr[cell];
         block_idx < m_matrix.row_ptr[cell + 1];
         ++block_idx )
    {
      if( m_matrix.col_ind[block_idx] != cell )
      {
        continue;
      }

      const int_t block_start = block_idx * block_size * block_size;
      for( int_t i = 0; i < block_size; ++i )
      {
        const int_t global_row = cell * block_size + i;
        diag_abs[global_row] = std::abs( m_matrix.values[block_start + i * block_size + i] );
      }
    }
  }

  const real_type min_norm = std::numeric_limits<real_type>::min();
  m_scaling.resize( num_rows, 1.0 );
  m_rowScaling.resize( num_rows, 1.0 );
  m_colScaling.resize( num_cols, 1.0 );
  for( int_t i = 0; i < num_rows; ++i )
  {
    if( diag_abs[i] > min_norm && std::isfinite( diag_abs[i] ) )
    {
      m_scaling[i] = 1.0 / std::sqrt( diag_abs[i] );
    }
    m_rowScaling[i] = m_scaling[i];
    m_colScaling[i] = m_scaling[i];
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

  logMGRConfigurationOnce( "MGR setup" );

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
  HYPRE_BoomerAMGSetTol( amg_precond, m_params.pressureAMGTolerance );
  HYPRE_BoomerAMGSetMaxIter( amg_precond,
                             static_cast<HYPRE_Int>(
                                 std::max<int_t>( m_params.pressureAMGMaxIter, 1 ) ) );
  HYPRE_BoomerAMGSetPrintLevel( amg_precond, 0 );

  // For non-symmetric systems
  HYPRE_BoomerAMGSetCoarsenType( amg_precond,
                                 static_cast<HYPRE_Int>( m_params.pressureAMGCoarsenType ) );
  HYPRE_BoomerAMGSetInterpType( amg_precond,
                                static_cast<HYPRE_Int>( m_params.pressureAMGInterpType ) );
  HYPRE_BoomerAMGSetRelaxType( amg_precond,
                               static_cast<HYPRE_Int>( m_params.pressureAMGRelaxType ) );
  HYPRE_BoomerAMGSetAggNumLevels(
      amg_precond,
      static_cast<HYPRE_Int>( std::max<int_t>( m_params.pressureAMGAggNumLevels, 0 ) ) );
  HYPRE_BoomerAMGSetAggInterpType(
      amg_precond,
      static_cast<HYPRE_Int>( m_params.pressureAMGAggInterpType ) );
  HYPRE_BoomerAMGSetAggPMaxElmts(
      amg_precond,
      static_cast<HYPRE_Int>( std::max<int_t>( m_params.pressureAMGAggPMaxElmts, 0 ) ) );
  HYPRE_BoomerAMGSetRelaxOrder(
      amg_precond,
      static_cast<HYPRE_Int>( m_params.pressureAMGRelaxOrder ) );
  if( m_params.pressureAMGStrongThreshold >= 0.0 )
  {
    HYPRE_BoomerAMGSetStrongThreshold(
        amg_precond,
        static_cast<HYPRE_Real>( m_params.pressureAMGStrongThreshold ) );
  }
  if( m_params.pressureAMGTruncFactor >= 0.0 )
  {
    HYPRE_BoomerAMGSetTruncFactor(
        amg_precond,
        static_cast<HYPRE_Real>( m_params.pressureAMGTruncFactor ) );
  }
  if( m_params.pressureAMGPMaxElmts >= 0 )
  {
    HYPRE_BoomerAMGSetPMaxElmts(
        amg_precond,
        static_cast<HYPRE_Int>( m_params.pressureAMGPMaxElmts ) );
  }
  if( m_params.pressureAMGMaxLevels > 0 )
  {
    HYPRE_BoomerAMGSetMaxLevels(
        amg_precond,
        static_cast<HYPRE_Int>( m_params.pressureAMGMaxLevels ) );
  }

  return amg_precond;
}

SolverResults LinearSolver::solveGMRES_MGR()
{

  SolverResults results;
  ::timer_node* mgr_timer = solveTimerNode( "GMRES", "MGR" );

  // Create GMRES solver
  HYPRE_Solver gmres_solver;
  HYPRE_ParCSRGMRESCreate( MPI_COMM_WORLD, &gmres_solver );

  HYPRE_ParCSRGMRESSetMaxIter( gmres_solver, m_params.maxIter );
  HYPRE_ParCSRGMRESSetTol( gmres_solver, m_params.tolerance );
  HYPRE_ParCSRGMRESSetKDim( gmres_solver, m_params.kdim );
  HYPRE_ParCSRGMRESSetPrintLevel( gmres_solver, m_params.logLevel );
  HYPRE_ParCSRGMRESSetLogging( gmres_solver, 1 );

  auto setup_start = std::chrono::high_resolution_clock::now();
  HYPRE_Solver mgr_precond = nullptr;
  {
    ScopedTimer mgr_total( mgr_timer );
    {
      ScopedTimer timer( mgr_timer ? &mgr_timer->node["create/config"] : nullptr );
      mgr_precond = setupMGRPreconditioner();
    }
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
    HYPRE_Int mgr_setup_rc = 0;
    {
      ScopedTimer timer( mgr_timer ? &mgr_timer->node["HYPRE_MGRSetup"] : nullptr );
      mgr_setup_rc = HYPRE_MGRSetup( mgr_precond, m_parMatrix, m_parRHS, m_parSol );
    }
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

    m_activeMGRPrecond = mgr_precond;
    m_activeKrylovName = "GMRES";
    if( m_params.compositeMode != CompositePreconditionerMode::mgrOnly &&
        blockLocalPreconditionerReady() )
    {
      HYPRE_ParCSRGMRESSetPrecond( gmres_solver,
                                    LinearSolver::compositePreconditionerSolve,
                                    LinearSolver::compositePreconditionerSetup,
                                    reinterpret_cast<HYPRE_Solver>( this ) );
    }
    else
    {
      HYPRE_ParCSRGMRESSetPrecond( gmres_solver,
                                    HYPRE_MGRSolve,
                                    MGRDummySetup,
                                    mgr_precond );
    }
  }

  auto setup_end = std::chrono::high_resolution_clock::now();
  results.setupTime = std::chrono::duration<double>( setup_end - setup_start ).count();
  m_setupTime = results.setupTime;  // Save to member for getSetupTime()

  // Setup and solve
  auto solve_start = std::chrono::high_resolution_clock::now();

  // Capture every HYPRE return code: a failed setup/solve makes the
  // statistics below meaningless (see recordKrylovOutcome).
  HYPRE_Int setup_rc = 0, solve_rc = 0;
  {
    ScopedTimer timer( solveTimerNode( "GMRES", "GMRES setup" ) );
    setup_rc = HYPRE_ParCSRGMRESSetup( gmres_solver, m_parMatrix, m_parRHS, m_parSol );
  }
  {
    ScopedTimer timer( solveTimerNode( "GMRES", "GMRES solve" ) );
    solve_rc = HYPRE_ParCSRGMRESSolve( gmres_solver, m_parMatrix, m_parRHS, m_parSol );
  }

  auto solve_end = std::chrono::high_resolution_clock::now();
  results.solveTime = std::chrono::duration<double>( solve_end - solve_start ).count();
  m_solveTime = results.solveTime;  // Save to member for getSolveTime()

  // Get statistics
  int_t num_iterations = 0;
  real_type final_res_norm = std::numeric_limits< real_type >::infinity();
  const HYPRE_Int iters_rc = HYPRE_GMRESGetNumIterations( gmres_solver, &num_iterations );
  const HYPRE_Int resid_rc = HYPRE_GMRESGetFinalRelativeResidualNorm( gmres_solver, &final_res_norm );

  recordKrylovOutcome( results, setup_rc, solve_rc, iters_rc, resid_rc,
                       num_iterations, final_res_norm, "HYPRE_GMRESGetNumIterations" );

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
  const bool apply_scaling = scalingActive( static_cast<int_t>( m_solution.size() ) );
  if( apply_scaling )
  {
    for( size_t i = 0; i < m_solution.size(); ++i )
    {
      m_solution[i] *= m_colScaling[i];
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
  m_activeMGRPrecond = nullptr;
  m_activeKrylovName.clear();
  HYPRE_MGRDestroy( mgr_precond );
  HYPRE_ParCSRGMRESDestroy( gmres_solver );

  return results;
}

SolverResults LinearSolver::solveFlexGMRES_MGR()
{

  SolverResults results;
  ::timer_node* mgr_timer = solveTimerNode( "FlexGMRES", "MGR" );

  // Create FlexGMRES solver
  HYPRE_Solver gmres_solver;
  HYPRE_ParCSRFlexGMRESCreate( MPI_COMM_WORLD, &gmres_solver );

  HYPRE_ParCSRFlexGMRESSetMaxIter( gmres_solver, m_params.maxIter );
  HYPRE_ParCSRFlexGMRESSetTol( gmres_solver, m_params.tolerance );
  HYPRE_ParCSRFlexGMRESSetKDim( gmres_solver, m_params.kdim );
  HYPRE_ParCSRFlexGMRESSetPrintLevel( gmres_solver, m_params.logLevel );
  HYPRE_ParCSRFlexGMRESSetLogging( gmres_solver, 1 );

  auto setup_start = std::chrono::high_resolution_clock::now();
  HYPRE_Solver mgr_precond = nullptr;
  {
    ScopedTimer mgr_total( mgr_timer );
    {
      ScopedTimer timer( mgr_timer ? &mgr_timer->node["create/config"] : nullptr );
      mgr_precond = setupMGRPreconditioner();
    }
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
    HYPRE_Int mgr_setup_rc = 0;
    {
      ScopedTimer timer( mgr_timer ? &mgr_timer->node["HYPRE_MGRSetup"] : nullptr );
      mgr_setup_rc = HYPRE_MGRSetup( mgr_precond, m_parMatrix, m_parRHS, m_parSol );
    }
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

    m_activeMGRPrecond = mgr_precond;
    m_activeKrylovName = "FlexGMRES";
    if( m_params.compositeMode != CompositePreconditionerMode::mgrOnly &&
        blockLocalPreconditionerReady() )
    {
      HYPRE_ParCSRFlexGMRESSetPrecond( gmres_solver,
                                        LinearSolver::compositePreconditionerSolve,
                                        LinearSolver::compositePreconditionerSetup,
                                        reinterpret_cast<HYPRE_Solver>( this ) );
    }
    else
    {
      HYPRE_ParCSRFlexGMRESSetPrecond( gmres_solver,
                                        HYPRE_MGRSolve,
                                        MGRDummySetup,
                                        mgr_precond );
    }
  }

  auto setup_end = std::chrono::high_resolution_clock::now();
  results.setupTime = std::chrono::duration<double>( setup_end - setup_start ).count();
  m_setupTime = results.setupTime;  // Save to member for getSetupTime()

  // Setup and solve
  auto solve_start = std::chrono::high_resolution_clock::now();

  // Capture every HYPRE return code: a failed setup/solve makes the
  // statistics below meaningless (see recordKrylovOutcome).
  HYPRE_Int setup_rc = 0, solve_rc = 0;
  {
    ScopedTimer timer( solveTimerNode( "FlexGMRES", "GMRES setup" ) );
    setup_rc = HYPRE_ParCSRFlexGMRESSetup( gmres_solver, m_parMatrix, m_parRHS, m_parSol );
  }
  {
    ScopedTimer timer( solveTimerNode( "FlexGMRES", "GMRES solve" ) );
    solve_rc = HYPRE_ParCSRFlexGMRESSolve( gmres_solver, m_parMatrix, m_parRHS, m_parSol );
  }

  auto solve_end = std::chrono::high_resolution_clock::now();
  results.solveTime = std::chrono::duration<double>( solve_end - solve_start ).count();
  m_solveTime = results.solveTime;  // Save to member for getSolveTime()

  // Get statistics
  int_t num_iterations = 0;
  real_type final_res_norm = std::numeric_limits< real_type >::infinity();
  const HYPRE_Int iters_rc = HYPRE_ParCSRFlexGMRESGetNumIterations( gmres_solver, &num_iterations );
  const HYPRE_Int resid_rc = HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm( gmres_solver, &final_res_norm );

  recordKrylovOutcome( results, setup_rc, solve_rc, iters_rc, resid_rc,
                       num_iterations, final_res_norm, "HYPRE_ParCSRFlexGMRESGetNumIterations" );

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
  const bool apply_scaling = scalingActive( static_cast<int_t>( m_solution.size() ) );
  if( apply_scaling )
  {
    for( size_t i = 0; i < m_solution.size(); ++i )
    {
      m_solution[i] *= m_colScaling[i];
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
  m_activeMGRPrecond = nullptr;
  m_activeKrylovName.clear();
  HYPRE_MGRDestroy( mgr_precond );
  HYPRE_ParCSRFlexGMRESDestroy( gmres_solver );

  return results;
}

SolverResults LinearSolver::solveGMRES_BCSRCPR()
{
  SolverResults results;
  if( !bcsrCPRPreconditionerReady() )
  {
    std::cerr << "Error: BCSR CPR preconditioner is not ready" << std::endl;
    results.converged = false;
    results.finalResidual = std::numeric_limits<real_type>::infinity();
    results.iterations = 0;
    return results;
  }

  HYPRE_Solver gmres_solver;
  HYPRE_ParCSRGMRESCreate( MPI_COMM_WORLD, &gmres_solver );
  HYPRE_ParCSRGMRESSetMaxIter( gmres_solver, m_params.maxIter );
  HYPRE_ParCSRGMRESSetTol( gmres_solver, m_params.tolerance );
  HYPRE_ParCSRGMRESSetKDim( gmres_solver, m_params.kdim );
  HYPRE_ParCSRGMRESSetPrintLevel( gmres_solver, m_params.logLevel );
  HYPRE_ParCSRGMRESSetLogging( gmres_solver, 1 );

  const auto setup_start = std::chrono::high_resolution_clock::now();
  m_activeKrylovName = "GMRES";
  HYPRE_ParCSRGMRESSetPrecond( gmres_solver,
                               LinearSolver::bcsrCPRPreconditionerSolve,
                               LinearSolver::bcsrCPRPreconditionerSetup,
                               reinterpret_cast<HYPRE_Solver>( this ) );
  const auto setup_end = std::chrono::high_resolution_clock::now();
  results.setupTime = std::chrono::duration<double>( setup_end - setup_start ).count();
  m_setupTime = results.setupTime;

  const auto solve_start = std::chrono::high_resolution_clock::now();
  // Capture every HYPRE return code: a failed setup/solve makes the
  // statistics below meaningless (see recordKrylovOutcome).
  HYPRE_Int setup_rc = 0, solve_rc = 0;
  {
    ScopedTimer timer( solveTimerNode( "GMRES", "GMRES setup" ) );
    setup_rc = HYPRE_ParCSRGMRESSetup( gmres_solver, m_parMatrix, m_parRHS, m_parSol );
  }
  {
    ScopedTimer timer( solveTimerNode( "GMRES", "GMRES solve" ) );
    solve_rc = HYPRE_ParCSRGMRESSolve( gmres_solver, m_parMatrix, m_parRHS, m_parSol );
  }
  const auto solve_end = std::chrono::high_resolution_clock::now();
  results.solveTime = std::chrono::duration<double>( solve_end - solve_start ).count();
  m_solveTime = results.solveTime;

  int_t num_iterations = 0;
  real_type final_res_norm = std::numeric_limits< real_type >::infinity();
  const HYPRE_Int iters_rc = HYPRE_GMRESGetNumIterations( gmres_solver, &num_iterations );
  const HYPRE_Int resid_rc = HYPRE_GMRESGetFinalRelativeResidualNorm( gmres_solver, &final_res_norm );
  if( recordKrylovOutcome( results, setup_rc, solve_rc, iters_rc, resid_rc,
                           num_iterations, final_res_norm, "HYPRE_GMRESGetNumIterations" ) )
  {
    recordBCSRCPRLinearIterations( num_iterations, results.converged );
  }

  const int_t num_rows = m_matrix.global_num_rows;
  m_solution.resize( num_rows );
  std::vector<bigint_t> rows( num_rows );
  for( int_t i = 0; i < num_rows; ++i )
  {
    rows[i] = i;
  }
  HYPRE_IJVectorGetValues( m_ijSol, num_rows, rows.data(), m_solution.data() );
  const bool apply_scaling = scalingActive( static_cast<int_t>( m_solution.size() ) );
  if( apply_scaling )
  {
    for( size_t i = 0; i < m_solution.size(); ++i )
    {
      m_solution[i] *= m_colScaling[i];
    }
  }

  m_activeKrylovName.clear();
  HYPRE_ParCSRGMRESDestroy( gmres_solver );
  return results;
}

SolverResults LinearSolver::solveFlexGMRES_BCSRCPR()
{
  SolverResults results;
  if( !bcsrCPRPreconditionerReady() )
  {
    std::cerr << "Error: BCSR CPR preconditioner is not ready" << std::endl;
    results.converged = false;
    results.finalResidual = std::numeric_limits<real_type>::infinity();
    results.iterations = 0;
    return results;
  }

  HYPRE_Solver gmres_solver;
  HYPRE_ParCSRFlexGMRESCreate( MPI_COMM_WORLD, &gmres_solver );
  HYPRE_ParCSRFlexGMRESSetMaxIter( gmres_solver, m_params.maxIter );
  HYPRE_ParCSRFlexGMRESSetTol( gmres_solver, m_params.tolerance );
  HYPRE_ParCSRFlexGMRESSetKDim( gmres_solver, m_params.kdim );
  HYPRE_ParCSRFlexGMRESSetPrintLevel( gmres_solver, m_params.logLevel );
  HYPRE_ParCSRFlexGMRESSetLogging( gmres_solver, 1 );

  const auto setup_start = std::chrono::high_resolution_clock::now();
  m_activeKrylovName = "FlexGMRES";
  HYPRE_ParCSRFlexGMRESSetPrecond( gmres_solver,
                                   LinearSolver::bcsrCPRPreconditionerSolve,
                                   LinearSolver::bcsrCPRPreconditionerSetup,
                                   reinterpret_cast<HYPRE_Solver>( this ) );
  const auto setup_end = std::chrono::high_resolution_clock::now();
  results.setupTime = std::chrono::duration<double>( setup_end - setup_start ).count();
  m_setupTime = results.setupTime;

  const auto solve_start = std::chrono::high_resolution_clock::now();
  // Capture every HYPRE return code: a failed setup/solve makes the
  // statistics below meaningless (see recordKrylovOutcome).
  HYPRE_Int setup_rc = 0, solve_rc = 0;
  {
    ScopedTimer timer( solveTimerNode( "FlexGMRES", "GMRES setup" ) );
    setup_rc = HYPRE_ParCSRFlexGMRESSetup( gmres_solver, m_parMatrix, m_parRHS, m_parSol );
  }
  {
    ScopedTimer timer( solveTimerNode( "FlexGMRES", "GMRES solve" ) );
    solve_rc = HYPRE_ParCSRFlexGMRESSolve( gmres_solver, m_parMatrix, m_parRHS, m_parSol );
  }
  const auto solve_end = std::chrono::high_resolution_clock::now();
  results.solveTime = std::chrono::duration<double>( solve_end - solve_start ).count();
  m_solveTime = results.solveTime;

  int_t num_iterations = 0;
  real_type final_res_norm = std::numeric_limits< real_type >::infinity();
  const HYPRE_Int iters_rc = HYPRE_ParCSRFlexGMRESGetNumIterations( gmres_solver, &num_iterations );
  const HYPRE_Int resid_rc = HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm( gmres_solver, &final_res_norm );
  if( recordKrylovOutcome( results, setup_rc, solve_rc, iters_rc, resid_rc,
                           num_iterations, final_res_norm, "HYPRE_ParCSRFlexGMRESGetNumIterations" ) )
  {
    recordBCSRCPRLinearIterations( num_iterations, results.converged );
  }

  const int_t num_rows = m_matrix.global_num_rows;
  m_solution.resize( num_rows );
  std::vector<bigint_t> rows( num_rows );
  for( int_t i = 0; i < num_rows; ++i )
  {
    rows[i] = i;
  }
  HYPRE_IJVectorGetValues( m_ijSol, num_rows, rows.data(), m_solution.data() );
  const bool apply_scaling = scalingActive( static_cast<int_t>( m_solution.size() ) );
  if( apply_scaling )
  {
    for( size_t i = 0; i < m_solution.size(); ++i )
    {
      m_solution[i] *= m_colScaling[i];
    }
  }

  m_activeKrylovName.clear();
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
  HYPRE_Solver amg_precond = nullptr;
  {
    ScopedTimer timer( solveTimerNode( "GMRES", "AMG create/config" ) );
    amg_precond = setupAMGPreconditioner();
  }

  HYPRE_ParCSRGMRESSetPrecond( gmres_solver,
                                HYPRE_BoomerAMGSolve,
                                HYPRE_BoomerAMGSetup,
                                amg_precond );

  auto setup_end = std::chrono::high_resolution_clock::now();
  results.setupTime = std::chrono::duration<double>( setup_end - setup_start ).count();
  m_setupTime = results.setupTime;  // Save to member for getSetupTime()

  // Setup and solve
  auto solve_start = std::chrono::high_resolution_clock::now();

  // Capture every HYPRE return code: a failed setup/solve makes the
  // statistics below meaningless (see recordKrylovOutcome).
  HYPRE_Int setup_rc = 0, solve_rc = 0;
  {
    ScopedTimer timer( solveTimerNode( "GMRES", "GMRES setup" ) );
    setup_rc = HYPRE_ParCSRGMRESSetup( gmres_solver, m_parMatrix, m_parRHS, m_parSol );
  }
  {
    ScopedTimer timer( solveTimerNode( "GMRES", "GMRES solve" ) );
    solve_rc = HYPRE_ParCSRGMRESSolve( gmres_solver, m_parMatrix, m_parRHS, m_parSol );
  }

  auto solve_end = std::chrono::high_resolution_clock::now();
  results.solveTime = std::chrono::duration<double>( solve_end - solve_start ).count();
  m_solveTime = results.solveTime;  // Save to member for getSolveTime()

  // Get statistics
  int_t num_iterations = 0;
  real_type final_res_norm = std::numeric_limits< real_type >::infinity();
  const HYPRE_Int iters_rc = HYPRE_GMRESGetNumIterations( gmres_solver, &num_iterations );
  const HYPRE_Int resid_rc = HYPRE_GMRESGetFinalRelativeResidualNorm( gmres_solver, &final_res_norm );

  recordKrylovOutcome( results, setup_rc, solve_rc, iters_rc, resid_rc,
                       num_iterations, final_res_norm, "HYPRE_GMRESGetNumIterations" );

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
  const bool apply_scaling = scalingActive( static_cast<int_t>( m_solution.size() ) );
  if( apply_scaling )
  {
    for( size_t i = 0; i < m_solution.size(); ++i )
    {
      m_solution[i] *= m_colScaling[i];
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
  HYPRE_Solver amg_precond = nullptr;
  {
    ScopedTimer timer( solveTimerNode( "FlexGMRES", "AMG create/config" ) );
    amg_precond = setupAMGPreconditioner();
  }

  HYPRE_ParCSRFlexGMRESSetPrecond( gmres_solver,
                                  HYPRE_BoomerAMGSolve,
                                  HYPRE_BoomerAMGSetup,
                                  amg_precond );

  auto setup_end = std::chrono::high_resolution_clock::now();
  results.setupTime = std::chrono::duration<double>( setup_end - setup_start ).count();
  m_setupTime = results.setupTime;  // Save to member for getSetupTime()

  // Setup and solve
  auto solve_start = std::chrono::high_resolution_clock::now();

  // Capture every HYPRE return code: a failed setup/solve makes the
  // statistics below meaningless (see recordKrylovOutcome).
  HYPRE_Int setup_rc = 0, solve_rc = 0;
  {
    ScopedTimer timer( solveTimerNode( "FlexGMRES", "GMRES setup" ) );
    setup_rc = HYPRE_ParCSRFlexGMRESSetup( gmres_solver, m_parMatrix, m_parRHS, m_parSol );
  }
  {
    ScopedTimer timer( solveTimerNode( "FlexGMRES", "GMRES solve" ) );
    solve_rc = HYPRE_ParCSRFlexGMRESSolve( gmres_solver, m_parMatrix, m_parRHS, m_parSol );
  }

  auto solve_end = std::chrono::high_resolution_clock::now();
  results.solveTime = std::chrono::duration<double>( solve_end - solve_start ).count();
  m_solveTime = results.solveTime;  // Save to member for getSolveTime()

  // Get statistics
  int_t num_iterations = 0;
  real_type final_res_norm = std::numeric_limits< real_type >::infinity();
  const HYPRE_Int iters_rc = HYPRE_ParCSRFlexGMRESGetNumIterations( gmres_solver, &num_iterations );
  const HYPRE_Int resid_rc = HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm( gmres_solver, &final_res_norm );

  recordKrylovOutcome( results, setup_rc, solve_rc, iters_rc, resid_rc,
                       num_iterations, final_res_norm, "HYPRE_ParCSRFlexGMRESGetNumIterations" );

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
  const bool apply_scaling = scalingActive( static_cast<int_t>( m_solution.size() ) );
  if( apply_scaling )
  {
    for( size_t i = 0; i < m_solution.size(); ++i )
    {
      m_solution[i] *= m_colScaling[i];
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
  clearBCSRCPRPreconditioner();
  clearHYPRESystemObjects();
  m_scaling.clear();
  m_rowScaling.clear();
  m_colScaling.clear();
  if( m_blockLocalPreconditioner )
  {
    m_blockLocalPreconditioner->clear();
  }
  clearCompositeWorkVectors();
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

  bool structure_changed =
      m_matrix.num_rows != num_rows ||
      m_matrix.num_cols != num_cols ||
      m_matrix.block_size != block_size ||
      m_matrix.num_nonzero_blocks != num_nonzero_blocks ||
      static_cast<int_t>( m_matrix.row_ptr.size() ) != num_rows + 1 ||
      static_cast<int_t>( m_matrix.col_ind.size() ) != num_nonzero_blocks;
  if( !structure_changed )
  {
    structure_changed =
        !std::equal( row_ptr, row_ptr + num_rows + 1, m_matrix.row_ptr.begin() ) ||
        !std::equal( col_ind, col_ind + num_nonzero_blocks, m_matrix.col_ind.begin() );
  }
  if( !structure_changed )
  {
    if( diag_ind )
    {
      structure_changed =
          static_cast<int_t>( m_matrix.diag_ind.size() ) != num_rows ||
          !std::equal( diag_ind, diag_ind + num_rows, m_matrix.diag_ind.begin() );
    }
    else
    {
      structure_changed = !m_matrix.diag_ind.empty();
    }
  }
  m_matrixStructureChanged = structure_changed;

  // Copy data into BlockCSRMatrix structure
  m_matrix.num_rows = num_rows;
  m_matrix.num_cols = num_cols;
  m_matrix.block_size = block_size;
  m_matrix.num_nonzero_blocks = num_nonzero_blocks;
  m_matrix.global_num_rows = num_rows * block_size;
  m_matrix.global_num_cols = num_cols * block_size;

  // Structural data (row_ptr / col_ind / diag_ind) is identical across Newton
  // iterations when the sparsity pattern is unchanged. structure_changed was
  // already detected above by std::equal-comparing the incoming pointers
  // against m_matrix, so when it is false we can skip the three structural
  // std::copy calls and only refresh the values -- the dominant per-Newton
  // ingest cost.
  if( structure_changed )
  {
    // Copy row_ptr (size = num_rows + 1)
    m_matrix.row_ptr.resize( num_rows + 1 );
    std::copy( row_ptr, row_ptr + num_rows + 1, m_matrix.row_ptr.begin() );

    // Copy col_ind (size = num_nonzero_blocks)
    m_matrix.col_ind.resize( num_nonzero_blocks );
    std::copy( col_ind, col_ind + num_nonzero_blocks, m_matrix.col_ind.begin() );

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
  }

  // Values change every Newton iteration -- always refresh.
  int_t values_size = num_nonzero_blocks * block_size * block_size;
  m_matrix.values.resize( values_size );
  std::copy( values, values + values_size, m_matrix.values.begin() );

  // New matrix values invalidate full-system HYPRE objects; CPR objects can be
  // reused when the block sparsity portrait above did not change.
  m_matrixLoaded = false;
  m_matrixAssembled = false;

  return true;
}

bool LinearSolver::setBCSRCPRSourceFromCSR( int_t num_rows,
                                             int_t num_cols,
                                             int_t block_size,
                                             int_t num_nonzero_blocks,
                                             const int_t * row_ptr,
                                             const int_t * col_ind,
                                             const double * values,
                                             const int_t * diag_ind,
                                             bool transpose_pressure_matrix )
{
  if( !row_ptr || !col_ind || !values )
  {
    std::cerr << "Error: Null pointer passed to setBCSRCPRSourceFromCSR"
              << std::endl;
    return false;
  }

  if( num_rows <= 0 || num_cols <= 0 || block_size <= 0 ||
      num_nonzero_blocks < 0 )
  {
    std::cerr << "Error: Invalid matrix dimensions passed to "
              << "setBCSRCPRSourceFromCSR" << std::endl;
    return false;
  }

  bool structure_changed =
      !m_bcsrCPRSourceMatrixReady ||
      m_bcsrCPRSourceMatrix.num_rows != num_rows ||
      m_bcsrCPRSourceMatrix.num_cols != num_cols ||
      m_bcsrCPRSourceMatrix.block_size != block_size ||
      m_bcsrCPRSourceMatrix.num_nonzero_blocks != num_nonzero_blocks ||
      static_cast<int_t>( m_bcsrCPRSourceMatrix.row_ptr.size() ) != num_rows + 1 ||
      static_cast<int_t>( m_bcsrCPRSourceMatrix.col_ind.size() ) != num_nonzero_blocks;
  if( !structure_changed )
  {
    structure_changed =
        !std::equal( row_ptr,
                     row_ptr + num_rows + 1,
                     m_bcsrCPRSourceMatrix.row_ptr.begin() ) ||
        !std::equal( col_ind,
                     col_ind + num_nonzero_blocks,
                     m_bcsrCPRSourceMatrix.col_ind.begin() );
  }
  if( !structure_changed )
  {
    if( diag_ind )
    {
      structure_changed =
          static_cast<int_t>( m_bcsrCPRSourceMatrix.diag_ind.size() ) != num_rows ||
          !std::equal( diag_ind,
                       diag_ind + num_rows,
                       m_bcsrCPRSourceMatrix.diag_ind.begin() );
    }
    else
    {
      structure_changed = !m_bcsrCPRSourceMatrix.diag_ind.empty();
    }
  }
  if( m_bcsrCPRSourcePressureTranspose != transpose_pressure_matrix )
  {
    structure_changed = true;
  }

  m_bcsrCPRSourceMatrix.num_rows = num_rows;
  m_bcsrCPRSourceMatrix.num_cols = num_cols;
  m_bcsrCPRSourceMatrix.block_size = block_size;
  m_bcsrCPRSourceMatrix.num_nonzero_blocks = num_nonzero_blocks;
  m_bcsrCPRSourceMatrix.global_num_rows = num_rows * block_size;
  m_bcsrCPRSourceMatrix.global_num_cols = num_cols * block_size;

  if( structure_changed )
  {
    m_bcsrCPRSourceMatrix.row_ptr.resize( num_rows + 1 );
    std::copy( row_ptr,
               row_ptr + num_rows + 1,
               m_bcsrCPRSourceMatrix.row_ptr.begin() );

    m_bcsrCPRSourceMatrix.col_ind.resize( num_nonzero_blocks );
    std::copy( col_ind,
               col_ind + num_nonzero_blocks,
               m_bcsrCPRSourceMatrix.col_ind.begin() );

    if( diag_ind )
    {
      m_bcsrCPRSourceMatrix.diag_ind.resize( num_rows );
      std::copy( diag_ind,
                 diag_ind + num_rows,
                 m_bcsrCPRSourceMatrix.diag_ind.begin() );
    }
    else
    {
      m_bcsrCPRSourceMatrix.diag_ind.clear();
    }
  }

  // Values change every Newton iteration -- always refresh.
  const int_t values_size = num_nonzero_blocks * block_size * block_size;
  m_bcsrCPRSourceMatrix.values.resize( values_size );
  std::copy( values,
             values + values_size,
             m_bcsrCPRSourceMatrix.values.begin() );

  m_bcsrCPRSourceMatrixReady = true;
  m_bcsrCPRSourcePressureTranspose = transpose_pressure_matrix;
  if( structure_changed )
  {
    clearBCSRCPRPreconditioner();
  }
  return true;
}

void LinearSolver::clearBCSRCPRSourceMatrix()
{
  if( m_bcsrCPRSourceMatrixReady )
  {
    m_bcsrCPRSourceMatrix = BlockCSRMatrix();
    m_bcsrCPRSourceMatrixReady = false;
    m_bcsrCPRSourcePressureTranspose = false;
    clearBCSRCPRPreconditioner();
  }
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

  bool structure_changed =
      m_matrix.num_rows != num_rows ||
      m_matrix.num_cols != num_cols ||
      m_matrix.block_size != block_size ||
      m_matrix.num_nonzero_blocks != num_nonzero_blocks ||
      static_cast<int_t>( m_matrix.row_ptr.size() ) < num_rows + 1 ||
      static_cast<int_t>( m_matrix.col_ind.size() ) < num_nonzero_blocks;
  if( !structure_changed )
  {
    structure_changed =
        !std::equal( row_ptr.begin(), row_ptr.begin() + num_rows + 1,
                     m_matrix.row_ptr.begin() ) ||
        !std::equal( col_ind.begin(), col_ind.begin() + num_nonzero_blocks,
                     m_matrix.col_ind.begin() );
  }
  if( !structure_changed )
  {
    if( !diag_ind.empty() )
    {
      structure_changed =
          static_cast<int_t>( m_matrix.diag_ind.size() ) < num_rows ||
          !std::equal( diag_ind.begin(), diag_ind.begin() + num_rows,
                       m_matrix.diag_ind.begin() );
    }
    else
    {
      structure_changed = !m_matrix.diag_ind.empty();
    }
  }
  m_matrixStructureChanged = structure_changed;

  // Copy data into BlockCSRMatrix structure
  m_matrix.num_rows = num_rows;
  m_matrix.num_cols = num_cols;
  m_matrix.block_size = block_size;
  m_matrix.num_nonzero_blocks = num_nonzero_blocks;
  m_matrix.global_num_rows = num_rows * block_size;
  m_matrix.global_num_cols = num_cols * block_size;

  if( structure_changed )
  {
    m_matrix.row_ptr = row_ptr;
    m_matrix.col_ind = col_ind;

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
  }

  // Values change every Newton iteration -- always refresh.
  m_matrix.values = values;

  // New matrix values invalidate full-system HYPRE objects; CPR objects can be
  // reused when the block sparsity portrait above did not change.
  m_matrixLoaded = false;
  m_matrixAssembled = false;

  return true;
}

int_t LinearSolver::setup( int_t max_iters, double tolerance )
{
  ScopedTimer timer( setupTimerNode( "setup total" ) );

  // Update solver parameters
  m_params.maxIter = max_iters;
  m_params.tolerance = tolerance;

  // The block portrait is fixed for most reservoir simulations. Preserve the
  // full-system ParCSR object when the portrait is unchanged and update only
  // values; fall back to full IJ rebuild when direct update is not available.
  if( m_matrixStructureChanged )
  {
    clearHYPRESystemObjects();
  }
  else
  {
    m_matrixLoaded = false;
    m_matrixAssembled = false;
    m_activeMGRPrecond = nullptr;
    m_activeKrylovName.clear();
  }
  if( !m_params.bcsrCPRReuseAMGHierarchy || m_matrixStructureChanged )
  {
    clearBCSRCPRPreconditioner();
  }

  // Compute matrix/RHS scaling if enabled.
  computeScaling();
  setupBlockLocalPreconditioner();
  if( !setupBCSRCPRPreconditioner() )
  {
    std::cerr << "Error: Failed to setup BCSR CPR preconditioner" << std::endl;
    return -1;
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
  m_matrixStructureChanged = false;

  return 0;
}

int LinearSolver::solve(mat_float* B, mat_float* X)
{
  if( !m_matrixLoaded || !m_matrixAssembled )
  {
    std::cerr << "Error: Matrix not set up. Call setup() first." << std::endl;
    m_lastResults.stopReason = SolverStopReason::backendError;
    return -1;
  }

  if( !B || !X )
  {
    std::cerr << "Error: Null pointer passed to solve()" << std::endl;
    m_lastResults.stopReason = SolverStopReason::backendError;
    return -1;
  }

  int_t num_rows = m_matrix.global_num_rows;
  const bool use_flex = ( m_params.krylovType == KrylovType::flexgmres );
  const std::string krylov_name = use_flex ? "FlexGMRES" : "GMRES";
  ScopedTimer krylov_timer( m_timerSolve ? &m_timerSolve->node[krylov_name] : nullptr );
  const bool apply_scaling = scalingActive( num_rows );

  std::vector<bigint_t> rows( num_rows );
  for( int_t i = 0; i < num_rows; ++i )
  {
    rows[i] = i;
  }

  std::vector<real_type> rhs_scaled;
  const real_type * rhs_values = B;
  {
    ScopedTimer timer( solveTimerNode( krylov_name, "RHS update" ) );
    updateResidualNormalization( B, nullptr, num_rows );

    if( apply_scaling )
    {
      rhs_scaled.resize( num_rows );
      for( int_t i = 0; i < num_rows; ++i )
      {
        rhs_scaled[i] = B[i] * m_rowScaling[i];
      }
      rhs_values = rhs_scaled.data();
    }
    HYPRE_IJVectorSetValues( m_ijRHS, num_rows, rows.data(), rhs_values );
    HYPRE_IJVectorAssemble( m_ijRHS );
  }

  {
    ScopedTimer timer( solveTimerNode( krylov_name, "initial guess" ) );
    std::vector<real_type> zeros( num_rows, 0.0 );
    HYPRE_IJVectorSetValues( m_ijSol, num_rows, rows.data(), zeros.data() );
    HYPRE_IJVectorAssemble( m_ijSol );
  }

  // Choose solver based on parameters
  if( m_params.useBCSRCPR )
  {
    m_lastResults = use_flex ? solveFlexGMRES_BCSRCPR() : solveGMRES_BCSRCPR();
  }
  else if( m_params.useMGR && m_strategy )
  {
    m_lastResults = use_flex ? solveFlexGMRES_MGR() : solveGMRES_MGR();
  }
  else
  {
    m_lastResults = use_flex ? solveFlexGMRES_AMG() : solveGMRES_AMG();
  }

  // Extract solution to X array
  {
    ScopedTimer timer( solveTimerNode( krylov_name, "solution copy" ) );
    HYPRE_IJVectorGetValues( m_ijSol, num_rows, rows.data(), X );
    if( apply_scaling )
    {
      for( int_t i = 0; i < num_rows; ++i )
      {
        X[i] *= m_colScaling[i];
      }
    }
  }

  // Classify WHY the solve ended. The sign of the value returned below cannot
  // carry this: a preconditioner setup failure (iterations == 0) and an
  // exhausted iteration budget both come out negative. Only an explicit
  // iteration-limit exhaustion that left a finite, non-regressing residual
  // yields a usable iterate; everything else is a hard failure. finalResidual
  // is HYPRE's RELATIVE residual, so "no worse than the initial guess" is
  // "<= 1" (with a small tolerance for round-off).
  if( m_lastResults.stopReason == SolverStopReason::backendError
      || m_lastResults.stopReason == SolverStopReason::breakdown )
  {
    // Already latched by recordKrylovOutcome: the statistics below are
    // untrustworthy and must NOT be allowed to promote this to iterationLimit
    // ("usable iterate").
  }
  else if( m_lastResults.converged )
  {
    m_lastResults.stopReason = SolverStopReason::converged;
  }
  else if( !std::isfinite( static_cast< double >( m_lastResults.finalResidual ) ) )
  {
    m_lastResults.stopReason = SolverStopReason::breakdown;
  }
  else if( m_lastResults.iterations >= m_params.maxIter
           && opendarts::linear_solvers::solve_result::residual_did_not_regress(
                  static_cast< double >( m_lastResults.finalResidual ), 1.0 ) )
  {
    m_lastResults.stopReason = SolverStopReason::iterationLimit;
  }
  else
  {
    // stalled/diverged before the budget, or the residual regressed past the
    // initial guess -- either way the iterate must not be applied
    m_lastResults.stopReason = SolverStopReason::breakdown;
  }

  // Return number of iterations (or negative error code). A failed
  // preconditioner setup reports converged == false with iterations == 0;
  // clamping the failure code to at least -1 keeps it distinguishable from
  // a successful 0-iteration solve (-0 == 0 would read as success in callers
  // that test the sign -- previously a setup failure surfaced as a
  // "successful" solve with a zero solution).
  if( m_lastResults.converged )
  {
    return m_lastResults.iterations;
  }
  else
  {
    return m_lastResults.iterations > 0 ? -m_lastResults.iterations : -1;
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
