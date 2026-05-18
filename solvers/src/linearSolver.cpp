/*
 * MGR Linear Solver - Linear Solver Implementation
 */

#include "LinearSolver.hpp"
#include "OpendartsJacobian.hpp"
#include "CompositionalFlowStrategy.hpp"
#include "timer_node.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <limits>
#include <algorithm>
#include <string>

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
              real_type fallback_shift_growth )
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
    m_pivotShift = std::max<real_type>( pivot_shift, 0.0 );
    m_fallbackStrategy = fallback_strategy;
    m_fallbackDiagonalTolerance = std::max<real_type>( fallback_diagonal_tolerance, 0.0 );
    m_fallbackShiftMax = std::max<real_type>( fallback_shift_max, 0.0 );
    m_fallbackShiftGrowth = std::max<real_type>( fallback_shift_growth, 1.0 );
    m_rowPtr = matrix.row_ptr;
    m_colInd = matrix.col_ind;
    m_diagInd.assign( m_numRows, -1 );

    if( static_cast<int_t>( m_rowPtr.size() ) < m_numRows + 1 ||
        static_cast<int_t>( m_colInd.size() ) < matrix.num_nonzero_blocks ||
        static_cast<int_t>( matrix.values.size() ) < matrix.num_nonzero_blocks * m_blockSizeSquared )
    {
      clear();
      return false;
    }

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

    m_originalValues.resize( matrix.num_nonzero_blocks * m_blockSizeSquared );
    m_luValues.resize( matrix.num_nonzero_blocks * m_blockSizeSquared );
    for( int_t row = 0; row < m_numRows; ++row )
    {
      for( int_t block = m_rowPtr[row]; block < m_rowPtr[row + 1]; ++block )
      {
        const int_t col = m_colInd[block];
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
    m_luValues = m_originalValues;

    m_diagInverse.assign( m_numRows * m_blockSizeSquared, 0.0 );
    m_blockWork.assign( m_blockSizeSquared, 0.0 );
    m_backwardBlock.assign( m_blockSize, 0.0 );
    if( m_type == LocalPreconditionerType::blockJacobi )
    {
      factorBlockJacobi();
    }
    else
    {
      factorBlockILU0();
    }

    m_forwardWork.assign( m_numRows * m_blockSize, 0.0 );
    m_ready = true;
    return true;
  }

  void clear()
  {
    m_type = LocalPreconditionerType::none;
    m_numRows = 0;
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
    m_ready = false;
    m_rowPtr.clear();
    m_colInd.clear();
    m_diagInd.clear();
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

  const char * name() const
  {
    return m_type == LocalPreconditionerType::blockJacobi ? "block Jacobi" : "block ILU(0)";
  }

  void matvec( const real_type * x, real_type * y ) const
  {
    const int_t n = m_numRows * m_blockSize;
    std::fill( y, y + n, 0.0 );
    for( int_t row = 0; row < m_numRows; ++row )
    {
      real_type * y_block = y + row * m_blockSize;
      for( int_t p = m_rowPtr[row]; p < m_rowPtr[row + 1]; ++p )
      {
        const real_type * block = &m_originalValues[p * m_blockSizeSquared];
        const real_type * x_block = x + m_colInd[p] * m_blockSize;
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

      if( m_type == LocalPreconditionerType::blockILU0 )
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

      if( m_type == LocalPreconditionerType::blockILU0 )
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

  void factorBlockJacobi()
  {
    for( int_t row = 0; row < m_numRows; ++row )
    {
      invertDiagonalBlock( row );
    }
  }

  void factorBlockILU0()
  {
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
    if( invertBlock( block, inverse, m_pivotShift ) )
    {
      return;
    }

    if( m_fallbackStrategy == LocalFallbackStrategy::shiftedDense ||
        m_fallbackStrategy == LocalFallbackStrategy::shiftedDenseThenDiagonal )
    {
      if( invertBlockWithShiftFallback( block, inverse ) )
      {
        ++m_shiftedDenseFallbackPivots;
        return;
      }
    }

    if( m_fallbackStrategy == LocalFallbackStrategy::boundedDiagonal ||
        m_fallbackStrategy == LocalFallbackStrategy::shiftedDenseThenDiagonal )
    {
      if( invertBlockDiagonal( block, inverse ) )
      {
        ++m_diagonalFallbackPivots;
        return;
      }
    }

    ++m_failedPivots;
    std::fill( inverse, inverse + m_blockSizeSquared, 0.0 );
    for( int_t i = 0; i < m_blockSize; ++i )
    {
      inverse[i * m_blockSize + i] = 1.0;
    }
  }

  bool invertBlockWithShiftFallback( const real_type * block, real_type * inverse ) const
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
                    real_type relative_shift ) const
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
    const real_type shift = relative_shift * std::max<real_type>( norm, 1.0 );
    const real_type pivot_tol = std::numeric_limits<real_type>::epsilon() *
                                std::max<real_type>( norm + std::abs( shift ), 1.0 ) * 100.0;

    if( m_blockSize == 1 )
    {
      const real_type pivot = block[0] + shift;
      if( std::abs( pivot ) <= pivot_tol || !std::isfinite( pivot ) )
      {
        return false;
      }
      inverse[0] = 1.0 / pivot;
      return std::isfinite( inverse[0] );
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
        return false;
      }
      const real_type inv_det = 1.0 / det;
      inverse[0] = d * inv_det;
      inverse[1] = -b * inv_det;
      inverse[2] = -c * inv_det;
      inverse[3] = a * inv_det;
      return std::isfinite( inverse[0] ) && std::isfinite( inverse[1] ) &&
             std::isfinite( inverse[2] ) && std::isfinite( inverse[3] );
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
          return false;
        }
      }
    }
    return true;
  }

  bool invertBlockDiagonal( const real_type * block, real_type * inverse ) const
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
      }
      else
      {
        inverse[i * m_blockSize + i] = 1.0;
      }
    }
    return used_diagonal_inverse;
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
  bool m_ready = false;
  std::vector<int_t> m_rowPtr;
  std::vector<int_t> m_colInd;
  std::vector<int_t> m_diagInd;
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

bool LinearSolver::blockLocalPreconditionerReady() const
{
  return m_blockLocalPreconditioner && m_blockLocalPreconditioner->ready();
}

void LinearSolver::setupBlockLocalPreconditioner()
{
  if( m_params.compositeMode == CompositePreconditionerMode::mgrOnly ||
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
      ? "block Jacobi setup" : "block ILU(0) setup";
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
                                                     m_params.localFallbackShiftGrowth );
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
    std::cerr << "[MGR] Info: " << m_blockLocalPreconditioner->name()
              << " used shifted dense fallback for " << shifted_fallback_pivots
              << " diagonal block(s)." << std::endl;
  }

  const int_t diagonal_fallback_pivots = m_blockLocalPreconditioner->diagonalFallbackPivots();
  if( diagonal_fallback_pivots > 0 )
  {
    std::cerr << "[MGR] Info: " << m_blockLocalPreconditioner->name()
              << " used bounded diagonal fallback for " << diagonal_fallback_pivots
              << " diagonal block(s)." << std::endl;
  }

  const int_t failed_pivots = m_blockLocalPreconditioner->failedPivots();
  if( failed_pivots > 0 )
  {
    std::cerr << "[MGR] Warning: " << m_blockLocalPreconditioner->name()
              << " used identity fallback for " << failed_pivots
              << " diagonal block(s)." << std::endl;
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

  if( m_params.compositeMode == CompositePreconditionerMode::localOnly )
  {
    ScopedTimer local_timer( mgr_timer ? &mgr_timer->node["block local solve"] : nullptr );
    m_blockLocalPreconditioner->apply( b_data, x_data );
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

  {
    ScopedTimer residual_timer( mgr_timer ? &mgr_timer->node["BCSR residual"] : nullptr );
    m_blockLocalPreconditioner->matvec( x_data, m_compositeAx.data() );
    for( int_t i = 0; i < local_size; ++i )
    {
      m_compositeResidual[i] = b_data[i] - m_compositeAx[i];
    }
  }

  {
    ScopedTimer local_timer( mgr_timer ? &mgr_timer->node["block local solve"] : nullptr );
    m_blockLocalPreconditioner->apply( m_compositeResidual.data(),
                                       m_compositeCorrection.data() );
  }

  for( int_t i = 0; i < local_size; ++i )
  {
    x_data[i] += m_compositeCorrection[i];
  }
  return 0;
}

bool LinearSolver::createHYPREMatrix()
{
  ScopedTimer timer( setupTimerNode( "HYPRE IJ matrix" ) );

  int_t num_rows = m_matrix.global_num_rows;
  int_t num_cols = m_matrix.global_num_cols;
  const bool apply_scaling = scalingActive( num_rows );
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
  ScopedTimer timer( setupTimerNode( "HYPRE vectors" ) );

  int_t num_rows = m_matrix.global_num_rows;
  const bool apply_scaling = scalingActive( num_rows );

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

  {
    ScopedTimer timer( solveTimerNode( "GMRES", "GMRES setup" ) );
    HYPRE_ParCSRGMRESSetup( gmres_solver, m_parMatrix, m_parRHS, m_parSol );
  }
  {
    ScopedTimer timer( solveTimerNode( "GMRES", "GMRES solve" ) );
    HYPRE_ParCSRGMRESSolve( gmres_solver, m_parMatrix, m_parRHS, m_parSol );
  }

  auto solve_end = std::chrono::high_resolution_clock::now();
  results.solveTime = std::chrono::duration<double>( solve_end - solve_start ).count();
  m_solveTime = results.solveTime;  // Save to member for getSolveTime()

  // Get statistics
  int_t num_iterations;
  real_type final_res_norm;
  HYPRE_GMRESGetNumIterations( gmres_solver, &num_iterations );
  HYPRE_GMRESGetFinalRelativeResidualNorm( gmres_solver, &final_res_norm );

  results.iterations = num_iterations;
  setConvergenceFromResidual( results, final_res_norm );

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

  {
    ScopedTimer timer( solveTimerNode( "FlexGMRES", "GMRES setup" ) );
    HYPRE_ParCSRFlexGMRESSetup( gmres_solver, m_parMatrix, m_parRHS, m_parSol );
  }
  {
    ScopedTimer timer( solveTimerNode( "FlexGMRES", "GMRES solve" ) );
    HYPRE_ParCSRFlexGMRESSolve( gmres_solver, m_parMatrix, m_parRHS, m_parSol );
  }

  auto solve_end = std::chrono::high_resolution_clock::now();
  results.solveTime = std::chrono::duration<double>( solve_end - solve_start ).count();
  m_solveTime = results.solveTime;  // Save to member for getSolveTime()

  // Get statistics
  int_t num_iterations;
  real_type final_res_norm;
  HYPRE_ParCSRFlexGMRESGetNumIterations( gmres_solver, &num_iterations );
  HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm( gmres_solver, &final_res_norm );

  results.iterations = num_iterations;
  setConvergenceFromResidual( results, final_res_norm );

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

  {
    ScopedTimer timer( solveTimerNode( "GMRES", "GMRES setup" ) );
    HYPRE_ParCSRGMRESSetup( gmres_solver, m_parMatrix, m_parRHS, m_parSol );
  }
  {
    ScopedTimer timer( solveTimerNode( "GMRES", "GMRES solve" ) );
    HYPRE_ParCSRGMRESSolve( gmres_solver, m_parMatrix, m_parRHS, m_parSol );
  }

  auto solve_end = std::chrono::high_resolution_clock::now();
  results.solveTime = std::chrono::duration<double>( solve_end - solve_start ).count();
  m_solveTime = results.solveTime;  // Save to member for getSolveTime()

  // Get statistics
  int_t num_iterations;
  real_type final_res_norm;
  HYPRE_GMRESGetNumIterations( gmres_solver, &num_iterations );
  HYPRE_GMRESGetFinalRelativeResidualNorm( gmres_solver, &final_res_norm );

  results.iterations = num_iterations;
  setConvergenceFromResidual( results, final_res_norm );

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

  {
    ScopedTimer timer( solveTimerNode( "FlexGMRES", "GMRES setup" ) );
    HYPRE_ParCSRFlexGMRESSetup( gmres_solver, m_parMatrix, m_parRHS, m_parSol );
  }
  {
    ScopedTimer timer( solveTimerNode( "FlexGMRES", "GMRES solve" ) );
    HYPRE_ParCSRFlexGMRESSolve( gmres_solver, m_parMatrix, m_parRHS, m_parSol );
  }

  auto solve_end = std::chrono::high_resolution_clock::now();
  results.solveTime = std::chrono::duration<double>( solve_end - solve_start ).count();
  m_solveTime = results.solveTime;  // Save to member for getSolveTime()

  // Get statistics
  int_t num_iterations;
  real_type final_res_norm;
  HYPRE_ParCSRFlexGMRESGetNumIterations( gmres_solver, &num_iterations );
  HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm( gmres_solver, &final_res_norm );

  results.iterations = num_iterations;
  setConvergenceFromResidual( results, final_res_norm );

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
  m_scaling.clear();
  m_rowScaling.clear();
  m_colScaling.clear();
  m_activeMGRPrecond = nullptr;
  m_activeKrylovName.clear();
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
  ScopedTimer timer( setupTimerNode( "setup total" ) );

  // Update solver parameters
  m_params.maxIter = max_iters;
  m_params.tolerance = tolerance;

  // Rebuild from the latest matrix contents on every setup().
  cleanup();

  // Compute matrix/RHS scaling if enabled.
  computeScaling();
  setupBlockLocalPreconditioner();

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
  if( m_params.useMGR && m_strategy )
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
