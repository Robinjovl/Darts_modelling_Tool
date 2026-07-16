/*
 * MGR Linear Solver - MGR Strategy Base Implementation
 */

#include "MGRStrategy.hpp"

namespace mgr {

MGRStrategy::MGRStrategy( int_t numLevels )
  : m_numLevels( numLevels )
  , m_numBlocks( 0 )
{
  m_levelParams.resize( numLevels );
}

} // namespace mgr
