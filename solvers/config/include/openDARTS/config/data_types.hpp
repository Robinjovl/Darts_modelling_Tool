//--------------------------------------------------------------------------
#ifndef OPENDARTS_CONFIG_DATA_TYPES_H
#define OPENDARTS_CONFIG_DATA_TYPES_H
//--------------------------------------------------------------------------

#include <climits>

namespace opendarts
{
  namespace config
  {
    // NOTE: This should be in an opendarts data_types.hpp file of some sort, global to all opendarts
    typedef int index_t; // define the matrix index type
    const unsigned long long int INDEX_T_MAX = INT_MAX;
    typedef double mat_float; // define data type of matrix values

    opendarts::config::index_t static_cast_check(size_t size_t_value);
  } // namespace config
} // namespace opendarts
//--------------------------------------------------------------------------
#endif // OPENDARTS_CONFIG_DATA_TYPES_H
//--------------------------------------------------------------------------
