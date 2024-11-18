#include <stdexcept>
#include <string>

#include "openDARTS/config/data_types.hpp"

namespace opendarts
{
  namespace config
  {
    // Casts size_t to int, but check first if  there will be no overflow
    opendarts::config::index_t static_cast_check(size_t size_t_value)
    {
      if (size_t_value >
          opendarts::config::INDEX_T_MAX) // check if it is larger than the maximum value of opendarts::config::index_t,
                                          // if so throw and exception since this cannot be handled in the loops
      {
        throw std::overflow_error("Size of vector is larger than what is supported by opendarts::config::index_t");
      }
      return static_cast<opendarts::config::index_t>(size_t_value);
    }

  } // namespace config
} // namespace opendarts
