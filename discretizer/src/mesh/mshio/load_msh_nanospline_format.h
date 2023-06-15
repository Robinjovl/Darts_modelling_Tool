#pragma once

#include "mshspec.h"

#include <istream>

namespace mshio
{

inline void load_nanospline_format(std::istream& in, MshSpec& spec)
{
    in >> spec.nanospline_format.version;
}

}
