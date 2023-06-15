#pragma once

#include "mshspec.h"

#include <istream>

namespace mshio {

void load_patches(std::istream& in, MshSpec& spec);

}
