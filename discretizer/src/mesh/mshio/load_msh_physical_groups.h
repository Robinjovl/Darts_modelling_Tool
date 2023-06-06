#pragma once
#include "mshspec.h"
#include <iostream>

namespace mshio {

void load_physical_groups(std::istream& in, MshSpec& spec);

} // namespace mshio
