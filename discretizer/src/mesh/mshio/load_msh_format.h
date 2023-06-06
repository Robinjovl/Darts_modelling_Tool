#pragma once
#include "mshspec.h"
#include <iostream>

namespace mshio {

void load_mesh_format(std::istream& in, MshSpec& spec);

}
