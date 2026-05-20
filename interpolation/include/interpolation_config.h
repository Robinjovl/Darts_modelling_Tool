#ifndef INTERPOLATION_CONFIG_H
#define INTERPOLATION_CONFIG_H

// Minimal shared configuration for the interpolation library.
// Provides timer_node, basic typedefs, and platform helpers
// without depending on engines/src/globals.h.

#include "timer_node.h"

#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <functional>
#include <string>
#include <limits>

typedef int index_t;
typedef double value_t;
typedef int interp_index_t;
typedef double interp_value_t;

// Maximum number of parameter-space dimensions for interpolator template instantiation
// It was MAX_NC before
#define MAX_DIMS 8

// workaround for vscode grammar checker
#ifdef __INTELLISENSE__
#define __global__
#define __constant__
#endif

#ifdef WITH_GPU
extern int device_num;
#endif

// __uint128_t support was removed once adaptive storage migrated to signed multi-index
// keys (cell_key_t<N_DIMS>). Previously it was needed to hold the mixed-radix packed
// integer for the dense enumeration of N_DIMS=20 hypercube grids; that enumeration
// is gone. uint64_t is sufficient for n_points_total / n_points_used counters; cells
// past the uint64 range still work via the multi-index path, with n_points_total_fp
// (double) as the authoritative diagnostic.

#endif /* INTERPOLATION_CONFIG_H */
