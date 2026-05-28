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

// Maximum number of parameter-space dimensions for interpolator template instantiation.
// Defaults to 8 (historical value, formerly named MAX_NC). Can be overridden at build
// time via the CMake variable OPENDARTS_MAX_DIMS (forwarded as -DMAX_DIMS=N to the
// interpolators target). The recursive_exposer loops stamp one class per (N_DIMS,
// N_OPS) pair for N_DIMS in 1..MAX_DIMS and interpolate_with_derivatives unrolls
// O(2^N_DIMS) per instantiation, so this is the dominant knob on compile-time memory.
#ifndef MAX_DIMS
#define MAX_DIMS 8
#endif

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
