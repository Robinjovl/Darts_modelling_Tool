/*
 * MGR Linear Solver - Standalone Linear Solver Library
 * Multigrid Reduction Solver for Block CSR Matrices
 * Copyright (c) 2025
 *
 * Type definitions and configuration
 * @author Xiaoming Tian
 */

#ifndef MGR_LINEAR_SOLVER_TYPES_HPP_
#define MGR_LINEAR_SOLVER_TYPES_HPP_

#include <cstddef>
#include <vector>
#include <cstdint>

// HYPRE types
// Note: _hypre_utilities.h already guards its own declarations with
// `#ifdef __cplusplus / extern "C"` internally. Wrapping the #include itself
// in an outer extern "C" block also pulls in HYPRE's own `#include <omp.h>`
// (which precedes HYPRE's internal guard) under C linkage. Newer libgomp
// omp.h (GCC 15+) declares C++ template overloads there, which is illegal
// under extern "C" ("template with C linkage").
#include <_hypre_utilities.h>

namespace mgr {

// Integer types (use HYPRE types directly for compatibility)
using int_t = int;
using bigint_t = HYPRE_BigInt;
using size_type = std::ptrdiff_t;

// Floating point type
using real_type = HYPRE_Real;

// Forward declarations
class LinearSolver;
class MGRStrategy;
class MGRPreconditioner;

} // namespace mgr

#endif // MGR_LINEAR_SOLVER_TYPES_HPP_
