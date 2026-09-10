//*************************************************************************
//    Copyright (c) 2022
//    Delft University of Technology, the Netherlands
//    Netherlands eScience Center
//
//    This file is part of the open Delft Advanced Research Terra Simulator (opendarts)
//
//    opendarts is free software: you can redistribute it and/or modify
//    it under the terms of the Apache License.
//
//    DARTS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// *************************************************************************

//--------------------------------------------------------------------------
#ifndef OPENDARTS_LINEAR_SOLVERS_OMP_PARTITION_HPP
#define OPENDARTS_LINEAR_SOLVERS_OMP_PARTITION_HPP
//--------------------------------------------------------------------------

// omp_partition -- the block-row thread partition used by the OpenMP-parallel
// engine Jacobian assembly (and the NUMA first-touch that precedes it).
//
// The engine's assembly parallel regions give every thread a contiguous range
// of block rows [row_thread_starts[id], row_thread_starts[id + 1]) and let it
// write only its own rows, so the assembly is race-free as long as the matrix
// hands back a partition sized to the team that will run those regions. This
// header centralises how that partition is built -- the open-source port of the
// reference CALCULATE_START_END(n_rows) default split in darts-linear-solvers'
// omp_tools.h (which the original open-source matrix had stubbed out to a single
// thread, disabling multi-threaded assembly).

#include "data_types.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace opendarts
{
  namespace linear_solvers
  {
    /** Number of threads the engine's OpenMP assembly / SpMV parallel regions
        will run with. This mirrors omp_get_max_threads(): the team size of a
        plain `#pragma omp parallel` when dynamic teams are disabled (the engine
        disables them at init so the running team matches this partition). Builds
        without OpenMP return 1, so the partition collapses to a single range. */
    inline int omp_assembly_n_threads()
    {
#ifdef _OPENMP
      const int nt = omp_get_max_threads();
      return nt > 0 ? nt : 1;
#else
      return 1;
#endif
    }

    /** Fills out[0 .. n_threads] with a contiguous, write-disjoint even split of
        the block-row range [0, n_block_rows): out[t] = n_block_rows * t /
        n_threads. Thread t owns rows [out[t], out[t + 1]); the union covers every
        row exactly once (empty ranges when n_block_rows < n_threads), which keeps
        the per-thread Jacobian assembly race-free. @p out must hold n_threads + 1
        entries. */
    inline void fill_even_row_partition(opendarts::config::index_t *out,
      opendarts::config::index_t n_block_rows, int n_threads)
    {
      if (n_threads < 1)
        n_threads = 1;
      for (int t = 0; t <= n_threads; ++t)
        out[t] = static_cast<opendarts::config::index_t>(
          static_cast<long long>(n_block_rows) * t / n_threads);
    }
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_OMP_PARTITION_HPP
//--------------------------------------------------------------------------
