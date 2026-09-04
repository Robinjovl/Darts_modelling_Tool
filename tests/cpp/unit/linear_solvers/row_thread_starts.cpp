#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>

#include "version.hpp"
#include "csr_matrix.hpp"
#include "linear_solvers_data_types.hpp"
#include "omp_partition.hpp"

#include "test_common.hpp"

int main()
{
  /*
    test_07__row_thread_starts
    Tests csr_matrix.get_row_thread_starts().
    Generates a tridiagonal matrix with block size 1 and checks that the row
    thread partition matches the contract documented in omp_partition.hpp: an
    even, write-disjoint split of the row range [0, n) across the engine's
    OpenMP assembly team (omp_assembly_n_threads()). The partition therefore
    starts at 0, ends at n, is monotonic non-decreasing, and equals
    n * t / n_threads at every boundary t. With OpenMP disabled (or
    OMP_NUM_THREADS=1) the team is a single thread and this collapses to the
    original single-range [0, n] expectation.

    Note: before OpenMP was enabled by default for the open-source build this
    test hard-coded row_thread_starts == [0, n]. That single-thread assumption
    breaks once csr_matrix::init sizes the partition to omp_get_max_threads()
    (> 1 on a multi-core runner), so the check is now thread-count aware.
  */

  int error_output = 0;
  opendarts::config::index_t n = 12;
  opendarts::config::index_t *row_thread_starts;

  // Generate the matrix
  opendarts::linear_solvers::csr_matrix<1> A;
  opendarts::linear_solvers::testing::generate_tridiagonal_matrix(A, n); // populate the matrix, in this case a
                                                                         // tridiagonal matrix with the values
                                                                         // -2, 1, 2 in the -2, 0, and 2 diagonals

  // csr_matrix::init sizes row_thread_starts to the same OpenMP team the engine
  // assembly regions run with, so reproduce that team size here to know how many
  // partition boundaries to expect.
  const int n_threads = opendarts::linear_solvers::omp_assembly_n_threads();

  // Get row_thread_starts and check it is the expected even split [0, ..., n].
  row_thread_starts = A.get_row_thread_starts();
  if (row_thread_starts[0] != 0) error_output += 1;
  if (row_thread_starts[n_threads] != n) error_output += 1;
  for (int t = 0; t <= n_threads; ++t)
  {
    const opendarts::config::index_t expected =
        static_cast<opendarts::config::index_t>(static_cast<long long>(n) * t / n_threads);
    if (row_thread_starts[t] != expected) error_output += 1;     // matches fill_even_row_partition()
    if (t > 0 && row_thread_starts[t] < row_thread_starts[t - 1]) error_output += 1; // monotonic non-decreasing
  }

  return error_output;
}
