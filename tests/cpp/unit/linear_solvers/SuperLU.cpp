#include <cstdio>
#include <stdio.h>
#include <string>

#include "slu_ddefs.h"

#include "openDARTS/config/version.hpp"
#include "test_common.hpp"

int test_SuperLU(std::string &output_filename, std::string &reference_filename);

int main()
{
  /*
    test_05__SuperLU
    Tests the correct linking to SuperLU in opendarts::linear_solvers. Implements
    an example from SuperLU (LU decomposition for a matrix A). The matrix A, and
    the LU decomposition are compared to a reference result.
  */

  // The output file to save the matrix to
  std::string output_filename("test_05__SuperLU.txt");

  //  The reference ascii file
  std::string data_path_prefix = opendarts::config::get_cmake_openDARTS_source_dir() +
                                 std::string("/tests/cpp/data/tests/linear_solvers/"); // the path to the data folder
  std::string reference_base_filename("test_05__SuperLU_ref.txt");           // the filename of the reference file
  std::string reference_filename = data_path_prefix +
                                   reference_base_filename; // get the full path to the reference file

  int error_output = 1;

  // Test SuperLU functionality
  error_output = test_SuperLU(output_filename, reference_filename);

  return error_output;
}

int test_SuperLU(std::string &output_filename, std::string &reference_filename)
{
  /*! \file
    Copyright (c) 2003, The Regents of the University of California, through
    Lawrence Berkeley National Laboratory (subject to receipt of any required
    approvals from U.S. Dept. of Energy)
    All rights reserved.
    The source code is distributed under BSD license, see the file License.txt
    at the top-level directory.
    */
  /*! @file superlu.c
   * \brief a small 5x5 example
   *
   * <pre>
   * * -- SuperLU routine (version 2.0) --
   * Univ. of California Berkeley, Xerox Palo Alto Research Center,
   * and Lawrence Berkeley National Lab.
   * November 15, 1997
   * </pre>
   */

  /*
    Generates a 5x5 matrix A and right hand size B with SuperLU. Then solves the system
        A x = B
    by computing the LU decomposition of A.
    The output (A, L, U) is sent to file, by redirecting the cout output (since we are using
    SuperLU functions).
  */
  // Redirect all output to file
  freopen(output_filename.data(), "w", stdout);

  // Error checking variables
  int error_output = 1;
  bool files_are_equal = false;

  // Start the test
  SuperMatrix A, L, U, B;
  double *a, *rhs;
  double s, u, p, e, r, l;
  int *asub, *xa;
  int *perm_r; /* row permutations from partial pivoting */
  int *perm_c; /* column permutation vector */
  int nrhs, info, i, m, n, nnz;
  superlu_options_t options;
  SuperLUStat_t stat;

  /* Initialize matrix A. */
  m = n = 5;
  nnz = 12;
  if (!(a = doubleMalloc(nnz)))
    ABORT("Malloc fails for a[].");
  if (!(asub = intMalloc(nnz)))
    ABORT("Malloc fails for asub[].");
  if (!(xa = intMalloc(n + 1)))
    ABORT("Malloc fails for xa[].");
  s = 19.0;
  u = 21.0;
  p = 16.0;
  e = 5.0;
  r = 18.0;
  l = 12.0;
  a[0] = s;
  a[1] = l;
  a[2] = l;
  a[3] = u;
  a[4] = l;
  a[5] = l;
  a[6] = u;
  a[7] = p;
  a[8] = u;
  a[9] = e;
  a[10] = u;
  a[11] = r;
  asub[0] = 0;
  asub[1] = 1;
  asub[2] = 4;
  asub[3] = 1;
  asub[4] = 2;
  asub[5] = 4;
  asub[6] = 0;
  asub[7] = 2;
  asub[8] = 0;
  asub[9] = 3;
  asub[10] = 3;
  asub[11] = 4;
  xa[0] = 0;
  xa[1] = 3;
  xa[2] = 6;
  xa[3] = 8;
  xa[4] = 10;
  xa[5] = 12;

  /* Create matrix A in the format expected by SuperLU. */
  dCreate_CompCol_Matrix(&A, m, n, nnz, a, asub, xa, SLU_NC, SLU_D, SLU_GE);

  /* Create right-hand side matrix B. */
  nrhs = 1;
  if (!(rhs = doubleMalloc(m * nrhs)))
    ABORT("Malloc fails for rhs[].");
  for (i = 0; i < m; ++i)
    rhs[i] = 1.0;
  dCreate_Dense_Matrix(&B, m, nrhs, rhs, m, SLU_DN, SLU_D, SLU_GE);

  if (!(perm_r = intMalloc(m)))
    ABORT("Malloc fails for perm_r[].");
  if (!(perm_c = intMalloc(n)))
    ABORT("Malloc fails for perm_c[].");

  /* Set the default input options. */
  set_default_options(&options);
  options.ColPerm = NATURAL;

  /* Initialize the statistics variables. */
  StatInit(&stat);

  /* Solve the linear system. */
  dgssv(&options, &A, perm_c, perm_r, &L, &U, &B, &stat, &info);

  std::string A_name("A");
  std::string U_name("U");
  std::string L_name("L");
  std::string perm_name("\nperm_r");
  dPrint_CompCol_Matrix(const_cast<char *>(A_name.data()), &A);
  dPrint_CompCol_Matrix(const_cast<char *>(U_name.data()), &U);
  dPrint_SuperNode_Matrix(const_cast<char *>(L_name.data()), &L);
  print_int_vec(const_cast<char *>(perm_name.data()), m, perm_r);

  /* De-allocate storage */
  SUPERLU_FREE(rhs);
  SUPERLU_FREE(perm_r);
  SUPERLU_FREE(perm_c);
  Destroy_CompCol_Matrix(&A);
  Destroy_SuperMatrix_Store(&B);
  Destroy_SuperNode_Matrix(&L);
  Destroy_CompCol_Matrix(&U);
  StatFree(&stat);

  std::cout << std::flush;

  // Compare the generated output to the reference output
  files_are_equal = opendarts::linear_solvers::testing::compare_files(output_filename,
      reference_filename); // check if equal to reference

  if (files_are_equal)
  {
    error_output = 0;
  }
  else
  {
    error_output = 1;
  }

  return error_output;
}
