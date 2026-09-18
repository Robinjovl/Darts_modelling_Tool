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
#ifndef OPENDARTS_LINEAR_SOLVERS_CSR_MATRIX_BASE_H
#define OPENDARTS_LINEAR_SOLVERS_CSR_MATRIX_BASE_H
//--------------------------------------------------------------------------

#include <string>

#include "data_types.hpp"
#include "linear_solvers_data_types.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** Abstract base sparse matrix for storage.
     *  Defines the base class for sparse matrix storage. This
     *  is a low level sparse matrix implementation containing only the storage
     *  of data in sparse matrix format. The objects of this class are not
     *  capable of performing linear algebra operations, i.e., matrix-matrix multiplication,
     *  matrix-vector multiplication.
     *
     *  This class
     */
    class csr_matrix_base
    {
    public:
      // Data members
      // The three fields below mirror the proprietary darts-linear-solvers
      // csr_matrix_base public layout. The engine and the BOS solvers are
      // compiled against whichever matrix the build selects (open-source here,
      // proprietary when ENABLE_BOS_SOLVERS=ON), so this class keeps the same
      // data members. They are load-bearing, not transitional -- do not
      // rename/retype while the BOS path is supported.
      opendarts::linear_solvers::sparse_matrix_type type;  // matrix-kind tag; proprietary stores it as int, here already the strong enum. Only ever set to MATRIX_TYPE_UNDEFINED.
      int is_square;  // 1 if n_rows == n_cols; int (not bool) for proprietary parity. Read by the adjoint assembly (engine_base.cpp).
      int n_row_size;  // rows per block (i.e. the block size); used e.g. for Jacobian value-array sizing on GPU (engine_base_gpu.cpp).
      opendarts::config::index_t n_rows;
      opendarts::config::index_t n_cols;
      opendarts::config::index_t n_non_zeros;

      // Member functions

      // Initialization member functions
      csr_matrix_base();
      virtual ~csr_matrix_base(){}; // needed when calling delete on an inherited class object


      // Data access Member Functions
      /** Provides direct access to the matrix values of the sparse matrix block CSR format.
          See csr_matrix::values for a detailed description.

          It is preferrable to directly access the data member csr_matrix::values .
          This member function is provided for backwards compatibility.

          @return pointer to the data memory array of the std::vector containing
          the nonzero values of the sparse array (csr_matrix::values).
      */
      virtual opendarts::config::mat_float *get_values() = 0; // mirrors proprietary csr_matrix; primary Jacobian accessor used throughout the engines

      /** Provides direct access to the row pointer data structure of the sparse matrix block CSR format.
          See csr_matrix::rows_ptr for a detailed description.

          It is preferrable to directly access the data member csr_matrix::rows_ptr .
          This member function is provided for backwards compatibility.

          @return pointer to the data memory array of the std::vector containing
          the rows pointers of the sparse array (csr_matrix::rows_ptr).
      */
      virtual opendarts::config::index_t *get_rows_ptr() = 0; // mirrors proprietary csr_matrix; primary Jacobian accessor used throughout the engines

      /** Provides direct access to the column index data structure of the sparse matrix block CSR format.
          See csr_matrix::cols_ind for a detailed description.

          It is preferrable to directly access the data member csr_matrix::cols_ind .
          This member function is provided for backwards compatibility.

          @return pointer to the data memory array of the std::vector containing
          the column indices of the sparse array (csr_matrix::rows_ptr).
      */
      virtual opendarts::config::index_t *get_cols_ind() = 0; // mirrors proprietary csr_matrix_base; primary Jacobian accessor used throughout the engines


      //! return rows_ptr array
      virtual opendarts::config::index_t *get_diag_ind() = 0; // mirrors proprietary csr_matrix_base; primary Jacobian accessor used throughout the engines


      /** Provides direct access to the start row index for each thread.

          This member function is provided for backwards compatibility. Currently no functionality
          exists for more than one thread.

          @return pointer to the data memory array of the std::vector containing
          the start row indices for each thread for the sparse array.
      */
      virtual opendarts::config::index_t *get_row_thread_starts() = 0;  // mirrors proprietary csr_matrix_base; used by the OpenMP assembly partition (see csr_matrix override)


      // Input/Output
      virtual int export_matrix_to_file(const std::string &filename,
          opendarts::linear_solvers::sparse_matrix_export_format export_format) = 0; // from original csr_matrix but
                                                                                     // generalised

      virtual int import_matrix_from_file(const std::string &filename,
          opendarts::linear_solvers::sparse_matrix_import_format import_format) = 0; // from original, but generalised


      // The three members below mirror the proprietary csr_matrix API. They must
      // resolve when the engine is compiled against either matrix, so they are
      // retained for BOS interface parity -- not slated for removal.

      int write_matrix_to_file(const char *filename, int sort_cols = 0);  // debug matrix dump; used live by the engines (e.g. engine_base_gpu.cpp:509, engine_pm_cpu.cpp:1715, engine_base::test_assembly)

      // Matrix-vector product r += A * v (v input, r output). NOTE: this is a
      // non-virtual no-op stub on the base, and csr_matrix<N> provides no host
      // matrix_vector_product of its own (only the transpose variant
      // matrix_vector_product_t), so it is non-functional in the open-source
      // build: its one live caller (engine_base::test_spmv, a benchmark) hits the
      // stub, and the mechanics-contact caller is commented out. Retained for BOS
      // interface parity (the proprietary base implements it).
      int matrix_vector_product(const double *v, double *r);

      // Linear combination r = alpha * A u + beta * v. Same status as
      // matrix_vector_product above: base no-op stub, no live open-source caller
      // (the impl emits a one-time deprecation note), kept for BOS interface parity.
      int calc_lin_comb(const double alpha, const double beta, double *u, double *v, double *r);

#ifdef WITH_GPU
      // GPU device layer, exposed polymorphically so the open-source GPU
      // solvers can drive a matrix held as a csr_matrix_base*. The concrete
      // implementations live in csr_matrix<N_BLOCK_SIZE>; see csr_matrix.hpp.
      virtual int matrix_vector_product_d(const double *v, double *r) = 0;   // r += A * v
      virtual int matrix_vector_product_d0(const double *v, double *r) = 0;  // r  = A * v
      virtual int matrix_vector_product_d_ell(const double *v, double *r) = 0;
      virtual int calc_lin_comb_d(const double alpha, const double beta, double *u, double *v, double *r) = 0;
      virtual int copy_struct_to_device() = 0;
      virtual int copy_values_to_device() = 0;

      // Transposed device SpMV -- the adjoint (CPRA) solve stack needs
      // r = A^T v on the device. Defaulted to "unsupported" (-1) so only the
      // matrix types that can serve it (block_csr_matrix, via its scalar-CSR
      // device view + generic cuSPARSE SpMV) opt in; the legacy csr_matrix<N>
      // and the matrix-free engine adapter are unaffected.
      // Contract: call refresh_transpose_spmv_d() once after each value
      // change (e.g. per solver setup) before using the _t_ products.
      virtual int matrix_vector_product_t_d0(const double * /*v*/, double * /*r*/) { return -1; } // r = A^T * v
      virtual int calc_lin_comb_t_d(const double /*alpha*/, const double /*beta*/, double * /*u*/,
        double * /*v*/, double * /*r*/) { return -1; } // r = alpha * A^T * u + beta * v
      virtual int refresh_transpose_spmv_d() { return -1; } // rebuild the cached transpose/scalar view

      // Direct device-pointer access to the block-CSR storage. Lets a
      // csr_matrix_base* be driven by the GPU engines/solvers (assembly
      // kernels, cuSPARSE SpMV) without knowing the concrete matrix type.
      virtual opendarts::config::mat_float *get_values_d() = 0;  // nonzero block values on device
      virtual opendarts::config::index_t *get_rows_ptr_d() = 0;  // block row pointers on device
      virtual opendarts::config::index_t *get_cols_ind_d() = 0;  // block column indices on device
      virtual opendarts::config::index_t *get_diag_ind_d() = 0;  // diagonal block indices on device
#endif // WITH_GPU
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_CSR_MATRIX_BASE_H
//--------------------------------------------------------------------------
