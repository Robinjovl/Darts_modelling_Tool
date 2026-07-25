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
#ifndef OPENDARTS_LINEAR_SOLVERS_CSR_MATRIX_H
#define OPENDARTS_LINEAR_SOLVERS_CSR_MATRIX_H
//--------------------------------------------------------------------------
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "data_types.hpp"
#include "csr_matrix_base.hpp"

#ifdef WITH_GPU
// GPU device storage and cuSPARSE-based linear algebra. Only pulled in for
// CUDA builds; CPU builds never see the CUDA headers. See the WITH_GPU block
// at the bottom of csr_matrix for the device API.
#include <cuda_runtime_api.h>
#include <cusparse.h>
#endif // WITH_GPU

namespace opendarts
{
  namespace linear_solvers
  {
    /** @brief Matrix storage class in block CSR format.

        This class implements the storage format of block CSR sparse matrices.
        Currently, only storage functionality is implemented. No linear algebra
        is avaible. This means that matrix-matrix and matrix-vector multiplication
        are not available, for example.
        @author apalha
        @date October 2022
    */
    template <uint8_t N_BLOCK_SIZE>
    class csr_matrix : public opendarts::linear_solvers::csr_matrix_base
    {
    public:
      // Data members
      // Names mirror the proprietary darts-linear-solvers csr_matrix exactly
      // (n_block_size_ and b_sqr are declared there too); do NOT rename while the
      // BOS path is supported -- the engine is compiled against either class.
      const opendarts::config::index_t n_block_size_ = N_BLOCK_SIZE;        // block size (== N_BLOCK_SIZE template parameter)
      const opendarts::config::index_t b_sqr = N_BLOCK_SIZE * N_BLOCK_SIZE; // block size squared (scalar values per block)
      opendarts::config::index_t n_total_non_zeros;                         // the total number of non zero values is
                                                                            // equal to n_block_size * n_block_size *
                                                                            // n_non_zeros, since n_non_zeros refers to
                                                                            // nonzero blocks

      std::vector<opendarts::config::mat_float> values;          // from original csr_matrix, but changed to std::vector
      std::vector<opendarts::config::index_t> diag_ind;          // from original csr_matrix, but changed to std::vector
      std::vector<opendarts::config::index_t> cols_ind;          // from original csr_matrix, but changed to std::vector
      std::vector<opendarts::config::index_t> rows_ptr;          // from original csr_matrix, but changed to std::vector
      std::vector<opendarts::config::index_t> row_thread_starts; // from original csr_matrix, but changed to std::vector

      // Member functions

      // Initialization member functions

      /** Default Constructor.
          Initializes an empty block csr matrix with block size 0x0 and dimension 0x0.
      */
      csr_matrix();

      /** Constructs an empty block csr matrix of specific dimension.
          Initializes an empty block csr matrix with of dimension \p n_rows x \p n_cols,
          \p n_non_zeros non zero elements, and block size of \p n_block_size x \p n_block_size
          and with memory storage for \p n_non_zeros nonzero coefficients.
          @param n_rows - Number of rows of the matrix.
          @param n_cols - Number of columns of the matrix.
          @param n_non_zeros - The number of nonzero elements in the matrix. This must be set once, preferably with the
         constructor, or using csr_matrix::init
      */
      csr_matrix(opendarts::config::index_t n_rows_input,
        opendarts::config::index_t n_cols_input,
        opendarts::config::index_t n_non_zeros_input);

      /** Constructs a block csr matrix from another block csr matrix.
          @param csr_matrix - The sparse block csr matrix to use to initialize this sparse matrix.
      */
      csr_matrix(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> &csr_matrix_in);

      /** Initializes the memory storage of block csr matrix with a specific dimension.
          Initializes an empty block csr matrix with of dimension \p n_rows x \p n_cols,
          \p n_non_zeros non zero elements, and block size of \p n_block_size x \p n_block_size
          and with memory storage for \p n_non_zeros nonzero coefficients.

          NOTE: This member function should be used only once, to initialize the
                sparse matrix. Further executions of this member function will reset
                the data. The matrix will be an empty matrix (with allocated data).

          @param n_rows - Number of rows of the matrix.
          @param n_cols - Number of columns of the matrix.
          @param n_block_size - The block size of the matrix, each indexed element will be a dense matrix of \p
         n_block_size x \p n_block_size .
          @param n_non_zeros - The number of nonzero elements in the matrix. This must be set once, preferably with the
         constructor, or using this member function.
      */
      int init(opendarts::config::index_t n_rows_input, opendarts::config::index_t n_cols_input, opendarts::config::index_t n_non_zeros_input);

      /** Initializes the block csr matrix with the raw data to populate the matrix.

          NOTE: This member function should be used only once, to initialize the
                sparse matrix. Further executions of this member function will break
                the underlying data if data has already been assigned to the matrix.

          @param n_rows - Number of rows of the matrix.
          @param n_cols - Number of columns of the matrix.
          @param n_block_size - The block size of the matrix, each indexed element will be a dense matrix of \p
         n_block_size x \p n_block_size .
          @param n_non_zeros - The number of nonzero elements in the matrix. This must be set once, preferably with the
         constructor, or using this member function.
          @param values - The vector containing the data of the n_non_zeros blocks, i.e., n_non_zeros * n_block_size *
         n_block_size values.
          @param rows_ptr - The vector with the index (pointer) to the start location of the list of column indices for
         the non zero values of the row.
          @param cols_ind - The vector with the indices of the column of the block.
      */
      int init(opendarts::config::index_t n_rows_input, opendarts::config::index_t n_cols_input, opendarts::config::index_t n_non_zeros_input, const std::vector<opendarts::config::mat_float> &values, const std::vector<opendarts::config::index_t> &rows_ptr, const std::vector<opendarts::config::index_t> &cols_ind);

      /** Initializes the sparse block CSR matrix with another block csr matrix.
          Initializes the matrix as a copy of the input csr matrix \p csr_matrix.

          @param csr_matrix - CSR matrix to copy.
      */
      int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> &csr_matrix_in);

      // Data access Member functions

      // This part should be removed in the future and direct access to the values
      // of the data members should be used, instead of accessing a pointer to
      // the underlying data of std::vector. This is done for backwards compatiblity.
      // The issue is that std::vector rescales
      // its size as needed. This hides potential errors. An alternative is to use
      // setters and getters that also do checks, but this adds an overhead that
      // penalizes performance.

      /** Provides direct access to the matrix values of the sparse matrix block CSR format.
          See csr_matrix::values for a detailed description.

          It is preferrable to directly access the data member csr_matrix::values .
          This member function is provided for backwards compatibility.

          @return pointer to the data memory array of the std::vector containing
          the nonzero values of the sparse array (csr_matrix::values).
      */
      opendarts::config::mat_float *get_values() override; // from original csr_matrix to keep compatibility

      /** Provides direct access to the row pointer data structure of the sparse matrix block CSR format.
          See csr_matrix::rows_ptr for a detailed description.

          It is preferrable to directly access the data member csr_matrix::rows_ptr .
          This member function is provided for backwards compatibility.

          @return pointer to the data memory array of the std::vector containing
          the rows pointers of the sparse array (csr_matrix::rows_ptr).
      */
      opendarts::config::index_t *get_rows_ptr() override; // from original csr_matrix to keep compatibility

      /** Provides direct access to the column index data structure of the sparse matrix block CSR format.
          See csr_matrix::cols_ind for a detailed description.

          It is preferrable to directly access the data member csr_matrix::cols_ind .
          This member function is provided for backwards compatibility.

          @return pointer to the data memory array of the std::vector containing
          the column indices of the sparse array (csr_matrix::rows_ptr).
      */
      opendarts::config::index_t *get_cols_ind() override; // from original csr_matrix to keep compatibility

      /** Provides direct access to the index data structure of the diagonal elements
          of the sparse matrix block CSR format.

          It is preferrable to directly access the data member csr_matrix::diag_ind .
          This member function is provided for backwards compatibility.

          @return pointer to the data memory array of the std::vector containing
          the indices of the diagonal elements of the sparse array (csr_matrix::diag_ind).
      */
      opendarts::config::index_t *get_diag_ind() override; // from original csr_matrix to keep compatibility

      /** The number of nonzero elements in the sparse matrix.
          See csr_matrix::cols_ind for a detailed description.

          It is preferrable to directly access the data member csr_matrix::n_non_zeros .
          This member function is provided for backwards compatibility.

          @return the number of nonzero elements in the spare matrix.
      */
      opendarts::config::index_t get_n_non_zeros(); // from original csr_matrix, direct access to
                                                    // this->n_non_zeros is more direct

      // Matrix transformation

      /** Computes a copy of this matrix with a block size 1.
          The block size 1 matrix is equivalent to this matrix, i.e.,
          it has the same nonzero values at the same (row, column) indices. The
          only difference is that formally it has a block size of 1 instead the
          block size of this matrix.

          @param csr_matrix_nb_1 - The block size 1 matrix where to convert this matrix.
      */
      void as_nb_1(opendarts::linear_solvers::csr_matrix<1> &csr_matrix_nb_1) const;

      /** Computes a copy of this matrix with a block size 1.
          The block size 1 matrix is equivalent to this matrix, i.e.,
          it has the same nonzero values at the same (row, column) indices. The
          only difference is that formally it has a block size of 1 instead the
          block size of this matrix.

          NOTE: this function takes as input a pointer, if this is a block size 1
                matrix then csr_matrix_nb_1 will point to this.

          @param csr_matrix_nb_1 - The block size 1 matrix where to convert this matrix.
      */
      template <uint8_t D = N_BLOCK_SIZE>
      typename std::enable_if<D == 1, void>::type as_nb_1(opendarts::linear_solvers::csr_matrix<1> *&csr_matrix_nb_1)
      {
        // Since the original matrix is already block size 1, just make csr_matrix_nb_1 point to this.
        csr_matrix_nb_1 = this;
      }

      /** Computes a copy of this matrix with a block size 1.
          The block size 1 matrix is equivalent to this matrix, i.e.,
          it has the same nonzero values at the same (row, column) indices. The
          only difference is that formally it has a block size of 1 instead the
          block size of this matrix.

          NOTE: this function takes as input a pointer, if this is a block size 1
                matrix then csr_matrix_nb_1 will point to this.

          @param csr_matrix_nb_1 - The block size 1 matrix where to convert this matrix.
      */
      template <uint8_t D = N_BLOCK_SIZE>
      typename std::enable_if<(D > 1), void>::type as_nb_1(opendarts::linear_solvers::csr_matrix<1> *&csr_matrix_nb_1)
      {
        // In this case we always need to copy, so just use the other function
        this->as_nb_1(*csr_matrix_nb_1);
      }

      /** Computes the transpose of this matrix.

          @param csr_matrix_transpose - The transpose matrix.
      */
      void transpose(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> &csr_matrix_transpose) const;

      /** Computes the transpose matrix vector product.
          r = A^t v

          @param v - The vector to apply the transpose of the matrix A^t.
          @param r - The vector with the result of the matrix vector product.
      */
      int matrix_vector_product_t(opendarts::config::mat_float *v, opendarts::config::mat_float *r);

      // Input/Output
      int export_matrix_to_file(const std::string &filename, opendarts::linear_solvers::sparse_matrix_export_format export_format) override;

      int import_matrix_from_file(const std::string &filename, opendarts::linear_solvers::sparse_matrix_import_format import_format) override;

    private:
      /** Writes the matrix to file in human_readable format.
          The sparse matrix will be written to file as a matrix, in the format
          -----------------------------
                  0      1     ...
          -----------------------------
            0 |   0      3.4    ...
            1 |   0.0    0      ...
          ... |   ...    ...    ...

          NOTE: The blocks are expanded. This means that a block matrix is shown as
                as if first converted to block size 1 and then saved.

          @param filename - The name of the file where to save the matrix.
          @return the error code:
             0  : no error
             10 : invalid matrix export format
      */
      int export_matrix_to_file_human_readable(const std::string &filename);

      /** Writes the matrix to file in csr format.
          The sparse matrix will be written to file in csr matrix format. For example
          The matrix n x n matrix

          1.0   0.0     0     0  -1.0  -3.0
          2.0   3.0     0     0  -2.0   0.0
            0     0     0     0     0     0
            0     0     0     0     0     0
            0     0     0     0   5.0   7.0
            0     0     0     0   6.0   8.0

          Can be stored as a 3 x 3 matrix of block size 2.

          NOTE 1: because we chose a block size 2, some of the zeros are real stored
          values and we have displayed them as 0.0. These are zeros within a
          block and will have to be stored. This is in contrast to the off block
          zeros, which are part of the sparsity pattern of the matrix, and therefore
          are represented as 0.

          NOTE 2: There are 3 non zero blocks in this matrix. Therefore n_non_zeros = 3.

          The output of this matrix will be:

          // N_ROWS	N_COLS	N_NON_ZEROS	N_BLOCK_SIZE
          3	3	3	2
          // Rows indexes[1..n_rows] (with out 0)
          2
          2
          4
          // END of Rows indexes
          // Values n_non_zeros elements
          // COLUMN	VALUE
          // ROW 0
          0	 1.0  0.0  2.0  3.0
          2	-1.0 -3.0 -2.0  0.0
          // ROW 1
          // ROW 2
          2	 5.0  7.0  6.0  8.0
          // END OF VALUES
          // END OF FILE

          @param filename - The name of the file where to save the matrix.
          @return the error code:
             0  : no error
             10 : invalid matrix export format
      */
      int export_matrix_to_file_csr(const std::string &filename);

      /** Reads the matrix from file in csr format.
          The sparse matrix will be read from file in csr matrix format. For example

          // N_ROWS	N_COLS	N_NON_ZEROS	N_BLOCK_SIZE
          3	3	3	2
          // Rows indexes[1..n_rows] (with out 0)
          2
          2
          4
          // END of Rows indexes
          // Values n_non_zeros elements
          // COLUMN	VALUE
          // ROW 0
          0	 1.0  0.0  2.0  3.0
          2	-1.0 -3.0 -2.0  0.0
          // ROW 1
          // ROW 2
          2	 5.0  7.0  6.0  8.0
          // END OF VALUES
          // END OF FILE

          Will results in a 3x3 matrix of block size 2x2:

          1.0   0.0     0     0  -1.0  -3.0
          2.0   3.0     0     0  -2.0   0.0
            0     0     0     0     0     0
            0     0     0     0     0     0
            0     0     0     0   5.0   7.0
            0     0     0     0   6.0   8.0

          NOTE 1: because we chose a block size 2, some of the zeros are real stored
          values and we have displayed them as 0.0. These are zeros within a
          block and are stored. This is in contrast to the off block
          zeros, which are part of the sparsity pattern of the matrix, and therefore
          are represented as 0.

          NOTE 2: There are 3 non zero blocks in this matrix. Therefore n_non_zeros = 3.

          @param filename - The name of the file where to save the matrix.
          @return the error code:
              0  : no error
             10  : invalid matrix import format
            100  : incompatible block size
      */
      int import_matrix_from_file_csr(const std::string &filename);

    public:
      // The initializers below mirror the proprietary darts-linear-solvers
      // csr_matrix so the engine compiles against either matrix. Retained for
      // BOS interface parity, not transitional.

      // Mirrors proprietary init(n_rows, n_cols, n_blok_size, n_non_zeros, row_thread_starts).
      // n_block_size_input is redundant here (the block size is the template parameter);
      // the implementation warns if it differs from n_block_size_ and otherwise ignores
      // it -- the argument exists only to match the proprietary signature.
      int init(opendarts::config::index_t n_rows_input, opendarts::config::index_t n_cols_input, opendarts::config::index_t n_block_size_input, opendarts::config::index_t n_non_zeros_input, opendarts::config::index_t *new_row_thread_starts = 0);

      int init_struct(opendarts::config::index_t n_rows_input, opendarts::config::index_t n_cols_input, opendarts::config::index_t n_non_zeros_input);

      /** Provides direct access to the start row index for each thread.

          This member function is provided for backwards compatibility. Currently no functionality
          exists for more than one thread.

          @return pointer to the data memory array of the std::vector containing
          the start row indices for each thread for the sparse array.
      */
      opendarts::config::index_t *get_row_thread_starts() override; // from original csr_matrix_base to keep compatibility

      // Pointer-taking init (mirrors proprietary init(csr_matrix*)); used by the
      // FS-CPR solver's sub-block matrices (linsolv_fs_cpr) and the solver tests.
      int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *csr_matrix_in);

      // build_transpose[_struct] mirror the proprietary csr_matrix; used by the
      // multi-point (super_mp) engine and the transpose unit tests. They wrap
      // transpose() (full values, resp. structure-only) behind the proprietary
      // signature (which takes a source pointer rather than a destination ref).
      int build_transpose(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *csr_matrix_in);

      int build_transpose_struct(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *csr_matrix_in);

      // Cross-block-size scalar (block-size-1) expansion; complements as_nb_1 (which
      // maps to this matrix's own block size). Mirrors proprietary to_nb_1; used by
      // the engines and FS-CPR. See also the polymorphic csr_matrix_base overload below.
      template <uint8_t M_BLOCK_SIZE>
      int to_nb_1(const opendarts::linear_solvers::csr_matrix<M_BLOCK_SIZE> *csr_matrix_in);

      // Polymorphic block-CSR -> scalar-CSR expansion: works for any
      // csr_matrix_base (legacy csr_matrix<N> or the unified block_csr_matrix)
      // through the get_*() accessor interface and n_row_size block size.
      int to_nb_1(opendarts::linear_solvers::csr_matrix_base *csr_matrix_in);

      // ----------------------------------------------------------------------
      // GPU device layer
      // ----------------------------------------------------------------------
      // Mirrors the host block-CSR storage on a CUDA device and provides
      // cuSPARSE-backed block sparse matrix-vector products. Ported from the
      // proprietary darts-linear-solvers csr_matrix; the open-source GPU
      // solvers (AMGX, cuSPARSE ILU, BiCGStab, ...) build on this layer.
      //
      // Set to 1 once device mode is enabled via init_device().
      // Declared unconditionally so non-GPU callers can cheaply test it.
      int gpu_mode = 0;

#ifdef WITH_GPU
      // Device copies of the block-CSR structure and values.
      opendarts::config::mat_float *values_d = nullptr;   // nonzero block values
      opendarts::config::index_t *rows_ptr_d = nullptr;   // block row pointers
      opendarts::config::index_t *cols_ind_d = nullptr;   // block column indices
      opendarts::config::index_t *diag_ind_d = nullptr;   // diagonal block indices

      // Scratch CSR copy used by convert_to_ELL (block-CSR expanded to scalar CSR).
      opendarts::config::index_t *csrRowPtrC = nullptr;
      opendarts::config::index_t *csrColIndC = nullptr;
      opendarts::config::mat_float *csrValC = nullptr;

      // Device work vectors (input v_d, result r_d) and a host check buffer.
      opendarts::config::mat_float *v_d = nullptr;
      opendarts::config::mat_float *r_d = nullptr;
      opendarts::config::mat_float *r_check = nullptr;

      cusparseMatDescr_t cus_descr = nullptr;
      cusparseHandle_t cus_handle = nullptr;

      /** Allocates the device buffers and initialises the cuSPARSE handle.
          @param n_rows_input - number of block rows.
          @param nnz - number of nonzero blocks.
          @return 0 on success, nonzero on failure. */
      int init_device(int n_rows_input, int nnz);

      /** Copies the block-CSR structure (rows_ptr, cols_ind, diag_ind) host->device. */
      int copy_struct_to_device() override;

      /** Copies the block values host->device. */
      int copy_values_to_device() override;

      // csr_matrix_base device-pointer accessors (block-CSR storage on device).
      opendarts::config::mat_float *get_values_d() override { return values_d; }
      opendarts::config::index_t *get_rows_ptr_d() override { return rows_ptr_d; }
      opendarts::config::index_t *get_cols_ind_d() override { return cols_ind_d; }
      opendarts::config::index_t *get_diag_ind_d() override { return diag_ind_d; }

      /** Copies an arbitrary host vector into a device buffer (n_rows * N_BLOCK_SIZE entries). */
      int copy_vector_to_device(const opendarts::config::mat_float *vector, opendarts::config::mat_float *vector_d);

      /** Copies a device buffer back to host (n_rows * N_BLOCK_SIZE entries). */
      int copy_vector_to_host(opendarts::config::mat_float *vector, opendarts::config::mat_float *vector_d);

      /** Expands the device block-CSR matrix into a scalar CSR copy (csr*C buffers). */
      int convert_to_ELL();

      /** r_d += A * v_d (cuSPARSE block SpMV). */
      int matrix_vector_product_d(const double *v_d, double *r_d) override;

      /** r_d  = A * v_d (cuSPARSE block SpMV, result overwritten). */
      int matrix_vector_product_d0(const double *v_d, double *r_d) override;

      /** Scalar-CSR SpMV; HYB path retired in CUDA 11+, kept for API parity. */
      int matrix_vector_product_d_ell(const double *v_d, double *r_d) override;

      /** r_d = alpha * A * v_d + beta * r_d (non-virtual, concrete-type helper). */
      int calc_lin_comb_d(double alpha, double *v_d, double beta, double *r_d);

      /** r = alpha * A * u + beta * v, all operands device pointers. */
      int calc_lin_comb_d(const double alpha, const double beta, double *u, double *v, double *r) override;

      /** Releases all device buffers and the cuSPARSE handle. */
      int free_device();
#endif // WITH_GPU
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_CSR_MATRIX_H
//--------------------------------------------------------------------------
