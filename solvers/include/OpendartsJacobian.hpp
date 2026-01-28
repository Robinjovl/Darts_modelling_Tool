/*
 * MGR Linear Solver - Open-Darts Jacobian Compatibility Layer
 *
 * This class mimics open-darts csr_matrix_base interface
 * for seamless integration with open-darts code
 * @author Xiaoming Tian
 */

#ifndef MGR_LINEAR_SOLVER_OPENDARTS_JACOBIAN_HPP_
#define MGR_LINEAR_SOLVER_OPENDARTS_JACOBIAN_HPP_

#include "Types.hpp"
#include <vector>
#include <memory>

namespace mgr {

/**
 * @brief Open-Darts Jacobian compatibility class
 *
 * Mimics open-darts csr_matrix_base interface:
 * - values: block values (size = nnz_blocks × block_size²)
 * - rows_ptr: row pointer (CSR format, block level)
 * - cols_ind: column block indices
 * - diag_ind: diagonal indices
 * - row_thread_starts: for OpenMP threading (can be empty)
 */
class OpendartsJacobian
{
public:

  // Type aliases for compatibility with open-darts
  using index_t = int_t;
  using value_t = real_type;

  /**
   * @brief Constructor
   */
  OpendartsJacobian()
    : type( 0 )
    , n_rows( 0 )
    , n_cols( 0 )
    , is_square( 0 )
    , n_row_size( 1 )
  {}

  /**
   * @brief Destructor
   */
  ~OpendartsJacobian() = default;

  // ========================================================================
  // open-darts csr_matrix_base interface methods
  // ========================================================================

  /**
   * @brief Get values array
   * @return Pointer to values array
   */
  value_t * get_values() { return values.data(); }

  /**
   * @brief Get values array (const)
   * @return Pointer to values array
   */
  const value_t * get_values() const { return values.data(); }

  /**
   * @brief Get rows pointer array
   * @return Pointer to rows_ptr array
   */
  index_t * get_rows_ptr() { return rows_ptr.data(); }

  /**
   * @brief Get rows pointer array (const)
   * @return Pointer to rows_ptr array
   */
  const index_t * get_rows_ptr() const { return rows_ptr.data(); }

  /**
   * @brief Get diagonal indices array
   * @return Pointer to diag_ind array
   */
  index_t * get_diag_ind() { return diag_ind.data(); }

  /**
   * @brief Get diagonal indices array (const)
   * @return Pointer to diag_ind array
   */
  const index_t * get_diag_ind() const { return diag_ind.data(); }

  /**
   * @brief Get columns indices array
   * @return Pointer to cols_ind array
   */
  index_t * get_cols_ind() { return cols_ind.data(); }

  /**
   * @brief Get columns indices array (const)
   * @return Pointer to cols_ind array
   */
  const index_t * get_cols_ind() const { return cols_ind.data(); }

  /**
   * @brief Get row thread starts array (for OpenMP)
   * @return Pointer to row_thread_starts array
   *
   * Note: If row_thread_starts is empty, returns rows_ptr as fallback
   * (OpenMP code expects a valid pointer, even if not used)
   */
  index_t * get_row_thread_starts()
  {
    if( row_thread_starts.empty() )
    {
      // Fallback: return rows_ptr to avoid nullptr
      // This is safe for non-OpenMP code
      return rows_ptr.data();
    }
    return row_thread_starts.data();
  }

  /**
   * @brief Get row thread starts array (const, for OpenMP)
   * @return Pointer to row_thread_starts array
   */
  const index_t * get_row_thread_starts() const
  {
    if( row_thread_starts.empty() )
    {
      // Fallback: return rows_ptr to avoid nullptr
      return rows_ptr.data();
    }
    return row_thread_starts.data();
  }

  // ========================================================================
  // Helper methods for initialization
  // ========================================================================

  /**
   * @brief Initialize from Block CSR matrix data
   * @param n_rows_ Number of block rows
   * @param n_cols_ Number of block columns
   * @param block_size Block size (variables per cell)
   * @param nnz Number of non-zero blocks
   * @param rows_ptr_ Row pointer array
   * @param cols_ind_ Column indices array
   * @param values_ Values array
   * @param diag_ind_ Diagonal indices array (optional)
   */
  void init( index_t n_rows_,
             index_t n_cols_,
             int block_size,
             index_t nnz,
             const index_t * rows_ptr_,
             const index_t * cols_ind_,
             const value_t * values_,
             const index_t * diag_ind_ = nullptr )
  {
    n_rows = n_rows_;
    n_cols = n_cols_;
    is_square = (n_rows == n_cols) ? 1 : 0;
    n_row_size = block_size;

    // Copy arrays
    rows_ptr.assign( rows_ptr_, rows_ptr_ + n_rows + 1 );
    cols_ind.assign( cols_ind_, cols_ind_ + nnz );

    // Values size: nnz blocks × block_size²
    index_t values_size = nnz * block_size * block_size;
    values.assign( values_, values_ + values_size );

    // Diagonal indices
    if( diag_ind_ ) {
      diag_ind.assign( diag_ind_, diag_ind_ + n_rows );
    } else {
      diag_ind.clear();
    }

    // row_thread_starts - empty for now (OpenMP not implemented)
    row_thread_starts.clear();
  }

  /**
   * @brief Get number of non-zero blocks
   * @return Number of non-zero blocks
   */
  index_t get_n_non_zeros() const { return cols_ind.size(); }

  /**
   * @brief Get block size
   * @return Block size (n_row_size)
   */
  int get_block_size() const { return n_row_size; }

  /**
   * @brief Get global DOF (total scalar rows)
   * @return n_rows × n_row_size
   */
  index_t get_global_dof() const { return n_rows * n_row_size; }

  // ========================================================================
  // Public members (matching csr_matrix_base)
  // ========================================================================

  int type;                       ///< Matrix type
  index_t n_rows;                 ///< Number of block rows
  index_t n_cols;                 ///< Number of block columns
  int is_square;                  ///< Square matrix flag
  int n_row_size;                 ///< Number of rows in each block (block_size)

  // Matrix storage
  std::vector<value_t> values;            ///< Block values
  std::vector<index_t> rows_ptr;          ///< Row pointers (CSR)
  std::vector<index_t> cols_ind;          ///< Column indices
  std::vector<index_t> diag_ind;          ///< Diagonal indices
  std::vector<index_t> row_thread_starts; ///< Row thread starts (for OpenMP)
};

} // namespace mgr

#endif // MGR_LINEAR_SOLVER_OPENDARTS_JACOBIAN_HPP_
