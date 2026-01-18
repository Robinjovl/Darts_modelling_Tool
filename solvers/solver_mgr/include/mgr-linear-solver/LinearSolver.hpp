/*
 * MGR Linear Solver - Main Linear Solver Interface
 * @author Xiaoming Tian
 */

#ifndef MGR_LINEAR_SOLVER_LINEAR_SOLVER_HPP_
#define MGR_LINEAR_SOLVER_LINEAR_SOLVER_HPP_

#include "Types.hpp"
#include "MGRStrategy.hpp"
#include <memory>
#include <string>
#include <vector>

// Forward declarations for HYPRE
extern "C" {
typedef struct hypre_ParCSRMatrix_struct *HYPRE_ParCSRMatrix;
typedef struct hypre_ParVector_struct *HYPRE_ParVector;
typedef struct hypre_Solver_struct *HYPRE_Solver;
typedef struct hypre_IJMatrix_struct *HYPRE_IJMatrix;
typedef struct hypre_IJVector_struct *HYPRE_IJVector;
}

namespace mgr {

// Open-darts compatible type aliases
using index_t = int_t;
using mat_float = real_type;

/**
 * @brief Block CSR matrix data structure (internal use only)
 *
 * Block CSR format for reservoir simulation matrices:
 * Each cell is a dense n×n block: [p, z0, z1, ...]
 * where p = pressure (elliptic), z = components (hyperbolic)
 */
struct BlockCSRMatrix
{
  int_t num_rows;           ///< Number of block rows (number of cells)
  int_t num_cols;           ///< Number of block columns
  int_t block_size;         ///< Size of each block (number of variables per cell)
  int_t num_nonzero_blocks; ///< Number of non-zero blocks

  std::vector<int_t> row_ptr;      ///< Row pointer (CSR format for blocks)
  std::vector<int_t> col_ind;      ///< Column indices (CSR format for blocks)
  std::vector<real_type> values;   ///< Block values (size = num_nonzero_blocks × block_size²)
  std::vector<int_t> diag_ind;     ///< Diagonal block indices (for open-darts compatibility)

  int_t global_num_rows; ///< Total number of scalar rows = num_rows × block_size
  int_t global_num_cols; ///< Total number of scalar cols = num_cols × block_size

  /// Default constructor
  BlockCSRMatrix()
    : num_rows( 0 ), num_cols( 0 ), block_size( 0 ),
      num_nonzero_blocks( 0 ), global_num_rows( 0 ), global_num_cols( 0 )
  {}
};

/**
 * @brief Solver parameters
 */
struct SolverParameters
{
  // GMRES parameters
  int_t maxIter = 100;        ///< Maximum GMRES iterations
  real_type tolerance = 1e-6; ///< Convergence tolerance
  int_t kdim = 30;            ///< Krylov subspace dimension

  // Preconditioner choice
  bool useMGR = true;         ///< Use MGR preconditioner (if false, use AMG)
  int_t logLevel = 1;         ///< Logging level (0=none, 1=basic, 2=detailed)
};

/**
 * @brief Solver results
 */
struct SolverResults
{
  int_t iterations;           ///< Number of iterations
  real_type finalResidual;    ///< Final residual norm
  bool converged;             ///< Convergence status
  real_type relError;         ///< Relative error vs reference (if available)
  double setupTime;           ///< Setup time in seconds
  double solveTime;           ///< Solve time in seconds

  SolverResults()
    : iterations( 0 )
    , finalResidual( 0.0 )
    , converged( false )
    , relError( 0.0 )
    , setupTime( 0.0 )
    , solveTime( 0.0 )
  {}
};

/**
 * @brief Main linear solver class
 *
 * Supports:
 *   - GMRES + BoomerAMG preconditioning
 *   - GMRES + MGR preconditioning with various strategies
 *   - Direct integration with open-darts (via setMatrixFromCSR/setMatrixFromVector)
 *   - open-darts compatible interface (init_timer_nodes, setup, solve)
 *   - Setting custom initial guess
 */
class LinearSolver
{
public:

  /**
   * @brief Constructor
   */
  LinearSolver();

  /**
   * @brief Destructor
   */
  ~LinearSolver();

  /**
   * @brief Set MGR strategy
   * @param strategy MGR strategy to use
   */
  void setStrategy( std::unique_ptr<MGRStrategy> strategy );

  /**
   * @brief Set solver parameters
   * @param params Solver parameters
   */
  void setParameters( const SolverParameters & params )
  {
    m_params = params;
  }

  /**
   * @brief Get solver parameters
   * @return Current solver parameters
   */
  const SolverParameters & getParameters() const
  {
    return m_params;
  }

  /**
   * @brief Set initial guess for the solution
   * @param initialGuess Initial solution vector (must match system size)
   *
   * If not set, the solver will use zero vector as initial guess.
   * This can be useful for:
   *   - Using reference solution as starting point
   *   - Warm-starting from previous time step solution
   *   - Providing physically meaningful initial values
   */
  void setInitialGuess( const std::vector<real_type> & initialGuess );

  /**
   * @brief Clear initial guess (reset to zero vector)
   */
  void clearInitialGuess();

  /**
   * @brief Solve the linear system
   * @return Solver results
   */
  SolverResults solve();

  /**
   * @brief Get solution vector
   */
  const std::vector<real_type> & getSolution() const
  {
    return m_solution;
  }

  // ========================================================================
  // open-darts Integration Interface
  // ========================================================================

  /**
   * @brief Set matrix from CSR raw pointers (open-darts style)
   * @param num_rows Number of block rows
   * @param num_cols Number of block columns
   * @param block_size Size of each block (variables per cell)
   * @param num_nonzero_blocks Number of non-zero blocks
   * @param row_ptr Row pointer array (CSR format)
   * @param col_ind Column index array (CSR format)
   * @param values Block values array (size = num_nonzero_blocks × block_size²)
   * @param diag_ind Diagonal indices (optional, can be nullptr)
   * @return true if successful
   *
   * This method accepts raw pointers compatible with open-darts csr_matrix format.
   * Data is copied into internal BlockCSRMatrix structure.
   */
  bool setMatrixFromCSR( int_t num_rows,
                         int_t num_cols,
                         int_t block_size,
                         int_t num_nonzero_blocks,
                         const int_t * row_ptr,
                         const int_t * col_ind,
                         const double * values,
                         const int_t * diag_ind = nullptr );

  /**
   * @brief Set matrix from std::vector (open-darts style)
   * @param num_rows Number of block rows
   * @param num_cols Number of block columns
   * @param block_size Size of each block (variables per cell)
   * @param num_nonzero_blocks Number of non-zero blocks
   * @param row_ptr Row pointer vector (CSR format)
   * @param col_ind Column index vector (CSR format)
   * @param values Block values vector (size = num_nonzero_blocks × block_size²)
   * @param diag_ind Diagonal indices vector (optional)
   * @return true if successful
   *
   * This method accepts std::vector containers compatible with open-darts csr_matrix format.
   * Data is copied into internal BlockCSRMatrix structure.
   */
  bool setMatrixFromVector( int_t num_rows,
                            int_t num_cols,
                            int_t block_size,
                            int_t num_nonzero_blocks,
                            const std::vector< int_t > & row_ptr,
                            const std::vector< int_t > & col_ind,
                            const std::vector< double > & values,
                            const std::vector< int_t > & diag_ind = {} );

  /**
   * @brief Initialize solver with Jacobian (open-darts style)
   * @param jacobian Pointer to Jacobian matrix (OpendartsJacobian*)
   * @param max_iters Maximum iterations
   * @param tolerance Convergence tolerance
   * @return 0 on success, non-zero on error
   *
   * This method provides open-darts compatible initialization.
   * For open-darts integration: use OpendartsJacobian wrapper class.
   * Alternatively, use setMatrixFromCSR() directly with csr_matrix data.
   */
  int init(void* jacobian, index_t max_iters, mat_float tolerance);

  /**
   * @brief Setup solver with Jacobian (open-darts style)
   * @param jacobian Pointer to Jacobian matrix (can be null if structure unchanged)
   * @return 0 on success, non-zero on error
   *
   * This method provides open-darts compatible setup interface.
   * If jacobian is null, reuses existing matrix structure.
   * Uses parameters saved from init() call.
   */
  int setup(void* jacobian);

  /**
   * @brief Setup the solver (open-darts style)
   * @param max_iters Maximum iterations
   * @param tolerance Convergence tolerance
   * @return 0 if successful, non-zero error code otherwise
   *
   * This method sets up the preconditioner and prepares the solver.
   * Must be called after setMatrix and before solve.
   */
  int_t setup( int_t max_iters, double tolerance );

  /**
   * @brief Solve the linear system (open-darts style)
   * @param B Right-hand side vector (size = global_num_rows)
   * @param X Solution vector (size = global_num_rows, pre-allocated)
   * @return Number of iterations, or negative value on error
   *
   * This method accepts raw pointers compatible with open-darts linsolv_iface::solve().
   * The solution is written to the X array.
   */
  int solve(mat_float* B, mat_float* X);

  /**
   * @brief Get number of iterations from last solve
   * @return Number of iterations
   */
  int_t get_n_iters() const;

  /**
   * @brief Get final residual from last solve
   * @return Final residual norm
   */
  double get_residual() const;

  /**
   * @brief Set preconditioner (placeholder for open-darts compatibility)
   * @param prec Preconditioner pointer (ignored)
   * @return 0 (always succeeds)
   *
   * Note: MGR is its own preconditioner, so external preconditioners
   * are not supported. This method exists only for interface compatibility.
   */
  int set_prec(void* prec);

  /**
   * @brief Set pressure system preconditioner (CPR-style, placeholder)
   * @param prec Preconditioner pointer (ignored)
   * @return 0 (always succeeds)
   *
   * Note: MGR handles pressure system internally.
   * This method exists only for interface compatibility with CPR-style solvers.
   */
  int set_p_system_prec(void* prec);

  /**
   * @brief Initialize timer nodes (open-darts compatibility placeholder)
   * @param timer_setup Timer node for setup (ignored, using std::chrono internally)
   * @param timer_solve Timer node for solve (ignored, using std::chrono internally)
   *
   * This method exists for interface compatibility with open-darts.
   * MGR Linear Solver uses std::chrono internally for timing.
   */
  void init_timer_nodes(void* timer_setup, void* timer_solve)
  {
    // Placeholder: ignore open-darts timer_node pointers
    (void)timer_setup;
    (void)timer_solve;
    // Use std::chrono internally instead
  }

  /**
   * @brief Get setup time from last solve
   * @return Setup time in seconds
   */
  double getSetupTime() const { return m_setupTime; }

  /**
   * @brief Get solve time from last solve
   * @return Solve time in seconds
   */
  double getSolveTime() const { return m_solveTime; }

private:

  BlockCSRMatrix m_matrix;              ///< Block CSR matrix
  std::vector<real_type> m_rhs;         ///< Right-hand side
  std::vector<real_type> m_reference;   ///< Reference solution (if available)
  std::vector<real_type> m_solution;    ///< Computed solution
  std::vector<real_type> m_initialGuess; ///< Initial guess (if set)
  bool m_hasInitialGuess;               ///< Flag for initial guess

  SolverParameters m_params;            ///< Solver parameters
  std::unique_ptr<MGRStrategy> m_strategy; ///< MGR strategy

  SolverResults m_lastResults;          ///< Results from most recent solve (for get_n_iters/get_residual)

  // Timing members (open-darts compatibility)
  double m_setupTime = 0.0;             ///< Setup time in seconds
  double m_solveTime = 0.0;             ///< Solve time in seconds

  // init() method parameters (for open-darts compatibility)
  int_t m_initMaxIters = 100;           ///< Max iterations from init()
  double m_initTolerance = 1e-6;        ///< Tolerance from init()

  HYPRE_IJMatrix m_ijMatrix;            ///< HYPRE IJ matrix
  HYPRE_IJVector m_ijRHS;               ///< HYPRE IJ RHS vector
  HYPRE_IJVector m_ijSol;               ///< HYPRE IJ solution vector
  HYPRE_ParCSRMatrix m_parMatrix;       ///< HYPRE parallel matrix
  HYPRE_ParVector m_parRHS;             ///< HYPRE parallel RHS vector
  HYPRE_ParVector m_parSol;             ///< HYPRE parallel solution vector

  bool m_matrixLoaded;                  ///< Matrix loaded flag
  bool m_matrixAssembled;               ///< Matrix assembled flag

  /**
   * @brief Create HYPRE matrix from Block CSR
   */
  bool createHYPREMatrix();

  /**
   * @brief Create HYPRE vectors
   */
  bool createHYPREVectors();

  /**
   * @brief Setup MGR preconditioner
   */
  HYPRE_Solver setupMGRPreconditioner();

  /**
   * @brief Setup BoomerAMG preconditioner
   */
  HYPRE_Solver setupAMGPreconditioner();

  /**
   * @brief Solve with GMRES + MGR
   */
  SolverResults solveGMRES_MGR();

  /**
   * @brief Solve with GMRES + AMG
   */
  SolverResults solveGMRES_AMG();

  /**
   * @brief Cleanup HYPRE objects
   */
  void cleanup();
};

} // namespace mgr

#endif // MGR_LINEAR_SOLVER_LINEAR_SOLVER_HPP_
