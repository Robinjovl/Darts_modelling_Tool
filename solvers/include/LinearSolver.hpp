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

class timer_node;

// Forward declarations for HYPRE
extern "C" {
typedef struct hypre_ParCSRMatrix_struct *HYPRE_ParCSRMatrix;
typedef struct hypre_ParVector_struct *HYPRE_ParVector;
typedef struct hypre_Solver_struct *HYPRE_Solver;
typedef struct hypre_IJMatrix_struct *HYPRE_IJMatrix;
typedef struct hypre_IJVector_struct *HYPRE_IJVector;
}

namespace mgr {

enum class KrylovType
{
  gmres,
  flexgmres
};

enum class ScalingType : int
{
  none = 0,
  physics = 1,
  rowColOneNorm = 2,
  diagonal = 3
};

enum class CompositePreconditionerMode : int
{
  mgrOnly = 0,
  mgrThenLocal = 1,
  localOnly = 2
};

enum class LocalPreconditionerType : int
{
  none = 0,
  blockJacobi = 1,
  blockILU0 = 2
};

enum class LocalFallbackStrategy : int
{
  identity = 0,
  shiftedDense = 1,
  boundedDiagonal = 2,
  shiftedDenseThenDiagonal = 3
};

enum class BCSRCPRReductionType : int
{
  pressureRow = 0,
  trueIMPES = 1
};

// Open-darts compatible type aliases
using index_t = int_t;
using mat_float = real_type;

class BlockLocalPreconditioner;

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
  KrylovType krylovType = KrylovType::flexgmres; ///< Krylov solver type (GMRES or FlexGMRES)

  // Preconditioner choice
  bool useMGR = true;         ///< Use MGR preconditioner (if false, use AMG)
  int_t logLevel = 1;         ///< Logging level (0=none, 1=basic, 2=detailed)

  // Physics-based scaling (Ahat = D * A * D, bhat = D * b, x = D * xhat)
  bool usePhysicsScaling = true; ///< Enable physics-based scaling (GEOS default)
  ScalingType scalingType = ScalingType::physics; ///< Matrix/RHS scaling mode

  // Optional full-system BCSR local correction used inside the Krylov preconditioner.
  CompositePreconditionerMode compositeMode = CompositePreconditionerMode::mgrOnly;
  LocalPreconditionerType localPreconditioner = LocalPreconditionerType::none;
  real_type localPivotShift = 1.0e-12;
  LocalFallbackStrategy localFallbackStrategy = LocalFallbackStrategy::identity;
  real_type localFallbackDiagonalTolerance = 1.0e-4;
  real_type localFallbackShiftMax = 1.0e-4;
  real_type localFallbackShiftGrowth = 100.0;
  real_type localCorrectionAlpha = 1.0;
  real_type localCorrectionAdaptiveFallbackThreshold = -1.0;
  real_type localCorrectionAdaptiveAlpha = 0.0;
  real_type localCorrectionAdaptiveFallbackThresholdHigh = -1.0;
  real_type localCorrectionAdaptiveAlphaHigh = 0.0;
  int_t localReservoirBlockCount = 0;

  // Experimental BCSR-native CPR prototype:
  // pressure AMG first stage + full-system BCSR local preconditioner second stage.
  bool useBCSRCPR = false;
  BCSRCPRReductionType bcsrCPRReduction = BCSRCPRReductionType::trueIMPES;
  int_t bcsrCPRPressureVariable = 0;
  real_type bcsrCPRWeightMax = 1.0e6;
  bool bcsrCPRReuseAMGHierarchy = false;
  int_t bcsrCPRAMGRebuildInterval = 1;
  bool bcsrCPRAdaptiveAMGRebuild = false;
  int_t bcsrCPRAdaptiveLIThreshold = 80;
  real_type bcsrCPRAdaptiveLIGrowthFactor = 2.0;
  int_t bcsrCPRAdaptiveMinReuseSetups = 1;
  int_t bcsrCPRAdaptiveMaxReuseSetups = 0;

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
   * @brief Set block size for MGR reduction (dofs per cell)
   * @param block_size Number of unknowns per cell
   */
  void setMGRBlockSize( int_t block_size );

  /**
   * @brief Set solver parameters
   * @param params Solver parameters
   */
  void setParameters( const SolverParameters & params )
  {
    m_params = params;
    if( !m_params.usePhysicsScaling )
    {
      m_params.scalingType = ScalingType::none;
    }
    else if( m_params.scalingType == ScalingType::none )
    {
      m_params.usePhysicsScaling = false;
    }
    if( m_params.compositeMode != CompositePreconditionerMode::mgrOnly &&
        m_params.localPreconditioner == LocalPreconditionerType::none )
    {
      m_params.localPreconditioner = LocalPreconditionerType::blockILU0;
    }
    if( m_params.useBCSRCPR &&
        m_params.localPreconditioner == LocalPreconditionerType::none )
    {
      m_params.localPreconditioner = LocalPreconditionerType::blockILU0;
    }
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
   * @brief Initialize timer nodes (open-darts compatibility)
   * @param timer_setup Timer node for setup
   * @param timer_solve Timer node for solve
   */
  void init_timer_nodes(::timer_node* timer_setup, ::timer_node* timer_solve);

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
  std::vector<real_type> m_scaling;     ///< Backward-compatible symmetric scaling vector
  std::vector<real_type> m_rowScaling;  ///< Left scaling vector for matrix/RHS
  std::vector<real_type> m_colScaling;  ///< Right scaling vector for matrix/solution
  bool m_hasInitialGuess;               ///< Flag for initial guess

  SolverParameters m_params;            ///< Solver parameters
  std::unique_ptr<MGRStrategy> m_strategy; ///< MGR strategy
  std::unique_ptr<BlockLocalPreconditioner> m_blockLocalPreconditioner;

  SolverResults m_lastResults;          ///< Results from most recent solve (for get_n_iters/get_residual)

  // Timing members (open-darts compatibility)
  double m_setupTime = 0.0;             ///< Setup time in seconds
  double m_solveTime = 0.0;             ///< Solve time in seconds
  ::timer_node* m_timerSetup = nullptr; ///< open-DARTS setup timer root
  ::timer_node* m_timerSolve = nullptr; ///< open-DARTS solve timer root

  // Residual normalization members
  real_type m_lastRhsNorm = 0.0;             ///< Norm of the HYPRE RHS used for convergence
  real_type m_lastInitialResidualNorm = 0.0; ///< Initial residual norm used when RHS is zero
  real_type m_lastResidualDenominator = 0.0; ///< Denominator for reported relative residual

  // init() method parameters (for open-darts compatibility)
  int_t m_initMaxIters = 100;           ///< Max iterations from init()
  double m_initTolerance = 1e-6;        ///< Tolerance from init()

  HYPRE_IJMatrix m_ijMatrix;            ///< HYPRE IJ matrix
  HYPRE_IJVector m_ijRHS;               ///< HYPRE IJ RHS vector
  HYPRE_IJVector m_ijSol;               ///< HYPRE IJ solution vector
  HYPRE_ParCSRMatrix m_parMatrix;       ///< HYPRE parallel matrix
  HYPRE_ParVector m_parRHS;             ///< HYPRE parallel RHS vector
  HYPRE_ParVector m_parSol;             ///< HYPRE parallel solution vector
  bool m_hypreSystemDirectUpdateReady = false;
  int_t m_hypreSystemMatrixCreateCount = 0;
  int_t m_hypreSystemMatrixSetValuesCount = 0;
  int_t m_hypreSystemMatrixAssembleCount = 0;
  int_t m_hypreSystemMatrixDirectUpdateCount = 0;
  std::vector<int_t> m_hypreSystemParCSRDiagDataIndex;

  HYPRE_IJMatrix m_cprPressureIJMatrix = nullptr;
  HYPRE_IJVector m_cprPressureIJRHS = nullptr;
  HYPRE_IJVector m_cprPressureIJSol = nullptr;
  HYPRE_ParCSRMatrix m_cprPressureParMatrix = nullptr;
  HYPRE_ParVector m_cprPressureParRHS = nullptr;
  HYPRE_ParVector m_cprPressureParSol = nullptr;
  HYPRE_Solver m_cprPressureAMG = nullptr;
  bool m_bcsrCPRReady = false;
  int_t m_cprPressureRows = 0;
  int_t m_cprPressureSetupCount = 0;
  bool m_matrixStructureChanged = true;
  bool m_cprPressurePatternReady = false;
  bool m_cprPressureMatrixAssembled = false;
  bool m_cprPressureVectorsReady = false;
  bool m_cprPressureAMGSetupDone = false;
  int_t m_cprClearCount = 0;
  int_t m_cprPressureMatrixCreateCount = 0;
  int_t m_cprPressureMatrixSetValuesCount = 0;
  int_t m_cprPressureMatrixAssembleCount = 0;
  int_t m_cprPressureMatrixUpdateCount = 0;
  int_t m_cprPressureMatrixDirectUpdateCount = 0;
  int_t m_cprPressureVectorCreateCount = 0;
  int_t m_cprPressureVectorReuseCount = 0;
  int_t m_cprPressureAMGCreateCount = 0;
  int_t m_cprPressureAMGSetupCount = 0;
  int_t m_cprPressureAMGReuseCount = 0;
  int_t m_cprPressureStructureReuseCount = 0;
  int_t m_cprPressureStructureResetCount = 0;
  int_t m_cprSetupsSinceAMGSetup = 0;
  int_t m_cprLastLinearIterations = -1;
  int_t m_cprLastAMGSetupLinearIterations = -1;
  bool m_cprLastLinearConverged = true;
  bool m_cprAMGSetupForCurrentSolve = false;
  std::string m_cprLastAMGRebuildReason;
  bool m_cprPressureDirectUpdateReady = false;
  std::vector<int_t> m_cprPressureParCSRDiagDataIndex;
  std::vector<real_type> m_cprPressureWeights;
  std::vector<bigint_t> m_cprPressureRowIndices;
  std::vector<int_t> m_cprPressureRowNCols;
  std::vector<int_t> m_cprPressureRowOffsets;
  std::vector<bigint_t> m_cprPressureCols;
  std::vector<real_type> m_cprPressureValues;
  std::vector<real_type> m_cprPressureRHSValues;
  std::vector<real_type> m_cprPressureSolution;
  std::vector<real_type> m_cprPressureCorrection;
  std::vector<real_type> m_cprResidual;
  std::vector<real_type> m_cprAx;
  std::vector<real_type> m_cprLocalCorrection;

  HYPRE_Solver m_activeMGRPrecond = nullptr; ///< MGR preconditioner used by composite callbacks
  std::string m_activeKrylovName;            ///< Current Krylov solver name for timer nesting
  std::vector<real_type> m_compositeResidual;
  std::vector<real_type> m_compositeAx;
  std::vector<real_type> m_compositeCorrection;

  bool m_matrixLoaded;                  ///< Matrix loaded flag
  bool m_matrixAssembled;               ///< Matrix assembled flag
  int_t m_mgrBlockSize = 0;             ///< Block size override for MGR (0 = use matrix)

  /**
   * @brief Create HYPRE matrix from Block CSR
   */
  bool createHYPREMatrix();

  /**
   * @brief Create HYPRE vectors
   */
  bool createHYPREVectors();

  /**
   * @brief Compute matrix/RHS scaling vectors
   */
  void computeScaling();
  void computePhysicsScaling();
  void computeRowColOneNormScaling();
  void computeDiagonalScaling();
  bool scalingActive(int_t num_rows) const;

  real_type computeScaledVectorNorm(const real_type* values, int_t num_rows) const;
  real_type computeInitialResidualNorm(const real_type* rhs,
                                       const real_type* initial_guess,
                                       int_t num_rows) const;
  void updateResidualNormalization(const real_type* rhs,
                                   const real_type* initial_guess,
                                   int_t num_rows);
  real_type normalizeFinalResidual(real_type hypre_final_residual) const;
  void setConvergenceFromResidual(SolverResults& results,
                                  real_type hypre_final_residual) const;

  ::timer_node* setupTimerNode(const std::string& name) const;
  ::timer_node* solveTimerNode(const std::string& krylov_name,
                               const std::string& name) const;

  void setupBlockLocalPreconditioner();
  void clearHYPRESystemObjects();
  bool prepareHYPRESystemDirectUpdate();
  bool updateHYPRESystemMatrixDirect();
  void clearCompositeWorkVectors();
  bool blockLocalPreconditionerReady() const;
  bool setupBCSRCPRPreconditioner();
  void clearBCSRCPRPreconditioner();
  bool bcsrCPRPreconditionerReady() const;
  void computeBCSRCPRPressureWeights();
  void buildBCSRCPRPressurePattern();
  void fillBCSRCPRPressureMatrixValues();
  bool prepareBCSRCPRPressureDirectUpdate();
  bool updateBCSRCPRPressureMatrixDirect();
  bool createBCSRCPRPressureMatrix();
  bool createBCSRCPRPressureVectors();
  void recordBCSRCPRLinearIterations(int_t iterations, bool converged);
  int applyBCSRCPRPreconditioner(HYPRE_ParCSRMatrix A,
                                 HYPRE_ParVector b,
                                 HYPRE_ParVector x);
  int applyCompositePreconditioner(HYPRE_ParCSRMatrix A,
                                   HYPRE_ParVector b,
                                   HYPRE_ParVector x);

  static int compositePreconditionerSetup(HYPRE_Solver solver,
                                          HYPRE_ParCSRMatrix A,
                                          HYPRE_ParVector b,
                                          HYPRE_ParVector x);
  static int compositePreconditionerSolve(HYPRE_Solver solver,
                                          HYPRE_ParCSRMatrix A,
                                          HYPRE_ParVector b,
                                          HYPRE_ParVector x);
  static int bcsrCPRPreconditionerSetup(HYPRE_Solver solver,
                                        HYPRE_ParCSRMatrix A,
                                        HYPRE_ParVector b,
                                        HYPRE_ParVector x);
  static int bcsrCPRPreconditionerSolve(HYPRE_Solver solver,
                                        HYPRE_ParCSRMatrix A,
                                        HYPRE_ParVector b,
                                        HYPRE_ParVector x);

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
   * @brief Solve with FlexGMRES + MGR
   */
  SolverResults solveFlexGMRES_MGR();

  /**
   * @brief Solve with GMRES + experimental BCSR-native CPR
   */
  SolverResults solveGMRES_BCSRCPR();

  /**
   * @brief Solve with FlexGMRES + experimental BCSR-native CPR
   */
  SolverResults solveFlexGMRES_BCSRCPR();

  /**
   * @brief Solve with GMRES + AMG
   */
  SolverResults solveGMRES_AMG();

  /**
   * @brief Solve with FlexGMRES + AMG
   */
  SolverResults solveFlexGMRES_AMG();

  /**
   * @brief Cleanup HYPRE objects
   */
  void cleanup();
};

} // namespace mgr

#endif // MGR_LINEAR_SOLVER_LINEAR_SOLVER_HPP_
