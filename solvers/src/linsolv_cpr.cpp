//*************************************************************************
//    Copyright (c) 2026
//    Delft University of Technology, the Netherlands
//
//    This file is part of the open Delft Advanced Research Terra Simulator
//    (open-DARTS). It is distributed under the Apache License.
// *************************************************************************

// Open-source CPR (Constrained Pressure Residual) two-stage preconditioner.
// See linsolv_cpr.hpp for the design rationale; algorithm follows the
// classical quasi-IMPES CPR (Wallis 1983) -- a pressure-subsystem AMG
// correction followed by a full-system ILU smoothing pass.
//
// HYPRE BoomerAMG / HYPRE_ILU are driven directly here (not through
// linsolv_hypre_amg / linsolv_hypre_ilu), with the same configuration the
// in-tree MGR uses for the pressure level of compositional flow
// (CompositionalFlowStrategy::setupPressureAMG). The wrappers are tuned for
// the elasticity engines (mechanical systems) and would be the wrong AMG
// setup for flow.

#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_utilities.h"
#include "_hypre_utilities.h"

#include "linsolv_cpr.hpp"

extern "C" {
HYPRE_Int HYPRE_Initialize(void);
HYPRE_Int HYPRE_Initialized(void);
}

namespace opendarts
{
  namespace linear_solvers
  {
    using opendarts::config::index_t;
    using opendarts::config::mat_float;

    namespace
    {
      // Abort-on-error helper for HYPRE calls. Mirrors the policy in
      // linsolv_hypre_amg / linsolv_hypre_ilu: HYPRE errors are not
      // recoverable here -- coming from a malformed pressure subsystem they
      // indicate a programmer bug, not runtime data.
      inline void check_hypre(int res, const char *where)
      {
        if (res)
        {
          char msg[256];
          HYPRE_DescribeError(res, msg);
          std::cerr << "linsolv_cpr: HYPRE error in " << where << " -- "
                    << std::string(msg) << std::endl;
          std::exit(-1);
        }
      }

      // r += A * v  (block-CSR, host, polymorphic via csr_matrix_base).
      template <uint8_t N>
      inline void block_csr_spmv_add(csr_matrix_base *A,
          const mat_float *v,
          mat_float *r)
      {
        const index_t *rows = A->get_rows_ptr();
        const index_t *cols = A->get_cols_ind();
        const mat_float *vals = A->get_values();
        const index_t n_block_rows = A->n_rows;
        constexpr int Ni = static_cast<int>(N);
        const std::size_t b2 = static_cast<std::size_t>(Ni) * Ni;
        for (index_t i = 0; i < n_block_rows; ++i)
        {
          mat_float *ri = r + static_cast<std::size_t>(i) * Ni;
          for (index_t jb = rows[i]; jb < rows[i + 1]; ++jb)
          {
            const mat_float *blk = vals + static_cast<std::size_t>(jb) * b2;
            const mat_float *vj = v + static_cast<std::size_t>(cols[jb]) * Ni;
            for (int e = 0; e < Ni; ++e)
            {
              mat_float acc = 0;
              for (int w = 0; w < Ni; ++w)
                acc += blk[e * Ni + w] * vj[w];
              ri[e] += acc;
            }
          }
        }
      }

      // r += A^T * v  (block-CSR, host). Walks rows of A and scatters each
      // block contribution to the corresponding row of A^T. Mirrors the
      // helper in linsolv_gmres.cpp.
      template <uint8_t N>
      inline void block_csr_spmv_t_add(csr_matrix_base *A,
          const mat_float *v,
          mat_float *r)
      {
        const index_t *rows = A->get_rows_ptr();
        const index_t *cols = A->get_cols_ind();
        const mat_float *vals = A->get_values();
        const index_t n_block_rows = A->n_rows;
        constexpr int Ni = static_cast<int>(N);
        const std::size_t b2 = static_cast<std::size_t>(Ni) * Ni;
        for (index_t i = 0; i < n_block_rows; ++i)
        {
          const mat_float *vi = v + static_cast<std::size_t>(i) * Ni;
          for (index_t jb = rows[i]; jb < rows[i + 1]; ++jb)
          {
            const mat_float *blk = vals + static_cast<std::size_t>(jb) * b2;
            mat_float *rj = r + static_cast<std::size_t>(cols[jb]) * Ni;
            for (int w = 0; w < Ni; ++w)
            {
              mat_float acc = 0;
              for (int e = 0; e < Ni; ++e)
                acc += blk[e * Ni + w] * vi[e];
              rj[w] += acc;
            }
          }
        }
      }

      // CSR transpose for csr_matrix<1>: builds B = A^T with the same data
      // layout. Output is initialised in place into dst.
      inline void csr_transpose_scalar(
          opendarts::linear_solvers::csr_matrix<1> &src,
          opendarts::linear_solvers::csr_matrix<1> &dst)
      {
        const index_t n_rows = src.n_rows;
        const index_t n_cols = src.n_cols;
        const index_t nnz = src.rows_ptr[n_rows];

        dst.init(n_cols, n_rows, nnz);

        // Step 1: column counts (= row counts of A^T).
        std::fill(dst.rows_ptr.begin(), dst.rows_ptr.end(), 0);
        for (index_t k = 0; k < nnz; ++k)
          dst.rows_ptr[src.cols_ind[k] + 1]++;
        // Cumulative sum to build row pointers.
        for (index_t i = 0; i < n_cols; ++i)
          dst.rows_ptr[i + 1] += dst.rows_ptr[i];

        // Step 2: scatter values, advancing a temporary write head per row.
        std::vector<index_t> head(dst.rows_ptr.begin(), dst.rows_ptr.begin() + n_cols);
        for (index_t i = 0; i < n_rows; ++i)
        {
          for (index_t jb = src.rows_ptr[i]; jb < src.rows_ptr[i + 1]; ++jb)
          {
            const index_t j = src.cols_ind[jb];
            const index_t w = head[j]++;
            dst.cols_ind[w] = i;
            dst.values[w] = src.values[jb];
          }
        }

        dst.n_non_zeros = nnz;
        dst.n_row_size = 1;
        dst.is_square = (n_rows == n_cols) ? 1 : 0;
        dst.type = MATRIX_TYPE_CSR_FIXED_STRUCTURE;
      }
    } // namespace

    template <uint8_t N_BLOCK_SIZE>
    linsolv_cpr<N_BLOCK_SIZE>::linsolv_cpr()
      : A_(nullptr),
        Ap_ij_(nullptr),
        Ap_parcsr_(nullptr),
        amg_b_ij_(nullptr),
        amg_x_ij_(nullptr),
        amg_b_par_(nullptr),
        amg_x_par_(nullptr),
        amg_(nullptr),
        amg_setup_done_(false),
        Ap_T_ij_(nullptr),
        Ap_T_parcsr_(nullptr),
        amg_T_b_ij_(nullptr),
        amg_T_x_ij_(nullptr),
        amg_T_b_par_(nullptr),
        amg_T_x_par_(nullptr),
        amg_T_(nullptr),
        amg_T_setup_done_(false),
        As_ij_(nullptr),
        As_parcsr_(nullptr),
        ilu_b_ij_(nullptr),
        ilu_x_ij_(nullptr),
        ilu_b_par_(nullptr),
        ilu_x_par_(nullptr),
        ilu_(nullptr),
        ilu_setup_done_(false),
        As_T_ij_(nullptr),
        As_T_parcsr_(nullptr),
        ilu_T_b_ij_(nullptr),
        ilu_T_x_ij_(nullptr),
        ilu_T_b_par_(nullptr),
        ilu_T_x_par_(nullptr),
        ilu_T_(nullptr),
        ilu_T_setup_done_(false),
        amg_max_iters_(2),       // AMG used as a prec -- a couple of V-cycles
        amg_tolerance_(1.0e-2),
        ilu_fill_level_(0),
        max_iters_(50),
        tolerance_(1.0e-5),
        n_iters_(0),
        first_setup_(true)
    {
    }

    template <uint8_t N_BLOCK_SIZE>
    linsolv_cpr<N_BLOCK_SIZE>::~linsolv_cpr()
    {
      destroy_hypre();
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_cpr<N_BLOCK_SIZE>::destroy_hypre()
    {
      if (amg_) { HYPRE_BoomerAMGDestroy(amg_); amg_ = nullptr; }
      if (Ap_ij_) { HYPRE_IJMatrixDestroy(Ap_ij_); Ap_ij_ = nullptr; }
      if (amg_b_ij_) { HYPRE_IJVectorDestroy(amg_b_ij_); amg_b_ij_ = nullptr; }
      if (amg_x_ij_) { HYPRE_IJVectorDestroy(amg_x_ij_); amg_x_ij_ = nullptr; }

      if (amg_T_) { HYPRE_BoomerAMGDestroy(amg_T_); amg_T_ = nullptr; }
      if (Ap_T_ij_) { HYPRE_IJMatrixDestroy(Ap_T_ij_); Ap_T_ij_ = nullptr; }
      if (amg_T_b_ij_) { HYPRE_IJVectorDestroy(amg_T_b_ij_); amg_T_b_ij_ = nullptr; }
      if (amg_T_x_ij_) { HYPRE_IJVectorDestroy(amg_T_x_ij_); amg_T_x_ij_ = nullptr; }

      if (ilu_) { HYPRE_ILUDestroy(ilu_); ilu_ = nullptr; }
      if (As_ij_) { HYPRE_IJMatrixDestroy(As_ij_); As_ij_ = nullptr; }
      if (ilu_b_ij_) { HYPRE_IJVectorDestroy(ilu_b_ij_); ilu_b_ij_ = nullptr; }
      if (ilu_x_ij_) { HYPRE_IJVectorDestroy(ilu_x_ij_); ilu_x_ij_ = nullptr; }

      if (ilu_T_) { HYPRE_ILUDestroy(ilu_T_); ilu_T_ = nullptr; }
      if (As_T_ij_) { HYPRE_IJMatrixDestroy(As_T_ij_); As_T_ij_ = nullptr; }
      if (ilu_T_b_ij_) { HYPRE_IJVectorDestroy(ilu_T_b_ij_); ilu_T_b_ij_ = nullptr; }
      if (ilu_T_x_ij_) { HYPRE_IJVectorDestroy(ilu_T_x_ij_); ilu_T_x_ij_ = nullptr; }

      amg_setup_done_ = false;
      amg_T_setup_done_ = false;
      ilu_setup_done_ = false;
      ilu_T_setup_done_ = false;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cpr<N_BLOCK_SIZE>::init(csr_matrix_base *A,
        int max_iters,
        mat_float tolerance)
    {
      A_ = A;
      max_iters_ = max_iters;
      tolerance_ = tolerance;
      n_iters_ = 0;
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_cpr<N_BLOCK_SIZE>::build_pressure_subsystem(csr_matrix_base *A)
    {
      const index_t *rows = A->get_rows_ptr();
      const index_t *cols = A->get_cols_ind();
      const mat_float *vals = A->get_values();
      const index_t n_block_rows = A->n_rows;
      const index_t n_block_cols = A->n_cols;
      const std::size_t b2 =
          static_cast<std::size_t>(N_BLOCK_SIZE) * N_BLOCK_SIZE;
      const index_t nnz = rows[n_block_rows];

      if (!Ap_)
        Ap_ = std::make_unique<opendarts::linear_solvers::csr_matrix<1>>();
      Ap_->init(n_block_rows, n_block_cols, nnz);

      // Copy the (block) sparsity pattern verbatim -- A_p has the same
      // sparsity as the block-CSR (one scalar entry per block).
      std::copy(rows, rows + n_block_rows + 1, Ap_->rows_ptr.data());
      std::copy(cols, cols + nnz, Ap_->cols_ind.data());

      // Extract the (0, 0) entry of every block as the A_p value.
      mat_float *ap_vals = Ap_->values.data();
      for (index_t jb = 0; jb < nnz; ++jb)
        ap_vals[jb] = vals[static_cast<std::size_t>(jb) * b2
            + static_cast<std::size_t>(P_VAR) * N_BLOCK_SIZE + P_VAR];

      Ap_->n_non_zeros = nnz;
      Ap_->n_row_size = 1;
      Ap_->is_square = (n_block_rows == n_block_cols) ? 1 : 0;
      Ap_->type = MATRIX_TYPE_CSR_FIXED_STRUCTURE;
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_cpr<N_BLOCK_SIZE>::build_hypre_ij(
        opendarts::linear_solvers::csr_matrix<1> &A,
        HYPRE_IJMatrix &A_ij,
        HYPRE_ParCSRMatrix &A_parcsr)
    {
      const index_t n_rows = A.n_rows;
      const index_t ilower = 0;
      const index_t iupper = n_rows - 1;

      std::vector<index_t> rows(n_rows), n_cols(n_rows);
      std::iota(rows.begin(), rows.end(), 0);
      for (index_t i = 0; i < n_rows; ++i)
        n_cols[i] = A.rows_ptr[i + 1] - A.rows_ptr[i];

      check_hypre(HYPRE_IJMatrixCreate(hypre_MPI_COMM_WORLD, ilower, iupper,
                      ilower, iupper, &A_ij),
          "IJMatrixCreate");
      check_hypre(HYPRE_IJMatrixSetPrintLevel(A_ij, 0), "IJMatrixSetPrintLevel");
      check_hypre(HYPRE_IJMatrixSetObjectType(A_ij, HYPRE_PARCSR),
          "IJMatrixSetObjectType");
      check_hypre(HYPRE_IJMatrixInitialize(A_ij), "IJMatrixInitialize");
      check_hypre(HYPRE_IJMatrixSetValues(A_ij, n_rows, n_cols.data(),
                      rows.data(), A.get_cols_ind(), A.get_values()),
          "IJMatrixSetValues");
      check_hypre(HYPRE_IJMatrixAssemble(A_ij), "IJMatrixAssemble");
      check_hypre(HYPRE_IJMatrixGetObject(A_ij, (void **) &A_parcsr),
          "IJMatrixGetObject");
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_cpr<N_BLOCK_SIZE>::refresh_hypre_ij(
        opendarts::linear_solvers::csr_matrix<1> &A,
        HYPRE_IJMatrix &A_ij,
        HYPRE_ParCSRMatrix &A_parcsr)
    {
      // Update values on an existing IJMatrix without destroying it.
      // HYPRE_IJMatrixInitialize re-opens the matrix for SetValues; the
      // sparsity pattern is preserved across calls.
      const index_t n_rows = A.n_rows;
      std::vector<index_t> rows(n_rows), n_cols(n_rows);
      std::iota(rows.begin(), rows.end(), 0);
      for (index_t i = 0; i < n_rows; ++i)
        n_cols[i] = A.rows_ptr[i + 1] - A.rows_ptr[i];

      check_hypre(HYPRE_IJMatrixInitialize(A_ij), "IJMatrixInitialize(refresh)");
      check_hypre(HYPRE_IJMatrixSetValues(A_ij, n_rows, n_cols.data(),
                      rows.data(), A.get_cols_ind(), A.get_values()),
          "IJMatrixSetValues(refresh)");
      check_hypre(HYPRE_IJMatrixAssemble(A_ij), "IJMatrixAssemble(refresh)");
      check_hypre(HYPRE_IJMatrixGetObject(A_ij, (void **) &A_parcsr),
          "IJMatrixGetObject(refresh)");
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_cpr<N_BLOCK_SIZE>::create_hypre_vectors(index_t n_rows,
        HYPRE_IJVector &b_ij,
        HYPRE_IJVector &x_ij)
    {
      const index_t ilower = 0;
      const index_t iupper = n_rows - 1;

      check_hypre(HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, ilower, iupper, &b_ij),
          "IJVectorCreate(b)");
      check_hypre(HYPRE_IJVectorSetPrintLevel(b_ij, 0), "IJVectorSetPrintLevel(b)");
      check_hypre(HYPRE_IJVectorSetObjectType(b_ij, HYPRE_PARCSR),
          "IJVectorSetObjectType(b)");

      check_hypre(HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, ilower, iupper, &x_ij),
          "IJVectorCreate(x)");
      check_hypre(HYPRE_IJVectorSetPrintLevel(x_ij, 0), "IJVectorSetPrintLevel(x)");
      check_hypre(HYPRE_IJVectorSetObjectType(x_ij, HYPRE_PARCSR),
          "IJVectorSetObjectType(x)");
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_cpr<N_BLOCK_SIZE>::set_hypre_vector(HYPRE_IJVector &v_ij,
        index_t n_rows,
        mat_float *vals,
        HYPRE_ParVector &v_par)
    {
      std::vector<index_t> rows(n_rows);
      std::iota(rows.begin(), rows.end(), 0);

      check_hypre(HYPRE_IJVectorInitialize(v_ij), "IJVectorInitialize");
      check_hypre(HYPRE_IJVectorSetValues(v_ij, n_rows, rows.data(), vals),
          "IJVectorSetValues");
      check_hypre(HYPRE_IJVectorAssemble(v_ij), "IJVectorAssemble");
      check_hypre(HYPRE_IJVectorGetObject(v_ij, (void **) &v_par),
          "IJVectorGetObject");
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cpr<N_BLOCK_SIZE>::setup(csr_matrix_base *A_input)
    {
      // Ensure HYPRE is initialised; when CPR is used without MGR alive in
      // the process the wrappers would otherwise hit "[Generic error]" out of
      // HYPRE's diagnostic layer.
      if (!HYPRE_Initialized())
        HYPRE_Initialize();
      HYPRE_ClearAllErrors();

      A_ = A_input;

      // Rebuild the (block-extracted) pressure subsystem and the scalar
      // expansion of the full system. Both are CSR<1> objects owned here;
      // their sparsity pattern is reused across nonlinear iterations.
      build_pressure_subsystem(A_input);
      if (!Ap_T_)
        Ap_T_ = std::make_unique<opendarts::linear_solvers::csr_matrix<1>>();
      csr_transpose_scalar(*Ap_, *Ap_T_);
      if (!As_)
        As_ = std::make_unique<opendarts::linear_solvers::csr_matrix<1>>();
      As_->to_nb_1(A_input);
      if (!As_T_)
        As_T_ = std::make_unique<opendarts::linear_solvers::csr_matrix<1>>();
      csr_transpose_scalar(*As_, *As_T_);

      if (first_setup_)
      {
        // First setup -- create all HYPRE handles fresh. Subsequent setup()
        // calls reuse them; a destroy/recreate cycle on every Newton iteration
        // crashes BoomerAMGSetup on the second call (HYPRE accumulates state
        // that BoomerAMGDestroy doesn't fully release).
        //
        // Forward AMG on A_p. Configuration mirrors
        // mgr::CompositionalFlowStrategy::setupPressureAMG: aggressive
        // coarsening + multipass interp + C-F relax, configured as a
        // preconditioner (MaxIter=amg_max_iters_, Tol=0). The
        // elasticity-tuned linsolv_hypre_amg wrapper is intentionally not
        // used here -- this is the flow-tuned configuration MGR uses.
        build_hypre_ij(*Ap_, Ap_ij_, Ap_parcsr_);
        create_hypre_vectors(Ap_->n_rows, amg_b_ij_, amg_x_ij_);

        check_hypre(HYPRE_BoomerAMGCreate(&amg_), "BoomerAMGCreate");
        check_hypre(HYPRE_BoomerAMGSetPrintLevel(amg_, 0),
            "BoomerAMGSetPrintLevel");
        check_hypre(HYPRE_BoomerAMGSetLogging(amg_, 0), "BoomerAMGSetLogging");
        check_hypre(HYPRE_BoomerAMGSetMaxIter(amg_, amg_max_iters_),
            "BoomerAMGSetMaxIter");
        check_hypre(HYPRE_BoomerAMGSetTol(amg_, 0.0), "BoomerAMGSetTol");
        check_hypre(HYPRE_BoomerAMGSetAggNumLevels(amg_, 1),
            "BoomerAMGSetAggNumLevels");
        check_hypre(HYPRE_BoomerAMGSetAggPMaxElmts(amg_, 20),
            "BoomerAMGSetAggPMaxElmts");
        check_hypre(HYPRE_BoomerAMGSetAggInterpType(amg_, 6),
            "BoomerAMGSetAggInterpType");
        check_hypre(HYPRE_BoomerAMGSetRelaxOrder(amg_, 1),
            "BoomerAMGSetRelaxOrder");

        check_hypre(
            HYPRE_BoomerAMGSetup(amg_, Ap_parcsr_, amg_b_par_, amg_x_par_),
            "BoomerAMGSetup");
        amg_setup_done_ = true;

        // Transpose AMG on A_p^T (for CPRA solve_transposed) -- separate
        // hierarchy because HYPRE_BoomerAMGSolveT supports only relax types
        // 7 / 9 and interacts poorly with our aggressive-coarsening flow
        // config.
        build_hypre_ij(*Ap_T_, Ap_T_ij_, Ap_T_parcsr_);
        create_hypre_vectors(Ap_T_->n_rows, amg_T_b_ij_, amg_T_x_ij_);

        check_hypre(HYPRE_BoomerAMGCreate(&amg_T_), "BoomerAMGCreate(T)");
        check_hypre(HYPRE_BoomerAMGSetPrintLevel(amg_T_, 0),
            "BoomerAMGSetPrintLevel(T)");
        check_hypre(HYPRE_BoomerAMGSetLogging(amg_T_, 0),
            "BoomerAMGSetLogging(T)");
        check_hypre(HYPRE_BoomerAMGSetMaxIter(amg_T_, amg_max_iters_),
            "BoomerAMGSetMaxIter(T)");
        check_hypre(HYPRE_BoomerAMGSetTol(amg_T_, 0.0), "BoomerAMGSetTol(T)");
        check_hypre(HYPRE_BoomerAMGSetAggNumLevels(amg_T_, 1),
            "BoomerAMGSetAggNumLevels(T)");
        check_hypre(HYPRE_BoomerAMGSetAggPMaxElmts(amg_T_, 20),
            "BoomerAMGSetAggPMaxElmts(T)");
        check_hypre(HYPRE_BoomerAMGSetAggInterpType(amg_T_, 6),
            "BoomerAMGSetAggInterpType(T)");
        check_hypre(HYPRE_BoomerAMGSetRelaxOrder(amg_T_, 1),
            "BoomerAMGSetRelaxOrder(T)");
        check_hypre(
            HYPRE_BoomerAMGSetup(amg_T_, Ap_T_parcsr_, amg_T_b_par_,
                amg_T_x_par_),
            "BoomerAMGSetup(T)");
        amg_T_setup_done_ = true;

        // Forward HYPRE_ILU on A_s (full-system scalar expansion).
        build_hypre_ij(*As_, As_ij_, As_parcsr_);
        create_hypre_vectors(As_->n_rows, ilu_b_ij_, ilu_x_ij_);

        check_hypre(HYPRE_ILUCreate(&ilu_), "ILUCreate");
        check_hypre(HYPRE_ILUSetPrintLevel(ilu_, 0), "ILUSetPrintLevel");
        check_hypre(HYPRE_ILUSetLogging(ilu_, 0), "ILUSetLogging");
        check_hypre(HYPRE_ILUSetMaxIter(ilu_, 1), "ILUSetMaxIter");
        check_hypre(HYPRE_ILUSetTol(ilu_, 0.0), "ILUSetTol");
        check_hypre(HYPRE_ILUSetType(ilu_, 0), "ILUSetType");
        check_hypre(HYPRE_ILUSetLevelOfFill(ilu_, ilu_fill_level_),
            "ILUSetLevelOfFill");

        check_hypre(HYPRE_ILUSetup(ilu_, As_parcsr_, ilu_b_par_, ilu_x_par_),
            "ILUSetup");
        ilu_setup_done_ = true;

        // Transpose HYPRE_ILU on A_s^T (HYPRE_ILU has no transpose-solve
        // entry point, so the adjoint path uses a second factorisation).
        build_hypre_ij(*As_T_, As_T_ij_, As_T_parcsr_);
        create_hypre_vectors(As_T_->n_rows, ilu_T_b_ij_, ilu_T_x_ij_);

        check_hypre(HYPRE_ILUCreate(&ilu_T_), "ILUCreate(T)");
        check_hypre(HYPRE_ILUSetPrintLevel(ilu_T_, 0),
            "ILUSetPrintLevel(T)");
        check_hypre(HYPRE_ILUSetLogging(ilu_T_, 0), "ILUSetLogging(T)");
        check_hypre(HYPRE_ILUSetMaxIter(ilu_T_, 1), "ILUSetMaxIter(T)");
        check_hypre(HYPRE_ILUSetTol(ilu_T_, 0.0), "ILUSetTol(T)");
        check_hypre(HYPRE_ILUSetType(ilu_T_, 0), "ILUSetType(T)");
        check_hypre(HYPRE_ILUSetLevelOfFill(ilu_T_, ilu_fill_level_),
            "ILUSetLevelOfFill(T)");
        check_hypre(
            HYPRE_ILUSetup(ilu_T_, As_T_parcsr_, ilu_T_b_par_, ilu_T_x_par_),
            "ILUSetup(T)");
        ilu_T_setup_done_ = true;

        first_setup_ = false;
      }
      else
      {
        // Subsequent setup -- refresh IJ matrix values, then re-run *Setup
        // on the existing solver handles. This rebuilds the AMG hierarchy /
        // ILU factorisation in place. Same code path the in-tree MGR uses.
        refresh_hypre_ij(*Ap_, Ap_ij_, Ap_parcsr_);
        refresh_hypre_ij(*Ap_T_, Ap_T_ij_, Ap_T_parcsr_);
        refresh_hypre_ij(*As_, As_ij_, As_parcsr_);
        refresh_hypre_ij(*As_T_, As_T_ij_, As_T_parcsr_);

        check_hypre(
            HYPRE_BoomerAMGSetup(amg_, Ap_parcsr_, amg_b_par_, amg_x_par_),
            "BoomerAMGSetup(re)");
        check_hypre(
            HYPRE_BoomerAMGSetup(amg_T_, Ap_T_parcsr_, amg_T_b_par_,
                amg_T_x_par_),
            "BoomerAMGSetup(T,re)");
        check_hypre(
            HYPRE_ILUSetup(ilu_, As_parcsr_, ilu_b_par_, ilu_x_par_),
            "ILUSetup(re)");
        check_hypre(
            HYPRE_ILUSetup(ilu_T_, As_T_parcsr_, ilu_T_b_par_, ilu_T_x_par_),
            "ILUSetup(T,re)");
      }

      // Scratch buffers for the per-apply CPR stages.
      const std::size_t n_scalar =
          static_cast<std::size_t>(A_input->n_rows) * N_BLOCK_SIZE;
      const std::size_t n_pressure = static_cast<std::size_t>(A_input->n_rows);
      // Layout: x_g[n_scalar], r_m[n_scalar], x_f[n_scalar],
      //         r_p[n_pressure], x_p[n_pressure].
      wksp_.assign(3 * n_scalar + 2 * n_pressure, 0.0);
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cpr<N_BLOCK_SIZE>::solve(mat_float *B, mat_float *X)
    {
      if (!A_ || !amg_setup_done_)
        return -1;

      const index_t n_block_rows = A_->n_rows;
      const std::size_t n_scalar =
          static_cast<std::size_t>(n_block_rows) * N_BLOCK_SIZE;
      const std::size_t n_pressure = static_cast<std::size_t>(n_block_rows);

      mat_float *x_g = wksp_.data();
      mat_float *r_m = x_g + n_scalar;
      mat_float *x_f = r_m + n_scalar;
      mat_float *r_p = x_f + n_scalar;
      mat_float *x_p = r_p + n_pressure;

      // Stage 1: pressure correction. Restrict B to the pressure subsystem,
      // solve A_p x_p = r_p with AMG, prolong to x_g (pressure component;
      // zero elsewhere).
      for (index_t i = 0; i < n_block_rows; ++i)
        r_p[i] = B[static_cast<std::size_t>(i) * N_BLOCK_SIZE + P_VAR];
      std::memset(x_p, 0, n_pressure * sizeof(mat_float));

      set_hypre_vector(amg_b_ij_, n_block_rows, r_p, amg_b_par_);
      set_hypre_vector(amg_x_ij_, n_block_rows, x_p, amg_x_par_);
      check_hypre(HYPRE_BoomerAMGSolve(amg_, Ap_parcsr_, amg_b_par_, amg_x_par_),
          "BoomerAMGSolve");
      {
        std::vector<index_t> rows(n_block_rows);
        std::iota(rows.begin(), rows.end(), 0);
        check_hypre(HYPRE_IJVectorGetValues(amg_x_ij_, n_block_rows,
                        rows.data(), x_p),
            "IJVectorGetValues(amg_x)");
      }

      std::memset(x_g, 0, n_scalar * sizeof(mat_float));
      for (index_t i = 0; i < n_block_rows; ++i)
        x_g[static_cast<std::size_t>(i) * N_BLOCK_SIZE + P_VAR] = x_p[i];

      // Stage 2: full-system smoothing on r_m = B - A x_g.
      std::memcpy(r_m, B, n_scalar * sizeof(mat_float));
      std::memset(x_f, 0, n_scalar * sizeof(mat_float));
      block_csr_spmv_add<N_BLOCK_SIZE>(A_, x_g, x_f);
      for (std::size_t k = 0; k < n_scalar; ++k)
        r_m[k] -= x_f[k];

      // Apply HYPRE-ILU to r_m for x_f.
      std::memset(x_f, 0, n_scalar * sizeof(mat_float));
      if (ilu_setup_done_)
      {
        const index_t n_s = static_cast<index_t>(n_scalar);
        set_hypre_vector(ilu_b_ij_, n_s, r_m, ilu_b_par_);
        set_hypre_vector(ilu_x_ij_, n_s, x_f, ilu_x_par_);
        check_hypre(HYPRE_ILUSolve(ilu_, As_parcsr_, ilu_b_par_, ilu_x_par_),
            "ILUSolve");
        std::vector<index_t> rows(n_s);
        std::iota(rows.begin(), rows.end(), 0);
        check_hypre(HYPRE_IJVectorGetValues(ilu_x_ij_, n_s, rows.data(), x_f),
            "IJVectorGetValues(ilu_x)");
      }

      // X = x_g + x_f.
      for (std::size_t k = 0; k < n_scalar; ++k)
        X[k] = x_g[k] + x_f[k];

      n_iters_ = 1;  // CPR as a preconditioner: one application
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cpr<N_BLOCK_SIZE>::solve_transposed(mat_float *B, mat_float *X)
    {
      if (!A_ || !amg_T_setup_done_ || !ilu_T_setup_done_)
        return -1;

      // CPRA (Han et al. 2013, eq 13). Each stage is the transpose of the
      // matching forward stage; the order is reversed:
      //   1. x_f = M̃^{-T} r              -- HYPRE_ILUSolve on A_s^T
      //   2. r_m = r - (Ã)^T x_f         -- block-CSR transpose SpMV
      //   3. r_p = C^T r_m                -- restrict to pressure
      //   4. x_p = (A_p^T)^{-1} r_p      -- HYPRE_BoomerAMGSolve on A_p^T
      //   5. X = C x_p + x_f             -- prolong + add
      const index_t n_block_rows = A_->n_rows;
      const std::size_t n_scalar =
          static_cast<std::size_t>(n_block_rows) * N_BLOCK_SIZE;
      const std::size_t n_pressure = static_cast<std::size_t>(n_block_rows);

      mat_float *x_g = wksp_.data();       // unused on transpose path
      mat_float *r_m = x_g + n_scalar;
      mat_float *x_f = r_m + n_scalar;
      mat_float *r_p = x_f + n_scalar;
      mat_float *x_p = r_p + n_pressure;

      const index_t n_s = static_cast<index_t>(n_scalar);
      std::vector<index_t> rows(n_scalar);
      std::iota(rows.begin(), rows.end(), 0);

      // Step 1: x_f = (A_s^T)^{-1} r
      std::memset(x_f, 0, n_scalar * sizeof(mat_float));
      set_hypre_vector(ilu_T_b_ij_, n_s, B, ilu_T_b_par_);
      set_hypre_vector(ilu_T_x_ij_, n_s, x_f, ilu_T_x_par_);
      check_hypre(HYPRE_ILUSolve(ilu_T_, As_T_parcsr_, ilu_T_b_par_, ilu_T_x_par_),
          "ILUSolve(T)");
      check_hypre(HYPRE_IJVectorGetValues(ilu_T_x_ij_, n_s, rows.data(), x_f),
          "IJVectorGetValues(ilu_T_x)");

      // Step 2: r_m = r - A^T x_f
      std::memcpy(r_m, B, n_scalar * sizeof(mat_float));
      {
        std::vector<mat_float> tmp(n_scalar, 0.0);
        block_csr_spmv_t_add<N_BLOCK_SIZE>(A_, x_f, tmp.data());
        for (std::size_t k = 0; k < n_scalar; ++k)
          r_m[k] -= tmp[k];
      }

      // Step 3: r_p = C^T r_m (pressure restriction).
      for (index_t i = 0; i < n_block_rows; ++i)
        r_p[i] = r_m[static_cast<std::size_t>(i) * N_BLOCK_SIZE + P_VAR];
      std::memset(x_p, 0, n_pressure * sizeof(mat_float));

      // Step 4: x_p = (A_p^T)^{-1} r_p via the AMG hierarchy on A_p^T.
      set_hypre_vector(amg_T_b_ij_, n_block_rows, r_p, amg_T_b_par_);
      set_hypre_vector(amg_T_x_ij_, n_block_rows, x_p, amg_T_x_par_);
      check_hypre(
          HYPRE_BoomerAMGSolve(amg_T_, Ap_T_parcsr_, amg_T_b_par_, amg_T_x_par_),
          "BoomerAMGSolve(T)");
      {
        std::vector<index_t> prows(n_block_rows);
        std::iota(prows.begin(), prows.end(), 0);
        check_hypre(HYPRE_IJVectorGetValues(amg_T_x_ij_, n_block_rows,
                        prows.data(), x_p),
            "IJVectorGetValues(amg_T_x)");
      }

      // Step 5: X = C x_p + x_f (prolong pressure component, add ILU update).
      std::memcpy(X, x_f, n_scalar * sizeof(mat_float));
      for (index_t i = 0; i < n_block_rows; ++i)
        X[static_cast<std::size_t>(i) * N_BLOCK_SIZE + P_VAR] += x_p[i];

      n_iters_ = 1;
      return 0;
    }

    // Explicit instantiations for the block sizes the engine uses.
    template class linsolv_cpr<1>;
    template class linsolv_cpr<2>;
    template class linsolv_cpr<3>;
    template class linsolv_cpr<4>;
    template class linsolv_cpr<5>;
    template class linsolv_cpr<6>;
    template class linsolv_cpr<7>;
    template class linsolv_cpr<8>;
    template class linsolv_cpr<9>;
    template class linsolv_cpr<10>;
    template class linsolv_cpr<11>;
    template class linsolv_cpr<12>;
    template class linsolv_cpr<13>;
  } // namespace linear_solvers
} // namespace opendarts
