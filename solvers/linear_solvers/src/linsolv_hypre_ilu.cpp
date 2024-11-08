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

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>

#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv.h"
#include "_hypre_parcsr_mv.h"

#include "openDARTS/config/data_types.hpp"
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#include "openDARTS/linear_solvers/linsolv_iface.hpp"
#include "openDARTS/linear_solvers/linsolv_hypre_ilu.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    void check_result(int res);
    
    template <uint8_t N_BLOCK_SIZE> 
    linsolv_hypre_ilu<N_BLOCK_SIZE>::linsolv_hypre_ilu()
    {
      // Initialize preconditioner (not used in this case, kept for compatibility)
      this->prec = 0;  // no preconditioner for this solver 
    }
    
    template <uint8_t N_BLOCK_SIZE> 
    linsolv_hypre_ilu<N_BLOCK_SIZE>::~linsolv_hypre_ilu()
    {
      check_result(HYPRE_ILUDestroy(this->solver));
      
      check_result(HYPRE_IJMatrixDestroy(this->A_ij));
      // check_result(HYPRE_ParCSRMatrixDestroy(this->A_parcsr));  // gives error
      
      check_result(HYPRE_IJVectorDestroy(this->b_ij));
      // check_result(HYPRE_ParVectorDestroy(this->b_par));  // gives error
      
      check_result(HYPRE_IJVectorDestroy(this->x_ij));
      // check_result(HYPRE_ParVectorDestroy(this->x_par));  // gives error
    }

    template <uint8_t N_BLOCK_SIZE> 
    int linsolv_hypre_ilu<N_BLOCK_SIZE>::set_prec(opendarts::linear_solvers::linsolv_iface *prec_in)
    {
      (void) prec_in;
      
      std::cout << "NOT IMPLEMENTED: linsolv_hypre_ilu::linsolv_hypre_ilu" << std::endl;
      
      return 1;
    }

    template <uint8_t N_BLOCK_SIZE> 
    int linsolv_hypre_ilu<N_BLOCK_SIZE>::init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_in, 
      opendarts::config::index_t max_iters, 
      opendarts::config::mat_float tolerance)
    {
      // max_iters: set to 1 if ILU is used as preconditioner
      // tolerance: set to 0.0 if ILU is used as preconditioner
       
      // Setup Hypre solver
      const int print_level = 2;  // print level
      
      check_result(HYPRE_ILUCreate(&(this->solver)));
    	check_result(HYPRE_ILUSetPrintLevel(this->solver, print_level));
    	check_result(HYPRE_ILUSetLogging(this->solver, print_level));
      
      check_result(HYPRE_ILUSetMaxIter(this->solver, max_iters));
      check_result(HYPRE_ILUSetTol(this->solver, tolerance));
      
      (void) A_in; // not used, kept to keep same interface
      
      return 0;
    }
    
    template <uint8_t N_BLOCK_SIZE> 
    int linsolv_hypre_ilu<N_BLOCK_SIZE>::setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_in)
    {
      // linsolv_iface::timer_setup->node["AMG"].start();
      
      // Store input system matrix
      this->A = A_in;
      
      // Setup right hand side and solution vectors
      const int print_level = 2;  // print level in Hypre
      opendarts::config::index_t n_rows = this->A->n_cols;;  // number of rows in vector must 
                                                             // be the same as number of columns 
                                                             // of system matrix
      
      opendarts::config::index_t ilower, iupper;
      ilower = 0;
      iupper = n_rows - 1;
      
      check_result(HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, ilower, iupper, &(this->b_ij)));
    	check_result(HYPRE_IJVectorSetPrintLevel(this->b_ij, print_level));
    	check_result(HYPRE_IJVectorSetObjectType(this->b_ij, HYPRE_PARCSR));
      
      check_result(HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, ilower, iupper, &(this->x_ij)));
    	check_result(HYPRE_IJVectorSetPrintLevel(this->x_ij, print_level));
    	check_result(HYPRE_IJVectorSetObjectType(this->x_ij, HYPRE_PARCSR));
      
      // Convert system matrix into Hypre required formats and store them
      linsolv_hypre_ilu<N_BLOCK_SIZE>::csr_matrix_to_hypre_ij(*(this->A), this->A_ij);  // convert input matrix to hypre ij matrix
      check_result(HYPRE_IJMatrixGetObject(this->A_ij, (void **)&(this->A_parcsr)));  // convert to hypre parCSR, needed by the solver
      
      // Setup Hypre ILU solver 
      // Note that in this function b_par and x_par are ignored
      check_result(HYPRE_ILUSetup(this->solver, this->A_parcsr, this->b_par, this->x_par));
      
      // linsolv_iface::timer_setup->node["AMG"].stop();
      
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE> 
    int linsolv_hypre_ilu<N_BLOCK_SIZE>::solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X)
    {
      
      // Generate Hypre right hand side vector b_ij
      opendarts::config::index_t n_rows = this->A->n_cols;;  // number of rows in vector must 
                                                             // be the same as number of columns 
                                                             // of system matrix
      
      std::vector<opendarts::config::index_t> rows(n_rows);
      std::iota(rows.begin(), rows.end(), 0);

      check_result(HYPRE_IJVectorInitialize(b_ij));
    	check_result(HYPRE_IJVectorSetValues(b_ij, n_rows, rows.data(), B));
    	check_result(HYPRE_IJVectorAssemble(b_ij));
    	check_result(HYPRE_IJVectorGetObject(b_ij, (void **)&b_par));
      
      // Generate Hypre solution vector x_ij
      check_result(HYPRE_IJVectorInitialize(x_ij));
    	check_result(HYPRE_IJVectorSetValues(x_ij, n_rows, rows.data(), X));
    	check_result(HYPRE_IJVectorAssemble(x_ij));
    	check_result(HYPRE_IJVectorGetObject(x_ij, (void **)&x_par));
    	
      // Solve the system
      check_result(HYPRE_ILUSolve(this->solver, this->A_parcsr, b_par, x_par));
      
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE> 
    opendarts::config::index_t linsolv_hypre_ilu<N_BLOCK_SIZE>::get_n_iters()
    {
      std::cout << "NOT IMPLEMENTED: linsolv_hypre_ilu::get_n_iters" << std::endl;
      
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE> 
    opendarts::config::mat_float linsolv_hypre_ilu<N_BLOCK_SIZE>::get_residual()
    {
      std::cout << "NOT IMPLEMENTED: linsolv_hypre_ilu::get_residual" << std::endl;
      
      return 1000.0;
    }
    
    template<>
    void linsolv_hypre_ilu<1>::csr_matrix_to_hypre_ij(
      opendarts::linear_solvers::csr_matrix<1> &A,
      HYPRE_IJMatrix &A_ij)
    {
      // NOTE: This function works only for N_BLOCK_SIZE = 1
      //       For other values of the block size a full copy of the data must be 
      //       done and some temporary storage needs to be arranged.
      
      const int print_level = 2;  // print level
      
      // Convert csr_matrix A to Hypre ij_matrix
      opendarts::config::index_t ilower, iupper;
      ilower = 0;
      iupper = A.n_rows - 1;
      
      std::vector<opendarts::config::index_t> rows(A.n_rows), n_cols(A.n_rows);
      std::iota(rows.begin(), rows.end(), 0);
      
      for (opendarts::config::index_t row_idx = 0; row_idx < A.n_rows; row_idx++)
    		n_cols[row_idx] = A.rows_ptr[row_idx + 1] - A.rows_ptr[row_idx];

      check_result(HYPRE_IJMatrixCreate(hypre_MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A_ij));
    	check_result(HYPRE_IJMatrixSetPrintLevel(A_ij, print_level));
    	check_result(HYPRE_IJMatrixSetObjectType(A_ij, HYPRE_PARCSR));
      check_result(HYPRE_IJMatrixInitialize(A_ij));
    	check_result(HYPRE_IJMatrixSetValues(A_ij, A.n_rows, n_cols.data(), rows.data(), A.get_cols_ind(), A.get_values()));
    	check_result(HYPRE_IJMatrixAssemble(A_ij));
    }
    
    void check_result(int res)
    {
      char err_msg_char[256];
      if (res)
      {
      	HYPRE_DescribeError(res, err_msg_char);
        std::string err_msg(err_msg_char);
      	std::cout << "\n" << err_msg << std::endl;
        exit(-1);
      }
    }
    
    template class linsolv_hypre_ilu<1>;
    // Note that for values of block size larger than 1 a matrix copy must be carried 
    // because conversion to a block matrix of size 1 is required
    // template class linsolv_hypre_amg<3>;
  } // namespace linear_solvers
} // namespace opendarts
