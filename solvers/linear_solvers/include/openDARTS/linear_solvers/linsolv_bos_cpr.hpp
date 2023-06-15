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

// *************************************************************************
// BOS CPR wrapper
// *************************************************************************

//--------------------------------------------------------------------------
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_BOS_CPR_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_BOS_CPR_HPP
//--------------------------------------------------------------------------

#include "openDARTS/config/data_types.hpp"
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#include "openDARTS/linear_solvers/csr_matrix_base.hpp"
#include "openDARTS/linear_solvers/linsolv_iface_bos.hpp"
#include "openDARTS/linear_solvers/linear_solver_base.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    template <uint8_t N_BLOCK_SIZE>
    class linsolv_bos_cpr : public opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>, public opendarts::linear_solvers::linear_solver_base
    {
      public:
        linsolv_bos_cpr();

        ~linsolv_bos_cpr();
        
        // TODO: This implementation is a complete mess.
        // There are two solves, one apparently is for the preconditioner and the 
        // other is for the system to solve. Why not make it explicit? Why not interconnect?
        // Why the two branches of inheritance from linsolv_iface_bos and linear_solver_base? 
        // Is this necessary? Or just legacy?
        // The same for setup...
        // Override should be used to make it clear that some of these functions are
        // overriding previously defined functions.
        // What is worse, is that pointers to base classes are cast to pointers of 
        // derived classes, e.g. setup just below: 
        //    csr_matrix_base --> csr_matrix<N_BLOCK_SIZE>
        // Nothing stops someone of doing something like:
        //    opendarts::linear_solvers::csr_matrix<4> *A_4 = new opendarts::linear_solvers::csr_matrix<4>;
        //    opendarts::linear_solvers::csr_matrix_base *A_base = A_4;
        //    
        //    opendarts::linear_solvers::linsolv_bos_cpr<1> bos_cpr_solver;
        //    bos_cpr_solver.setup(A_base);
        //
        // This is perfectly valid, but is catastrophic, and will keep people scratching their heads.
        
        //////////////////////
        // linear_solver_base
        //////////////////////
        
        // solve preconditioner
        int solve(opendarts::linear_solvers::csr_matrix_base *matrix, 
            opendarts::config::mat_float *v, 
            opendarts::config::mat_float *r) override;

        // setup preconditioner
        int setup(opendarts::linear_solvers::csr_matrix_base *matrix) override;

        //////////////////////
        // linsolv_iface
        //////////////////////

        virtual int set_prec(opendarts::linear_solvers::linsolv_iface *prec_in) override;

        virtual int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_in, 
            opendarts::config::index_t max_iters, 
            opendarts::config::mat_float tolerance) override;
          
        virtual int setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_in) override;

        virtual int solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X) override;
      
        virtual opendarts::config::index_t get_n_iters()  override;

        virtual opendarts::config::mat_float get_residual() override;
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_BOS_CPR_HPP
//--------------------------------------------------------------------------
