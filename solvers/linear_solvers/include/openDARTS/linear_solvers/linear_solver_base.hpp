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
#ifndef OPENDARTS_LINEAR_SOLVERS_LINEAR_SOLVER_BASE_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINEAR_SOLVER_BASE_HPP
//--------------------------------------------------------------------------

#include "openDARTS/config/data_types.hpp"
#include "openDARTS/linear_solvers/csr_matrix_base.hpp"
#include "openDARTS/linear_solvers/linear_solver_prop.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    class linear_solver_base
    {
      //-----------------------------------------
      //  METHODS
      //-----------------------------------------
      public:
        // constructor
        linear_solver_base(){};
        
        // destructor
        virtual ~linear_solver_base(){};

        // solve
        virtual int solve(opendarts::linear_solvers::csr_matrix_base *A, 
          opendarts::config::mat_float *B, opendarts::config::mat_float *X);
        
        // setup
        virtual int setup(opendarts::linear_solvers::csr_matrix_base *matrix) = 0;
        
        //! set up preconditioner
        void set_prec(opendarts::linear_solvers::linear_solver_base *new_prec);
        
        opendarts::linear_solvers::linear_solver_base *get_prec() const;

      private:
      //-----------------------------------------
      //  VARIABLES
      //-----------------------------------------
      public:
        opendarts::linear_solvers::linear_solver_prop prop;            //!< properties for solvers
        
      protected:
        opendarts::config::mat_float *wksp;                       //!< workspace array
        opendarts::config::index_t n_memory_allocated;             //!< total amount of memory allocated for workspace
        opendarts::linear_solvers::linear_solver_base *prec;           //!< pointer to the preconditioner
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINEAR_SOLVER_BASE_HPP
//--------------------------------------------------------------------------
