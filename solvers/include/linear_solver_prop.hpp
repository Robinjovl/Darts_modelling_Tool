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
#ifndef OPENDARTS_LINEAR_SOLVERS_LINEAR_SOLVER_PROP_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINEAR_SOLVER_PROP_HPP
//--------------------------------------------------------------------------

#include "data_types.hpp"
#include "csr_matrix_base.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /**
     * @brief properties for linear solvers
     */
    class linear_solver_prop
    {
      //-----------------------------------------
      //  METHODS
      //-----------------------------------------
      public:
        // constructor
        linear_solver_prop(){};
        // destructor
        ~linear_solver_prop(){};

        // set maximum number of iterations
        int set_max_iters(opendarts::config::index_t n_iters){(void) n_iters; return 0;};

        //! set tolerance
        void set_tolerance(opendarts::config::mat_float new_tol) {if (tol > 10e-16) tol = new_tol;};

        //! return != 0 if method successfully converged
        int check_convergence() {return success;}

        //! return number of iteration
        opendarts::config::index_t get_iters () {return iters;}

        //! set number of iters
        void set_iters (opendarts::config::index_t n_iters) {iters = n_iters;}

        //! return relativ residual denominator
        opendarts::config::mat_float get_relative_factor () {return relative_factor;}

        //! return resid array (0 ... iters)
        opendarts::config::mat_float *get_residuals () const {return resid;}
        //! return convergence rate (0 ... iters)
        opendarts::config::mat_float *get_convergence_rate () const {return convergence_rate;}

        //! return maximum alloved number of iterations
        opendarts::config::index_t get_max_iters () {return max_iters;}

        //! return tolerance
        opendarts::config::mat_float get_tolerance () {return tol;}



      //-----------------------------------------
      //  VARIABLES
      //-----------------------------------------
      private:
        // All members carried over from the legacy bos class uninitialised;
        // initialise them so accessors never read indeterminate values.
        // max_iters/tol take the proprietary constructor's defaults (30 /
        // 1e-5) -- set_tolerance() only assigns when the CURRENT tol exceeds
        // 1e-15, so a zero init would have made it a permanent no-op.
        opendarts::config::index_t max_iters = 30;                 //!< maximum number of iteration
        opendarts::config::mat_float tol = 1.0e-5;                //!< tolerance

        opendarts::config::mat_float *resid = nullptr;            //!< array of L2 residuals sqrt (<b - Ax, b - Ax>) (resid[0] -- initial residual,
                                                                  //!< resid[iters] -- final residual)
        opendarts::config::mat_float *convergence_rate = nullptr; //!< array of convergence rate
      public:
        opendarts::config::mat_float final_resid = 0.0;
        opendarts::config::mat_float relative_factor = 1.0;       //!< denominator for resid array to get relative residual

        opendarts::config::index_t success = 0;                    //!< != 0 if successivly converged
        opendarts::config::index_t iters = 0;                      //!< number of iteration spend for convergence
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINEAR_SOLVER_PROP_HPP
//--------------------------------------------------------------------------
