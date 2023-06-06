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

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "openDARTS/linear_solvers/csr_matrix_base.hpp"
#include "openDARTS/linear_solvers/linear_solver_base.hpp"
#include "openDARTS/linear_solvers/data_types.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    int linear_solver_base::solve(opendarts::linear_solvers::csr_matrix_base *A, 
      opendarts::config::mat_float *B, opendarts::config::mat_float *X)
    {
      (void)A; // to suppress warning of unused parameter
      (void)B;
      (void)X;
      
      return 0;
    }
    
    void linear_solver_base::set_prec(opendarts::linear_solvers::linear_solver_base *new_prec)
    {
      this->prec = new_prec;
    }
    
    opendarts::linear_solvers::linear_solver_base *linear_solver_base::get_prec() const
    {
      return this->prec;
    }
  } // namespace linear_solvers
} // namespace opendarts
