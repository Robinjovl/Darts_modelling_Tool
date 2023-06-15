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
#include <string>
#include <vector>

#include "openDARTS/linear_solvers/csr_matrix_base.hpp"
#include "openDARTS/linear_solvers/data_types.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    // Initialization member functions
    csr_matrix_base::csr_matrix_base()
    {
      this->type = opendarts::linear_solvers::sparse_matrix_type::MATRIX_TYPE_UNDEFINED;  
      this->n_rows = 0;
      this->n_cols = 0;
      this->n_non_zeros = 0;
      this->n_row_size = 1;  // TODO: this is redudant and confusing with n_block_size, see the header, kept for backwards compatibility
      this->is_square = 0;
    }
    
    
    // TODO: Kept for backwards compatibility, need to check if to keep or not or restructure
    
    int csr_matrix_base::write_matrix_to_file(const char *filename, int sort_cols)
    {
      std::cout << "csr_matrix_base::write_matrix_to_file will be deprecated in the future." << std::endl;
      
      (void) sort_cols;
      if (sort_cols != 0) 
        std::cout << "Sorting columns not implemented." << std::endl;
      
      std::string filename_string(filename);
      
      return this->export_matrix_to_file(filename_string, opendarts::linear_solvers::sparse_matrix_export_format::csr); 
    }
    
    int csr_matrix_base::matrix_vector_product(const double *v, double *r)
    {
      std::cout << "csr_matrix_base::matrix_vector_product will be deprecated in the future." << std::endl;
      
      (void) v;
      (void) r;
      
      return 1;
    }
    
    // calculate linear combination r = alpha * Au + beta * v
    // TODO: Implemented for backwards compatibility, need to check if this is kept or not and how
    int csr_matrix_base::calc_lin_comb(const double alpha, const double beta, double *u, double *v, double *r)
    {
      std::cout << "csr_matrix_base::calc_lin_comb will be deprecated in the future." << std::endl;
      
      (void) alpha;
      (void) beta;
      (void) u;
      (void) v;
      (void) r;
      
      return 1;
    }
    
  } // namespace linear_solvers
} // namespace opendarts
