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

#include "openDARTS/config/data_types.hpp"
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#include "openDARTS/linear_solvers/csr_matrix_base.hpp"
#include "openDARTS/linear_solvers/linsolv_iface.hpp"
#include "openDARTS/linear_solvers/linsolv_bos_cpr.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    template <uint8_t N_BLOCK_SIZE> 
    linsolv_bos_cpr<N_BLOCK_SIZE>::linsolv_bos_cpr()
    {
      std::cout << "NOT IMPLEMENTED: linsolv_bos_cpr::linsolv_bos_cpr" << std::endl;
    }
    
    template <uint8_t N_BLOCK_SIZE> 
    linsolv_bos_cpr<N_BLOCK_SIZE>::~linsolv_bos_cpr()
    {
      std::cout << "NOT IMPLEMENTED: linsolv_bos_cpr::~linsolv_bos_cpr" << std::endl;
    }
    
    template <uint8_t N_BLOCK_SIZE> 
    int linsolv_bos_cpr<N_BLOCK_SIZE>::solve(opendarts::linear_solvers::csr_matrix_base *matrix, 
        opendarts::config::mat_float *v, 
        opendarts::config::mat_float *r)
    {
      std::cout << "NOT IMPLEMENTED: linsolv_bos_cpr::solve(csr_matrix_base)" << std::endl;
      
      // TODO: This cannot be like this. Matrix is not used, this is confusing for user.
      (void) matrix;
  
      return this->solve(v, r);
    };
    
    template <uint8_t N_BLOCK_SIZE> 
    int linsolv_bos_cpr<N_BLOCK_SIZE>::setup(opendarts::linear_solvers::csr_matrix_base *matrix)
    {
      std::cout << "NOT IMPLEMENTED: linsolv_bos_cpr::setup(csr_matrix_base)" << std::endl;
      // TODO: This cannot be like this. Why is this needed? The whole classes need 
      //       to be redesigned.
      return this->setup((opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *)matrix);
    }

    template <uint8_t N_BLOCK_SIZE> 
    int linsolv_bos_cpr<N_BLOCK_SIZE>::set_prec(opendarts::linear_solvers::linsolv_iface *prec_in)
    {
      (void) prec_in;
      
      std::cout << "NOT IMPLEMENTED: linsolv_bos_cpr::set_prec" << std::endl;
      
      return 1;
    }

    template <uint8_t N_BLOCK_SIZE> 
    int linsolv_bos_cpr<N_BLOCK_SIZE>::init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_in, 
      opendarts::config::index_t max_iters, 
      opendarts::config::mat_float tolerance)
    {
      (void) A_in;
      (void) max_iters;
      (void) tolerance; 
      
      std::cout << "NOT IMPLEMENTED: linsolv_bos_cpr::init" << std::endl;
      
      return 1;
    }
    
    template <uint8_t N_BLOCK_SIZE> 
    int linsolv_bos_cpr<N_BLOCK_SIZE>::setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_in)
    {
      (void) A_in;
      
      std::cout << "NOT IMPLEMENTED: linsolv_bos_cpr::setup" << std::endl;
      
      return 1;
    }

    template <uint8_t N_BLOCK_SIZE> 
    int linsolv_bos_cpr<N_BLOCK_SIZE>::solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X)
    {
      (void) B; 
      (void) X;
      
      std::cout << "NOT IMPLEMENTED: linsolv_bos_cpr::solve" << std::endl;
      
      return 1;
    }

    template <uint8_t N_BLOCK_SIZE> 
    opendarts::config::index_t linsolv_bos_cpr<N_BLOCK_SIZE>::get_n_iters()
    {
      std::cout << "NOT IMPLEMENTED: linsolv_bos_cpr::get_n_iters" << std::endl;
      
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE> 
    opendarts::config::mat_float linsolv_bos_cpr<N_BLOCK_SIZE>::get_residual()
    {
      std::cout << "NOT IMPLEMENTED: linsolv_bos_cpr::get_residual" << std::endl;
      
      return 0.0;
    }
    
    template class linsolv_bos_cpr<2>;
    template class linsolv_bos_cpr<3>;
    template class linsolv_bos_cpr<4>;
    template class linsolv_bos_cpr<5>;
    template class linsolv_bos_cpr<6>;
    template class linsolv_bos_cpr<7>;
    template class linsolv_bos_cpr<8>;
    template class linsolv_bos_cpr<9>;
    template class linsolv_bos_cpr<10>;
    template class linsolv_bos_cpr<11>;
    template class linsolv_bos_cpr<12>;
    template class linsolv_bos_cpr<13>;
  } // namespace linear_solvers
} // namespace opendarts
