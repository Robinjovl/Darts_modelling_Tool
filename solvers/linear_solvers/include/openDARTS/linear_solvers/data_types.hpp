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
#ifndef OPENDARTS_LINEAR_SOLVERS_DATA_TYPES_H
#define OPENDARTS_LINEAR_SOLVERS_DATA_TYPES_H
//--------------------------------------------------------------------------

#include <iostream>
#include <type_traits>

namespace opendarts
{
  namespace linear_solvers
  {
    // TODO: These names should change from MATRIX_TYPE_XXXXX simply to XXXXX since it is
    // enum class of type sparse_matrix_type, it is just duplicating names
    enum class sparse_matrix_type
    {
      MATRIX_TYPE_UNDEFINED = 0,
      MATRIX_TYPE_BANDED = 1,
      MATRIX_TYPE_CSR,
      MATRIX_TYPE_2_IN_1,
      MATRIX_TYPE_MPI_CSR_SIMPLE,
      MATRIX_TYPE_MPI_CSR_DIAGOFFD,
      MATRIX_TYPE_MPI_2_IN_1,
      MATRIX_TYPE_2_CSR_IN_1,
      MATRIX_TYPE_CSR_FIXED_STRUCTURE,
    };

    // For backwards compatibility with previous version of linear_solvers
    // TODO: to remove
    constexpr auto MATRIX_TYPE_UNDEFINED = sparse_matrix_type::MATRIX_TYPE_UNDEFINED;
    constexpr auto MATRIX_TYPE_BANDED = sparse_matrix_type::MATRIX_TYPE_BANDED;
    constexpr auto MATRIX_TYPE_CSR = sparse_matrix_type::MATRIX_TYPE_CSR;
    constexpr auto MATRIX_TYPE_2_IN_1 = sparse_matrix_type::MATRIX_TYPE_2_IN_1;
    constexpr auto MATRIX_TYPE_MPI_CSR_SIMPLE = sparse_matrix_type::MATRIX_TYPE_MPI_CSR_SIMPLE;
    constexpr auto MATRIX_TYPE_MPI_CSR_DIAGOFFD = sparse_matrix_type::MATRIX_TYPE_MPI_CSR_DIAGOFFD;
    constexpr auto MATRIX_TYPE_MPI_2_IN_1 = sparse_matrix_type::MATRIX_TYPE_MPI_2_IN_1;
    constexpr auto MATRIX_TYPE_2_CSR_IN_1 = sparse_matrix_type::MATRIX_TYPE_2_CSR_IN_1;
    constexpr auto MATRIX_TYPE_CSR_FIXED_STRUCTURE = sparse_matrix_type::MATRIX_TYPE_CSR_FIXED_STRUCTURE;
    
    enum class Preconditioner
    { 
      FS_UP,
      FS_UPG 
    };

    // Matrix export and import formats
    enum class sparse_matrix_export_format
    {
      human_readable,
      csr,
      csr_sorted,
      ij,
      mm
    };

    enum class sparse_matrix_import_format
    {
      csr
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_DATA_TYPES_H
//--------------------------------------------------------------------------
