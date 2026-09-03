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
#ifndef OPENDARTS_LINEAR_SOLVERS_BLOCK_CSR_IO_HPP
#define OPENDARTS_LINEAR_SOLVERS_BLOCK_CSR_IO_HPP
//--------------------------------------------------------------------------

// Debugging I/O for block_csr_matrix (SOLVER_REFACTORING_PLAN.md section
// 12.10, phase A5). Used only for diagnostic dumps of the Jacobian; not on
// any performance path.

#include <string>

#include "block_csr_matrix.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** Writes @p matrix to @p filename in a human-readable block-CSR text
        format (header line + one line per stored block: "row col : values").
        Returns 0 on success, nonzero if the file could not be opened. */
    int write_block_csr_matrix(const block_csr_matrix &matrix, const std::string &filename);
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_BLOCK_CSR_IO_HPP
//--------------------------------------------------------------------------
