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

#include <cstddef>

#include "scalar_csr_adapter.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    scalar_csr_adapter::scalar_csr_adapter(const block_csr_matrix &matrix)
      : matrix_(&matrix),
        expansion_(&matrix.structure().scalar_csr(matrix.block_size())),
        values_(static_cast<std::size_t>(expansion_->nnz()))
    {
      refresh();
    }

    void scalar_csr_adapter::refresh()
    {
      expansion_->gather_values(matrix_->values(), values_.data());
    }
  } // namespace linear_solvers
} // namespace opendarts
