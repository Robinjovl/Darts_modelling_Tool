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
#ifndef OPENDARTS_LINEAR_SOLVERS_GPU_TOOLS_HPP
#define OPENDARTS_LINEAR_SOLVERS_GPU_TOOLS_HPP
//--------------------------------------------------------------------------

#ifdef WITH_GPU

namespace opendarts
{
  namespace linear_solvers
  {
    /** Element-wise device-to-device copy with type conversion.
        Used by the mixed-precision GPU solvers to move data between float
        and double buffers without a host round-trip.
        @param dst - destination device buffer.
        @param src - source device buffer.
        @param len - number of elements to copy. */
    template <typename fp_type1, typename fp_type2>
    void copy_device_data(fp_type1 *dst, fp_type2 *src, int len);
  } // namespace linear_solvers
} // namespace opendarts

#endif // WITH_GPU

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_GPU_TOOLS_HPP
//--------------------------------------------------------------------------
