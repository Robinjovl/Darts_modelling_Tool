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

#ifdef WITH_GPU

#include "gpu_tools.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    template <typename fp_type1, typename fp_type2>
    __global__ void copy_fp_data(fp_type1 *dst, fp_type2 *src, int len)
    {
      int i = threadIdx.x + blockIdx.x * blockDim.x;
      if (i >= len)
        return;
      dst[i] = src[i];
    }

    template <typename fp_type1, typename fp_type2>
    void copy_device_data(fp_type1 *dst, fp_type2 *src, int len)
    {
      const int n_threads_block = 1024;
      const int n_blocks = (len + n_threads_block - 1) / n_threads_block;
      copy_fp_data<fp_type1, fp_type2><<<n_blocks, n_threads_block>>>(dst, src, len);
    }

    // Explicit instantiations for the float<->double conversions in use.
    template void copy_device_data<>(float *, double *, int);
    template void copy_device_data<>(double *, float *, int);
  } // namespace linear_solvers
} // namespace opendarts

#endif // WITH_GPU
