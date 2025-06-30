#ifndef D89802AE_4C88_4BCD_88D1_1B45D12E933F
#define D89802AE_4C88_4BCD_88D1_1B45D12E933F

// define those to avoid warning indication in syntax check for non-nvcc compilers
#ifndef __NVCC__
#define __forceinline__
#define __host__
#define __device__
#endif

/**
 * @brief Get the index of interval, which containes specified point coordinate, for specified axis
 * 
 * @tparam service_value_t - floating point type used for axis parameters
 * @param axis_values - state values, coordinates of point to interpolate
 * @param i - index of axis 
 * @param axis_min - array of minimim values for axes (left parametrization limit)
 * @param axis_max - array of maximum values for axes (right parametrization limit)
 * @param axis_step_inv - array of inverted values of axes intervals lengths
 * @param axis_points - array of numbers of supporting points for axes
 * @return axis interval index 
 */
template <typename service_value_t>
__forceinline__ __host__ __device__ unsigned int get_axis_idx(const value_t *axis_values, int i,
                                                              service_value_t *axis_min, service_value_t *axis_max,
                                                              service_value_t *axis_step_inv,
                                                              uint32_t *axis_points)
{
  int axis_idx = int((axis_values[i] - axis_min[i]) * axis_step_inv[i]);

  // check that axis_idx is within interpolation interval: valid axis_idx is between [0; axis_points[i] - 2],
  // since there are axis_points[i]-1 intervals along the axis
  if (axis_idx < 0)
  {
    axis_idx = 0;
    if (axis_values[i] < axis_min[i])
    {
      printf("Interpolation warning: axis %d is out of limits (%lf; %lf) with value %lf, extrapolation is applied\n", i, axis_min[i], axis_max[i], axis_values[i]);
    }
  }
  else if (axis_idx > (axis_points[i] - 2))
  {
    axis_idx = axis_points[i] - 2;
    if (axis_values[i] > axis_max[i])
    {
      printf("Interpolation warning: axis %d is out of limits (%lf; %lf) with value %lf, extrapolation is applied\n", i, axis_min[i], axis_max[i], axis_values[i]);
    }
  }

  return axis_idx;
}

template <typename value_t>
__forceinline__ __host__ __device__ int get_axis_interval_index(double axis_value,
                                                                value_t axis_min, value_t axis_max,
                                                                value_t axis_step_inv,
                                                                int axis_points)
{
  int axis_interval_index = int((axis_value - axis_min) * axis_step_inv);

  // check that axis_idx is within interpolation interval: valid axis_idx is between [0; axis_points - 2],
  // since there are axis_points-1 intervals along the axis
  if (axis_interval_index < 0)
  {
    axis_interval_index = 0;
    if (axis_value < axis_min)
    {
      printf("Interpolation warning: axis is out of limits (%lf; %lf) with value %lf, extrapolation is applied\n", axis_min, axis_max, axis_value);
    }
  }
  else if (axis_interval_index > (axis_points - 2))
  {
    axis_interval_index = axis_points - 2;
    if (axis_value > axis_max)
    {
      printf("Interpolation warning: axis is out of limits (%lf; %lf) with value %lf, extrapolation is applied\n", axis_min, axis_max, axis_value);
    }
  }

  return axis_interval_index;
}

template <uint8_t N_DIMS, typename service_value_t, typename service_index_t>
__forceinline__ __host__ __device__ service_index_t get_body_idx(const value_t *axis_values, int i,
                                                                 service_value_t *axis_min, service_value_t *axis_max,
                                                                 service_value_t *axis_step_inv, service_index_t *axis_body_mult,
                                                                 uint32_t *axis_points)
{
  service_index_t body_idx = 0;
  for (int i = 0; i < N_DIMS; ++i)
  {
    body_idx += get_axis_idx(axis_values, i, axis_min, axis_max, axis_step_inv, axis_points) * axis_body_mult[i];
  }

  return body_idx;
}

template <typename service_value_t, typename interp_value_t>
__forceinline__ __host__ __device__ unsigned int get_axis_idx_low_mult(const value_t *axis_values, int i,
                                                                       service_value_t *axis_min, service_value_t *axis_max,
                                                                       service_value_t *axis_step, service_value_t *axis_step_inv,
                                                                       uint32_t *axis_points,
                                                                       // OUTPUT:
                                                                       interp_value_t *axis_low,
                                                                       interp_value_t *axis_mult)
{
  int axis_idx = get_axis_idx(axis_values, i, axis_min, axis_max, axis_step_inv, axis_points);

  axis_low[i] = axis_idx * axis_step[i] + axis_min[i];
  axis_mult[i] = (axis_values[i] - axis_low[i]) * axis_step_inv[i];
  return axis_idx;
}

template <typename value_t>
__forceinline__ __host__ __device__ int get_axis_interval_index_low_mult(double axis_value,
                                                                         value_t axis_min, value_t axis_max,
                                                                         value_t axis_step, value_t axis_step_inv,
                                                                         int axis_points,
                                                                         // OUTPUT:
                                                                         value_t *axis_low,
                                                                         value_t *axis_mult)
{
  int axis_interval_index = get_axis_interval_index(axis_value, axis_min, axis_max, axis_step_inv, axis_points);

  *axis_low = axis_interval_index * axis_step + axis_min;
  *axis_mult = (axis_value - *axis_low) * axis_step_inv;
  return axis_interval_index;
}

template <typename service_value_t, typename interp_value_t, uint16_t N_DIMS, uint16_t N_OPS>
__forceinline__ __host__ __device__ void interpolate_with_derivatives(const value_t *axis_values,
                                                                      const interp_value_t *body_data,
                                                                      interp_value_t *axis_low,
                                                                      interp_value_t *axis_mult,
                                                                      service_value_t *axis_step_inv,
                                                                      // OUTPUT:
                                                                      value_t *interp_values, value_t *interp_derivs)
{
  static const uint16_t N_VERTS = 1 << N_DIMS;
  uint16_t pwr = N_VERTS / 2; // distance between high and low values
  interp_value_t workspace[(2 * N_VERTS - 1) * N_OPS];

  // copy operator values for all vertices
  for (int i = 0; i < N_VERTS * N_OPS; ++i)
  {
    workspace[i] = body_data[i];
  }

  for (int i = 0; i < N_DIMS; ++i)
  {
    //printf ("i = %d, N_VERTS = %d, New offset: %d\n", i, N_VERTS, 2 * N_VERTS - (N_VERTS>>i));

    for (int j = 0; j < pwr; ++j)
    {
      for (int op = 0; op < N_OPS; ++op)
      {
        // update own derivative
        workspace[(2 * N_VERTS - (N_VERTS >> i) + j) * N_OPS + op] = (workspace[(j + pwr) * N_OPS + op] - workspace[j * N_OPS + op]) * axis_step_inv[i];
      }

      // update all dependent derivatives
      for (int k = 0; k < i; k++)
      {
        for (int op = 0; op < N_OPS; ++op)
        {
          workspace[(2 * N_VERTS - (N_VERTS >> k) + j) * N_OPS + op] = workspace[(2 * N_VERTS - (N_VERTS >> k) + j) * N_OPS + op] + axis_mult[i] * (workspace[(2 * N_VERTS - (N_VERTS >> k) + j + pwr) * N_OPS + op] - workspace[(2 * N_VERTS - (N_VERTS >> k) + j) * N_OPS + op]);
        }
      }

      for (int op = 0; op < N_OPS; ++op)
      {
        // interpolate value
        workspace[j * N_OPS + op] = workspace[j * N_OPS + op] + (axis_values[i] - axis_low[i]) * workspace[(2 * N_VERTS - (N_VERTS >> i) + j) * N_OPS + op];
      }
    }
    pwr /= 2;
  }
  for (int op = 0; op < N_OPS; ++op)
  {
    interp_values[op] = workspace[op];
    for (int i = 0; i < N_DIMS; ++i)
    {
      interp_derivs[op * N_DIMS + i] = workspace[(2 * N_VERTS - (N_VERTS >> i)) * N_OPS + op];
    }
  }
}

template <typename value_t, uint16_t N_DIMS, uint16_t N_OPS>
__forceinline__ __host__ __device__ void interpolate_point_with_derivatives(const double *axis_values,
                                                                            const value_t *body_data,
                                                                            const value_t *axis_low,
                                                                            const value_t *axis_mult,
                                                                            const value_t *axis_step_inv,
                                                                            // OUTPUT:
                                                                            double *interp_values, double *interp_derivs)
{
  static const uint32_t N_VERTS = 1 << N_DIMS;
  uint32_t pwr = N_VERTS / 2; // distance between high and low values
  static_assert(N_DIMS <= 24, "N_DIMS is too large and exceeds memory limits.");
  std::vector<value_t> workspace((2 * N_VERTS - 1) * N_OPS);

  // copy operator values for all vertices
  for (int i = 0; i < N_VERTS * N_OPS; ++i)
  {
    workspace[i] = body_data[i];
  }

  for (int i = 0; i < N_DIMS; ++i)
  {
    //printf ("i = %d, N_VERTS = %d, New offset: %d\n", i, N_VERTS, 2 * N_VERTS - (N_VERTS>>i));

    for (int j = 0; j < pwr; ++j)
    {
      for (int op = 0; op < N_OPS; ++op)
      {
        // update own derivative
        workspace[(2 * N_VERTS - (N_VERTS >> i) + j) * N_OPS + op] = (workspace[(j + pwr) * N_OPS + op] - workspace[j * N_OPS + op]) * axis_step_inv[i];
      }

      // update all dependent derivatives
      for (int k = 0; k < i; k++)
      {
        for (int op = 0; op < N_OPS; ++op)
        {
          workspace[(2 * N_VERTS - (N_VERTS >> k) + j) * N_OPS + op] = workspace[(2 * N_VERTS - (N_VERTS >> k) + j) * N_OPS + op] + axis_mult[i] * (workspace[(2 * N_VERTS - (N_VERTS >> k) + j + pwr) * N_OPS + op] - workspace[(2 * N_VERTS - (N_VERTS >> k) + j) * N_OPS + op]);
        }
      }

      for (int op = 0; op < N_OPS; ++op)
      {
        // interpolate value
        workspace[j * N_OPS + op] = workspace[j * N_OPS + op] + (axis_values[i] - axis_low[i]) * workspace[(2 * N_VERTS - (N_VERTS >> i) + j) * N_OPS + op];
      }
    }
    pwr /= 2;
  }
  for (int op = 0; op < N_OPS; ++op)
  {
    interp_values[op] = workspace[op];
    for (int i = 0; i < N_DIMS; ++i)
    {
      interp_derivs[op * N_DIMS + i] = workspace[(2 * N_VERTS - (N_VERTS >> i)) * N_OPS + op];
    }
  }
}

template <typename value_t, uint16_t N_DIMS, uint16_t N_OPS>
__forceinline__ __host__ __device__ void interpolate_operator_with_derivatives(const double *axis_values,
                                                                               const value_t *body_data,
                                                                               const value_t *axis_low,
                                                                               const value_t *axis_mult,
                                                                               const value_t *axis_step_inv,
                                                                               const int operator_idx,
                                                                               // OUTPUT:
                                                                               double *interp_values, double *interp_derivs)
{
  static const uint16_t N_VERTS = 1 << N_DIMS;
  uint16_t pwr = N_VERTS / 2; // distance between high and low values
  value_t workspace[2 * N_VERTS - 1];

  // copy operator values for all vertices
  for (int i = 0; i < N_VERTS; ++i)
  {
    workspace[i] = body_data[N_OPS * i + operator_idx];
  }

  for (int i = 0; i < N_DIMS; ++i)
  {
    //printf ("i = %d, N_VERTS = %d, New offset: %d\n", i, N_VERTS, 2 * N_VERTS - (N_VERTS>>i));

    for (int j = 0; j < pwr; ++j)
    {
      // update own derivative
      workspace[2 * N_VERTS - (N_VERTS >> i) + j] = (workspace[j + pwr] - workspace[j]) * axis_step_inv[i];

      // update all dependent derivatives
      for (int k = 0; k < i; k++)
      {
        workspace[2 * N_VERTS - (N_VERTS >> k) + j] = workspace[2 * N_VERTS - (N_VERTS >> k) + j] + axis_mult[i] * (workspace[2 * N_VERTS - (N_VERTS >> k) + j + pwr] - workspace[2 * N_VERTS - (N_VERTS >> k) + j]);
      }

      // interpolate value
      workspace[j] = workspace[j] + (axis_values[i] - axis_low[i]) * workspace[2 * N_VERTS - (N_VERTS >> i) + j];
    }
    pwr /= 2;
  }
  interp_values[operator_idx] = workspace[0];
  for (int i = 0; i < N_DIMS; ++i)
  {
    interp_derivs[operator_idx * N_DIMS + i] = workspace[2 * N_VERTS - (N_VERTS >> i)];
  }
}
#endif /* D89802AE_4C88_4BCD_88D1_1B45D12E933F */
