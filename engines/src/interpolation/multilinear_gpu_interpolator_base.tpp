#include <fstream>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <limits>
#include <algorithm>

#include "multilinear_gpu_interpolator_base.hpp"

using namespace std;

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::multilinear_gpu_interpolator_base(operator_set_evaluator_iface *supporting_point_evaluator,
                                                                                                      const std::vector<int> &axes_points,
                                                                                                      const std::vector<double> &axes_min,
                                                                                                      const std::vector<double> &axes_max)
    : supporting_point_evaluator(supporting_point_evaluator), axes_points(axes_points), axes_min(axes_min), axes_max(axes_max)

{
  assert(axes_min.size() == axes_points.size());
  assert(axes_max.size() == axes_points.size());
  int n_dims_interpolator = N_DIMS;
  assert(n_dims_interpolator == axes_points.size());

  axes_step.resize(N_DIMS);
  axes_step_inv.resize(N_DIMS);
  for (int dim = 0; dim < N_DIMS; dim++)
  {
    axes_step[dim] = (axes_max[dim] - axes_min[dim]) / (axes_points[dim] - 1);
    axes_step_inv[dim] = 1 / axes_step[dim];
  }

  //use double to avoid overflow
  n_points_total_fp = 1;
  for (int dim = 0; dim < N_DIMS; dim++)
    n_points_total_fp *= axes_points[dim];

  n_points_total = n_points_total_fp;
  n_points_used = 0;
  n_interpolations = 0;

  if (n_points_total_fp > std::numeric_limits<index_t>::max())
  {
    std::string error = "Error: The total requested amount of points (" + std::to_string(n_points_total_fp) +
                        ") exceeds the limit in index type (" + std::to_string(std::numeric_limits<index_t>::max()) + ")\n";
    throw std::range_error(error);
  }
  axis_point_mult.resize(N_DIMS);
  axis_hypercube_mult.resize(N_DIMS);
  axis_point_mult[N_DIMS - 1] = 1;
  axis_hypercube_mult[N_DIMS - 1] = 1;
  for (int i = N_DIMS - 2; i >= 0; --i)
  {
    axis_point_mult[i] = axis_point_mult[i + 1] * axes_points[i + 1];
    axis_hypercube_mult[i] = axis_hypercube_mult[i + 1] * (axes_points[i + 1] - 1);
  }
  axes_points_d = axes_points;
  axes_min_d = axes_min;
  axes_max_d = axes_max;
  axes_step_d = axes_step;
  axes_step_inv_d = axes_step_inv;
  axis_point_mult_d = axis_point_mult;
  axis_hypercube_mult_d = axis_hypercube_mult;

  new_point_coords.resize(N_DIMS);
  new_operator_values.resize(N_OPS);
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
int multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::evaluate_d(double *state_d, double *values_d)
{
  thrust::device_vector<int> index_d(1);
  thrust::device_vector<double> derivatives_d(N_OPS * N_DIMS);
  index_d[0] = 0;
  evaluate_with_derivatives_d(1, state_d, thrust::raw_pointer_cast(index_d.data()),
                              values_d, thrust::raw_pointer_cast(derivatives_d.data()));
  return 0;
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
void multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::get_point_coordinates(index_t point_index, point_coordinates_t &coordinates)
{
  auto remainder_idx = point_index;
  for (auto i = 0; i < N_DIMS; ++i)
  {
    index_t axis_idx = remainder_idx / axis_point_mult[i];
    remainder_idx = remainder_idx % axis_point_mult[i];
    coordinates[i] = this->axes_min[i] + this->axes_step[i] * axis_idx;
  }
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
void multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::get_hypercube_points(index_t hypercube_idx, hypercube_points_index_t &hypercube_points)
{
  auto remainder_idx = hypercube_idx;
  auto pwr = N_VERTS;
  hypercube_points.fill(0);

  for (auto i = 0; i < N_DIMS; ++i)
  {

    index_t axis_idx = remainder_idx / axis_hypercube_mult[i];
    remainder_idx = remainder_idx % axis_hypercube_mult[i];

    pwr /= 2;

    for (auto j = 0; j < N_VERTS; ++j)
    {
      auto zero_or_one = (j / pwr) % 2;
      hypercube_points[j] += (axis_idx + zero_or_one) * axis_point_mult[i];
    }
  }
}
template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
int multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::get_axis_n_points(int axis) const
{
  return axes_points[axis];
}
template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
double multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::get_axis_max(int axis) const
{
  return axes_max[axis];
}
template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
double multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::get_axis_min(int axis) const 
{
  return axes_min[axis];
}
template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
uint64_t multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::get_n_interpolations() const 
{
  return n_interpolations;
}
template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
uint64_t multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::get_n_points_total() const
{
  return n_points_total;
}
template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
uint64_t multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::get_n_points_used() const
{
  return n_points_used;
}