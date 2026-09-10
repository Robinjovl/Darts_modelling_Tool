#include <fstream>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <limits>
#include <algorithm>

#include "multilinear_gpu_interpolator_base.hpp"

// Unbounded grid (adaptive GPU): (origin, step) only. The static-only device arrays
// (axes_points_d, axes_max_d, axis_*_mult_d) are left empty — the adaptive kernels use
// only axes_origin_d / axes_step_d / axes_step_inv_d with signed multi-index keys.
template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::multilinear_gpu_interpolator_base(operator_set_evaluator_iface *supporting_point_evaluator,
                                                                                                      const std::vector<double> &axes_origin_,
                                                                                                      const std::vector<double> &axes_step_)
    : supporting_point_evaluator(supporting_point_evaluator), axes_origin(axes_origin_), axes_step(axes_step_)

{
  assert(axes_step.size() == axes_origin.size());
  assert(N_DIMS == axes_origin.size());

  axes_step_inv.resize(N_DIMS);
  for (int dim = 0; dim < N_DIMS; dim++)
    axes_step_inv[dim] = 1.0 / axes_step[dim];

  // Unbounded grid: no finite supporting-point count, and no flat mixed-radix index is
  // ever formed (adaptive storage keys on cell_key_t / a 64-bit hash of it). The old
  // 1024^N_DIMS product therefore cannot occur. n_points_total stays 0 (diagnostic only).
  n_points_total_fp = 0.0;
  n_points_total = 0;
  n_points_used = 0;
  n_interpolations = 0;

  axes_origin_d = axes_origin;
  axes_step_d = axes_step;
  axes_step_inv_d = axes_step_inv;

  new_point_coords.resize(N_DIMS);
  new_operator_values.resize(N_OPS);
}

// Bounded dense grid (static GPU): builds the flat mixed-radix multipliers + axes_max
// and uploads the bounded device arrays used by the static kernel.
template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::multilinear_gpu_interpolator_base(operator_set_evaluator_iface *supporting_point_evaluator,
                                                                                                      const std::vector<double> &axes_origin_,
                                                                                                      const std::vector<double> &axes_step_,
                                                                                                      const std::vector<int> &axes_points_)
    : supporting_point_evaluator(supporting_point_evaluator), axes_origin(axes_origin_), axes_step(axes_step_),
      axes_points(axes_points_),
      axes_max([&] {
        std::vector<double> mx(axes_origin_.size());
        for (std::size_t d = 0; d < axes_origin_.size(); ++d)
          mx[d] = axes_origin_[d] + (axes_points_[d] - 1) * axes_step_[d];
        return mx;
      }())

{
  assert(axes_step.size() == axes_origin.size());
  assert(axes_points.size() == axes_origin.size());
  assert(N_DIMS == axes_origin.size());

  axes_step_inv.resize(N_DIMS);
  for (int dim = 0; dim < N_DIMS; dim++)
    axes_step_inv[dim] = 1.0 / axes_step[dim];

  //use double to avoid overflow
  n_points_total_fp = 1;
  for (int dim = 0; dim < N_DIMS; dim++)
    n_points_total_fp *= axes_points[dim];

  // Bounded grid: n_points_total is the real dense product. The static GPU init()
  // guards it against index_t range before allocating dense storage.
  n_points_total = (n_points_total_fp > static_cast<double>(std::numeric_limits<index_t>::max()))
                       ? std::numeric_limits<index_t>::max()
                       : static_cast<index_t>(n_points_total_fp);
  n_points_used = 0;
  n_interpolations = 0;

  axis_point_mult.resize(N_DIMS);
  axis_hypercube_mult.resize(N_DIMS);
  axis_point_mult[N_DIMS - 1] = 1;
  axis_hypercube_mult[N_DIMS - 1] = 1;
  for (int i = N_DIMS - 2; i >= 0; --i)
  {
    axis_point_mult[i] = axis_point_mult[i + 1] * axes_points[i + 1];
    axis_hypercube_mult[i] = axis_hypercube_mult[i + 1] * (axes_points[i + 1] - 1);
  }
  axes_origin_d = axes_origin;
  axes_step_d = axes_step;
  axes_step_inv_d = axes_step_inv;
  axes_points_d = axes_points;
  axes_max_d = axes_max;
  axis_point_mult_d = axis_point_mult;
  axis_hypercube_mult_d = axis_hypercube_mult;

  new_point_coords.resize(N_DIMS);
  new_operator_values.resize(N_OPS);
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
int multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::evaluate_d(double *state_d, double *values_d)
{
  thrust::device_vector<int> index_d(1);
  thrust::device_vector<double> derivatives_d(N_OPS * N_DIMS);
  index_d[0] = 0;
  evaluate_with_derivatives_d(1, state_d, thrust::raw_pointer_cast(index_d.data()),
                              values_d, thrust::raw_pointer_cast(derivatives_d.data()));
  return 0;
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
void multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::get_point_coordinates(index_t point_index, point_coordinates_t &coordinates)
{
  auto remainder_idx = point_index;
  for (auto i = 0; i < N_DIMS; ++i)
  {
    index_t axis_idx = remainder_idx / axis_point_mult[i];
    remainder_idx = remainder_idx % axis_point_mult[i];
    coordinates[i] = this->axes_origin[i] + this->axes_step[i] * axis_idx;
  }
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
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
template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
int multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::get_axis_n_points(int axis) const
{
  // Unbounded grids have no finite point count; report 0 (the OBL-stats print guards
  // against this). Bounded grids return the real per-axis count.
  return axes_points.empty() ? 0 : axes_points[axis];
}
template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
double multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::get_axis_max(int axis) const
{
  // Unbounded grids have no upper bound; report +inf so any engine-side axis clamp
  // becomes a no-op on the upper side — consistent with the adaptive cache extrapolating.
  return axes_max.empty() ? std::numeric_limits<double>::max() : axes_max[axis];
}
template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
double multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::get_axis_min(int axis) const
{
  return axes_origin[axis];
}
template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
uint64_t multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::get_n_interpolations() const
{
  return n_interpolations;
}
template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
uint64_t multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::get_n_points_total() const
{
  return n_points_total;
}
template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
uint64_t multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::get_n_points_used() const
{
  return n_points_used;
}
