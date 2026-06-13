#include <fstream>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <limits>
#include <algorithm>

#include "multilinear_interpolator_base.hpp"
#include "multilinear_interpolator_common.h"


// Unbounded grid (adaptive): (origin, step) only. The bounded-path helpers
// (axis_point_mult, axis_hypercube_mult, axes_max_internal) are left empty —
// adaptive subclasses override interpolate()/interpolate_with_derivatives() and
// never touch them.
template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
multilinear_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::multilinear_interpolator_base(operator_set_evaluator_iface *supporting_point_evaluator,
                                                                                              const std::vector<double> &axes_origin_,
                                                                                              const std::vector<double> &axes_step_)
    : interpolator_base(supporting_point_evaluator, axes_origin_, axes_step_),
      axes_origin_internal(this->axes_origin.begin(), this->axes_origin.end()),
      axes_step_internal(this->axes_step.begin(), this->axes_step.end()),
      axes_step_inv_internal(this->axes_step_inv.begin(), this->axes_step_inv.end())
{
}

// Bounded dense grid (static): builds the flat mixed-radix multipliers used by the
// dense storage path, and copies the derived axes_max into value_t precision.
template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
multilinear_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::multilinear_interpolator_base(operator_set_evaluator_iface *supporting_point_evaluator,
                                                                                              const std::vector<double> &axes_origin_,
                                                                                              const std::vector<double> &axes_step_,
                                                                                              const std::vector<int> &axes_points_)
    : interpolator_base(supporting_point_evaluator, axes_origin_, axes_step_, axes_points_),
      axes_origin_internal(this->axes_origin.begin(), this->axes_origin.end()),
      axes_step_internal(this->axes_step.begin(), this->axes_step.end()),
      axes_step_inv_internal(this->axes_step_inv.begin(), this->axes_step_inv.end()),
      axes_max_internal(this->axes_max.begin(), this->axes_max.end())
{
  // The dense static grid is bounded; n_points_total is finite (the static init()
  // validates it against index_t range before allocating). Per-axis index overflow on
  // the adaptive (cell_key_t::idx is int32_t) path is detected separately at evaluation
  // time in get_axis_interval_index_unbounded — it does not apply to this dense path.
  axis_point_mult.resize(N_DIMS);
  axis_hypercube_mult.resize(N_DIMS);
  axis_point_mult[N_DIMS - 1] = 1;
  axis_hypercube_mult[N_DIMS - 1] = 1;
  for (int i = N_DIMS - 2; i >= 0; --i)
  {
    axis_point_mult[i] = axis_point_mult[i + 1] * this->axes_points[i + 1];
    axis_hypercube_mult[i] = axis_hypercube_mult[i + 1] * (this->axes_points[i + 1] - 1);
  }
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
void multilinear_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::get_point_coordinates(index_t point_index, point_coordinates_t &coordinates)
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
void multilinear_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::get_hypercube_points(index_t hypercube_idx, hypercube_points_index_t &hypercube_points)
{
  index_t remainder_idx = hypercube_idx;
  index_t pwr = N_VERTS;
  hypercube_points.fill(0);

  for (auto i = 0; i < N_DIMS; ++i)
  {

    index_t axis_idx = remainder_idx / axis_hypercube_mult[i];
    remainder_idx = remainder_idx % axis_hypercube_mult[i];

    pwr /= 2;

    for (uint64_t j = 0; j < N_VERTS; ++j)
    {
      index_t zero_or_one = (static_cast<index_t>(j) / pwr) % 2;
      hypercube_points[j] += (axis_idx + zero_or_one) * axis_point_mult[i];
    }
  }
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
int multilinear_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::interpolate(const std::vector<double> &point, std::vector<double> &values)
{
  // let it be a but less efficient but general,
  // use the same routine and compute derivatives as well, despite we don`t need them.
  double derivatives[N_OPS * N_DIMS];

  if (point.size() != N_DIMS)
    {
	  printf("Inconsistence in interpolation! Point size = %zu should be equal to N_DIMS = %d\n", point.size(), N_DIMS);
	}

  interpolate_with_derivatives(point.data(), values.data(), &derivatives[0]);

  return 0;
}


template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
int multilinear_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::interpolate_with_derivatives(const double *point,
                                                                                                 double *values,
                                                                                                 double *derivatives)
{
  index_t hypercube_idx = 0;
  value_t axis_low[N_DIMS];
  value_t mult[N_DIMS];

  for (int i = 0; i < N_DIMS; ++i)
  {
    int axis_idx = get_axis_interval_index_low_mult<value_t>(point[i],
                                                             this->axes_origin_internal[i], this->axes_max_internal[i], this->axes_step_internal[i],
                                                             this->axes_step_inv_internal[i], axes_points[i],
                                                             &axis_low[i], &mult[i]);
    hypercube_idx += static_cast<index_t>(axis_idx) * axis_hypercube_mult[i];
  }
  const hypercube_data_t &hypercube = this->get_hypercube_data(hypercube_idx);
  interpolate_point_with_derivatives<value_t, N_DIMS, N_OPS>(point, hypercube.data(),
                                                             &axis_low[0], &mult[0], this->axes_step_inv_internal.data(),
                                                             values,
                                                             derivatives);

  return 0;
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
int multilinear_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::interpolate_with_derivatives(const std::vector<double> &points, const std::vector<int> &points_idxs,
                                                                                                 std::vector<double> &values, std::vector<double> &derivatives)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < static_cast<int64_t>(points_idxs.size()); ++i)
  {

    index_t offset = points_idxs[i];
    interpolate_with_derivatives(points.data() + offset * N_DIMS,
                                 values.data() + offset * N_OPS,
                                 derivatives.data() + offset * N_OPS * N_DIMS);
  }

  return 0;
}
