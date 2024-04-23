#include <fstream>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <limits>
#include <algorithm>

#include "multilinear_static_nested_cpu_interpolator.hpp"
#include "multilinear_interpolator_common.h"

using namespace std;

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
multilinear_static_nested_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::
  multilinear_static_nested_cpu_interpolator(operator_set_evaluator_iface *supporting_point_evaluator,
                                          const std::vector<int> &axes_points,
                                          const std::vector<double> &axes_min,
                                          const std::vector<double> &axes_max,
                                          const index_t _nest_lvl)
    : multilinear_static_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>(supporting_point_evaluator, axes_points, axes_min, axes_max),
      nest_lvl(_nest_lvl)
{
  // compute the total number of hypercubes
  n_hypercubes_total = 1;
  for (int i = 0; i < N_DIMS; i++)
  {
    n_hypercubes_total *= this->axes_points[i] - 1;
  }
  hypercube_interpolators.resize(n_hypercubes_total, nullptr);
  active_hypercube_counter.resize(n_hypercubes_total, 0);
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
multilinear_static_nested_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::~multilinear_static_nested_cpu_interpolator()
{
  for (auto& interp : hypercube_interpolators)
  {
    if (interp != nullptr)
      delete interp;
  }
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
int multilinear_static_nested_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::init_nested(const index_t init_nest_lvl, 
                                                                            const std::vector<int>& nested_axes_points)
{
  int res = 0;
  index_t axis_idx[N_DIMS];
  vector<double> axis_low(N_DIMS), axis_high(N_DIMS);

  if (init_nest_lvl == nest_lvl + 1) // initialize nested interpolators here
  {
    for (index_t i = 0; i < n_hypercubes_total; i++)
    {
      const auto& counter = active_hypercube_counter[i];
      auto& hc_interp = hypercube_interpolators[i];
      if (counter > 0 && hc_interp == nullptr)
      {
        decode_hypercube_idx<index_t, N_DIMS>(i, &axis_idx[0], this->axis_hypercube_mult.data());
        // calculate axis_low using the decoded axis index
        for (int i = 0; i < N_DIMS; i++)
        {
          axis_low[i] = axis_idx[i] * this->axes_step_internal[i] + this->axes_min_internal[i];
          axis_high[i] = (axis_idx[i] + 1) * this->axes_step_internal[i] + this->axes_min_internal[i];
        }

        // need to find axes_min, axes_max for a given hypercube
        hc_interp = new multilinear_static_nested_cpu_interpolator(this->supporting_point_evaluator, nested_axes_points, 
                                                                    axis_low, axis_high, nest_lvl + 1);
        hc_interp->init();
      }
    }
  }
  else if (init_nest_lvl > nest_lvl + 1) // pass to next level intepolators
  {
    for (auto& interp : hypercube_interpolators)
    {
      if (interp != nullptr)
        res += interp->init_nested(init_nest_lvl, nested_axes_points);
    }
  }

  return res;
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
const typename multilinear_static_nested_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::hypercube_data_t& multilinear_static_nested_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::get_hypercube_data(const index_t hypercube_index)
{

  return this->hypercube_data[hypercube_index];
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
int multilinear_static_nested_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::interpolate_with_derivatives(const std::vector<double>& points, const std::vector<int>& points_idxs,
  std::vector<double>& values, std::vector<double>& derivatives)
{
#pragma omp parallel for
  for (int i = 0; i < points_idxs.size(); i++)
  {

    index_t offset = points_idxs[i];
    interpolate_with_derivatives(points.data() + offset * N_DIMS,
      values.data() + offset * N_OPS,
      derivatives.data() + offset * N_OPS * N_DIMS);
  }

  return 0;
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
int multilinear_static_nested_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::interpolate_with_derivatives(const double* point, double* values, double* derivatives)
{
  index_t hypercube_idx = 0;
  value_t axis_low[N_DIMS];
  value_t mult[N_DIMS];

  for (int i = 0; i < N_DIMS; ++i)
  {
    int axis_idx = get_axis_interval_index_low_mult<value_t>(point[i],
      this->axes_min_internal[i], this->axes_max_internal[i], this->axes_step_internal[i],
      this->axes_step_inv_internal[i], this->axes_points[i],
      &axis_low[i], &mult[i]);
    hypercube_idx += axis_idx * this->axis_hypercube_mult[i];
  }
  active_hypercube_counter[hypercube_idx]++;

  auto& nested_interp = hypercube_interpolators[hypercube_idx];
  if (nested_interp != nullptr) // delegate interpolation to nested if exists
  {
    nested_interp->interpolate_with_derivatives(point, values, derivatives);
  }
  else // do interpolation on your own
  {
    const hypercube_data_t& hypercube = this->get_hypercube_data(hypercube_idx);
    interpolate_point_with_derivatives<value_t, N_DIMS, N_OPS>(point, hypercube.data(),
      &axis_low[0], &mult[0], this->axes_step_inv_internal.data(),
      values,
      derivatives);
  }

  return 0;
}