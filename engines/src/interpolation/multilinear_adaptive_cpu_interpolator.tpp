//*************************************************************************
//    Copyright (c) 2018
//            Mark Khait         M.Khait@tudelft.nl
//            Denis Voskov    D.V.Voskov@tudelft.nl
//    Delft University of Technology, the Netherlands
//
//    This file is part of the Delft Advanced Research Terra Simulator (DARTS)
//
//    DARTS is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as
//    published by the Free Software Foundation, either version 3 of the
//    License, or (at your option) any later version.
//
//    DARTS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public
//    License along with DARTS. If not, see <http://www.gnu.org/licenses/>.
// *************************************************************************

#include <fstream>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <limits>
#include <algorithm>

#include "multilinear_adaptive_cpu_interpolator.hpp"

using namespace std;

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::
    multilinear_adaptive_cpu_interpolator(operator_set_evaluator_iface *supporting_point_evaluator,
                                          const std::vector<int> &axes_points,
                                          const std::vector<double> &axes_min,
                                          const std::vector<double> &axes_max)
    : multilinear_interpolator_base<index_t, value_t, N_DIMS, N_OPS>(supporting_point_evaluator, axes_points, axes_min, axes_max)

{
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
const typename multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::point_data_t &
multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::get_point_data(const index_t point_index)
{
  auto item = point_data.find(point_index);
  typename multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::point_data_t new_point;
  if (item == point_data.end())
  {
    this->timer->node["body generation"].node["point generation"].start();
    this->get_point_coordinates(point_index, this->new_point_coords);
    this->supporting_point_evaluator->evaluate(this->new_point_coords, this->new_operator_values);
    // check operator values
    for (int op = 0; op < N_OPS; op++)
    {
      new_point[op] = this->new_operator_values[op];
      if (isnan(this->new_operator_values[op]))
      {
        printf("OBL generation warning: nan operator detected! Operator %d for point (", op);
        for (int a = 0; a < N_DIMS; a++)
        {
          printf("%lf, ", this->new_point_coords[a]);
        }
        printf(") is %lf\n", this->new_operator_values[op]);
      }
    }
    point_data[point_index] = new_point;
    this->n_points_used++;
    this->timer->node["body generation"].node["point generation"].stop();
    return point_data[point_index];
  }
  else
    return item->second;
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
const typename multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::hypercube_data_t &
multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::get_hypercube_data(const index_t hypercube_index)
{
  auto item = hypercube_data.find(hypercube_index);
  if (item == hypercube_data.end())
  {
    this->timer->node["body generation"].start();
    hypercube_points_index_t points;
    typename multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::hypercube_data_t new_hypercube;

    this->get_hypercube_points(hypercube_index, points);

    for (int i = 0; i < this->N_VERTS; ++i)
    {
      // obtain point data and copy it to hypercube data
      const typename multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::point_data_t &p_data = this->get_point_data(points[i]);
      for (int op = 0; op < N_OPS; op++)
      {
        new_hypercube[i * N_OPS + op] = p_data[op];
      }
    }
    hypercube_data[hypercube_index] = new_hypercube;
    this->timer->node["body generation"].stop();
    return hypercube_data[hypercube_index];
  }
  else
    return item->second;
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
int multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::interpolate_with_derivatives(const std::vector<double> &points, const std::vector<int> &points_idxs,
                                                                                                         std::vector<double> &values, std::vector<double> &derivatives)
{
// First, all missing points and hypercubes need to be generated in a single thread mode
// this guarantees correct data insertion into point_data and hypercube_data,
// and also allows for not thread-safe operator generation
#pragma omp single
  for (index_t p = 0; p < points_idxs.size(); p++)
  {
    index_t offset = points_idxs[p];
    index_t hypercube_idx = 0;

    for (int i = 0; i < N_DIMS; ++i)
    {
      int axis_idx = get_axis_interval_index<value_t>(points[offset * N_DIMS + i],
                                                      this->axes_min_internal[i], this->axes_max_internal[i],
                                                      this->axes_step_inv_internal[i], this->axes_points[i]);
      hypercube_idx += axis_idx * this->axis_hypercube_mult[i];
    }
    const hypercube_data_t &hypercube = this->get_hypercube_data(hypercube_idx);
  }

  // Now data storages are filled with required data, and parallel interpolation can be launched
  multilinear_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::interpolate_with_derivatives(points, points_idxs, values, derivatives);

  return 0;
}