//*************************************************************************
//    Copyright (c) 2021
//            Bowen Zhu         B.Zhu-3@student.tudelft.nl
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

#include <numeric>
#include <algorithm>
#include "linear_cpu_interpolator_base.hpp"

template <typename index_t, int N_DIMS, int N_OPS>
linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::linear_cpu_interpolator_base(operator_set_evaluator_iface *supporting_point_evaluator,
                                                                                   const std::vector<int> &axes_points,
                                                                                   const std::vector<double> &axes_min, const std::vector<double> &axes_max)
    : interpolator_base(supporting_point_evaluator, axes_points, axes_min, axes_max)
{

    axes_mult[N_DIMS - 1] = 1;
    for (int dim{N_DIMS - 2}; dim >= 0; dim--)
        axes_mult[dim] = axes_mult[dim + 1] * axes_points[dim + 1];

    // initialize the values with 0
    standard_simplex = {};
    // and then set some to 1
    for (int vertex_i = 0; vertex_i < N_DIMS; vertex_i++)
        for (int dim_i = vertex_i; dim_i < N_DIMS; dim_i++)
            standard_simplex[vertex_i][dim_i] = 1;

    if (n_points_total_fp > std::numeric_limits<index_t>::max())
    {
        std::string error = "Error: The total requested amount of points (" + std::to_string(n_points_total_fp) +
                            ") exceeds the limit in index type (" + std::to_string(std::numeric_limits<index_t>::max()) + ")\n";
        throw std::range_error(error);
    }

    transform_last_axis = 1;
}

template <typename index_t, int N_DIMS, int N_OPS>
void linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::find_hypercube(const std::vector<double> &points,
                                                                          std::array<int, N_DIMS> &hypercube,
                                                                          std::array<double, N_DIMS> &scaled_point,
                                                                          const int point_index)
{

    for (int i = 0; i < N_DIMS; i++)
    {
        double point = points[point_index + i];
        if (transform_last_axis && i == (N_DIMS - 1))
        {
            point = axes_max[i] - (point - axes_min[i]);
        }
        scaled_point[i] = (point - axes_min[i]) * axes_step_inv[i];
        hypercube[i] = (int)scaled_point[i];
        scaled_point[i] -= hypercube[i];
    }
}

template <typename index_t, int N_DIMS, int N_OPS>
void linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::find_simplex(const std::array<int, N_DIMS> &hypercube,
                                                                        const std::array<double, N_DIMS> &scaled_point,
                                                                        std::array<int, N_DIMS> &tri_order,
                                                                        std::array<std::array<int, N_DIMS>, N_DIMS + 1> &simplex)
{
    std::iota(tri_order.begin(), tri_order.end(), 0);
    std::sort(tri_order.begin(), tri_order.end(),
              [&scaled_point](int i1, int i2) { return scaled_point[i1] < scaled_point[i2]; });

    for (int vertex_i = 0; vertex_i <= N_DIMS; vertex_i++)
    {
        for (int dim_i = 0; dim_i < N_DIMS; dim_i++)
            simplex[vertex_i][dim_i] = hypercube[dim_i];
        for (int dim_i = 0; dim_i < N_DIMS; dim_i++)
            simplex[vertex_i][tri_order[dim_i]] += standard_simplex[vertex_i][dim_i];
    }
}

template <typename index_t, int N_DIMS, int N_OPS>
int linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::interpolate(const std::vector<value_t> &point, std::vector<value_t> &values)
{
    std::array<int, N_DIMS> hypercube;
    std::array<double, N_DIMS> scaled_point;
    find_hypercube(point, hypercube, scaled_point);
    std::array<std::array<int, N_DIMS>, N_DIMS + 1> simplex;
    std::array<int, N_DIMS> tri_order;
    find_simplex(hypercube, scaled_point, tri_order, simplex);
    std::array<double, N_DIMS + 1> weights;
    weights[0] = scaled_point[tri_order[0]];
    weights[N_DIMS] = 1 - scaled_point[tri_order[N_DIMS - 1]];

    for (int dim_i = 1; dim_i < N_DIMS; dim_i++)
        weights[dim_i] = scaled_point[tri_order[dim_i]] - scaled_point[tri_order[dim_i - 1]];
    values.assign(N_OPS, 0);
    for (int dim_i = 0; dim_i <= N_DIMS; dim_i++)
    {
        std::array<double, N_OPS> supp_values;
        get_supporting_point(simplex[dim_i], supp_values);
        for (int op = 0; op < N_OPS; op++)
        {
            values[op] += weights[dim_i] * supp_values[op];
        }
    }
    return 0;
}

template <typename index_t, int N_DIMS, int N_OPS>
int linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::interpolate_with_derivatives(const std::vector<double> &points,
                                                                                       const std::vector<int> &points_idxs,
                                                                                       std::vector<double> &values,
                                                                                       std::vector<double> &derivatives)
{
    for (std::size_t point_i = 0; point_i < points_idxs.size(); point_i++)
    {
        int point_offset = points_idxs[point_i];

        std::array<int, N_DIMS> hypercube;
        std::array<double, N_DIMS> scaled_point;
        find_hypercube(points, hypercube, scaled_point, point_offset * N_DIMS);

        std::array<std::array<int, N_DIMS>, N_DIMS + 1> simplex;
        std::array<int, N_DIMS> tri_order;
        find_simplex(hypercube, scaled_point, tri_order, simplex);

        std::array<std::array<double, N_OPS>, N_DIMS + 1> simplex_vertex_values;
        for (int i = 0; i <= N_DIMS; i++)
            get_supporting_point(simplex[i], simplex_vertex_values[i]);

        for (int i = 0; i < N_OPS; i++)
        {
            values[point_offset * N_OPS + i] = simplex_vertex_values[0][i];
        }
        double simplex_vertex_values_gap;
        for (int dim_i = 0; dim_i < N_DIMS; dim_i++)
        {
            for (int op_i = 0; op_i < N_OPS; op_i++)
            {
                simplex_vertex_values_gap = simplex_vertex_values[dim_i + 1][op_i] - simplex_vertex_values[dim_i][op_i];
                values[point_offset * N_OPS + op_i] += (1 - scaled_point[tri_order[dim_i]]) * simplex_vertex_values_gap;
                if (transform_last_axis && tri_order[dim_i] == (N_DIMS - 1))
                    simplex_vertex_values_gap *= -1;
                derivatives[(point_offset * N_OPS + op_i) * N_DIMS + tri_order[dim_i]] = -simplex_vertex_values_gap * axes_step_inv[tri_order[dim_i]];
            }
        }
    }
    return 0;
}

template <typename index_t, int N_DIMS, int N_OPS>
index_t linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::get_index_from_vertex(const std::array<int, N_DIMS> &vertex)
{
    index_t index = 0;
    for (int dim_i = 0; dim_i < N_DIMS; dim_i++)
        index += vertex[dim_i] * this->axes_mult[dim_i];
    return index;
}

template <typename index_t, int N_DIMS, int N_OPS>
void linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::get_point_from_vertex(const std::array<int, N_DIMS> &vertex,
                                                                                 std::vector<double> &point)
{
    for (int i = 0; i < N_DIMS; i++)
        point[i] = vertex[i] * axes_step[i] + axes_min[i];
    if (transform_last_axis)
        point[N_DIMS - 1] = axes_max[N_DIMS - 1] - (point[N_DIMS - 1] - axes_min[N_DIMS - 1]);
}
