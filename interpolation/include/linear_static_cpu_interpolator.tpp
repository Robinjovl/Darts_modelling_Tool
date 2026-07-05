#include <limits>
#include <stdexcept>
#include <string>

#include "linear_static_cpu_interpolator.hpp"

template <int N_DIMS, int N_OPS>
linear_static_cpu_interpolator<N_DIMS, N_OPS>::linear_static_cpu_interpolator(
    operator_set_evaluator_iface *supporting_point_evaluator,
    const std::vector<double> &axes_origin,
    const std::vector<double> &axes_step,
    const std::vector<int> &axes_points,
    bool _use_barycentric_interpolation)
    : linear_cpu_interpolator_base<N_DIMS, N_OPS>(supporting_point_evaluator, axes_origin, axes_step, axes_points, _use_barycentric_interpolation)
{
    this->n_points_used = this->n_points_total;
}

template <int N_DIMS, int N_OPS>
int linear_static_cpu_interpolator<N_DIMS, N_OPS>::init()
{
    // initialize base class first
    interpolator_base::init();

    // Static interpolator requires a bounded grid (dense vector storage).
    {
        double int_type_max = static_cast<double>(std::numeric_limits<uint64_t>::max());
        if (this->n_points_total_fp > int_type_max)
        {
            throw std::range_error(
                "static linear interpolator requires a bounded grid; n_points_total (" +
                std::to_string(this->n_points_total_fp) +
                ") exceeds the dense-index (uint64) range (" + std::to_string(int_type_max) +
                "). Use the adaptive variant for unbounded grids.");
        }
    }

    // now evaluate points unless they were already assigned via Python
    if (point_data.size() == 0)
    {
        uint64_t n_points = this->axes_mult[0] * this->axes_points[0];
        std::cout << "Computing " << n_points << " supporting points for static storage..." << std::endl;
        point_data.resize(n_points * N_OPS);

        for (uint64_t point_i = 0; point_i < n_points; point_i++)
        {
            vertex_t vertex;
            get_vertex_from_index(point_i, vertex);

            this->get_point_from_vertex(vertex, this->new_point_coords);

            this->supporting_point_evaluator->evaluate(this->new_point_coords, this->new_operator_values);
            for (int i = 0; i < N_OPS; i++)
            {
                point_data[N_OPS * point_i + i] = this->new_operator_values[i];
            }
        }
    }
    return 0;
}

template <int N_DIMS, int N_OPS>
void linear_static_cpu_interpolator<N_DIMS, N_OPS>::get_vertex_from_index(uint64_t index,
                                                                          vertex_t &vertex)
{
    for (int dim = 0; dim < N_DIMS; dim++)
    {
        vertex[dim] = static_cast<int32_t>(index / this->axes_mult[dim]);
        index %= this->axes_mult[dim];
    }
}

template <int N_DIMS, int N_OPS>
void linear_static_cpu_interpolator<N_DIMS, N_OPS>::get_supporting_point(const vertex_t &vertex, std::array<double, N_OPS> &values)
{
    uint64_t index = this->get_index_from_vertex(vertex);
    for (int op = 0; op < N_OPS; op++)
    {
        values[op] = point_data[index * N_OPS + op];
    }
}
