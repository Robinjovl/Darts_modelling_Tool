#include "linear_static_cpu_interpolator.hpp"

template <typename index_t, int N_DIMS, int N_OPS>
linear_static_cpu_interpolator<index_t, N_DIMS, N_OPS>::linear_static_cpu_interpolator(
    operator_set_evaluator_iface *supporting_point_evaluator, 
    const std::vector<int> &axes_points,
    const std::vector<double> &axes_min, 
    const std::vector<double> &axes_max, 
    bool _use_barycentric_interpolation)
    : linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>(supporting_point_evaluator, axes_points, axes_min, axes_max, _use_barycentric_interpolation)
{
    this->n_points_used = this->n_points_total;
}

template <typename index_t, int N_DIMS, int N_OPS>
int linear_static_cpu_interpolator<index_t, N_DIMS, N_OPS>::init()
{
    // initialize base class first
    interpolator_base::init();

    // now evaluate points unless they were already assigned via Python
    if (point_data.size() == 0)
    {
        index_t n_points = this->axes_mult[0] * this->axes_points[0];
        cout << "Computing " << n_points << " supporting points for static storage..." << std::endl;
        point_data.resize(n_points * N_OPS);

        for (index_t point_i = 0; point_i < n_points; point_i++)
        {
            std::array<int, N_DIMS> vertex;
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

template <typename index_t, int N_DIMS, int N_OPS>
void linear_static_cpu_interpolator<index_t, N_DIMS, N_OPS>::get_vertex_from_index(index_t index,
                                                                                   std::array<int, N_DIMS> &vertex)
{
    for (int dim = 0; dim < N_DIMS; dim++)
    {
        vertex[dim] = index / this->axes_mult[dim];
        index %= this->axes_mult[dim];
    }
}

template <typename index_t, int N_DIMS, int N_OPS>
void linear_static_cpu_interpolator<index_t, N_DIMS, N_OPS>::get_supporting_point(const std::array<int, N_DIMS> &vertex, std::array<double, N_OPS> &values)
{
    index_t index = this->get_index_from_vertex(vertex);
    for (int op = 0; op < N_OPS; op++)
    {
        values[op] = point_data[index * N_OPS + op];
    }
}