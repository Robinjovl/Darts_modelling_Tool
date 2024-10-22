#include "linear_adaptive_cpu_interpolator.hpp"

template <typename index_t, int N_DIMS, int N_OPS>
linear_adaptive_cpu_interpolator<index_t, N_DIMS, N_OPS>::linear_adaptive_cpu_interpolator(
    operator_set_evaluator_iface *supporting_point_evaluator, 
    const std::vector<int> &axesPoints,
    const std::vector<double> &axesMin, 
    const std::vector<double> &axesMax, 
    bool _use_barycentric_interpolation)
    : linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>(supporting_point_evaluator, axesPoints, axesMin, axesMax, _use_barycentric_interpolation)
{
}

template <typename index_t, int N_DIMS, int N_OPS>
void linear_adaptive_cpu_interpolator<index_t, N_DIMS, N_OPS>::get_supporting_point(const std::array<index_t, N_DIMS> &vertex, std::array<double, N_OPS> &values)
{
    index_t index = this->get_index_from_vertex(vertex);
    auto search = point_data.find(index);
    if (search == point_data.end()) ///< std::unordered_map<...>::contains is supported since C++20
    {
        this->timer->node["point generation"].start();
        this->get_point_from_vertex(vertex, this->new_point_coords);
        this->supporting_point_evaluator->evaluate(this->new_point_coords, this->new_operator_values);
        for (int j = 0; j < N_OPS; j++)
        {
            point_data[index][j] = this->new_operator_values[j];
            values[j] = this->new_operator_values[j];
            if (isnan(this->new_operator_values[j]))
            {
                printf("OBL generation warning: nan operator detected! Operator %d for point (", j);
                for (int a = 0; a < N_DIMS; a++)
                {
                    printf("%lf, ", this->new_point_coords[a]);
                }
                printf(") is %lf\n", this->new_operator_values[j]);
            }
        }
        this->timer->node["point generation"].stop();
        this->n_points_used++;
    }
    else
    {
        for (int j = 0; j < N_OPS; j++)
        {
            values[j] = search->second[j];
        }
    }
}
