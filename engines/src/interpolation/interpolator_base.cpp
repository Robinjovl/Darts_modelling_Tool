#include <numeric>
#include <algorithm>
#include <assert.h>
#include "interpolator_base.hpp"

interpolator_base::interpolator_base(operator_set_evaluator_iface *supporting_point_evaluator_,
                                     const std::vector<int> &axes_points,
                                     const std::vector<double> &axes_min, const std::vector<double> &axes_max)
    : supporting_point_evaluator(supporting_point_evaluator_), axes_points(axes_points), axes_min(axes_min), axes_max(axes_max)
{
    n_dims = static_cast<int>(axes_points.size());
    assert(axes_min.size() == axes_points.size());
    assert(axes_max.size() == axes_points.size());
    axes_step.resize(n_dims);
    axes_step_inv.resize(n_dims);
    for (int dim = 0; dim < n_dims; dim++)
    {
        axes_step[dim] = (axes_max[dim] - axes_min[dim]) / (axes_points[dim] - 1);
        axes_step_inv[dim] = 1 / axes_step[dim];
    }

    //use double to avoid overflow
    n_points_total_fp = 1;
    n_points_total = 1;
    for (int dim = 0; dim < n_dims; dim++)
      n_points_total *= axes_points[dim];

    n_points_total_fp = static_cast<double>(n_points_total);
    n_points_used = 0;
    n_interpolations = 0;
}

int interpolator_base::init()
{
    // take n_dims from derived interpolator class
    int n_dims_interpolator = this->get_n_dims();
    // take n_dims from parameter space passed during cunstruction
    int n_dims_parameter_space = n_dims;
    // make sure they match
    assert(n_dims_interpolator == n_dims_parameter_space);
    n_ops = this->get_n_ops();

    new_point_coords.resize(n_dims_interpolator);
    new_operator_values.resize(n_ops);

    return 0;
}

int interpolator_base::evaluate(const std::vector<value_t> &state, std::vector<value_t> &values)
{
    timer->start();
    // call implementation of a derived class
    this->interpolate(state, values);
    timer->stop();
    n_interpolations += n_ops;
    return 0;
}

int interpolator_base::evaluate_with_derivatives(const std::vector<double> &states,
                                                 const std::vector<int> &states_idxs,
                                                 std::vector<double> &values,
                                                 std::vector<double> &derivatives)
{
    // check consistency of input arrays
    assert(n_dims * values.size() == derivatives.size());
    assert(states.size() > (*std::max_element(states_idxs.begin(), states_idxs.end())) * n_dims);

    timer->start();
    // call implementation of a derived class
    this->interpolate_with_derivatives(states, states_idxs, values, derivatives);
    timer->stop();
    n_interpolations += states_idxs.size() * n_ops;
    return 0;
}

int interpolator_base::get_axis_n_points(int axis) const
{
    return axes_points[axis];
}

double interpolator_base::get_axis_max(int axis) const
{
    return axes_max[axis];
}

double interpolator_base::get_axis_min(int axis) const
{
    return axes_min[axis];
}

uint64_t interpolator_base::get_n_interpolations() const
{
    return n_interpolations;
}

uint64_t interpolator_base::get_n_points_total() const
{
    return static_cast<uint64_t>(n_points_total);
}

uint64_t interpolator_base::get_n_points_used() const
{
    return static_cast<uint64_t>(n_points_used);
}
