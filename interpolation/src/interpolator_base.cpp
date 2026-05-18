#include <numeric>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <assert.h>
#include "interpolator_base.hpp"

interpolator_base::interpolator_base(operator_set_evaluator_iface *supporting_point_evaluator_,
                                     const std::vector<int> &axes_points_,
                                     const std::vector<double> &axes_min_,
                                     const std::vector<double> &axes_max_)
    : axes_points(axes_points_),
      axes_min(axes_min_),
      axes_max(axes_max_),
      supporting_point_evaluator(supporting_point_evaluator_)
{
    n_dims = static_cast<int>(axes_points_.size());
    assert(axes_min_.size() == axes_points_.size());
    assert(axes_max_.size() == axes_points_.size());
    axes_step.resize(n_dims);
    axes_step_inv.resize(n_dims);
    for (int dim = 0; dim < n_dims; dim++)
    {
        axes_step[dim] = (axes_max_[dim] - axes_min_[dim]) / (axes_points_[dim] - 1);
        axes_step_inv[dim] = 1 / axes_step[dim];
    }

    //use double to avoid overflow
    n_points_total_fp = 1;
    n_points_total = 1;
    for (int dim = 0; dim < n_dims; dim++)
      n_points_total *= axes_points_[dim];

    n_points_total_fp = static_cast<double>(n_points_total);
    n_points_used = 0;
    n_interpolations = 0;
}

int interpolator_base::init()
{
    // take n_dims from derived interpolator class
    int n_dims_interpolator = this->get_n_dims();
    // take n_dims from parameter space passed during cunstruction
    // make sure they match
    assert(n_dims_interpolator == n_dims);
    n_ops = this->get_n_ops();

    new_point_coords.resize(n_dims_interpolator);
    new_operator_values.resize(n_ops);

    return 0;
}

int interpolator_base::evaluate(const std::vector<value_t> &state, std::vector<value_t> &values)
{
    if (state.size() < static_cast<size_t>(n_dims))
    {
        throw std::invalid_argument("Interpolator state buffer is too small for n_dims=" + std::to_string(n_dims));
    }
    if (values.size() < static_cast<size_t>(n_ops))
    {
        throw std::invalid_argument("Interpolator values buffer is too small for n_ops=" + std::to_string(n_ops));
    }

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
    size_t required_states_size = 0;
    size_t required_values_size = 0;
    size_t required_derivatives_size = 0;

    if (!states_idxs.empty())
    {
        int max_state_idx = states_idxs[0];
        for (const int state_idx : states_idxs)
        {
            if (state_idx < 0)
            {
                throw std::invalid_argument("Interpolator state index cannot be negative");
            }
            max_state_idx = std::max(max_state_idx, state_idx);
        }

        const size_t n_required_states = static_cast<size_t>(max_state_idx) + 1;
        required_states_size = n_required_states * static_cast<size_t>(n_dims);
        required_values_size = n_required_states * static_cast<size_t>(n_ops);
        required_derivatives_size = required_values_size * static_cast<size_t>(n_dims);
    }

    if (states.size() < required_states_size)
    {
        throw std::invalid_argument("Interpolator states buffer is too small for n_dims=" + std::to_string(n_dims));
    }
    if (values.size() < required_values_size)
    {
        throw std::invalid_argument("Interpolator values buffer is too small for n_ops=" + std::to_string(n_ops));
    }
    if (derivatives.size() < required_derivatives_size)
    {
        throw std::invalid_argument("Interpolator derivatives buffer is too small for n_ops=" + std::to_string(n_ops));
    }

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
