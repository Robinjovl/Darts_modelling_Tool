#include <numeric>
#include <algorithm>
#include <limits>
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

    // Compute n_points_total_fp (double, overflow-safe) first as the authoritative
    // diagnostic. n_points_total mirrors it in uint64_t for legacy callers, with
    // saturation at uint64::max for huge grids — adaptive code paths don't depend
    // on n_points_total being exact (cell_key_t is the identity).
    n_points_total_fp = 1.0;
    for (int dim = 0; dim < n_dims; dim++)
      n_points_total_fp *= static_cast<double>(axes_points_[dim]);

    if (n_points_total_fp > static_cast<double>(std::numeric_limits<uint64_t>::max()))
      n_points_total = std::numeric_limits<uint64_t>::max();
    else
      n_points_total = static_cast<uint64_t>(n_points_total_fp);

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
    if (timer) timer->start();
    // call implementation of a derived class
    this->interpolate(state, values);
    if (timer) timer->stop();
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
    if (!states_idxs.empty())
    {
        assert(states.size() > static_cast<size_t>(*std::max_element(states_idxs.begin(), states_idxs.end()) * n_dims));
    }

    if (timer) timer->start();
    // call implementation of a derived class
    this->interpolate_with_derivatives(states, states_idxs, values, derivatives);
    if (timer) timer->stop();
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
    return n_points_total;
}

uint64_t interpolator_base::get_n_points_used() const
{
    return n_points_used;
}
