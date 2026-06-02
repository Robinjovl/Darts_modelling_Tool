#include <numeric>
#include <algorithm>
#include <limits>
#include <assert.h>
#include "interpolator_base.hpp"

namespace
{
// Derive the per-axis upper bound of a finite dense grid from (origin, step, points).
// Only used by the bounded (static) ctor; unbounded grids leave axes_max empty.
std::vector<double> make_axes_max(const std::vector<double> &origin,
                                  const std::vector<double> &step,
                                  const std::vector<int> &points)
{
    std::vector<double> mx(origin.size());
    for (std::size_t d = 0; d < origin.size(); ++d)
        mx[d] = origin[d] + (points[d] - 1) * step[d];
    return mx;
}
} // namespace

interpolator_base::interpolator_base(operator_set_evaluator_iface *supporting_point_evaluator_,
                                     const std::vector<double> &axes_origin_,
                                     const std::vector<double> &axes_step_)
    : axes_origin(axes_origin_),
      axes_step(axes_step_),
      axes_points(),
      axes_max(),
      supporting_point_evaluator(supporting_point_evaluator_)
{
    n_dims = static_cast<int>(axes_origin_.size());
    assert(axes_step_.size() == axes_origin_.size());
    axes_step_inv.resize(n_dims);
    for (int dim = 0; dim < n_dims; dim++)
        axes_step_inv[dim] = 1.0 / axes_step_[dim];

    // Unbounded grid: there is no finite supporting-point count. n_points_total is a
    // diagnostic only (the "% generated" OBL stat); leave it 0 so callers can detect
    // the unbounded case. Adaptive storage keys on cell_key_t and never forms a flat
    // mixed-radix product, so the old 1024^n_dims overflow cannot occur here.
    n_points_total_fp = 0.0;
    n_points_total = 0;
    n_points_used = 0;
    n_interpolations = 0;
}

interpolator_base::interpolator_base(operator_set_evaluator_iface *supporting_point_evaluator_,
                                     const std::vector<double> &axes_origin_,
                                     const std::vector<double> &axes_step_,
                                     const std::vector<int> &axes_points_)
    : axes_origin(axes_origin_),
      axes_step(axes_step_),
      axes_points(axes_points_),
      axes_max(make_axes_max(axes_origin_, axes_step_, axes_points_)),
      supporting_point_evaluator(supporting_point_evaluator_)
{
    n_dims = static_cast<int>(axes_origin_.size());
    assert(axes_step_.size() == axes_origin_.size());
    assert(axes_points_.size() == axes_origin_.size());
    axes_step_inv.resize(n_dims);
    for (int dim = 0; dim < n_dims; dim++)
        axes_step_inv[dim] = 1.0 / axes_step_[dim];

    // Bounded dense grid: n_points_total is the real product (overflow-safe in fp,
    // saturated in uint64_t). The static init() validates it against index_t range
    // before allocating dense storage.
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
    // Unbounded grids have no finite point count; report 0 (the OBL-stats print
    // guards against this). Bounded grids return the real per-axis count.
    return axes_points.empty() ? 0 : axes_points[axis];
}

double interpolator_base::get_axis_max(int axis) const
{
    // Unbounded grids have no upper bound; report +inf so any engine-side axis clamp
    // (e.g. the engine_super_cpu temperature clamp) becomes a no-op on the upper side
    // — consistent with the adaptive cache being free to extrapolate past the window.
    return axes_max.empty() ? std::numeric_limits<double>::max() : axes_max[axis];
}

double interpolator_base::get_axis_min(int axis) const
{
    return axes_origin[axis];
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
