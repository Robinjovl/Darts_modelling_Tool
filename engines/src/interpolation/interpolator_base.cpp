#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <assert.h>
#include "interpolator_base.hpp"

interpolator_base::interpolator_base(operator_set_evaluator_iface *supporting_point_evaluator_,
                                     const std::vector<int> &axes_points,
                                     const std::vector<double> &axes_min, const std::vector<double> &axes_max,
                                     const std::vector<std::vector<double>> *axis_nodes)
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

    axis_nodes_offset.resize(n_dims + 1, 0);
    axis_cells_offset.resize(n_dims + 1, 0);
    axis_bin_offset.resize(n_dims + 1, 0);
    axis_bin_count.resize(n_dims, 0);
    axis_bin_inv_width.resize(n_dims, 0.0);

    // Each axis is stored consecutively inside the flattened buffers.  Pre-compute offsets
    // so we can slice axis_nodes_flat/axis_inv_dx_flat without extra multiplications inside hot loops.
    for (int dim = 0; dim < n_dims; ++dim)
    {
        axis_nodes_offset[dim + 1] = axis_nodes_offset[dim] + static_cast<size_t>(axes_points[dim]);
        axis_cells_offset[dim + 1] = axis_cells_offset[dim] + static_cast<size_t>(std::max(axes_points[dim] - 1, 0));
    }

    axis_nodes_flat.resize(axis_nodes_offset.back());
    axis_inv_dx_flat.resize(axis_cells_offset.back());

    const bool has_custom_nodes = axis_nodes != nullptr && axis_nodes->size() == static_cast<size_t>(n_dims);
    nonuniform_axes_enabled = has_custom_nodes;
    if (axis_nodes != nullptr && !has_custom_nodes)
    {
        throw std::invalid_argument("axis_nodes must either be null or contain entries for all axes");
    }

    for (int dim = 0; dim < n_dims; ++dim)
    {
        const int n_points_dim = axes_points[dim];
        double *axis_nodes_ptr = axis_nodes_flat.data() + axis_nodes_offset[dim];
        double *axis_inv_dx_ptr = axis_inv_dx_flat.data() + axis_cells_offset[dim];

        if (has_custom_nodes)
        {
            const auto &nodes = axis_nodes->at(dim);
            if (static_cast<int>(nodes.size()) != n_points_dim)
            {
                throw std::invalid_argument("axis_nodes entry size mismatch for axis " + std::to_string(dim));
            }
            std::copy(nodes.begin(), nodes.end(), axis_nodes_ptr);
        }
        else
        {
            for (int i = 0; i < n_points_dim; ++i)
            {
                axis_nodes_ptr[i] = axes_min[dim] + axes_step[dim] * i;
            }
        }

        // Per-cell inverse spacing (1/dx) is stored explicitly to avoid divisions during interpolation.
        for (int i = 0; i < n_points_dim - 1; ++i)
        {
            double dx = axis_nodes_ptr[i + 1] - axis_nodes_ptr[i];
            if (dx <= 0.0)
            {
                throw std::invalid_argument("axis_nodes must be strictly increasing for each axis");
            }
            axis_inv_dx_ptr[i] = 1.0 / dx;
        }

        if (nonuniform_axes_enabled)
        {
            // Build a compact coarse bin table that quickly guesses the interval which contains a point.
            // The table resolution scales with the number of intervals but is capped to keep memory bounded.
            const int intervals = std::max(n_points_dim - 1, 1);
            const int proposed_bins = std::max(1, std::min(MAX_AXIS_BINS, intervals * 4));
            axis_bin_count[dim] = proposed_bins;
            axis_bin_offset[dim + 1] = axis_bin_offset[dim] + static_cast<size_t>(proposed_bins);

            const double axis_span = axis_nodes_ptr[n_points_dim - 1] - axis_nodes_ptr[0];
            const double bin_width = proposed_bins > 0 ? axis_span / proposed_bins : 0.0;
            axis_bin_inv_width[dim] = (bin_width > 0.0) ? (1.0 / bin_width) : 0.0;
        }
    }

    if (nonuniform_axes_enabled)
    {
        // Materialize all bin tables in a single contiguous vector for cache-friendly access at runtime.
        axis_bin_left_idx_flat.resize(axis_bin_offset.back());
        for (int dim = 0; dim < n_dims; ++dim)
        {
            const int n_points_dim = axes_points[dim];
            if (n_points_dim < 2)
                continue;
            const double *axis_nodes_ptr = axis_nodes_flat.data() + axis_nodes_offset[dim];
            uint32_t *bin_ptr = axis_bin_left_idx_flat.data() + axis_bin_offset[dim];
            const int bins = axis_bin_count[dim];
            if (bins <= 0)
                continue;
            const double min_v = axis_nodes_ptr[0];
            const double max_v = axis_nodes_ptr[n_points_dim - 1];
            const double span = max_v - min_v;
            const double bin_width = (bins > 0) ? span / bins : 0.0;
            int current_idx = 0;
            for (int b = 0; b < bins; ++b)
            {
                double x = min_v + b * bin_width;
                while (current_idx + 1 < n_points_dim && axis_nodes_ptr[current_idx + 1] <= x)
                {
                    ++current_idx;
                }
                if (current_idx > n_points_dim - 2)
                {
                    current_idx = n_points_dim - 2;
                }
                bin_ptr[b] = static_cast<uint32_t>(current_idx);
            }
        }
    }
    else
    {
        axis_bin_left_idx_flat.clear();
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
    if (!states_idxs.empty())
    {
        auto max_idx = *std::max_element(states_idxs.begin(), states_idxs.end());
        assert(states.size() > max_idx * n_dims);
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
