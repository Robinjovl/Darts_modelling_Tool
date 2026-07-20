#include <cmath>
#include "linear_adaptive_cpu_interpolator.hpp"

template <int N_DIMS, int N_OPS>
linear_adaptive_cpu_interpolator<N_DIMS, N_OPS>::linear_adaptive_cpu_interpolator(
    operator_set_evaluator_iface *supporting_point_evaluator_,
    const std::vector<double> &axes_origin_,
    const std::vector<double> &axes_step_,
    bool _use_barycentric_interpolation)
    : linear_cpu_interpolator_base<N_DIMS, N_OPS>(supporting_point_evaluator_, axes_origin_, axes_step_, _use_barycentric_interpolation)
{
    // Enable signed-floor axis indexing in find_hypercube / get_point_from_vertex so the
    // adaptive cache can grow freely (the grid is unbounded — origin + step only).
    this->use_unbounded_axis_index = true;
}

// ─── multi-index key utilities ──────────────────────────────────────────────────

template <int N_DIMS, int N_OPS>
typename linear_adaptive_cpu_interpolator<N_DIMS, N_OPS>::key_t
linear_adaptive_cpu_interpolator<N_DIMS, N_OPS>::key_from_vertex(const vertex_t &vertex) const
{
    key_t k;
    // vertex_t and cell_key_t share the int32 element type — plain copy, no decoding.
    for (int i = 0; i < N_DIMS; ++i)
        k.idx[i] = vertex[i];
    return k;
}

// ─── adaptive supporting-point lookup ──────────────────────────────────────────

template <int N_DIMS, int N_OPS>
void linear_adaptive_cpu_interpolator<N_DIMS, N_OPS>::get_supporting_point(
    const vertex_t &vertex, std::array<double, N_OPS> &values)
{
    const key_t k = this->key_from_vertex(vertex);
    auto search = point_data.find(k);
    if (search != point_data.end())
    {
        for (int j = 0; j < N_OPS; j++)
            values[j] = search->second[j];
        return;
    }

    if (this->timer) this->timer->node["point generation"].start();
    this->get_point_from_vertex(vertex, this->new_point_coords);
    this->supporting_point_evaluator->evaluate(this->new_point_coords, this->new_operator_values);
    auto &slot = point_data[k];
    for (int j = 0; j < N_OPS; j++)
    {
        slot[j] = this->new_operator_values[j];
        values[j] = this->new_operator_values[j];
        if (std::isnan(this->new_operator_values[j]))
        {
            printf("OBL generation warning: nan operator detected! Operator %d for point (", j);
            for (int a = 0; a < N_DIMS; a++)
            {
                printf("%lf, ", this->new_point_coords[a]);
            }
            printf(") is %lf\n", this->new_operator_values[j]);
        }
    }
    if (this->timer) this->timer->node["point generation"].stop();
    this->n_points_used++;
    // Mark for append-only cache flush after this new point is materialized.
    dirty_point_data.insert(k);
    // Stamp the point with the current batch-interpolation (nonlinear-iteration) index.
    dirty_point_epochs[k] = eval_index;
}

// ─── batch materialization ─────────────────────────────────────────────────────

template <int N_DIMS, int N_OPS>
void linear_adaptive_cpu_interpolator<N_DIMS, N_OPS>::materialize_missing_points(
    const std::vector<double> &points, const std::vector<int> &points_idxs)
{
    std::unordered_set<key_t, key_hash_t> missing_set;
    std::vector<key_t> missing_keys;
    std::vector<double> batch_coords;

    for (std::size_t point_i = 0; point_i < points_idxs.size(); point_i++)
    {
        int point_offset = points_idxs[point_i];
        vertex_t hypercube;
        std::array<double, N_DIMS> scaled_point;
        this->find_hypercube(points, hypercube, scaled_point, point_offset * N_DIMS);

        std::array<vertex_t, N_DIMS + 1> simplex;

        if (this->use_barycentric_interpolation)
        {
            py::gil_scoped_acquire acquire;
            constexpr int n_size = N_DIMS + 1;
            auto numpy_point = py::array_t<double>(N_DIMS, scaled_point.data());
            int simplex_id = this->tri_info.tri.attr("find_simplex")(numpy_point).template cast<int>();

            double* vertices = static_cast<double*>(
                this->tri_info.tri.attr("points").template cast<py::array_t<double>>().request().ptr);
            int* simplices = &(static_cast<int*>(
                this->tri_info.tri.attr("simplices").template cast<py::array_t<int>>().request().ptr))[n_size * simplex_id];

            for (int vertex_i = 0; vertex_i <= N_DIMS; vertex_i++)
            {
                double* vertex = &vertices[simplices[vertex_i] * N_DIMS];
                for (int dim_i = 0; dim_i < N_DIMS; dim_i++)
                    simplex[vertex_i][dim_i] = hypercube[dim_i] + static_cast<int32_t>(vertex[dim_i]);
            }
        }
        else
        {
            std::array<int, N_DIMS> tri_order;
            this->find_simplex(hypercube, scaled_point, tri_order, simplex);
        }

        // Check each simplex vertex for cache miss
        for (int v = 0; v <= N_DIMS; v++)
        {
            const key_t k = this->key_from_vertex(simplex[v]);
            if (point_data.find(k) == point_data.end() && missing_set.find(k) == missing_set.end())
            {
                missing_set.insert(k);
                missing_keys.push_back(k);
                // Compute physical coordinates for this vertex
                this->get_point_from_vertex(simplex[v], this->new_point_coords);
                batch_coords.insert(batch_coords.end(),
                                    this->new_point_coords.begin(),
                                    this->new_point_coords.end());
            }
        }
    }

    if (missing_keys.empty()) return;

    if (this->timer) this->timer->node["point generation"].start();

    const size_t n_missing = missing_keys.size();
    std::vector<double> batch_values(n_missing * N_OPS);
    this->supporting_point_evaluator->evaluate_batch(
        batch_coords, static_cast<int>(n_missing), batch_values, N_OPS);

    point_data.reserve(point_data.size() + n_missing);
    for (size_t i = 0; i < n_missing; i++)
    {
        auto &slot = point_data[missing_keys[i]];
        for (int op = 0; op < N_OPS; op++)
        {
            double val = batch_values[i * N_OPS + op];
            slot[op] = val;
            if (std::isnan(val))
            {
                printf("OBL generation warning: nan operator detected! Operator %d for point (", op);
                for (int a = 0; a < N_DIMS; a++)
                    printf("%lf, ", batch_coords[i * N_DIMS + a]);
                printf(") is %lf\n", val);
            }
        }
        this->n_points_used++;
        // Mark for append-only cache flush after this new point is materialized.
        dirty_point_data.insert(missing_keys[i]);
        // Stamp the point with the current batch-interpolation (nonlinear-iteration) index.
        dirty_point_epochs[missing_keys[i]] = eval_index;
    }

    if (this->timer) this->timer->node["point generation"].stop();
}

template <int N_DIMS, int N_OPS>
int linear_adaptive_cpu_interpolator<N_DIMS, N_OPS>::interpolate_with_derivatives(
    const std::vector<double> &points, const std::vector<int> &points_idxs,
    std::vector<double> &values, std::vector<double> &derivatives)
{
    // One batch interpolation call == one nonlinear-iteration assembly of this operator
    // set: advance the epoch stamp applied to points materialized during this call.
    eval_index++;

    // Pre-fetch all missing supporting points via batch evaluation.
    // After this, all get_supporting_point() calls in the base class will be cache hits.
    materialize_missing_points(points, points_idxs);

    // Delegate to base class for actual interpolation
    return linear_cpu_interpolator_base<N_DIMS, N_OPS>::interpolate_with_derivatives(
        points, points_idxs, values, derivatives);
}
