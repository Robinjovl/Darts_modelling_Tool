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
        if (this->timer) this->timer->node["point generation"].start();
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
        if (this->timer) this->timer->node["point generation"].stop();
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

template <typename index_t, int N_DIMS, int N_OPS>
void linear_adaptive_cpu_interpolator<index_t, N_DIMS, N_OPS>::materialize_missing_points(
    const std::vector<double> &points, const std::vector<int> &points_idxs)
{
    // Phase 1: Scan all cells, find their simplex vertices, collect unique missing point indices.
    std::unordered_set<index_t> missing_set;
    std::vector<index_t> missing_indices;
    std::vector<double> batch_coords;

    for (std::size_t point_i = 0; point_i < points_idxs.size(); point_i++)
    {
        int point_offset = points_idxs[point_i];
        std::array<index_t, N_DIMS> hypercube;
        std::array<double, N_DIMS> scaled_point;
        this->find_hypercube(points, hypercube, scaled_point, point_offset * N_DIMS);

        std::array<std::array<index_t, N_DIMS>, N_DIMS + 1> simplex;

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
                    simplex[vertex_i][dim_i] = hypercube[dim_i] + static_cast<index_t>(vertex[dim_i]);
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
            index_t idx = this->get_index_from_vertex(simplex[v]);
            if (point_data.find(idx) == point_data.end() && missing_set.find(idx) == missing_set.end())
            {
                missing_set.insert(idx);
                missing_indices.push_back(idx);
                // Compute and store coordinates for this vertex
                this->get_point_from_vertex(simplex[v], this->new_point_coords);
                batch_coords.insert(batch_coords.end(),
                                    this->new_point_coords.begin(),
                                    this->new_point_coords.end());
            }
        }
    }

    if (missing_indices.empty()) return;

    // Phase 2: Batch-evaluate all missing points through a single evaluate_batch() call.
    //          The evaluator's evaluate_batch() may dispatch to a multiprocessing pool
    //          (ParallelEvaluator) or fall back to serial per-point evaluate() (default).
    if (this->timer) this->timer->node["point generation"].start();

    const size_t n_missing = missing_indices.size();
    std::vector<double> batch_values(n_missing * N_OPS);
    this->supporting_point_evaluator->evaluate_batch(
        batch_coords, static_cast<int>(n_missing), batch_values, N_OPS);

    // Unpack results into point_data cache
    point_data.reserve(point_data.size() + n_missing);
    for (size_t i = 0; i < n_missing; i++)
    {
        const index_t idx = missing_indices[i];
        for (int op = 0; op < N_OPS; op++)
        {
            double val = batch_values[i * N_OPS + op];
            point_data[idx][op] = val;
            if (isnan(val))
            {
                printf("OBL generation warning: nan operator detected! Operator %d for point (", op);
                for (int a = 0; a < N_DIMS; a++)
                    printf("%lf, ", batch_coords[i * N_DIMS + a]);
                printf(") is %lf\n", val);
            }
        }
        this->n_points_used++;
    }

    if (this->timer) this->timer->node["point generation"].stop();
}

template <typename index_t, int N_DIMS, int N_OPS>
int linear_adaptive_cpu_interpolator<index_t, N_DIMS, N_OPS>::interpolate_with_derivatives(
    const std::vector<double> &points, const std::vector<int> &points_idxs,
    std::vector<double> &values, std::vector<double> &derivatives)
{
    // Pre-fetch all missing supporting points via batch evaluation.
    // After this, all get_supporting_point() calls in the base class will be cache hits.
    materialize_missing_points(points, points_idxs);

    // Delegate to base class for actual interpolation
    return linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::interpolate_with_derivatives(
        points, points_idxs, values, derivatives);
}
