#ifndef B2B2062C_7916_498E_8F40_7EFBD5E778F1
#define B2B2062C_7916_498E_8F40_7EFBD5E778F1
#include <unordered_map>
#include <unordered_set>
#include "multi_index_key.hpp"
#include "linear_cpu_interpolator_base.hpp"
#include "point_data_store.hpp"

/**
 * Adaptive piecewise linear interpolator.
 *
 * Storage is keyed on a signed multi-index (cell_key_t<N_DIMS>) so the adaptive cache
 * no longer depends on axes_min/axes_max to bound the index space — out-of-window
 * supporting points are evaluated and cached on demand. Python cache I/O is exposed
 * via the tuple-keyed `point_data_full` property (signed multi-index → operator tuple);
 * the legacy integer-keyed `point_data` shim was retired alongside the multi-index
 * migration since the unbounded grid has no integer-key packing.
 *
 * @tparam index_t legacy packed-integer index type (used by pybind/cache compat layer)
 * @tparam N_DIMS  number of dimensions in parameter space
 * @tparam N_OPS   number of operators to be interpolated
 */
template <typename index_t, int N_DIMS, int N_OPS>
class linear_adaptive_cpu_interpolator : public linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>
{
public:
    typedef cell_key_t<N_DIMS> key_t;
    typedef cell_key_hash<N_DIMS> key_hash_t;

    linear_adaptive_cpu_interpolator(operator_set_evaluator_iface *base_points_generator,
                                     const std::vector<double> &axes_origin,
                                     const std::vector<double> &axes_step,
                                     bool _use_barycentric_interpolation);

    /**
     * Adaptive supporting-point storage, keyed on signed multi-index (cell_key_t).
     */
    // Phase F (FC03): hybrid mmap'd-arena + in-RAM overlay store (drop-in).
    point_data_store<N_DIMS, N_OPS, double, key_hash_t> point_data;
    /**
     * Multi-index keys of supporting points materialized since the last external cache
     * flush. Mirrors the development-branch dirty-set tracker but in the new key space,
     * so the OBL cache writer can persist only the newly evaluated points.
     */
    std::unordered_set<key_t, key_hash_t> dirty_point_data;

    /**
     * @brief Evaluation epoch of each dirty (unflushed) supporting point: maps the
     * point multi-index key to the batch-interpolation index (≈ time step /
     * nonlinear iteration) at which it was first materialized. Persisted next to the
     * append-only delta so the OBL cache records when each point entered the sampling.
     * Cleared together with dirty_point_data on every external cache flush.
     */
    std::unordered_map<key_t, uint64_t, key_hash_t> dirty_point_epochs;

    /**
     * @brief Number of batch interpolation calls processed so far. Incremented once
     * at the start of every interpolate_with_derivatives() (one nonlinear-iteration
     * assembly of this operator set) and used to stamp newly materialized points.
     */
    uint64_t eval_index = 0;

    /**
     * @brief Interpolate with batch pre-fetching of missing supporting points.
     */
    int interpolate_with_derivatives(const std::vector<double> &points, const std::vector<int> &points_idxs,
                                     std::vector<double> &values, std::vector<double> &derivatives) override;

    /**
     * @brief Build the multi-index key from a vertex (with int32 bit pattern packed in index_t).
     */
    key_t key_from_vertex(const std::array<index_t, N_DIMS> &vertex) const;

    size_t get_n_cached_points() const { return point_data.size(); }

private:
    /**
     * @brief Get values of operators at a supporting point identified by vertex array.
     *
     * Decodes the vertex into a multi-index, then looks up the cell_key map. On miss,
     * evaluates through the supporting_point_evaluator and stores.
     */
    void get_supporting_point(const std::array<index_t, N_DIMS> &vertex, std::array<double, N_OPS> &values) override;

    /**
     * @brief Pre-fetch all missing supporting points for the given set of cells via batch evaluation.
     */
    void materialize_missing_points(const std::vector<double> &points, const std::vector<int> &points_idxs);
};

#include "linear_adaptive_cpu_interpolator.tpp"
#endif /* B2B2062C_7916_498E_8F40_7EFBD5E778F1 */
