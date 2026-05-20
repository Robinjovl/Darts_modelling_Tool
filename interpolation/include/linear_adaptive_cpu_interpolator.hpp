#ifndef B2B2062C_7916_498E_8F40_7EFBD5E778F1
#define B2B2062C_7916_498E_8F40_7EFBD5E778F1
#include <unordered_map>
#include <unordered_set>
#include "multi_index_key.hpp"
#include "linear_cpu_interpolator_base.hpp"

/**
 * Adaptive piecewise linear interpolator.
 *
 * Storage is keyed on a signed multi-index (cell_key_t<N_DIMS>) so the adaptive cache
 * no longer depends on axes_min/axes_max to bound the index space — out-of-window
 * supporting points are evaluated and cached on demand. The Python-visible point_data
 * property keeps the legacy integer-keyed format for in-bounds cells (pickle cache
 * compatibility); out-of-bounds cells live in memory only.
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
                                     const std::vector<int> &axesPoints,
                                     const std::vector<double> &axesMin,
                                     const std::vector<double> &axesMax,
                                     bool _use_barycentric_interpolation);

    /**
     * Adaptive supporting-point storage, keyed on signed multi-index.
     */
    std::unordered_map<key_t, std::array<double, N_OPS>, key_hash_t> point_data;

    /**
     * @brief Interpolate with batch pre-fetching of missing supporting points.
     */
    int interpolate_with_derivatives(const std::vector<double> &points, const std::vector<int> &points_idxs,
                                     std::vector<double> &values, std::vector<double> &derivatives) override;

    /**
     * @brief Build the multi-index key from a vertex (with int32 bit pattern packed in index_t).
     */
    key_t key_from_vertex(const std::array<index_t, N_DIMS> &vertex) const;

    /**
     * @brief Legacy packing: multi-index → integer key (axes_mult). Used for pickle export.
     */
    index_t to_int_key(const key_t &k) const;

    /**
     * @brief Legacy unpacking: integer key → multi-index. Used for pickle import.
     */
    key_t from_int_key(index_t int_key) const;

    /**
     * @brief True iff every component is within [0, axes_points[i]-1]. Out-of-bounds cells
     * are kept in memory but not exported through the legacy point_data property.
     */
    bool is_in_bounds(const key_t &k) const;

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
