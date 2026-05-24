#ifndef B2B2062C_7916_498E_8F40_7EFBD5E778F1
#define B2B2062C_7916_498E_8F40_7EFBD5E778F1
#include <unordered_map>
#include <unordered_set>
#include "linear_cpu_interpolator_base.hpp"

/**
 * Adaptive piecewise linear interpolator
 *
 * @tparam index_t index type used for supporting point indexing
 * @tparam N_DIMS The number of dimensions in paramter space
 * @tparam N_OPS The number of operators to be interpolated
 */
template <typename index_t, int N_DIMS, int N_OPS>
class linear_adaptive_cpu_interpolator : public linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>
{
public:
    linear_adaptive_cpu_interpolator(operator_set_evaluator_iface *base_points_generator,
                                     const std::vector<int> &axesPoints,
                                     const std::vector<double> &axesMin,
                                     const std::vector<double> &axesMax,
                                     bool _use_barycentric_interpolation);

    std::unordered_map<index_t, std::array<double, N_OPS>> point_data; ///< adaptive storage: the values of operators at supporting points actually required

    /**
     * @brief Interpolate with batch pre-fetching of missing supporting points.
     *
     * Overrides the base class to add a two-pass approach:
     *   Pass 1: scan all cells, identify simplex vertices, collect unique missing point indices,
     *           batch-evaluate them via evaluate_batch() (which may use ParallelEvaluator).
     *   Pass 2: delegate to base class interpolate_with_derivatives() — all cache hits.
     */
    int interpolate_with_derivatives(const std::vector<double> &points, const std::vector<int> &points_idxs,
                                     std::vector<double> &values, std::vector<double> &derivatives) override;

private:
    /**
     * @brief Get values of operators at supporting point
     * The function checks whether the values at the given vertex are computed before,
     *      if yes, the values are directly returned;
     *      if not, the function computes the values, stores them and then returns.
     *
     * @param[in] vertex The index of the given supporting point along axes
     * @param[out] values The values of operators at a given point
     */
    void get_supporting_point(const std::array<index_t, N_DIMS> &vertex, std::array<double, N_OPS> &values) override;

    /**
     * @brief Pre-fetch all missing supporting points for the given set of cells via batch evaluation.
     *
     * Scans all cells to find their simplex vertices, collects unique missing point indices,
     * then evaluates them all in one evaluate_batch() call. After this, all subsequent
     * get_supporting_point() calls in the interpolation loop will be cache hits.
     */
    void materialize_missing_points(const std::vector<double> &points, const std::vector<int> &points_idxs);
};

#include "linear_adaptive_cpu_interpolator.tpp"
#endif /* B2B2062C_7916_498E_8F40_7EFBD5E778F1 */
