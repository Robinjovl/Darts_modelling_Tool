#ifndef BF53EBFF_8376_4642_9B52_6A604A3467C3
#define BF53EBFF_8376_4642_9B52_6A604A3467C3

#include <functional>
#include <vector>
#include "linear_cpu_interpolator_base.hpp"

/**
 * Static piecewise linear interpolator
 *
 * Vertices are int32-native (base vertex_t); the dense point storage is addressed by a
 * uint64 mixed-radix index (get_index_from_vertex).
 *
 * @tparam N_DIMS The number of dimensions in paramter space
 * @tparam N_OPS The number of operators to be interpolated
 */
template <int N_DIMS, int N_OPS>
class linear_static_cpu_interpolator : public linear_cpu_interpolator_base<N_DIMS, N_OPS>
{
public:
    using typename linear_cpu_interpolator_base<N_DIMS, N_OPS>::vertex_t;
    /**
     * @brief Construct the interpolator with a finite dense grid (origin, step, points).
     *
     * @param[in] supporting_point_evaluator      Object used to compute operators values at supporting points
     * @param[in] axes_origin                     Grid origin (lower corner) for each axis
     * @param[in] axes_step                       Cell size for each axis
     * @param[in] axes_points                     Number of supporting points (minimum 2) along each axis
     * @param[in] _use_barycentric_interpolation  Flag to turn on barycentric interpolation on Delaunay triangulation
     */
    linear_static_cpu_interpolator(operator_set_evaluator_iface *supporting_point_evaluator,
                                   const std::vector<double> &axes_origin,
                                   const std::vector<double> &axes_step,
                                   const std::vector<int> &axes_points,
                                   bool _use_barycentric_interpolation);

    /**
     * @brief Initialize the interpolator by computing all values of supporting points if the storage was not already initialized
     *
     * @return int 0 if successful
     */
    int init() override;

    std::vector<double> point_data; ///< static storage: the values of operators at all supporting points
private:
    /**
     * @brief This function transfers index of supporting point to its indices along axes.
     *
     * For example, in the 2-dimensional grid with axes_points = [4, 3],
     * the supporting point with index 0 is [0, 0],
     * the supporting point with index 2 is [0, 2],
     * the supporting point with index 4 is [1, 0],
     * the supporting point with index 5 is [1, 1],
     *
     * @param[in] index Index of the supporting point in the storage
     * @param[out] vertex The supporting point's index along each axis
     */
    void get_vertex_from_index(uint64_t index, vertex_t &vertex);
    /**
     * @brief Get the operator values at the given supporting point using static storage
     *
     * @param[in] vertex The indexes of coordinates the given supporting point along axes
     * @param[out] values The values of operators at a given point
     */
    void get_supporting_point(const vertex_t &vertex, std::array<double, N_OPS> &values) override;
};

#include "linear_static_cpu_interpolator.tpp"
#endif /* BF53EBFF_8376_4642_9B52_6A604A3467C3 */
