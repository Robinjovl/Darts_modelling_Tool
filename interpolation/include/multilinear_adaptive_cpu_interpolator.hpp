#ifndef BFD055ED_0AA6_4F1C_A5A5_9157EE0F34FF
#define BFD055ED_0AA6_4F1C_A5A5_9157EE0F34FF

#include <vector>
#include <array>
#include <unordered_set>
#include <algorithm>

#include "multilinear_interpolator_base.hpp"

/**
 * @brief  Piecewise mulitlinear interpolator with adaptive storage
 *
 * Adaptive evaluation is split into three explicit phases:
 *   Phase 1 – parallel computation of hypercube indices for every requested cell
 *   Phase 2 – serial materialization of missing supporting points (Python callback)
 *             and parallel assembly of missing hypercube payloads
 *   Phase 3 – parallel read-only interpolation with thread-local workspace
 *
 * After Phase 2 completes, point_data and hypercube_data are read-only for Phase 3,
 * eliminating any concurrent mutation of shared containers.
 *
 * @tparam index_t type used for indexing of supporting points and hypercubes
 * @tparam value_t value type used for supporting point storage, hypercube storage and interpolation
 * @tparam N_DIMS The number of dimensions in paramter space
 * @tparam N_OPS The number of operators to be interpolated
 */
template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
class multilinear_adaptive_cpu_interpolator : public multilinear_interpolator_base<index_t, value_t, N_DIMS, N_OPS>
{
public:
   using typename multilinear_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::point_coordinates_t;
   using typename multilinear_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::point_data_t;
   using typename multilinear_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::hypercube_data_t;
   using typename multilinear_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::hypercube_points_index_t;
   /**
     * @brief Construct the interpolator with specified parametrization space
     *
     * @param[in] supporting_point_evaluator    Object used to compute operators values at supporting points
     * @param[in] axes_points               Number of supporting points (minimum 2) along axes
     * @param[in] axes_min                  Minimum value for each axis
     * @param[in] axes_max                  Maximum for each axis
     */
   multilinear_adaptive_cpu_interpolator(operator_set_evaluator_iface *supporting_point_evaluator,
                                         const std::vector<int> &axes_points,
                                         const std::vector<double> &axes_min,
                                         const std::vector<double> &axes_max);

   /**
   * @brief adaptive point storage: the values of operators at requested supporting points
   * Storage is grown dynamically in the process of simulation.
   * Only supporting points that are required for interpolation are computed and added
   *
   */
   std::unordered_map<index_t, point_data_t> point_data;

   /**
   * @brief Get indexes of all evaluated hypercubes
   *
   * @return array of indexes of hypercubes
   */
   std::vector<index_t> get_hypercube_indexes() const;
protected:
   /**
     * @brief Get values of operators at a given point
     * Provide a reference to correct location in the adaptive point storage.
     * If the point is not found, compute it first, and then return the reference.
     * Used by the single-point interpolation path and by materialize_missing_cache.
     *
     * @param[in] point_index index of point
     * @return operator values at given point
     */
   const point_data_t &get_point_data(const index_t point_index);
   /**
     * @brief Get values of operators at all vertices of the hypercube.
     * Provide a reference to correct location in the adaptive hypercube storage.
     * If the hypercube is not found, compute it first, and then return the reference.
     * Used by the single-point interpolation path.
     *
     * @param[in] hypercube_index index of hypercube
     * @return operator values at all vertices of the hypercube
     */
   const hypercube_data_t &get_hypercube_data(const index_t hypercube_index);
   /**
     * @brief Compute interpolation and its gradient for all operators at every specified point
     *        Uses three-phase parallel approach: discover, materialize, interpolate.
     *
     * @param[in]   points        Array of coordinates in parametrization space
     * @param[in]   points_idxs   Indexes of points in the points array which are marked for interpolation
     * @param[out]  values        Interpolated values
     * @param[out]  derivatives   Interpolation gradients
     * @return 0 if interpolation is successful
     */
   int interpolate_with_derivatives(const std::vector<double> &points, const std::vector<int> &points_idxs,
                                    std::vector<double> &values, std::vector<double> &derivatives) override;

   /**
     * @brief Materialize all missing supporting points and hypercubes for the given hypercube indices.
     *        Missing points are evaluated serially through the supporting_point_evaluator (Python-safe).
     *        Missing hypercube payloads are assembled in parallel from the now-complete point cache.
     *
     * @param[in] missing_hc  Sorted, unique vector of hypercube indices not yet in hypercube_data
     */
   void materialize_missing_cache(const std::vector<index_t> &missing_hc);

   /**
   * @brief adaptive hypercube storage: the values of operators at every vertex of reqested hypercubes
   * Storage is grown dynamically in the process of simulation
   * Only hypercubes that are required for interpolation are computed and added
   *
   * In fact it is an excess storage used to reduce memory accesses during interpolation.
   * Here all values of all vertexes of requested hypercube are stored consecutevely and are accessed via a single index
   * Usage of point_data for interpolation directly would require N_VERTS memory accesses (>1000 accesses for 10-dimensional space)
   *  *
   */
   std::unordered_map<index_t, hypercube_data_t> hypercube_data;
};

// now include implementation of the templated class from tpp file
#include "multilinear_adaptive_cpu_interpolator.tpp"

#endif /* BFD055ED_0AA6_4F1C_A5A5_9157EE0F34FF */
