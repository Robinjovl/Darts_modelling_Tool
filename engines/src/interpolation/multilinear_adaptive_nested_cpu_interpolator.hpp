#ifndef MULTILINEAR_ADAPTIVE_NESTED_CPU_INTERPOLATOR_HPP_
#define MULTILINEAR_ADAPTIVE_NESTED_CPU_INTERPOLATOR_HPP_

#include "multilinear_adaptive_cpu_interpolator.hpp"

/**
 * @brief  Piecewise mulitlinear interpolator with adaptive storage and nested interpolation for hypercubes
 * 
 * 
 * @tparam index_t type used for indexing of supporting points and hypercubes
 * @tparam value_t value type used for supporting point storage, hypercube storage and interpolation
 * @tparam N_DIMS The number of dimensions in paramter space
 * @tparam N_OPS The number of operators to be interpolated
 */
template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
class multilinear_adaptive_nested_cpu_interpolator : public multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>
{
public:
   using typename multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::point_coordinates_t;
   using typename multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::point_data_t;
   using typename multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::hypercube_data_t;
   using typename multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::hypercube_points_index_t;
   typedef multilinear_adaptive_nested_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS> nested_interpolator_t;

   /**
     * @brief Construct the interpolator with specified parametrization space
     * 
     * @param[in] supporting_point_evaluator    Object used to compute operators values at supporting points
     * @param[in] axes_points               Number of supporting points (minimum 2) along axes
     * @param[in] axes_min                  Minimum value for each axis
     * @param[in] axes_max                  Maximum for each axis
     */
   multilinear_adaptive_nested_cpu_interpolator(operator_set_evaluator_iface *supporting_point_evaluator,
                                               const std::vector<int> &axes_points,
                                               const std::vector<double> &axes_min,
                                               const std::vector<double> &axes_max);
protected:
   /**
     * @brief Get values of operators at a given point 
     * Provide a reference to correct location in the adaptive point storage. 
     * If the point is not found, compute it first, and then return the reference.
     *
     * @param[in] point_index index of point 
     * @return operator values at given point
     */
   const point_data_t &get_point_data(const index_t point_index);

   /**
     * @brief Get values of operators at all vertices of the hypercube. 
     * Provide a reference to correct location in the adaptive hypercube storage. 
     * If the hypercube is not found, compute it first, and then return the reference.
     *
     * @param[in] hypercube_index index of hypercube 
     * @return operator values at all vertices of the hypercube
     */
   const hypercube_data_t &get_hypercube_data(const index_t hypercube_index);
   /**
     * @brief Compute interpolation and its gradient for all operators at every specified point
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
   * @brief adaptive hypercube storage: the values of operators at every vertex of reqested hypercubes
   * Storage is grown dynamically in the process of simulation
   * Only hypercubes that are required for interpolation are computed and added
   * 
   * In fact it is an excess storage used to reduce memory accesses during interpolation. 
   * Here all values of all vertexes of requested hypercube are stored consecutevely and are accessed via a single index
   * Usage of point_data for interpolation directly would require N_VERTS memory accesses (>1000 accesses for 10-dimensional space)
   *  * 
   */
   std::unordered_map<index_t, nested_interpolator_t*> hypercube_interpolators;
};

// now include implementation of the templated class from tpp file
#include "multilinear_adaptive_nested_cpu_interpolator.tpp"

#endif /* MULTILINEAR_ADAPTIVE_NESTED_CPU_INTERPOLATOR_HPP_ */
