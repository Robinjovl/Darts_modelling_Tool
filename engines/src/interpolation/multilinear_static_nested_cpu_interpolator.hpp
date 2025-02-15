#ifndef MULTILINEAR_STATIC_NESTED_CPU_INTERPOLATOR_HPP_
#define MULTILINEAR_STATIC_NESTED_CPU_INTERPOLATOR_HPP_

#include "multilinear_static_cpu_interpolator.hpp"

/**
 * @brief Piecewise mulitlinear interpolator with static storage and nested interpolation for hypercubes
 *
 * Static storage is initialized in init() method. Two-level storage is used:
 * with operator data at every supporting point and with operator data at all vertices of every hypercube
 * point data may be assigned externally after construction and before init() call to save time
 * hypercube storage then is initialized only  and much faster, as does not involve computation of supporting points,
 * only copying
 *
 * @tparam index_t type used for indexing of supporting points and hypercubes
 * @tparam value_t value type used for supporting point storage, hypercube storage and interpolation
 * @tparam N_DIMS The number of dimensions in paramter space
 * @tparam N_OPS The number of operators to be interpolated
 */
template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
class multilinear_static_nested_cpu_interpolator : public multilinear_static_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>
{
public:
   using typename multilinear_static_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::point_coordinates_t;
   using typename multilinear_static_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::point_data_t;
   using typename multilinear_static_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::hypercube_data_t;
   using typename multilinear_static_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::hypercube_points_index_t;
   typedef multilinear_static_nested_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS> nested_interpolator_t;
   /**
     * @brief Construct the interpolator with specified parametrization space
     *
     * @param[in] supporting_point_evaluator    Object used to compute operators values at supporting points
     * @param[in] axes_points               Number of supporting points (minimum 2) along axes
     * @param[in] axes_min                  Minimum value for each axis
     * @param[in] axes_max                  Maximum for each axis
     * @param[in] _nest_lvl                 Level of nested interpolator
     */
   multilinear_static_nested_cpu_interpolator(operator_set_evaluator_iface* supporting_point_evaluator,
     const std::vector<int>& axes_points,
     const std::vector<double>& axes_min,
     const std::vector<double>& axes_max,
     const index_t _nest_lvl = 0);

   /**
    * @brief Destructor
    */
   ~multilinear_static_nested_cpu_interpolator();

   /**
     * @brief Initialize nested interpolators
     * @return int 0 if successful
     */
   int init_nested(const index_t init_nest_lvl, const std::vector<int>& nested_axes_points);

   std::vector<index_t> active_hypercube_counter; ///< Indices of hypercubes used for interpolation
   const index_t nest_lvl; ///< Level of nested interpolator in the hierarchy of interpolators
   index_t max_nest_lvl; ///< Maximum level of nested interpolator
   uint64_t n_hypercubes_total; ///< Total number of hypercubes

   /**
   * @brief Compute interpolation and its gradient for all operators at every specified point
   *
   * @param[in]   points        Array of coordinates in parametrization space
   * @param[in]   points_idxs   Indexes of points in the points array which are marked for interpolation
   * @param[out]  values        Interpolated values
   * @param[out]  derivatives   Interpolation gradients
   * @return 0 if interpolation is successful
   */
   int interpolate_with_derivatives(const std::vector<double>& points, const std::vector<int>& points_idxs,
     std::vector<double>& values, std::vector<double>& derivatives) override;

   /**
   * @brief Compute interpolation and its gradient for all operators at the given point point
   *
   * @param[in]   points        Coordinates of a point where interpolation is requested
   * @param[out]  values        Interpolated values
   * @param[out]  derivatives   Interpolation gradients
   * @return 0 if interpolation is successful
   */
   int interpolate_with_derivatives(const double* point, double* values, double* derivatives) override;

protected:
  /**
    * @brief Get values of operators at all vertices of the hypercube.
    * Simply provide a reference to correct location in static storage - all values have been already computed
    *
    * @param[in] hypercube_index index of hypercube
    * @return operator values at all vertices of the hypercube
    */
  const hypercube_data_t& get_hypercube_data(const index_t hypercube_index) override;

  std::vector<nested_interpolator_t*> hypercube_interpolators;
};

// now include implementation of the templated class from tpp file
#include "multilinear_static_nested_cpu_interpolator.tpp"

#endif /* MULTILINEAR_STATIC_NESTED_CPU_INTERPOLATOR_HPP_ */
