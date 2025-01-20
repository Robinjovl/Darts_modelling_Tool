#ifndef A25773F7_8039_4BB3_ADDB_810683F83CD7
#define A25773F7_8039_4BB3_ADDB_810683F83CD7

#include <vector>
#include <array>

#include "interpolator_base.hpp"

/**
 * @brief  Interpolator base for static/adaptive piecewise mulitlinear interpolator
 * 
 * Interpolation is performed simulataneously for several functions (operators) in multidimensional parameter space
 * In order to do that, the space is uniformly parametrized within range of interest
 * That range along each axis is devided by specific number of equal intervals, forming uniform mesh 
 * Each vertex of the mesh represents a supporting point, where operator values are evaluated exactly
 * Using data at supporting points, interpolation is performed
 * 
 * @tparam index_t type used for indexing of supporting points and hypercubes
 * @tparam value_t value type used for supporting point storage, hypercube storage and interpolation
 * @tparam N_DIMS The number of dimensions in paramter space
 * @tparam N_OPS The number of operators to be interpolated
 */
template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
class multilinear_interpolator_base : public interpolator_base
{
public:
  const static uint16_t N_VERTS = (1 << N_DIMS); ///< number of vertexes in interpolation hypercube - N_DIMS-th power of 2

  typedef typename std::array<value_t, N_OPS> point_data_t;    ///< values of all operators at a given (supporting) point
  typedef typename std::vector<double> point_coordinates_t;    ///< coordinates of a given point in N_DIMS-dimensional space
  typedef typename std::array<int, N_DIMS> point_axes_index_t; ///< indexes of axes of point in parametrized space for each axis

  typedef typename std::array<value_t, N_VERTS * N_OPS> hypercube_data_t; ///< type for keeping values of all operators at all vertexes of a hypercube
  typedef typename std::array<index_t, N_VERTS> hypercube_points_index_t; ///< type for indexing vertexes of a hypercube

  /**
     * @brief Construct the interpolator with specified parametrization space
     * 
     * @param[in] supporting_point_evaluator    Object used to compute operators values at supporting points
     * @param[in] axes_points               Number of supporting points (minimum 2) along axes
     * @param[in] axes_min                  Minimum value for each axis
     * @param[in] axes_max                  Maximum for each axis
     */
  multilinear_interpolator_base(operator_set_evaluator_iface *supporting_point_evaluator,
                                const std::vector<int> &axes_points,
                                const std::vector<double> &axes_min,
                                const std::vector<double> &axes_max);

  /**
     * @brief Get the number of dimensions in interpolation space
     *
     */
  int get_n_dims() const { return N_DIMS; };

  /**
     * @brief Get the number of operators to be interpolated
     *
     */
  int get_n_ops() const { return N_OPS; };

  /**
     * @brief Compute interpolation for all operators at the given point
     *
     * @param[in]   point   Coordinates in parametrization space
     * @param[out]  values  Interpolated values
     * @return 0 if interpolation is successful
     */
  int interpolate(const std::vector<double> &point, std::vector<double> &values) override;

  /**
     * @brief Compute interpolation and its gradient for all operators at the given point point
     *
     * @param[in]   points        Coordinates of a point where interpolation is requested
     * @param[out]  values        Interpolated values
     * @param[out]  derivatives   Interpolation gradients
     * @return 0 if interpolation is successful
     */
  int interpolate_with_derivatives(const double *point,
                                   double *values,
                                   double *derivatives);

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

protected:
  /**
    * @brief Get point coordinates in space for given point index
    * 
    * @param[in] point_index index of the point
    * @param[out] coordinates coordinates along all axes
    */
  void inline get_point_coordinates(index_t point_index, point_coordinates_t &coordinates);

  // calculate point indexes for given hypercube
  /**
   * @brief Get indexes of all vertices for given hypercube
   * 
   * @param[in] index index of the hyporcube
   * @param[out] hypercube_points indexes of all vertices of hypercube
   */
  void inline get_hypercube_points(index_t index, hypercube_points_index_t &hypercube_points);

  /**
     * @brief Get values of operators at all vertices of the hypercube
     * Implementation depends on underlying storage. 
     *
     * @param[in] hypercube_index index of hypercube 
     * @return operator values at all vertices of the hypercube
     */
  virtual const hypercube_data_t &get_hypercube_data(const index_t hypercube_index) = 0;

  // decalare a copy of parametrization parameters in value_t precision to perform maximum computations with this precision
  const std::vector<value_t> axes_min_internal;      ///< minimum at each axis in value_t type
  const std::vector<value_t> axes_max_internal;      ///< maximum of each axis in value_t type
  const std::vector<value_t> axes_step_internal;     ///< the distance between neighbor supporting points for each axis in value_t type
  const std::vector<value_t> axes_step_inv_internal; ///< inverse of step (to avoid division) in value_t type

  std::vector<index_t> axis_point_mult;     ///< mult factor for each axis (for points) to compute global point index
  std::vector<index_t> axis_hypercube_mult; ///< mult factor for each axis (for hypercubes) to compute global hypercubes index
};

// now include implementation of the templated class from tpp file
#include "multilinear_interpolator_base.tpp"

#endif /* A25773F7_8039_4BB3_ADDB_810683F83CD7 */
