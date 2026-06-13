#ifndef DC436B8D_1BB3_4BF3_98FE_9A7C961A6B53
#define DC436B8D_1BB3_4BF3_98FE_9A7C961A6B53

#include <thrust/device_vector.h>

#include "evaluator_iface.h"

/**
 * @brief  Piecewise mulitlinear GPU interpolator base class
 *
 * Introduces and initialize basic interpolation data on GPU device
 *
 * @tparam index_t type used for indexing of supporting points and hypercubes
 * @tparam value_t value type used for supporting point storage, hypercube storage and interpolation
 * @tparam N_DIMS The number of dimensions in paramter space
 * @tparam N_OPS The number of operators to be interpolated
 */
// N_OPS widened to uint16_t — see multilinear_interpolator_base.hpp.
template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
class multilinear_gpu_interpolator_base : public operator_set_gradient_evaluator_gpu
{
public:
   const static uint32_t N_VERTS = (1 << N_DIMS); ///< number of vertexes in interpolation hypercube - N_DIMS-th power of 2

   typedef typename std::array<value_t, N_OPS> point_data_t;               ///< values of all operators at a given (supporting) point
   typedef typename std::vector<double> point_coordinates_t;               ///< coordinates of a given point in N_DIMS-dimensional space
   typedef typename std::array<index_t, N_VERTS> hypercube_points_index_t; ///< type for indexing vertexes of a hypercube

   /**
     * @brief Construct an unbounded interpolator parametrized by (origin, step).
     *        Used by adaptive GPU storage; the grid has no upper bound.
     *
     * @param[in] supporting_point_evaluator    Object used to compute operators values at supporting points
     * @param[in] axes_origin              Grid origin (lower corner) for each axis
     * @param[in] axes_step                Cell size for each axis
     */
   multilinear_gpu_interpolator_base(operator_set_evaluator_iface *supporting_point_evaluator,
                                     const std::vector<double> &axes_origin,
                                     const std::vector<double> &axes_step);

   /**
     * @brief Construct a bounded interpolator with a finite dense grid.
     *        Used by static GPU storage; builds the flat mixed-radix multipliers.
     *
     * @param[in] supporting_point_evaluator    Object used to compute operators values at supporting points
     * @param[in] axes_origin              Grid origin (lower corner) for each axis
     * @param[in] axes_step                Cell size for each axis
     * @param[in] axes_points              Number of supporting points (minimum 2) along each axis
     */
   multilinear_gpu_interpolator_base(operator_set_evaluator_iface *supporting_point_evaluator,
                                     const std::vector<double> &axes_origin,
                                     const std::vector<double> &axes_step,
                                     const std::vector<int> &axes_points);
   /**
     * @brief Get the number of supporting points for the given axis
     *
     * @param axis index of axis in question
     */
   int get_axis_n_points(int axis) const;

   /**
     * @brief Get the parametrization minimum value for given axis
     *
     * @param axis index of axis in question
     */
   double get_axis_min(int axis) const;

   /**
     * @brief Get the parametrization maximum value for given axis
     *
     * @param axis index of axis in question
     */
   double get_axis_max(int axis) const;

   /**
     * @brief Get the number of interpolations that took place
     *
     */
   uint64_t get_n_interpolations() const;

   /**
     * @brief Get the total number of supporting points in parameter space
     *
     * @return the total number of supporting points
     */
   uint64_t get_n_points_total() const;

   /**
     * @brief Get the number of supporting points used (evaluated through supporting_point_evaluator)
     *        The number is equal to n_points_total for static interpolation methods
     *
     * @return the number of supporting points used
     */
   uint64_t get_n_points_used() const;
   /**
   * @brief Compute operators values for specified state on device
   *
   * @param state Coordinates in parameter space, where operators to be evaluated
   * @param values Evaluated operators values
   * @return int 0 if evaluation is successful
   */
   int evaluate_d(double *state_d, double *values_d);

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

   operator_set_evaluator_iface *supporting_point_evaluator; ///< object which computes operator values for supporting points

   const std::vector<double> axes_origin;    ///< grid origin (lower corner) at each axis
   std::vector<double> axes_step;            ///< the distance between neighbor supporting points for each axis
   std::vector<double> axes_step_inv;        ///< inverse of step (to avoid division)
   // The following are populated only for bounded (static) grids; empty for unbounded
   // adaptive grids, whose kernels use only (origin, step) with signed multi-index keys.
   const std::vector<int> axes_points;       ///< number of supporting points along each axis (bounded grids only)
   const std::vector<double> axes_max;       ///< maximum of each axis (bounded grids only)
   std::vector<index_t> axis_point_mult;     ///< mult factor for each axis (for points) to compute global point index (bounded grids only)
   std::vector<index_t> axis_hypercube_mult; ///< mult factor for each axis (for hypercubes) to compute global hypercubes index (bounded grids only)

   uint64_t n_interpolations; ///< Number of interpolations that took place
   uint64_t n_points_total;   ///< Total number of parametrization points
   double n_points_total_fp;  ///< Total number of parametrization points in floating point format, to detect index overflow in derived classes
   uint64_t n_points_used;    ///< Number of parametrization points which were used (equal to n_points_total for static interpolators)

   std::vector<double> new_point_coords;    ///< intermediate storage for supporting point generation
   std::vector<double> new_operator_values; ///< intermediate storage for supporting point generation

   thrust::device_vector<value_t> axes_origin_d;   ///< grid origin at each axis in value_t type on device
   thrust::device_vector<value_t> axes_step_d;     ///< the distance between neighbor supporting points for each axis in value_t type on device
   thrust::device_vector<value_t> axes_step_inv_d; ///< inverse of step (to avoid division) in value_t type on device

   // bounded (static) grids only:
   thrust::device_vector<int> axes_points_d;       ///< number of parametrization points for each axis on device
   thrust::device_vector<value_t> axes_max_d;      ///< maximum of each axis in value_t type on device
   thrust::device_vector<index_t> axis_point_mult_d;     ///< mult factor for each axis (for points) to compute global point index on device
   thrust::device_vector<index_t> axis_hypercube_mult_d; ///< mult factor for each axis (for hypercubes) to compute global hypercubes index on device

   int kernel_block_size; /// GPU thread block size for interpolation kernel
};

// now include implementation of the templated class from tpp file
#include "multilinear_gpu_interpolator_base.tpp"

#endif /* DC436B8D_1BB3_4BF3_98FE_9A7C961A6B53 */
