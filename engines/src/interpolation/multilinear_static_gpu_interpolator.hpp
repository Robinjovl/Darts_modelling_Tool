//*************************************************************************
//    Copyright (c) 2018
//            Mark Khait         M.Khait@tudelft.nl
//            Denis Voskov    D.V.Voskov@tudelft.nl
//    Delft University of Technology, the Netherlands
//
//    This file is part of the Delft Advanced Research Terra Simulator (DARTS)
//
//    DARTS is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as
//    published by the Free Software Foundation, either version 3 of the
//    License, or (at your option) any later version.
//
//    DARTS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public
//    License along with DARTS. If not, see <http://www.gnu.org/licenses/>.
// *************************************************************************

#ifndef CFF5C8D4_2C19_48B8_9962_4985676A4DFC
#define CFF5C8D4_2C19_48B8_9962_4985676A4DFC

#include <vector>
#include <array>

#include "multilinear_gpu_interpolator_base.hpp"

/**
 * @brief  Piecewise mulitlinear interpolator for GPU with static storage
 * 
 * Static storage is initialized in init() method. Two-level storage is used: 
 * with operator data at every supporting point (on host only) and with operator data at all vertices of every hypercube (on device only)
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
class multilinear_static_gpu_interpolator : public multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>
{
public:
  using typename multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::point_coordinates_t;
  using typename multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::point_data_t;
  using typename multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::hypercube_points_index_t;
  /**
     * @brief Construct the interpolator with specified parametrization space
     * 
     * @param[in] supporting_point_evaluator    Object used to compute operators values at supporting points
     * @param[in] axes_points               Number of supporting points (minimum 2) along axes
     * @param[in] axes_min                  Minimum value for each axis
     * @param[in] axes_max                  Maximum for each axis
     */
  multilinear_static_gpu_interpolator(operator_set_evaluator_iface *supporting_point_evaluator,
                                      const std::vector<int> &axes_points,
                                      const std::vector<double> &axes_min,
                                      const std::vector<double> &axes_max);

  /**
     * @brief Initialize the interpolator by:
     * 1. computing all values of supporting points on host (if point_data storage was not already initialized)
     * 2. populating hypercube static storage (on device) from point storage
     * 
     * @return int 0 if successful
     */
  int init();

  /**
   * @brief static point storage: the values of operators at all supporting points
   * 
   * Used to store all computed supporting points and to initialize hypercube_data_d
   * Is initialized during init() or externally from Python
   */
  std::vector<point_data_t> point_data;

protected:
  /**
     * @brief Compute interpolation and its gradient on the device for all operators at every specified point
     *
     * @param[in]   points        Array of coordinates in parametrization space
     * @param[in]   points_idxs   Indexes of points in the points array which are marked for interpolation
     * @param[out]  values        Interpolated values
     * @param[out]  derivatives   Interpolation gradients
     * @return 0 if interpolation is successful
     */
  virtual int evaluate_with_derivatives_d(int n_states_idxs, double *state_d, int *states_idxs_d,
                                          double *values_d, double *derivatives_d) override;

  /**
   * @brief static hypercube storage on device: the values of operators at every vertex of all hypercubes
   * 
   * In fact it is an excess storage used to reduce memory accesses during interpolation. 
   * Here all values of all vertexes of every hypercube are stored consecutevely and are accessed via a single index
   * Usage of point_data for interpolation directly would require N_VERTS memory accesses (>1000 accesses for 10-dimensional space)
   */
  thrust::device_vector<value_t> hypercube_data_d; ///< Device hypercube data
};

// now include implementation of the templated class from tpp file
#include "multilinear_static_gpu_interpolator.tpp"

#endif /* CFF5C8D4_2C19_48B8_9962_4985676A4DFC */
