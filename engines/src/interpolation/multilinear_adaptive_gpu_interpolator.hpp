#ifndef C2EE8F5F_DDE3_4485_90BA_A6C32B67EB4D
#define C2EE8F5F_DDE3_4485_90BA_A6C32B67EB4D

#include <vector>
#include <array>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#if 1 //def CUDA12
#include <thrust/system/cuda/memory_resource.h>
#else
#include <pinned_allocator.h>
#endif

#include "gpu_hashmap_async.h"
#include "multilinear_gpu_interpolator_base.hpp"

/**
 * @brief  Piecewise mulitlinear interpolator for GPU with adaptive storage
 * 
 * Two-level storage is used: 
 * with operator data at every computed supporting point (on host and on device) and with operator data at all vertices of every requested hypercube (on device only)
 * point data may be assigned externally after construction and before init() call to save time
 * 
 * @tparam index_t type used for indexing of supporting points and hypercubes
 * @tparam value_t value type used for supporting point storage, hypercube storage and interpolation
 * @tparam N_DIMS The number of dimensions in paramter space
 * @tparam N_OPS The number of operators to be interpolated
 */

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
class multilinear_adaptive_gpu_interpolator : public multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>
{
public:
  const static uint16_t N_VERTS = (1 << N_DIMS); ///< number of vertexes in interpolation hypercube - N_DIMS-th power of 2

  using typename multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::point_coordinates_t;
  using typename multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::point_data_t;
  using typename multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::hypercube_points_index_t;
  #if 1 //CUDA12
  using mr = thrust::cuda::universal_host_pinned_memory_resource;
  using index_pinned_allocator = thrust::mr::stateless_resource_allocator<index_t, mr >;
  using value_pinned_allocator = thrust::mr::stateless_resource_allocator<value_t, mr >;
  using int_pinned_allocator = thrust::mr::stateless_resource_allocator<int, mr >;
  typedef thrust::host_vector<index_t, index_pinned_allocator> pinned_index_vector_t;
  typedef thrust::host_vector<value_t, value_pinned_allocator> pinned_value_vector_t;
  typedef thrust::host_vector<int, int_pinned_allocator> pinned_int_vector_t;
  #else
  typedef thrust::host_vector<index_t, thrust::cuda::experimental::pinned_allocator<index_t>> pinned_index_vector_t;
  typedef thrust::host_vector<value_t, thrust::cuda::experimental::pinned_allocator<value_t>> pinned_value_vector_t;
  typedef thrust::host_vector<int, thrust::cuda::experimental::pinned_allocator<int>> pinned_int_vector_t;
  #endif
  /**
     * @brief Construct the interpolator with specified parametrization space
     * 
     * @param[in] supporting_point_evaluator    Object used to compute operators values at supporting points
     * @param[in] axes_points               Number of supporting points (minimum 2) along axes
     * @param[in] axes_min                  Minimum value for each axis
     * @param[in] axes_max                  Maximum for each axis
     */
  multilinear_adaptive_gpu_interpolator(operator_set_evaluator_iface *supporting_point_evaluator,
                                        const std::vector<int> &axes_points,
                                        const std::vector<double> &axes_min,
                                        const std::vector<double> &axes_max);

  ~multilinear_adaptive_gpu_interpolator();
  /**
     * @brief Initialize the interpolator, if point_data storage was already initialized from Python
     * 
     * @return int 0 if successful
     */
  int init();

  /**
	 * @brief Write interpolator data to file
	 *
	 * @param filename name of the file
	 * @return int error code
	 */
  int write_to_file(const std::string filename);

  /**
   * @brief adaptive point storage on host: the values of operators at specific points
   * 
   * Used to store all computed supporting points and to initialize point_data_d
   * Can be initialized externally from Python
   */
  std::unordered_map<index_t, point_data_t> point_data;

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
    * @brief Generate hypercube data 
    * 
    * @param[in] hypercube_idx 
    * @param[out] new_hypercube operator values at all vertices of generated hypercube
    * @return 0 if success 
    */
  int generate_hypercube(index_t hypercube_idx, value_t *new_hypercube);

  /**
     * @brief Compute interpolation and its gradient on the device for all operators at every specified point
     *
     * @param[in]   points        Array of coordinates in parametrization space
     * @param[in]   points_idxs   Indexes of points in the points array which are marked for interpolation
     * @param[out]  values        Interpolated values
     * @param[out]  derivatives   Interpolation gradients
     * @return 0 if interpolation is successful
     */
  virtual int
  evaluate_with_derivatives_d(int n_states_idxs, double *state_d, int *states_idxs_d,
                              double *values_d, double *derivatives_d) override;

  // **** HOST DATA ****
  /**
   * @brief adaptive hypercube index storage on host: indexes of already generated hypercubes
   * 
   * Used to check if a hypercube was already computed
   */
  std::unordered_set<index_t> generated_hypercubes;

  pinned_index_vector_t hypercubes_to_compute;      ///< New hypercubes required for each state to interpolate (on host)
  pinned_value_vector_t new_hypercube_data_buffer;  ///< Generated hypercube data buffer to copy to device (on host)
  pinned_index_vector_t new_hypercube_index_buffer; ///< Generated hypercube indexes buffer to copy to device (on host)
  pinned_int_vector_t hashmap_expansion_needed;     ///< Flag showing if hashmap expansion needed (on host)

  std::vector<value_t> new_hypercube_data;  ///< Data storage for generated hypercubes
  std::vector<index_t> new_hypercube_index; ///< Index storage for generated hypercubes

  // **** DEVICE DATA ****
  /**
   * @brief adaptive hypercube storage on device
   * 
   * In fact it is an excess storage used to reduce memory accesses during interpolation. 
   * Here, all values of all vertexes of every stored hypercube are stored consecutevely and are accessed via a single index
   * Usage of point_data for interpolation directly would require N_VERTS memory accesses (>1000 accesses for 10-dimensional space)
   */
  gpu_hashmap_async::gpu_hash_map<value_t, N_VERTS * N_OPS> *hypercube_data_d; ///< Hypercube data on device

  gpu_hashmap_async::gpu_hash_map<value_t, N_OPS> *point_data_d; ///< Point data on device

  thrust::device_vector<index_t> state_markers_d;              ///< Marker array filled with -1 for states scheduled for stage 1 and hypercube index for those for states scheduled for stage 2 (on device)
  thrust::device_vector<value_t> new_hypercube_data_buffer_d;  ///< Generated hypercube data buffer to receive data from host (on device)
  thrust::device_vector<index_t> new_hypercube_index_buffer_d; ///< Generated hypercube index buffer to receive data from host (on device)
  thrust::device_vector<int> hashmap_expansion_needed_d;       ///< Flag showing if hashmap expansion needed (on device)

  cudaStream_t hypercube_generation_stream; ///< Stream to collect required hypercube indices, send them to host, compute them and send the result back to device
  cudaStream_t stage1_interpolation_stream; ///< Stream to interpolate those states which do not require new hypercubes
};
// now include implementation of the templated class from tpp file
#include "multilinear_adaptive_gpu_interpolator.tpp"

#endif /* C2EE8F5F_DDE3_4485_90BA_A6C32B67EB4D */
