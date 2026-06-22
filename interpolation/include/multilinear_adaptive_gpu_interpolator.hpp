#ifndef C2EE8F5F_DDE3_4485_90BA_A6C32B67EB4D
#define C2EE8F5F_DDE3_4485_90BA_A6C32B67EB4D

#include <vector>
#include <array>
#include <unordered_set>
#include <unordered_map>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#if 1 //def CUDA12
#include <thrust/system/cuda/memory_resource.h>
#else
#include <pinned_allocator.h>
#endif

#include "multi_index_key.hpp"
#include "gpu_hashmap_async.h"
#include "multilinear_gpu_interpolator_base.hpp"

/**
 * @brief  Piecewise mulitlinear interpolator for GPU with adaptive storage.
 *
 * Storage is keyed on signed multi-index (cell_key_t<N_DIMS>) on the host side; the
 * device hashmap uses a 64-bit content hash of the multi-index as its lookup key.
 * The adaptive cache therefore no longer depends on axes_min/axes_max — the solver
 * may explore state space freely; cells outside the prescribed window are evaluated
 * and cached on demand.
 *
 * @tparam index_t legacy integer index type (kept for ABI compat; unused on the GPU hot path)
 * @tparam value_t value type used for supporting point and hypercube storage
 * @tparam N_DIMS  number of dimensions in parameter space
 * @tparam N_OPS   number of operators to be interpolated
 */
// N_OPS widened to uint16_t — must match base class.
template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
class multilinear_adaptive_gpu_interpolator : public multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>
{
public:
  const static uint32_t N_VERTS = (1 << N_DIMS);

  using typename multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::point_coordinates_t;
  using typename multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::point_data_t;
  using typename multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::hypercube_points_index_t;

  typedef cell_key_t<N_DIMS> key_t;
  typedef cell_key_hash<N_DIMS> key_hash_t;

#if 1 //CUDA12
  using mr = thrust::cuda::universal_host_pinned_memory_resource;
  using index_pinned_allocator = thrust::mr::stateless_resource_allocator<index_t, mr>;
  using value_pinned_allocator = thrust::mr::stateless_resource_allocator<value_t, mr>;
  using int_pinned_allocator = thrust::mr::stateless_resource_allocator<int, mr>;
  using key_pinned_allocator = thrust::mr::stateless_resource_allocator<key_t, mr>;
  typedef thrust::host_vector<index_t, index_pinned_allocator> pinned_index_vector_t;
  typedef thrust::host_vector<value_t, value_pinned_allocator> pinned_value_vector_t;
  typedef thrust::host_vector<int, int_pinned_allocator> pinned_int_vector_t;
  typedef thrust::host_vector<key_t, key_pinned_allocator> pinned_key_vector_t;
#else
  typedef thrust::host_vector<index_t, thrust::cuda::experimental::pinned_allocator<index_t>> pinned_index_vector_t;
  typedef thrust::host_vector<value_t, thrust::cuda::experimental::pinned_allocator<value_t>> pinned_value_vector_t;
  typedef thrust::host_vector<int, thrust::cuda::experimental::pinned_allocator<int>> pinned_int_vector_t;
  typedef thrust::host_vector<key_t, thrust::cuda::experimental::pinned_allocator<key_t>> pinned_key_vector_t;
#endif

  multilinear_adaptive_gpu_interpolator(operator_set_evaluator_iface *supporting_point_evaluator,
                                        const std::vector<double> &axes_origin,
                                        const std::vector<double> &axes_step);

  ~multilinear_adaptive_gpu_interpolator();

  int init();
  int write_to_file(const std::string filename);

  /**
   * @brief Adaptive point storage on host, keyed on signed multi-index.
   *        The grid is unbounded; the Python `point_data_full` view exports the full
   *        cell-key map as tuple keys.
   */
  std::unordered_map<key_t, point_data_t, key_hash_t> point_data;

  size_t get_n_cached_points() const { return point_data.size(); }
  size_t get_n_cached_hypercubes() const { return generated_hypercubes.size(); }

  /**
   * @brief Supporting points materialized since the last external cache flush.
   *
   * Mirrors the CPU adaptive interpolators so Python can persist only newly
   * computed points (append-only OBL cache) instead of rewriting the full map.
   * Keyed on cell_key_t to match the GPU point_data storage above.
   */
  std::unordered_set<key_t, key_hash_t> dirty_point_data;

  /**
   * @brief Evaluation epoch of each dirty (unflushed) supporting point: maps the
   * point multi-index key to the batch-interpolation index (≈ time step /
   * nonlinear iteration) at which it was first materialized. Mirrors the CPU
   * adaptive interpolators so Python can persist, next to the append-only delta,
   * when each point entered the OBL sampling. Cleared together with
   * dirty_point_data on every external cache flush.
   */
  std::unordered_map<key_t, uint64_t, key_hash_t> dirty_point_epochs;

  /**
   * @brief Number of batch interpolation calls processed so far. Incremented once
   * at the start of every evaluate_with_derivatives_d() (one nonlinear-iteration
   * assembly of this operator set) and used to stamp newly materialized points.
   */
  uint64_t eval_index = 0;

  /**
   * @brief Optional cap on the number of cached hypercubes (0 = unbounded; default).
   *
   * The 130 KiB hypercube payloads live in the device hashmap (hypercube_data_d),
   * tracked host-side by generated_hypercubes. The device open-addressed map has no
   * selective-erase API, so the cap is enforced coarsely: when the tracked count
   * exceeds hypercube_cap, the whole device map and host tracker are dropped (see
   * clear_hypercube_data) and rebuilt on demand from point_data — no flash, and the
   * persisted supporting-point cache is untouched. This is the device analog of the
   * CPU adaptive interpolator's LRU bound.
   */
  size_t hypercube_cap = 0;

  /**
   * @brief Bound the device hypercube cache to ~cap entries (0 = unbounded).
   */
  void set_hypercube_cap(size_t cap) { hypercube_cap = cap; }
  size_t get_hypercube_cap() const { return hypercube_cap; }

  /**
   * @brief Drop all device hypercube payloads + the host key tracker, releasing
   *        device memory. Rebuilt on demand from point_data (no flash). point_data
   *        and the on-disk cache are untouched. Call only between evaluate() calls.
   */
  void clear_hypercube_data();

protected:
  /**
   * @brief Get values of operators at a given supporting point. Cell-key-driven.
   */
  const point_data_t &get_point_data(const key_t &point_key);

  /**
   * @brief Compute the hypercube payload (N_VERTS × N_OPS doubles) for a given multi-index.
   */
  int generate_hypercube(const key_t &hc_key, value_t *new_hypercube);

  virtual int evaluate_with_derivatives_d(int n_states_idxs, double *state_d, int *states_idxs_d,
                                          double *values_d, double *derivatives_d) override;

  // **** HOST DATA ****
  /**
   * @brief Adaptive hypercube tracking on host — multi-indices already generated.
   */
  std::unordered_set<key_t, key_hash_t> generated_hypercubes;

  pinned_key_vector_t hypercubes_to_compute;        ///< Multi-indices of new hypercubes required per state (host)
  pinned_value_vector_t new_hypercube_data_buffer;  ///< Generated hypercube data buffer to copy to device (host)
  pinned_key_vector_t new_hypercube_index_buffer;   ///< Generated hypercube multi-indices buffer to copy to device (host)
  pinned_int_vector_t hashmap_expansion_needed;     ///< Flag showing if hashmap expansion needed (host)

  std::vector<value_t> new_hypercube_data;  ///< Data storage for generated hypercubes (host stable buffer)
  std::vector<key_t> new_hypercube_index;   ///< Multi-index storage for generated hypercubes (host stable buffer)

  // **** DEVICE DATA ****
  /**
   * @brief Adaptive hypercube storage on device. Keyed on a 64-bit content hash of
   * the multi-index. Theoretical collision rate ~N^2/2^64 per pair — negligible at
   * the cell counts realistic for GPU simulations (~10^6 unique cells → ~3e-8 per
   * pair). Cache growth past the prescribed window is unaffected.
   */
  gpu_hashmap_async::gpu_hash_map<value_t, N_VERTS * N_OPS> *hypercube_data_d;

  /**
   * @brief Per-state hypercube multi-index (device). Sentinel idx[0] == INT32_MIN means "no new hypercube".
   */
  thrust::device_vector<key_t> state_hc_keys_d;
  thrust::device_vector<value_t> new_hypercube_data_buffer_d;  ///< Generated hypercube data buffer (device)
  thrust::device_vector<key_t> new_hypercube_index_buffer_d;   ///< Generated hypercube multi-indices buffer (device)
  thrust::device_vector<int> hashmap_expansion_needed_d;

  cudaStream_t hypercube_generation_stream;
  cudaStream_t stage1_interpolation_stream;
};

#include "multilinear_adaptive_gpu_interpolator.tpp"

#endif /* C2EE8F5F_DDE3_4485_90BA_A6C32B67EB4D */
