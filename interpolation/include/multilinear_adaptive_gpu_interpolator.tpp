#include <fstream>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <limits>
#include <algorithm>
#include <climits>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include "multilinear_adaptive_gpu_interpolator.hpp"
#include "gpu_tools.h"

#define USE_THREAD_PER_OPERATOR_KERNEL
#define HYPERCUBE_BUFFER_SIZE 100

// ─── kernel forward declarations ───────────────────────────────────────────────

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
__global__ void multilinear_adaptive3_check_hypercube_ready_kernel(
    const unsigned int n_states_idxs, const int *states_idxs_d, const double *states_d,
    const value_t *axis_min_d, const value_t *axis_step_inv_d,
    gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d,
    cell_key_t<N_DIMS> *state_hc_keys);

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS, bool FIRST_STAGE>
__global__ void multilinear_adaptive_interpolate_thread_per_state_stages_kernel(
    const unsigned int n_states_idxs, const int *states_idxs_d, const double *states_d,
    const value_t *axis_min_d, const value_t *axis_step_d, const value_t *axis_step_inv_d,
    gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d,
    const cell_key_t<N_DIMS> *state_hc_keys, double *values_d, double *derivatives_d);

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS, bool FIRST_STAGE>
__global__ void multilinear_adaptive_interpolate_thread_per_operator_stages_kernel(
    const unsigned int n_states_idxs, const int *states_idxs_d, const double *states_d,
    const value_t *axis_min_d, const value_t *axis_step_d, const value_t *axis_step_inv_d,
    gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d,
    const cell_key_t<N_DIMS> *state_hc_keys, double *values_d, double *derivatives_d);

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
__global__ void add_hypercubes_to_hashmap(
    const unsigned int n_new_hypercubes, const cell_key_t<N_DIMS> *new_hypercube_keys_d,
    const value_t *new_hypercube_data_d,
    gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d);

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
__global__ void check_if_hashmap_expansion_needed(
    const float threshold,
    gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d,
    int *expansion_needed);

// ─── constructor / destructor / init ───────────────────────────────────────────

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
multilinear_adaptive_gpu_interpolator<value_t, N_DIMS, N_OPS>::multilinear_adaptive_gpu_interpolator(
    operator_set_evaluator_iface *supporting_point_evaluator,
    const std::vector<double> &axes_origin,
    const std::vector<double> &axes_step)
    : multilinear_gpu_interpolator_base<uint64_t, value_t, N_DIMS, N_OPS>(supporting_point_evaluator, axes_origin, axes_step)
{
  this->kernel_block_size = 128;

  // initial hashmap size: ~50 MB
  int max_hypercube_capacity = 50 * 1024 * 1024 / (N_VERTS * N_OPS * sizeof(value_t));
  hypercube_data_d = gpu_hashmap_async::create_hashmap<value_t, N_VERTS * N_OPS>(max_hypercube_capacity);

  cudaStreamCreate(&hypercube_generation_stream);
  cudaStreamCreate(&stage1_interpolation_stream);

  new_hypercube_data_buffer.resize(HYPERCUBE_BUFFER_SIZE * N_OPS * N_VERTS);
  new_hypercube_index_buffer.resize(HYPERCUBE_BUFFER_SIZE);
  new_hypercube_data_buffer_d.resize(HYPERCUBE_BUFFER_SIZE * N_OPS * N_VERTS);
  new_hypercube_index_buffer_d.resize(HYPERCUBE_BUFFER_SIZE);
  new_hypercube_data.resize(HYPERCUBE_BUFFER_SIZE * N_OPS * N_VERTS);
  new_hypercube_index.resize(HYPERCUBE_BUFFER_SIZE);
  hashmap_expansion_needed.resize(1);
  hashmap_expansion_needed_d.resize(1);
  axis_overflow_count_host.resize(1);
  axis_overflow_count_d.resize(1);
}

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
multilinear_adaptive_gpu_interpolator<value_t, N_DIMS, N_OPS>::~multilinear_adaptive_gpu_interpolator()
{
  gpu_hashmap_async::delete_hashmap(hypercube_data_d);
}

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
int multilinear_adaptive_gpu_interpolator<value_t, N_DIMS, N_OPS>::init()
{
  return 0;
}

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
void multilinear_adaptive_gpu_interpolator<value_t, N_DIMS, N_OPS>::clear_hypercube_data()
{
  // Drop the device hypercube map and the host key tracker; both are rebuilt on
  // demand from point_data (no flash). Mirrors the destructor + constructor so the
  // fresh map starts at the same ~50 MB capacity. point_data and the on-disk cache
  // are untouched.
  gpu_hashmap_async::delete_hashmap(hypercube_data_d);
  int max_hypercube_capacity = 50 * 1024 * 1024 / (N_VERTS * N_OPS * sizeof(value_t));
  hypercube_data_d = gpu_hashmap_async::create_hashmap<value_t, N_VERTS * N_OPS>(max_hypercube_capacity);
  generated_hypercubes.clear();
}

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
int multilinear_adaptive_gpu_interpolator<value_t, N_DIMS, N_OPS>::write_to_file(const std::string filename)
{
  return 0;
}

// ─── supporting-point access (host) ────────────────────────────────────────────

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
const typename multilinear_adaptive_gpu_interpolator<value_t, N_DIMS, N_OPS>::point_data_t &
multilinear_adaptive_gpu_interpolator<value_t, N_DIMS, N_OPS>::get_point_data(const key_t &point_key)
{
  auto item = point_data.find(point_key);
  if (item != point_data.end())
    return item->second;

  // Compute physical coordinates from multi-index, evaluate, store, return.
  point_data_t new_point;
  for (uint8_t i = 0; i < N_DIMS; ++i)
    this->new_point_coords[i] = this->axes_origin[i] + this->axes_step[i] * static_cast<double>(point_key.idx[i]);
  this->supporting_point_evaluator->evaluate(this->new_point_coords, this->new_operator_values);
  for (int op = 0; op < N_OPS; op++)
  {
    new_point[op] = this->new_operator_values[op];
    if (isnan(this->new_operator_values[op]))
    {
      printf("OBL generation warning: nan operator detected! Operator %d for point (", op);
      for (int a = 0; a < N_DIMS; a++)
        printf("%lf, ", this->new_point_coords[a]);
      printf(") is %lf\n", this->new_operator_values[op]);
    }
  }
  auto insert_result = point_data.emplace(point_key, new_point);
  // Mark for append-only cache flush after this new point is materialized.
  dirty_point_data.insert(point_key);
  // Stamp the point with the current batch-interpolation (nonlinear-iteration) index.
  dirty_point_epochs[point_key] = eval_index;
  this->n_points_used++;
  return insert_result.first->second;
}

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
int multilinear_adaptive_gpu_interpolator<value_t, N_DIMS, N_OPS>::generate_hypercube(
    const key_t &hc_key, value_t *new_hypercube)
{
  // Build the N_VERTS supporting-point keys from the lower corner.
  static const uint32_t NV = (1u << N_DIMS);
  for (uint32_t v = 0; v < NV; ++v)
  {
    key_t pt_key;
    for (uint8_t i = 0; i < N_DIMS; ++i)
    {
      const int32_t bit = static_cast<int32_t>((v >> (N_DIMS - 1 - i)) & 1u);
      pt_key.idx[i] = hc_key.idx[i] + bit;
    }
    const point_data_t &p_data = this->get_point_data(pt_key);
    for (int op = 0; op < N_OPS; op++)
      new_hypercube[v * N_OPS + op] = p_data[op];
  }
  return 0;
}

// ─── batch evaluation ──────────────────────────────────────────────────────────

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
int multilinear_adaptive_gpu_interpolator<value_t, N_DIMS, N_OPS>::evaluate_with_derivatives_d(
    int n_states_idxs, double *states_d, int *states_idxs_d,
    double *values_d, double *derivatives_d)
{
  this->timer->start();
  this->timer->node["gpu interpolation"].start();
  static int detailed_timing = 0;

  // One batch interpolation call == one nonlinear-iteration assembly of this operator
  // set: advance the epoch stamp applied to points materialized during this call.
  eval_index++;

  // Bound the device hypercube cache: when it exceeds hypercube_cap, drop all cached
  // hypercubes (host tracker + device map) and let this batch rebuild only its working
  // set from point_data (no flash). Coarse vs the CPU LRU because the device
  // open-addressed map has no selective-erase API; the per-batch working set is
  // bounded by n_states_idxs, so size hypercube_cap >> cells.
  if (hypercube_cap != 0 && generated_hypercubes.size() > hypercube_cap)
    clear_hypercube_data();

  state_hc_keys_d.resize(n_states_idxs);
  hypercubes_to_compute.resize(n_states_idxs);

  // Reset the per-batch int32 cell-index overflow accumulator; the check kernel
  // atomicAdds into it, and it is read back off the hot path near the end of the batch.
  axis_overflow_count_d[0] = 0;

  if (detailed_timing)
    this->timer->node["gpu interpolation"].node["check"].start_gpu();
  multilinear_adaptive3_check_hypercube_ready_kernel<value_t, N_DIMS, N_OPS>
      KERNEL_1D_THREAD(n_states_idxs, this->kernel_block_size)(
          n_states_idxs, states_idxs_d, states_d,
          thrust::raw_pointer_cast(this->axes_origin_d.data()),
          thrust::raw_pointer_cast(this->axes_step_inv_d.data()),
          hypercube_data_d,
          thrust::raw_pointer_cast(state_hc_keys_d.data()),
          thrust::raw_pointer_cast(axis_overflow_count_d.data()));
  if (detailed_timing)
    this->timer->node["gpu interpolation"].node["check"].stop_gpu();

  if (detailed_timing)
    this->timer->node["gpu interpolation"].node["1st stage + gen"].start_gpu();
  if (detailed_timing)
    this->timer->node["gpu interpolation"].node["1st stage"].start_gpu(stage1_interpolation_stream);

#ifdef USE_THREAD_PER_OPERATOR_KERNEL
  multilinear_adaptive_interpolate_thread_per_operator_stages_kernel<value_t, N_DIMS, N_OPS, true>
      KERNEL_1D_THREAD_STREAM(n_states_idxs * N_OPS, this->kernel_block_size, stage1_interpolation_stream)(
          n_states_idxs, states_idxs_d, states_d,
          thrust::raw_pointer_cast(this->axes_origin_d.data()),
          thrust::raw_pointer_cast(this->axes_step_d.data()),
          thrust::raw_pointer_cast(this->axes_step_inv_d.data()),
          hypercube_data_d,
          thrust::raw_pointer_cast(state_hc_keys_d.data()),
          values_d, derivatives_d);
#else
  multilinear_adaptive_interpolate_thread_per_state_stages_kernel<value_t, N_DIMS, N_OPS, true>
      KERNEL_1D_THREAD_STREAM(n_states_idxs, this->kernel_block_size, stage1_interpolation_stream)(
          n_states_idxs, states_idxs_d, states_d,
          thrust::raw_pointer_cast(this->axes_origin_d.data()),
          thrust::raw_pointer_cast(this->axes_step_d.data()),
          thrust::raw_pointer_cast(this->axes_step_inv_d.data()),
          hypercube_data_d,
          thrust::raw_pointer_cast(state_hc_keys_d.data()),
          values_d, derivatives_d);
#endif

  if (detailed_timing)
    this->timer->node["gpu interpolation"].node["1st stage"].stop_gpu(stage1_interpolation_stream);

  if (detailed_timing)
    this->timer->node["gpu interpolation"].node["gen"].start_gpu(hypercube_generation_stream);

  // Copy hypercube multi-indices to host (DTOH).
  cudaMemcpyAsync(thrust::raw_pointer_cast(hypercubes_to_compute.data()),
                  thrust::raw_pointer_cast(state_hc_keys_d.data()),
                  n_states_idxs * sizeof(key_t),
                  cudaMemcpyDeviceToHost, hypercube_generation_stream);

  cudaStreamSynchronize(hypercube_generation_stream);
  int new_hypercubes_generated = 0;

  for (int i = 0, h = 0; i < n_states_idxs; i++)
  {
    const key_t &hc_key = hypercubes_to_compute[i];
    // sentinel: idx[0] == INT32_MIN means "already cached, skip"
    const bool is_sentinel = (hc_key.idx[0] == INT32_MIN);
    if (!is_sentinel && !generated_hypercubes.count(hc_key))
    {
      generated_hypercubes.insert(hc_key);
      new_hypercube_index[h] = hc_key;
      generate_hypercube(hc_key, new_hypercube_data.data() + h * N_OPS * N_VERTS);
      new_hypercubes_generated = 1;
      h++;
    }

    if (h == HYPERCUBE_BUFFER_SIZE || (i == (n_states_idxs - 1) && h))
    {
      thrust::copy(new_hypercube_data.begin(), new_hypercube_data.end(), new_hypercube_data_buffer.begin());
      thrust::copy(new_hypercube_index.begin(), new_hypercube_index.end(), new_hypercube_index_buffer.begin());

      cudaMemcpyAsync(thrust::raw_pointer_cast(new_hypercube_data_buffer_d.data()),
                      thrust::raw_pointer_cast(new_hypercube_data_buffer.data()),
                      h * sizeof(value_t) * N_OPS * N_VERTS,
                      cudaMemcpyHostToDevice, hypercube_generation_stream);
      cudaMemcpyAsync(thrust::raw_pointer_cast(new_hypercube_index_buffer_d.data()),
                      thrust::raw_pointer_cast(new_hypercube_index_buffer.data()),
                      h * sizeof(key_t),
                      cudaMemcpyHostToDevice, hypercube_generation_stream);

      add_hypercubes_to_hashmap<value_t, N_DIMS, N_OPS>
          KERNEL_1D_THREAD_STREAM(h * N_OPS * N_VERTS, this->kernel_block_size, hypercube_generation_stream)(
              h, thrust::raw_pointer_cast(new_hypercube_index_buffer_d.data()),
              thrust::raw_pointer_cast(new_hypercube_data_buffer_d.data()),
              hypercube_data_d);

      CUDA_CHECK_RETURN(cudaStreamSynchronize(hypercube_generation_stream));
      h = 0;
    }
  }
  if (detailed_timing)
    this->timer->node["gpu interpolation"].node["gen"].stop_gpu(hypercube_generation_stream);
  if (detailed_timing)
    this->timer->node["gpu interpolation"].node["wait for 1st stage"].start_gpu();

  cudaDeviceSynchronize();

  if (detailed_timing)
    this->timer->node["gpu interpolation"].node["wait for 1st stage"].stop_gpu();
  if (detailed_timing)
    this->timer->node["gpu interpolation"].node["1st stage + gen"].stop_gpu();

  if (detailed_timing)
    this->timer->node["gpu interpolation"].node["expansion"].start_gpu();

  check_if_hashmap_expansion_needed<value_t, N_DIMS, N_OPS>
      <<<1, 1, 0>>>(0.7, hypercube_data_d, thrust::raw_pointer_cast(hashmap_expansion_needed_d.data()));

  cudaMemcpy(thrust::raw_pointer_cast(hashmap_expansion_needed.data()),
             thrust::raw_pointer_cast(hashmap_expansion_needed_d.data()), sizeof(int),
             cudaMemcpyDeviceToHost);

  if (hashmap_expansion_needed[0])
    hypercube_data_d = gpu_hashmap_async::expand_hashmap(hypercube_data_d, 2);

  if (detailed_timing)
    this->timer->node["gpu interpolation"].node["expansion"].stop_gpu();
  if (detailed_timing)
    this->timer->node["gpu interpolation"].node["2nd stage"].start_gpu();

  if (new_hypercubes_generated)
  {
#ifdef USE_THREAD_PER_OPERATOR_KERNEL
    multilinear_adaptive_interpolate_thread_per_operator_stages_kernel<value_t, N_DIMS, N_OPS, false>
        KERNEL_1D_THREAD(n_states_idxs * N_OPS, this->kernel_block_size)(
            n_states_idxs, states_idxs_d, states_d,
            thrust::raw_pointer_cast(this->axes_origin_d.data()),
            thrust::raw_pointer_cast(this->axes_step_d.data()),
            thrust::raw_pointer_cast(this->axes_step_inv_d.data()),
            hypercube_data_d,
            thrust::raw_pointer_cast(state_hc_keys_d.data()),
            values_d, derivatives_d);
#else
    multilinear_adaptive_interpolate_thread_per_state_stages_kernel<value_t, N_DIMS, N_OPS, false>
        KERNEL_1D_THREAD(n_states_idxs, this->kernel_block_size)(
            n_states_idxs, states_idxs_d, states_d,
            thrust::raw_pointer_cast(this->axes_origin_d.data()),
            thrust::raw_pointer_cast(this->axes_step_d.data()),
            thrust::raw_pointer_cast(this->axes_step_inv_d.data()),
            hypercube_data_d,
            thrust::raw_pointer_cast(state_hc_keys_d.data()),
            values_d, derivatives_d);
#endif
    if (detailed_timing)
      this->timer->node["gpu interpolation"].node["2nd stage"].stop_gpu();
  }

  this->timer->node["gpu interpolation"].stop();

  // Off the hot path: read back this batch's int32 cell-index overflow tally and hand
  // it to the unified reporter (blocking copy also guarantees the check kernel, which
  // ran on the default stream, has completed). Same warning path as the CPU backend.
  cudaMemcpy(thrust::raw_pointer_cast(axis_overflow_count_host.data()),
             thrust::raw_pointer_cast(axis_overflow_count_d.data()), sizeof(int),
             cudaMemcpyDeviceToHost);
  this->report_axis_index_overflows(static_cast<uint64_t>(axis_overflow_count_host[0]),
                                    this->axes_origin, this->axes_step);

  this->n_interpolations += N_OPS * n_states_idxs;
  this->timer->stop();
  return 0;
}

// ─── kernel definitions ────────────────────────────────────────────────────────

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
__global__ void multilinear_adaptive3_check_hypercube_ready_kernel(
    const unsigned int n_states_idxs, const int *states_idxs_d, const double *states_d,
    const value_t *axis_min_d, const value_t *axis_step_inv_d,
    gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d,
    cell_key_t<N_DIMS> *state_hc_keys, int *overflow_counter_d)
{
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > n_states_idxs - 1)
    return;

  int state_idx = states_idxs_d[i];

  // Per-thread int32 cell-index overflow tally, accumulated branchlessly into a
  // register. This is the canonical index computation for the batch (mirrors CPU
  // Phase 1); the interpolate kernels re-derive the index without counting so cells
  // are tallied once. Reduced to the device accumulator below only if nonzero — a
  // warp-coherent, almost-always-not-taken branch, so effectively free on the hot path.
  int cell_overflows = 0;
  cell_key_t<N_DIMS> hc_key;
  for (int d = 0; d < N_DIMS; ++d)
  {
    hc_key.idx[d] = get_axis_interval_index_unbounded<value_t>(
        states_d[state_idx * N_DIMS + d], axis_min_d[d], axis_step_inv_d[d], &cell_overflows);
  }
  if (cell_overflows)
    atomicAdd(overflow_counter_d, cell_overflows);

  value_t *hypercube_data;
  if (lookup_data(hypercube_data_d, gpu_hashmap_async::key_from_cell<N_DIMS>(hc_key), &hypercube_data))
  {
    // hypercube not available; write multi-index for stage-2 generation
    state_hc_keys[i] = hc_key;
  }
  else
  {
    // hypercube cached; mark sentinel
    state_hc_keys[i].idx[0] = INT32_MIN;
  }
}

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS, bool FIRST_STAGE>
__global__ void multilinear_adaptive_interpolate_thread_per_state_stages_kernel(
    const unsigned int n_states_idxs, const int *states_idxs_d, const double *states_d,
    const value_t *axis_min_d, const value_t *axis_step_d, const value_t *axis_step_inv_d,
    gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d,
    const cell_key_t<N_DIMS> *state_hc_keys, double *values_d, double *derivatives_d)
{
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > n_states_idxs - 1)
    return;

  const bool is_sentinel = (state_hc_keys[i].idx[0] == INT32_MIN);
  if (FIRST_STAGE)
  {
    // Stage 1: process states whose hypercube is already cached
    if (!is_sentinel)
      return;
  }
  else
  {
    // Stage 2: process states whose hypercube was just generated
    if (is_sentinel)
      return;
  }

  int state_idx = states_idxs_d[i];

  value_t axis_low[N_DIMS];
  value_t mult[N_DIMS];
  cell_key_t<N_DIMS> hc_key;
  for (int d = 0; d < N_DIMS; ++d)
  {
    hc_key.idx[d] = get_axis_interval_index_low_mult_unbounded<value_t>(
        states_d[state_idx * N_DIMS + d],
        axis_min_d[d], axis_step_d[d], axis_step_inv_d[d],
        &axis_low[d], &mult[d]);
  }

  value_t *hypercube_data;
  // hashmap is keyed by the 63-bit content hash of cell_key_t (key_from_cell);
  // this GPU path is collision-PRONE, not collision-safe — unlike the CPU
  // std::unordered_map<cell_key_t> path. See gpu_hashmap_async.h for the tradeoff.
  if (lookup_data(hypercube_data_d, gpu_hashmap_async::key_from_cell<N_DIMS>(hc_key), &hypercube_data))
  {
    if (FIRST_STAGE)
      printf("Thread %u error s1: hypercube missing\n", i);
    else
      printf("Thread %u error s2: hypercube missing\n", i);
  }
  else
  {
    interpolate_point_with_derivatives<value_t, N_DIMS, N_OPS>(
        states_d + state_idx * N_DIMS, hypercube_data,
        &axis_low[0], &mult[0], axis_step_inv_d,
        values_d + state_idx * N_OPS,
        derivatives_d + state_idx * N_OPS * N_DIMS);
  }
}

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS, bool FIRST_STAGE>
__global__ void multilinear_adaptive_interpolate_thread_per_operator_stages_kernel(
    const unsigned int n_states_idxs, const int *states_idxs_d, const double *states_d,
    const value_t *axis_min_d, const value_t *axis_step_d, const value_t *axis_step_inv_d,
    gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d,
    const cell_key_t<N_DIMS> *state_hc_keys, double *values_d, double *derivatives_d)
{
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned operator_idx = i % N_OPS;
  const unsigned state_idx_idx = i / N_OPS;
  if (state_idx_idx > n_states_idxs - 1)
    return;

  const bool is_sentinel = (state_hc_keys[state_idx_idx].idx[0] == INT32_MIN);
  if (FIRST_STAGE)
  {
    if (!is_sentinel)
      return;
  }
  else
  {
    if (is_sentinel)
      return;
  }

  int state_idx = states_idxs_d[state_idx_idx];

  value_t axis_low[N_DIMS];
  value_t mult[N_DIMS];
  cell_key_t<N_DIMS> hc_key;
  for (int d = 0; d < N_DIMS; ++d)
  {
    hc_key.idx[d] = get_axis_interval_index_low_mult_unbounded<value_t>(
        states_d[state_idx * N_DIMS + d],
        axis_min_d[d], axis_step_d[d], axis_step_inv_d[d],
        &axis_low[d], &mult[d]);
  }

  value_t *hypercube_data;
  // hashmap is keyed by the 63-bit content hash of cell_key_t (key_from_cell);
  // this GPU path is collision-PRONE, not collision-safe — unlike the CPU
  // std::unordered_map<cell_key_t> path. See gpu_hashmap_async.h for the tradeoff.
  if (lookup_data(hypercube_data_d, gpu_hashmap_async::key_from_cell<N_DIMS>(hc_key), &hypercube_data))
  {
    if (FIRST_STAGE)
      printf("Thread %u error s1: hypercube missing\n", i);
    else
      printf("Thread %u error s2: hypercube missing\n", i);
  }
  else
  {
    interpolate_operator_with_derivatives<value_t, N_DIMS, N_OPS>(
        states_d + state_idx * N_DIMS, hypercube_data,
        &axis_low[0], &mult[0], axis_step_inv_d,
        operator_idx,
        values_d + state_idx * N_OPS,
        derivatives_d + state_idx * N_OPS * N_DIMS);
  }
}

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
__global__ void add_hypercubes_to_hashmap(
    const unsigned int n_new_hypercubes, const cell_key_t<N_DIMS> *new_hypercube_keys_d,
    const value_t *new_hypercube_data_d,
    gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d)
{
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  static const uint32_t NV = (1u << N_DIMS);
  const unsigned hypercube_index = i / (N_OPS * NV);

  if (hypercube_index > n_new_hypercubes - 1)
    return;

  if (hypercube_data_d->occupied + n_new_hypercubes > hypercube_data_d->size)
  {
    if (i == 0)
    {
      printf("Should not have happened: hashmap overflow occured! Decrease hashmap expansion threshold below %f\n",
             (double)hypercube_data_d->occupied / hypercube_data_d->size);
      hypercube_data_d->occupied += n_new_hypercubes;
    }
    return;
  }
  else if (i == 0)
  {
    hypercube_data_d->occupied += n_new_hypercubes;
  }

  const unsigned vector_index = i % (N_OPS * NV);
  const gpu_hashmap_async::hash_key_t k =
      gpu_hashmap_async::key_from_cell<N_DIMS>(new_hypercube_keys_d[hypercube_index]);
  if (insert_vector_element(hypercube_data_d, k,
                            &new_hypercube_data_d[hypercube_index * N_OPS * NV], vector_index))
    return;
}

template <typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
__global__ void check_if_hashmap_expansion_needed(
    const float threshold,
    gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d,
    int *expansion_needed)
{
  if (hypercube_data_d->occupied > hypercube_data_d->size * threshold)
    *expansion_needed = 1;
  else
    *expansion_needed = 0;
}
