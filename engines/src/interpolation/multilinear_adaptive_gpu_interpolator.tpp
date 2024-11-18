#include <fstream>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <limits>
#include <algorithm>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include "multilinear_adaptive_gpu_interpolator.hpp"
#include "gpu_tools.h"

#define USE_THREAD_PER_OPERATOR_KERNEL
#define HYPERCUBE_BUFFER_SIZE 100

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
__global__ void multilinear_adaptive3_check_hypercube_ready_kernel(const unsigned int n_states_idxs, const int *states_idxs_d, const double *states_d,
                                                                   const int *axis_points_d, const index_t *axis_hypercube_mult_d,
                                                                   const value_t *axis_min_d, const value_t *axis_max_d,
                                                                   const value_t *axis_step_inv_d,
                                                                   gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d,
                                                                   index_t *hypercubes_to_compute);

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS, bool FIRST_STAGE>
__global__ void multilinear_adaptive_interpolate_thread_per_state_stages_kernel(const unsigned int n_states_idxs, const int *states_idxs_d, const double *states_d,
                                                                                const int *axis_points_d, const index_t *axis_hypercube_mult_d,
                                                                                const value_t *axis_min_d, const value_t *axis_max_d,
                                                                                const value_t *axis_step_d, const value_t *axis_step_inv_d,
                                                                                gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d,
                                                                                index_t *state_markers, double *values_d, double *derivatives_d);

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS, bool FIRST_STAGE>
__global__ void multilinear_adaptive_interpolate_thread_per_operator_stages_kernel(const unsigned int n_states_idxs, const int *states_idxs_d, const double *states_d,
                                                                                   const int *axis_points_d, const index_t *axis_hypercube_mult_d,
                                                                                   const value_t *axis_min_d, const value_t *axis_max_d,
                                                                                   const value_t *axis_step_d, const value_t *axis_step_inv_d,
                                                                                   gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d,
                                                                                   index_t *state_markers, double *values_d, double *derivatives_d);

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
__global__ void
add_hypercubes_to_hashmap(const unsigned int n_new_hypercubes, const index_t *new_hypercube_index_d,
                          const value_t *new_hypercube_data_d,
                          gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d);

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
__global__ void
check_if_hashmap_expansion_needed(const float threshold, gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d, int *expansion_needed);

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
multilinear_adaptive_gpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::multilinear_adaptive_gpu_interpolator(operator_set_evaluator_iface *supporting_point_evaluator,
                                                                                                              const std::vector<int> &axes_points,
                                                                                                              const std::vector<double> &axes_min,
                                                                                                              const std::vector<double> &axes_max)
    : multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>(supporting_point_evaluator, axes_points, axes_min, axes_max)

{
  int min_job_size;
#ifdef USE_THREAD_PER_OPERATOR_KERNEL
  // this->kernel_block_size = get_kernel_thread_block_size(
  //     multilinear_adaptive_interpolate_thread_per_operator_kernel<index_t, value_t, N_DIMS, N_OPS>, min_job_size);
  // cout << "multilinear adaptive gpu interpolator thread_per_operator kernel block size is " << this->kernel_block_size
  //      << ", minimum " << min_job_size / N_OPS << " states needed to reach full occupancy" << std::endl;
  // custom choice of block size results in better performance
  this->kernel_block_size = 128;
#else
  // this->kernel_block_size = get_kernel_thread_block_size(multilinear_adaptive_interpolate_thread_per_state_s1_kernel<index_t, value_t, N_DIMS, N_OPS>, min_job_size);
  // cout << "multilinear adaptive gpu interpolator _thread_per_state kernel block size is " << this->kernel_block_size
  //      << ", minimum " << min_job_size << " states needed to reach full occupancy" << std::endl;
  // custom choice of block size results in better performance
  this->kernel_block_size = 128;
#endif
  // set initial hasmap size to ~ 50 Mb
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
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
multilinear_adaptive_gpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::~multilinear_adaptive_gpu_interpolator()
{
  gpu_hashmap_async::delete_hashmap(hypercube_data_d);
  //gpu_hashmap_async::delete_hashmap(point_data_d);
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
int multilinear_adaptive_gpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::init()
{
  // we could populate hypercube data here if supporting point was assigned via Python
  // but its not obvious how to translate computed point indexes into hypercube indexes
  // so do nothing here and only save time on point generation, letting hypercube generation happen from scratch

  // std::vector<value_t> hypercube_data;
  // if (insert_data(hypercube_data_d, 0, hypercube_data.data()))
  //   printf("Hashmap insert fail. Larger size is needed\n");

  return 0;
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
int multilinear_adaptive_gpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::write_to_file(const std::string filename)
{
  return 0;
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
const typename multilinear_adaptive_gpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::point_data_t &
multilinear_adaptive_gpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::get_point_data(const index_t point_index)
{
  auto item = point_data.find(point_index);
  typename multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::point_data_t new_point;
  if (item == point_data.end())
  {
    //this->timer->node["gpu interpolation"].node["hypercube generation"].node["point generation"].start_gpu();
    this->get_point_coordinates(point_index, this->new_point_coords);
    this->supporting_point_evaluator->evaluate(this->new_point_coords, this->new_operator_values);
    // check operator values
    for (int op = 0; op < N_OPS; op++)
    {
      new_point[op] = this->new_operator_values[op];
      if (isnan(this->new_operator_values[op]))
      {
        printf("OBL generation warning: nan operator detected! Operator %d for point (", op);
        for (int a = 0; a < N_DIMS; a++)
        {
          printf("%lf, ", this->new_point_coords[a]);
        }
        printf(") is %lf\n", this->new_operator_values[op]);
      }
    }
    point_data[point_index] = new_point;
    this->n_points_used++;
    //this->timer->node["gpu interpolation"].node["hypercube generation"].node["point generation"].stop_gpu();
    return point_data[point_index];
  }
  else
    return item->second;
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
int multilinear_adaptive_gpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::
    generate_hypercube(index_t hypercube_idx, value_t *new_hypercube)
{
  // printf("Hypercube %d generation start\n", hypercube_idx);
  //this->timer->node["gpu interpolation"].node["hypercube generation"].start_gpu();
  hypercube_points_index_t points;
  this->get_hypercube_points(hypercube_idx, points);

  for (auto j = 0; j < this->N_VERTS; ++j)
  {
    // obtain point data and copy it to hypercube data
    const typename multilinear_adaptive_gpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::point_data_t &p_data =
        this->get_point_data(points[j]);

    for (int op = 0; op < N_OPS; op++)
    {
      new_hypercube[j * N_OPS + op] = p_data[op];
    }
  }

  //this->timer->node["gpu interpolation"].node["hypercube generation"].stop_gpu();
  // printf("Hypercube %d generation done\n", hypercube_idx);
  // print_device_vector_kernel(new_hypercube.data(), N_VERTS * N_OPS, "Hypercube data");
  return 0;
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
int multilinear_adaptive_gpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::
    evaluate_with_derivatives_d(int n_states_idxs, double *states_d, int *states_idxs_d,
                                double *values_d, double *derivatives_d)
{
  this->timer->start();
  this->timer->node["gpu interpolation"].start();
  static int detailed_timing = 0;
  // this->timer->node["gpu interpolation"].node["round1"].start_gpu();
  // cudaProfilerStart();

  state_markers_d.resize(n_states_idxs);
  hypercubes_to_compute.resize(n_states_idxs);

  if (detailed_timing)
    this->timer->node["gpu interpolation"].node["check"].start_gpu();
  multilinear_adaptive3_check_hypercube_ready_kernel<index_t, value_t, N_DIMS, N_OPS>
      KERNEL_1D_THREAD(n_states_idxs, this->kernel_block_size)(n_states_idxs, states_idxs_d, states_d,
                                                               thrust::raw_pointer_cast(this->axes_points_d.data()),
                                                               thrust::raw_pointer_cast(this->axis_hypercube_mult_d.data()),
                                                               thrust::raw_pointer_cast(this->axes_min_d.data()),
                                                               thrust::raw_pointer_cast(this->axes_max_d.data()),
                                                               thrust::raw_pointer_cast(this->axes_step_inv_d.data()),
                                                               hypercube_data_d,
                                                               thrust::raw_pointer_cast(state_markers_d.data()));
  if (detailed_timing)
  {
    this->timer->node["gpu interpolation"].node["check"].stop_gpu();
  }
  //cudaDeviceSynchronize();
  //  printf("Fist stage: %d, second stage %d, total %d\n", n_first_stage, n_second_stage, n_states_idxs);
  //int n_new_hypercubes = thrust::unique(thrust::device, hypercubes_to_compute_d.begin(), hypercubes_to_compute_d.end()) - hypercubes_to_compute_d.begin();

  // for (int i = 0; i < n_first_stage; i++)
  //   std::cout << "FS[" << i << "] = " << states_idxs_first_stage_d[i] << std::endl;

  // for (int i = 0; i < n_second_stage; i++)
  //   std::cout << "SS[" << i << "] = " << states_idxs_second_stage_d[i] << std::endl;

  // Run the first stage stream

  if (detailed_timing)
    this->timer->node["gpu interpolation"].node["1st stage + gen"].start_gpu();
  if (detailed_timing)
    this->timer->node["gpu interpolation"].node["1st stage"].start_gpu(stage1_interpolation_stream);

#ifdef USE_THREAD_PER_OPERATOR_KERNEL

  multilinear_adaptive_interpolate_thread_per_operator_stages_kernel<index_t, value_t, N_DIMS, N_OPS, true>
      KERNEL_1D_THREAD_STREAM(n_states_idxs * N_OPS, this->kernel_block_size,
                              stage1_interpolation_stream)(n_states_idxs, states_idxs_d, states_d,
                                                           thrust::raw_pointer_cast(this->axes_points_d.data()), thrust::raw_pointer_cast(this->axis_hypercube_mult_d.data()),
                                                           thrust::raw_pointer_cast(this->axes_min_d.data()), thrust::raw_pointer_cast(this->axes_max_d.data()),
                                                           thrust::raw_pointer_cast(this->axes_step_d.data()), thrust::raw_pointer_cast(this->axes_step_inv_d.data()),
                                                           hypercube_data_d,
                                                           thrust::raw_pointer_cast(state_markers_d.data()), values_d, derivatives_d);
#else
  multilinear_adaptive_interpolate_thread_per_state_stages_kernel<index_t, value_t, N_DIMS, N_OPS, true>
      KERNEL_1D_THREAD_STREAM(n_states_idxs,
                              this->kernel_block_size,
                              stage1_interpolation_stream)(n_states_idxs, states_idxs_d, states_d,
                                                           thrust::raw_pointer_cast(this->axes_points_d.data()), thrust::raw_pointer_cast(this->axis_hypercube_mult_d.data()),
                                                           thrust::raw_pointer_cast(this->axes_min_d.data()), thrust::raw_pointer_cast(this->axes_max_d.data()),
                                                           thrust::raw_pointer_cast(this->axes_step_d.data()), thrust::raw_pointer_cast(this->axes_step_inv_d.data()),
                                                           hypercube_data_d,
                                                           thrust::raw_pointer_cast(state_markers_d.data()), values_d, derivatives_d);
#endif

  if (detailed_timing)
    this->timer->node["gpu interpolation"].node["1st stage"].stop_gpu(stage1_interpolation_stream);
  // Organize hypercube generation stream
  if (detailed_timing)
    this->timer->node["gpu interpolation"].node["gen"].start_gpu(hypercube_generation_stream);

  // 1. Copy marker array (with hypercube indexes required for stage 2) to host
  cudaMemcpyAsync(thrust::raw_pointer_cast(hypercubes_to_compute.data()),
                  thrust::raw_pointer_cast(state_markers_d.data()), n_states_idxs * sizeof(index_t),
                  cudaMemcpyDeviceToHost, hypercube_generation_stream);

  // 2. Wait till copy completes
  cudaStreamSynchronize(hypercube_generation_stream);
  int new_hypercubes_generated = 0;
  // 3. Generate all required hypercubes
  for (int i = 0, h = 0; i < n_states_idxs; i++)
  {
    // generate if not -1 (marker for 1 stage interpolation) and not already generated
    if (hypercubes_to_compute[i] != -1 && !generated_hypercubes.count(hypercubes_to_compute[i]))
    {
      generated_hypercubes.insert(hypercubes_to_compute[i]);
      new_hypercube_index[h] = hypercubes_to_compute[i];
      generate_hypercube(hypercubes_to_compute[i], new_hypercube_data.data() + h * N_OPS * N_VERTS);
      new_hypercubes_generated = 1;
      //printf("Generated hypercube #%d: %u\n", h, hypercubes_to_compute[i]);

      h++;
    }

    // copy if buffer is filled or the last iteration is reached and buffer is not empty
    if (h == HYPERCUBE_BUFFER_SIZE || (i == (n_states_idxs - 1) && h))
    {
      //printf("Flushing buffer size %d!\n", h);

#if 1 //CUDA12
      thrust::copy(new_hypercube_data.begin(), new_hypercube_data.end(), new_hypercube_data_buffer.begin());
      thrust::copy(new_hypercube_index.begin(), new_hypercube_index.end(), new_hypercube_index_buffer.begin());
#else
      memcpy(new_hypercube_data_buffer.data(), new_hypercube_data.data(), h * sizeof(value_t) * N_OPS * N_VERTS);
      memcpy(new_hypercube_index_buffer.data(), new_hypercube_index.data(), h * sizeof(index_t));
#endif

      cudaMemcpyAsync(thrust::raw_pointer_cast(new_hypercube_data_buffer_d.data()),
                      thrust::raw_pointer_cast(new_hypercube_data_buffer.data()), h * sizeof(value_t) * N_OPS * N_VERTS,
                      cudaMemcpyHostToDevice, hypercube_generation_stream);
      cudaMemcpyAsync(thrust::raw_pointer_cast(new_hypercube_index_buffer_d.data()),
                      thrust::raw_pointer_cast(new_hypercube_index_buffer.data()), h * sizeof(index_t),
                      cudaMemcpyHostToDevice, hypercube_generation_stream);

      add_hypercubes_to_hashmap<index_t, value_t, N_DIMS, N_OPS>
          KERNEL_1D_THREAD_STREAM(h * N_OPS * N_VERTS,
                                  this->kernel_block_size,
                                  hypercube_generation_stream)(h, thrust::raw_pointer_cast(new_hypercube_index_buffer_d.data()),
                                                               thrust::raw_pointer_cast(new_hypercube_data_buffer_d.data()),
                                                               hypercube_data_d);
      // we have to wait before the previous data portion is processed on device before overwriting it

      CUDA_CHECK_RETURN(cudaStreamSynchronize(hypercube_generation_stream));
      h = 0;
    }
  }
  if (detailed_timing)
    this->timer->node["gpu interpolation"].node["gen"].stop_gpu(hypercube_generation_stream);
  if (detailed_timing)
    this->timer->node["gpu interpolation"].node["wait for 1st stage"].start_gpu();

  // Now wait until both streams are done
  cudaDeviceSynchronize();

  if (detailed_timing)
    this->timer->node["gpu interpolation"].node["wait for 1st stage"].stop_gpu();
  if (detailed_timing)
    this->timer->node["gpu interpolation"].node["1st stage + gen"].stop_gpu();

  if (detailed_timing)
  {
    this->timer->node["gpu interpolation"].node["expansion"].start_gpu();
  }

  // check if current hashmap is 70% full, then expansion initiated
  check_if_hashmap_expansion_needed<index_t, value_t, N_DIMS, N_OPS>
      <<<1, 1, 0>>>(0.7, hypercube_data_d, thrust::raw_pointer_cast(hashmap_expansion_needed_d.data()));

  cudaMemcpy(thrust::raw_pointer_cast(hashmap_expansion_needed.data()),
             thrust::raw_pointer_cast(hashmap_expansion_needed_d.data()), sizeof(int),
             cudaMemcpyDeviceToHost);

  // increase the hashmap size 2 times
  if (hashmap_expansion_needed[0])
  {
    hypercube_data_d = expand_hashmap(hypercube_data_d, 2);
  }
  if (detailed_timing)
  {
    this->timer->node["gpu interpolation"].node["expansion"].stop_gpu();
  }

  if (detailed_timing)
    this->timer->node["gpu interpolation"].node["2nd stage"].start_gpu();

  // Launch the 2nd stage only if needed
  if (new_hypercubes_generated)
  {
    //printf("Starting stage 2\n");
#ifdef USE_THREAD_PER_OPERATOR_KERNEL

    multilinear_adaptive_interpolate_thread_per_operator_stages_kernel<index_t, value_t, N_DIMS, N_OPS, false>
        KERNEL_1D_THREAD(n_states_idxs * N_OPS,
                         this->kernel_block_size)(n_states_idxs, states_idxs_d, states_d,
                                                  thrust::raw_pointer_cast(this->axes_points_d.data()), thrust::raw_pointer_cast(this->axis_hypercube_mult_d.data()),
                                                  thrust::raw_pointer_cast(this->axes_min_d.data()), thrust::raw_pointer_cast(this->axes_max_d.data()),
                                                  thrust::raw_pointer_cast(this->axes_step_d.data()), thrust::raw_pointer_cast(this->axes_step_inv_d.data()),
                                                  hypercube_data_d,
                                                  thrust::raw_pointer_cast(state_markers_d.data()), values_d, derivatives_d);
#else
    multilinear_adaptive_interpolate_thread_per_state_stages_kernel<index_t, value_t, N_DIMS, N_OPS, false>
        KERNEL_1D_THREAD(n_states_idxs,
                         this->kernel_block_size)(n_states_idxs, states_idxs_d, states_d,
                                                  thrust::raw_pointer_cast(this->axes_points_d.data()), thrust::raw_pointer_cast(this->axis_hypercube_mult_d.data()),
                                                  thrust::raw_pointer_cast(this->axes_min_d.data()), thrust::raw_pointer_cast(this->axes_max_d.data()),
                                                  thrust::raw_pointer_cast(this->axes_step_d.data()), thrust::raw_pointer_cast(this->axes_step_inv_d.data()),
                                                  hypercube_data_d,
                                                  thrust::raw_pointer_cast(state_markers_d.data()), values_d, derivatives_d);
#endif

    if (detailed_timing)
      this->timer->node["gpu interpolation"].node["2nd stage"].stop_gpu();
    // for (int i = 0; i < new_end - hypercubes_to_compute_d.begin(); i++)
    //   std::cout << "H[" << i << "] = " << hypercubes_to_compute_d[i] << std::endl;
  }

  this->timer->node["gpu interpolation"].stop();
  this->n_interpolations += N_OPS * n_states_idxs;
  this->timer->stop();
  return 0;
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
__global__ void
multilinear_adaptive3_check_hypercube_ready_kernel(const unsigned int n_states_idxs, const int *states_idxs_d, const double *states_d,
                                                   const int *axis_points_d, const index_t *axis_hypercube_mult_d,
                                                   const value_t *axis_min_d, const value_t *axis_max_d,
                                                   const value_t *axis_step_inv_d,
                                                   gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d,
                                                   index_t *hypercubes_to_compute)
{
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  if (i > n_states_idxs - 1)
    return;

  index_t state_idx = states_idxs_d[i];

  index_t hypercube_idx = 0;
  value_t *hypercube_data;

  for (int i = 0; i < N_DIMS; ++i)
  {
    unsigned int axis_idx = get_axis_interval_index<value_t>(states_d[state_idx * N_DIMS + i],
                                                             axis_min_d[i], axis_max_d[i],
                                                             axis_step_inv_d[i], axis_points_d[i]);
    hypercube_idx += axis_idx * axis_hypercube_mult_d[i];
  }
  if (lookup_data(hypercube_data_d, hypercube_idx, &hypercube_data))
  {
    // hypercube not available, fill its index
    hypercubes_to_compute[i] = hypercube_idx;
    //enqueue(hypercube_generation_queue, hypercube_idx);
    //printf("Thread %d: Requesting hypercube %d\n", i, hypercube_idx);
  }
  else
  {
    // hypercube is available, mark state to be computed next
    hypercubes_to_compute[i] = -1;
  }
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS, bool FIRST_STAGE>
__global__ void
multilinear_adaptive_interpolate_thread_per_state_stages_kernel(const unsigned int n_states_idxs, const int *states_idxs_d, const double *states_d,
                                                                const int *axis_points_d, const index_t *axis_hypercube_mult_d,
                                                                const value_t *axis_min_d, const value_t *axis_max_d,
                                                                const value_t *axis_step_d, const value_t *axis_step_inv_d,
                                                                gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d,
                                                                index_t *state_markers, double *values_d, double *derivatives_d)
{
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  if (i > n_states_idxs - 1)
    return;

  if (FIRST_STAGE)
  {
    // do not process states corresponding to new hypercubes
    if (state_markers[i] != -1)
      return;
  }
  else
  {
    // do not process states NOT corresponding to new hypercubes
    if (state_markers[i] == -1)
      return;
  }

  index_t state_idx = states_idxs_d[i];

  index_t hypercube_idx = 0;
  value_t axis_low[N_DIMS];
  value_t mult[N_DIMS];
  value_t *hypercube_data;

  for (int i = 0; i < N_DIMS; ++i)
  {
    unsigned int axis_idx = get_axis_interval_index_low_mult<value_t>(states_d[state_idx * N_DIMS + i],
                                                                      axis_min_d[i], axis_max_d[i],
                                                                      axis_step_d[i], axis_step_inv_d[i], axis_points_d[i],
                                                                      &axis_low[i], &mult[i]);
    hypercube_idx += axis_idx * axis_hypercube_mult_d[i];
  }
  if (lookup_data(hypercube_data_d, hypercube_idx, &hypercube_data))
  {
    if (FIRST_STAGE)
    {
      printf("Thread %d error s1: Requesting hypercube %d\n", i, hypercube_idx);
    }
    else
    {
      printf("Thread %d error s2: Requesting hypercube %d\n", i, hypercube_idx);
    }
  }
  else
  {
    //printf("Thread %d: Found hypercube %d\n", i, hypercube_idx);
    interpolate_point_with_derivatives<value_t, N_DIMS, N_OPS>(states_d + state_idx * N_DIMS, hypercube_data,
                                                               &axis_low[0], &mult[0], axis_step_inv_d,
                                                               values_d + state_idx * N_OPS,
                                                               derivatives_d + state_idx * N_OPS * N_DIMS);
  }
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS, bool FIRST_STAGE>
__global__ void
multilinear_adaptive_interpolate_thread_per_operator_stages_kernel(const unsigned int n_states_idxs, const int *states_idxs_d, const double *states_d,
                                                                   const int *axis_points_d, const index_t *axis_hypercube_mult_d,
                                                                   const value_t *axis_min_d, const value_t *axis_max_d,
                                                                   const value_t *axis_step_d, const value_t *axis_step_inv_d,
                                                                   gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d,
                                                                   index_t *state_markers, double *values_d, double *derivatives_d)
{
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  const unsigned operator_idx = i % N_OPS;
  const unsigned state_idx_idx = i / N_OPS;
  if (state_idx_idx > n_states_idxs - 1)
    return;

  

  if (FIRST_STAGE)
  {
    // do not process states corresponding to new hypercubes
    if (state_markers[state_idx_idx] != -1)
      return;
  }
  else
  {
    // do not process states NOT corresponding to new hypercubes
    if (state_markers[state_idx_idx] == -1)
      return;
  }
  
  index_t state_idx = states_idxs_d[state_idx_idx];
  index_t hypercube_idx = 0;
  value_t axis_low[N_DIMS];
  value_t mult[N_DIMS];
  value_t *hypercube_data;

  for (int i = 0; i < N_DIMS; ++i)
  {
    unsigned int axis_idx = get_axis_interval_index_low_mult<value_t>(states_d[state_idx * N_DIMS + i],
                                                                      axis_min_d[i], axis_max_d[i],
                                                                      axis_step_d[i], axis_step_inv_d[i], axis_points_d[i],
                                                                      &axis_low[i], &mult[i]);
    hypercube_idx += axis_idx * axis_hypercube_mult_d[i];
  }

  if (lookup_data(hypercube_data_d, hypercube_idx, &hypercube_data))
  {
    if (FIRST_STAGE)
    {
      printf("Thread %d error s1: Requesting hypercube %d\n", i, hypercube_idx);
    }
    else
    {
      int idx = hypercube_idx;
      printf("Thread %d error s2: Requesting hypercube %d\n", i, idx);
    }
  }
  else
  {
    //printf("Thread %d: Found hypercube %d\n", i, hypercube_idx);
    interpolate_operator_with_derivatives<value_t, N_DIMS, N_OPS>(states_d + state_idx * N_DIMS, hypercube_data,
                                                                  &axis_low[0], &mult[0], axis_step_inv_d,
                                                                  operator_idx,
                                                                  values_d + state_idx * N_OPS,
                                                                  derivatives_d + state_idx * N_OPS * N_DIMS);
  }
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
__global__ void
add_hypercubes_to_hashmap(const unsigned int n_new_hypercubes, const index_t *new_hypercube_index_d,
                          const value_t *new_hypercube_data_d,
                          gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d)
{
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  static const uint16_t N_VERTS = 1 << N_DIMS;

  const unsigned hypercube_index = i / (N_OPS * N_VERTS);

  if (hypercube_index > n_new_hypercubes - 1)
    return;

  if (hypercube_data_d->occupied + n_new_hypercubes > hypercube_data_d->size)
  {
    if (i == 0)
    {
      printf("Should not have happened: hashmap overflow occured! Decrease hashmap expansion threshold below %lf\n", hypercube_data_d->occupied / hypercube_data_d->size);
      hypercube_data_d->occupied += n_new_hypercubes;
    }
    return;
  }
  else if (i == 0)
  {
    hypercube_data_d->occupied += n_new_hypercubes;
  }

  const unsigned vector_index = i % (N_OPS * N_VERTS);

  if (insert_vector_element(hypercube_data_d, new_hypercube_index_d[hypercube_index], &new_hypercube_data_d[hypercube_index * N_OPS * N_VERTS], vector_index))
    return;
  //printf("thread %d insertion failed\n", threadIdx.x);
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
__global__ void
check_if_hashmap_expansion_needed(const float threshold, gpu_hashmap_async::gpu_hash_map<value_t, (1 << N_DIMS) * N_OPS> *hypercube_data_d, int *expansion_needed)
{

  if (hypercube_data_d->occupied > hypercube_data_d->size * threshold)
  {
    *expansion_needed = 1;
  }
  else
  {
    *expansion_needed = 0;
  }
}