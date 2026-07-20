#include <fstream>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <thrust/host_vector.h>

#include "multilinear_static_gpu_interpolator.hpp"
#include "gpu_tools.h"

#define USE_THREAD_PER_OPERATOR_KERNEL

template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
__global__ void
multilinear_static_interpolate_thread_per_state_kernel(const unsigned int n_states_idxs, const int *states_idxs_d, const double *states_d,
                                                       const int *axis_points_d, const index_t *axis_hypercube_mult_d,
                                                       const value_t *axis_min_d, const value_t *axis_max_d,
                                                       const value_t *axis_step_d, const value_t *axis_step_inv_d,
                                                       const value_t *hypercube_data_d,
                                                       double *values_d, double *derivatives_d);

template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
__global__ void
multilinear_static_interpolate_thread_per_operator_kernel(const unsigned int n_states_idxs, const int *states_idxs_d, const double *states_d,
                                                          const int *axis_points_d, const index_t *axis_hypercube_mult_d,
                                                          const value_t *axis_min_d, const value_t *axis_max_d,
                                                          const value_t *axis_step_d, const value_t *axis_step_inv_d,
                                                          const value_t *hypercube_data_d,
                                                          double *values_d, double *derivatives_d);

template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
multilinear_static_gpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::multilinear_static_gpu_interpolator(operator_set_evaluator_iface *supporting_point_evaluator,
                                                                                                          const std::vector<double> &axes_origin,
                                                                                                          const std::vector<double> &axes_step,
                                                                                                          const std::vector<int> &axes_points)
    : multilinear_gpu_interpolator_base<index_t, value_t, N_DIMS, N_OPS>(supporting_point_evaluator, axes_origin, axes_step, axes_points)

{
  this->n_points_used = this->n_points_total;
  int min_job_size;
#ifdef USE_THREAD_PER_OPERATOR_KERNEL
  // this->kernel_block_size = get_kernel_thread_block_size(
  //     multilinear_static_interpolate_thread_per_operator_kernel<index_t, value_t, N_DIMS, N_OPS>, min_job_size);
  //  cout << "multilinear static gpu interpolator thread_per_operator kernel block size is " << this->kernel_block_size
  //      << ", minimum " << min_job_size / N_OPS << " states needed to reach full occupancy" << std::endl;
  // custom choice of block size results in better performance
  this->kernel_block_size = 128;

#else
  this->kernel_block_size = get_kernel_thread_block_size(multilinear_static_interpolate_thread_per_state_kernel<index_t, value_t, N_DIMS, N_OPS>, min_job_size);
  // the block size from get_kernel_thread_block_size sometimes leads to underperforming up to 10x times...
  // so set it here to 64
  this->kernel_block_size = 128;
  std::cout << "multilinear static gpu interpolator _thread_per_state kernel block size is " << this->kernel_block_size
       << ", minimum " << min_job_size << " states needed to reach full occupancy" << std::endl;

#endif
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
int multilinear_static_gpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::init()
{
  // Static interpolator strictly requires a bounded grid (dense vector storage). The
  // base ctor no longer throws on overflow (the adaptive variant tolerates it), so
  // validate here before allocating - mirrors the CPU static interpolator.
  {
    double int_type_max = static_cast<double>(std::numeric_limits<index_t>::max());
    if (this->n_points_total_fp > int_type_max)
    {
      throw std::range_error(
          "static GPU interpolator requires a bounded grid; n_points_total (" +
          std::to_string(this->n_points_total_fp) +
          ") exceeds index_t range (" + std::to_string(int_type_max) +
          "). Use the adaptive variant for unbounded grids.");
    }
  }

  // evaluate supporting point data unless it was already assigned via Python
  if (point_data.size() == 0)
  {
    std::cout << "Computing " << this->n_points_total << " supporting points for static storage..." << std::endl;
    point_data.resize(this->n_points_total);

    for (auto i = 0; i < this->n_points_total; i++)
    {
      // let generator fill the vector
      this->get_point_coordinates(i, this->new_point_coords);
      this->supporting_point_evaluator->evaluate(this->new_point_coords, this->new_operator_values);
      // and move the data to the new place
      std::copy_n(std::make_move_iterator(this->new_operator_values.begin()), N_OPS, point_data[i].begin());
    }
  }
  // populate hypercube data on host - unlike CPU interpolators it is just a flat array
  thrust::host_vector<value_t> hypercube_data;
  hypercube_points_index_t points;

  // compute the total number of hypercubes
  uint64_t n_hypercubes_total = 1;
  for (int i = 0; i < N_DIMS; i++)
  {
    n_hypercubes_total *= this->axes_points[i] - 1;
  }
  hypercube_data.resize(n_hypercubes_total * this->N_VERTS * N_OPS);
  // fill each hypercube directly on the device with corresponding data from point_data
  for (auto i = 0; i < n_hypercubes_total; i++)
  {
    this->get_hypercube_points(i, points);

    for (auto j = 0; j < this->N_VERTS; ++j)
    {
      thrust::copy_n(point_data[points[j]].begin(), N_OPS, hypercube_data.begin() + N_OPS * (i * this->N_VERTS + j));
    }
  }
  // and copy it to device
  hypercube_data_d = hypercube_data;
  return 0;
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
int multilinear_static_gpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::
    evaluate_with_derivatives_d(int n_states_idxs, double *states_d, int *states_idxs_d,
                                double *values_d, double *derivatives_d)
{
  this->timer->start();
  this->timer->node["gpu interpolation"].start();

#ifdef USE_THREAD_PER_OPERATOR_KERNEL

  multilinear_static_interpolate_thread_per_operator_kernel<index_t, value_t, N_DIMS, N_OPS>
      KERNEL_1D_THREAD(n_states_idxs * N_OPS, this->kernel_block_size)(n_states_idxs, states_idxs_d, states_d,
                                                                       thrust::raw_pointer_cast(this->axes_points_d.data()), thrust::raw_pointer_cast(this->axis_hypercube_mult_d.data()),
                                                                       thrust::raw_pointer_cast(this->axes_origin_d.data()), thrust::raw_pointer_cast(this->axes_max_d.data()),
                                                                       thrust::raw_pointer_cast(this->axes_step_d.data()), thrust::raw_pointer_cast(this->axes_step_inv_d.data()),
                                                                       thrust::raw_pointer_cast(hypercube_data_d.data()),
                                                                       values_d, derivatives_d);
#else
  multilinear_static_interpolate_thread_per_state_kernel<index_t, value_t, N_DIMS, N_OPS>
      KERNEL_1D_THREAD(n_states_idxs, this->kernel_block_size)(n_states_idxs, states_idxs_d, states_d,
                                                               thrust::raw_pointer_cast(this->axes_points_d.data()), thrust::raw_pointer_cast(this->axis_hypercube_mult_d.data()),
                                                               thrust::raw_pointer_cast(this->axes_origin_d.data()), thrust::raw_pointer_cast(this->axes_max_d.data()),
                                                               thrust::raw_pointer_cast(this->axes_step_d.data()), thrust::raw_pointer_cast(this->axes_step_inv_d.data()),
                                                               thrust::raw_pointer_cast(hypercube_data_d.data()),
                                                               values_d, derivatives_d);
#endif
  this->timer->node["gpu interpolation"].stop();
  this->n_interpolations += N_OPS * n_states_idxs;
  this->timer->stop();
  return 0;
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
__global__ void
multilinear_static_interpolate_thread_per_state_kernel(const unsigned int n_states_idxs, const int *states_idxs_d, const double *states_d,
                                                       const int *axis_points_d, const index_t *axis_hypercube_mult_d,
                                                       const value_t *axis_min_d, const value_t *axis_max_d,
                                                       const value_t *axis_step_d, const value_t *axis_step_inv_d,
                                                       const value_t *hypercube_data_d,
                                                       double *values_d, double *derivatives_d)
{
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  if (i > n_states_idxs - 1)
    return;

  index_t state_idx = states_idxs_d[i];
  static const uint32_t N_VERTS = 1 << N_DIMS;

  index_t hypercube_idx = 0;
  value_t axis_low[N_DIMS];
  value_t mult[N_DIMS];

  for (int i = 0; i < N_DIMS; ++i)
  {
    unsigned int axis_idx = get_axis_interval_index_low_mult<value_t>(states_d[state_idx * N_DIMS + i],
                                                                      axis_min_d[i], axis_max_d[i],
                                                                      axis_step_d[i], axis_step_inv_d[i], axis_points_d[i],
                                                                      &axis_low[i], &mult[i]);
    hypercube_idx += axis_idx * axis_hypercube_mult_d[i];
  }

  interpolate_point_with_derivatives<value_t, N_DIMS, N_OPS>(states_d + state_idx * N_DIMS, hypercube_data_d + hypercube_idx * N_VERTS * N_OPS,
                                                             &axis_low[0], &mult[0], axis_step_inv_d,
                                                             values_d + state_idx * N_OPS,
                                                             derivatives_d + state_idx * N_OPS * N_DIMS);
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
__global__ void
multilinear_static_interpolate_thread_per_operator_kernel(const unsigned int n_states_idxs, const int *states_idxs_d, const double *states_d,
                                                          const int *axis_points_d, const index_t *axis_hypercube_mult_d,
                                                          const value_t *axis_min_d, const value_t *axis_max_d,
                                                          const value_t *axis_step_d, const value_t *axis_step_inv_d,
                                                          const value_t *hypercube_data_d,
                                                          double *values_d, double *derivatives_d)
{
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  const unsigned operator_idx = i % N_OPS;
  const unsigned state_idx_idx = i / N_OPS;
  if (state_idx_idx > n_states_idxs - 1)
    return;

  index_t state_idx = states_idxs_d[state_idx_idx];
  static const uint32_t N_VERTS = 1 << N_DIMS;

  index_t hypercube_idx = 0;
  value_t axis_low[N_DIMS];
  value_t mult[N_DIMS];

  for (int i = 0; i < N_DIMS; ++i)
  {
    unsigned int axis_idx = get_axis_interval_index_low_mult<value_t>(states_d[state_idx * N_DIMS + i],
                                                                      axis_min_d[i], axis_max_d[i],
                                                                      axis_step_d[i], axis_step_inv_d[i], axis_points_d[i],
                                                                      &axis_low[i], &mult[i]);
    hypercube_idx += axis_idx * axis_hypercube_mult_d[i];
  }

  interpolate_operator_with_derivatives<value_t, N_DIMS, N_OPS>(states_d + state_idx * N_DIMS, hypercube_data_d + hypercube_idx * N_VERTS * N_OPS,
                                                                &axis_low[0], &mult[0], axis_step_inv_d,
                                                                operator_idx,
                                                                values_d + state_idx * N_OPS,
                                                                derivatives_d + state_idx * N_OPS * N_DIMS);
}
