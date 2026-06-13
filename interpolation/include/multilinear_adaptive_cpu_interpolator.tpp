#include <fstream>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <limits>
#include <algorithm>
#include <unordered_set>
#include <utility>
#include <cmath>

#include "multilinear_adaptive_cpu_interpolator.hpp"

template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::
    multilinear_adaptive_cpu_interpolator(operator_set_evaluator_iface *supporting_point_evaluator_,
                                          const std::vector<double> &axes_origin_,
                                          const std::vector<double> &axes_step_)
    : multilinear_interpolator_base<index_t, value_t, N_DIMS, N_OPS>(supporting_point_evaluator_, axes_origin_, axes_step_)

{
}

// ─── multi-index key utilities ──────────────────────────────────────────────────

template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
void multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::get_point_coordinates_from_key(
    const key_t &k, point_coordinates_t &coordinates) const
{
  for (uint8_t i = 0; i < N_DIMS; ++i)
  {
    coordinates[i] = this->axes_origin[i] + this->axes_step[i] * static_cast<double>(k.idx[i]);
  }
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
void multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::get_hypercube_vertex_keys(
    const key_t &hc_key, hypercube_vertex_keys_t &vertex_keys) const
{
  // Vertex layout matches the legacy MSB-first ordering in get_hypercube_points()
  // so an old integer hypercube index decoded into hc_key yields the same vertex set.
  static const uint32_t N_VERTS = (1u << N_DIMS);
  for (uint32_t v = 0; v < N_VERTS; ++v)
  {
    for (uint8_t i = 0; i < N_DIMS; ++i)
    {
      const int32_t bit = static_cast<int32_t>((v >> (N_DIMS - 1 - i)) & 1u);
      vertex_keys[v].idx[i] = hc_key.idx[i] + bit;
    }
  }
}

// ─── cache accessors ────────────────────────────────────────────────────────────

template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
const typename multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::point_data_t &
multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::get_point_data(const key_t &point_key)
{
  auto item = point_data.find(point_key);
  if (item != point_data.end())
    return item->second;

  if (this->timer) this->timer->node["body generation"].node["point generation"].start();
  point_data_t new_point;
  this->get_point_coordinates_from_key(point_key, this->new_point_coords);
  this->supporting_point_evaluator->evaluate(this->new_point_coords, this->new_operator_values);
  for (int op = 0; op < N_OPS; op++)
  {
    new_point[op] = this->new_operator_values[op];
    if (std::isnan(this->new_operator_values[op]))
    {
      printf("OBL generation warning: nan operator detected! Operator %d for point (", op);
      for (int a = 0; a < N_DIMS; a++)
      {
        printf("%lf, ", this->new_point_coords[a]);
      }
      printf(") is %lf\n", this->new_operator_values[op]);
    }
  }
  auto insert_result = point_data.emplace(point_key, new_point);
  this->n_points_used++;
  if (this->timer) this->timer->node["body generation"].node["point generation"].stop();
  return insert_result.first->second;
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
const typename multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::hypercube_data_t &
multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::get_hypercube_data(const key_t &hypercube_key)
{
  auto item = hypercube_data.find(hypercube_key);
  if (item != hypercube_data.end())
    return item->second;

  if (this->timer) this->timer->node["body generation"].start();
  hypercube_vertex_keys_t vertex_keys;
  hypercube_data_t new_hypercube;
  this->get_hypercube_vertex_keys(hypercube_key, vertex_keys);

  static const uint32_t N_VERTS = (1u << N_DIMS);
  for (uint32_t i = 0; i < N_VERTS; ++i)
  {
    const point_data_t &p_data = this->get_point_data(vertex_keys[i]);
    for (int op = 0; op < N_OPS; op++)
    {
      new_hypercube[i * N_OPS + op] = p_data[op];
    }
  }
  auto insert_result = hypercube_data.emplace(hypercube_key, new_hypercube);
  if (this->timer) this->timer->node["body generation"].stop();
  return insert_result.first->second;
}

// ─── hypercube-key export (multi-index view) ───────────────────────────────────

template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
std::vector<typename multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::key_t>
multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::get_hypercube_keys() const
{
  std::vector<key_t> keys;
  keys.reserve(hypercube_data.size());
  for (const auto &pair : hypercube_data)
    keys.push_back(pair.first);
  return keys;
}

// ─── single-point interpolation (multi-index path) ─────────────────────────────

template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
int multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::interpolate(
    const std::vector<double> &point, std::vector<double> &values)
{
  if (point.size() != N_DIMS)
  {
    printf("Inconsistence in interpolation! Point size = %zu should be equal to N_DIMS = %d\n", point.size(), N_DIMS);
  }

  double derivatives[N_OPS * N_DIMS];
  value_t axis_low[N_DIMS];
  value_t mult[N_DIMS];
  key_t hc_key;

  for (uint8_t i = 0; i < N_DIMS; ++i)
  {
    hc_key.idx[i] = get_axis_interval_index_low_mult_unbounded<value_t>(
        point[i],
        this->axes_origin_internal[i],
        this->axes_step_internal[i],
        this->axes_step_inv_internal[i],
        &axis_low[i], &mult[i]);
  }

  const hypercube_data_t &hc = this->get_hypercube_data(hc_key);

  interpolate_point_with_derivatives<value_t, N_DIMS, N_OPS>(
      point.data(), hc.data(), &axis_low[0], &mult[0],
      this->axes_step_inv_internal.data(),
      values.data(), &derivatives[0]);

  return 0;
}

// ─── batch materialization ─────────────────────────────────────────────────────

template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
void multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::materialize_missing_cache(
    const std::vector<key_t> &missing_hc)
{
  // Phase 2a: collect all unique missing supporting-point keys.
  std::vector<key_t> all_point_keys;
  static const uint32_t N_VERTS = (1u << N_DIMS);
  all_point_keys.reserve(missing_hc.size() * N_VERTS);

  hypercube_vertex_keys_t hc_vertices;
  for (size_t h = 0; h < missing_hc.size(); ++h)
  {
    this->get_hypercube_vertex_keys(missing_hc[h], hc_vertices);
    for (uint32_t v = 0; v < N_VERTS; ++v)
    {
      if (point_data.find(hc_vertices[v]) == point_data.end())
      {
        all_point_keys.push_back(hc_vertices[v]);
      }
    }
  }

  // Deduplicate
  std::sort(all_point_keys.begin(), all_point_keys.end());
  all_point_keys.erase(std::unique(all_point_keys.begin(), all_point_keys.end()),
                       all_point_keys.end());

  // Phase 2b: batch-evaluate missing points
  if (!all_point_keys.empty())
  {
    if (this->timer) this->timer->node["body generation"].node["point generation"].start();

    const size_t n_missing = all_point_keys.size();

    std::vector<double> batch_coords(n_missing * N_DIMS);
    point_coordinates_t local_coords(N_DIMS);
    for (size_t i = 0; i < n_missing; ++i)
    {
      this->get_point_coordinates_from_key(all_point_keys[i], local_coords);
      std::copy(local_coords.begin(), local_coords.end(),
                batch_coords.begin() + i * N_DIMS);
    }

    std::vector<double> batch_values(n_missing * N_OPS);
    this->supporting_point_evaluator->evaluate_batch(
        batch_coords, static_cast<int>(n_missing), batch_values, N_OPS);

    point_data.reserve(point_data.size() + n_missing);
    for (size_t i = 0; i < n_missing; ++i)
    {
      const key_t &pt_key = all_point_keys[i];
      point_data_t new_point;
      for (int op = 0; op < N_OPS; op++)
      {
        double val = batch_values[i * N_OPS + op];
        new_point[op] = val;
        if (std::isnan(val))
        {
          printf("OBL generation warning: nan operator detected! Operator %d for point (", op);
          for (int a = 0; a < N_DIMS; a++)
          {
            printf("%lf, ", batch_coords[i * N_DIMS + a]);
          }
          printf(") is %lf\n", val);
        }
      }
      point_data.emplace(pt_key, new_point);
      this->n_points_used++;
    }

    if (this->timer) this->timer->node["body generation"].node["point generation"].stop();
  }

  // Phase 2c: assemble missing hypercube payloads in parallel from the now-complete point cache.
  if (this->timer) this->timer->node["body generation"].node["hypercube generation"].start();

  std::vector<std::pair<key_t, hypercube_data_t>> new_hc_entries(missing_hc.size());

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int h = 0; h < static_cast<int>(missing_hc.size()); ++h)
  {
    hypercube_vertex_keys_t vk;
    this->get_hypercube_vertex_keys(missing_hc[h], vk);

    hypercube_data_t payload;
    for (uint32_t v = 0; v < N_VERTS; ++v)
    {
      const point_data_t &p_data = point_data.at(vk[v]);
      for (int op = 0; op < N_OPS; op++)
      {
        payload[v * N_OPS + op] = p_data[op];
      }
    }
    new_hc_entries[h] = std::make_pair(missing_hc[h], std::move(payload));
  }

  // Serial commit
  hypercube_data.reserve(hypercube_data.size() + missing_hc.size());
  for (auto &entry : new_hc_entries)
  {
    hypercube_data.emplace(entry.first, std::move(entry.second));
  }

  if (this->timer) this->timer->node["body generation"].node["hypercube generation"].stop();
}

// ─── batch interpolation (multi-index path) ────────────────────────────────────

template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
int multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::interpolate_with_derivatives(
    const std::vector<double> &points, const std::vector<int> &points_idxs,
    std::vector<double> &values, std::vector<double> &derivatives)
{
  const size_t n_cells = points_idxs.size();
  if (n_cells == 0)
    return 0;

  // Phase 1: parallel computation of multi-index hypercube key for every requested cell.
  if (this->timer) this->timer->node["body generation"].node["cache lookup"].start();

  std::vector<key_t> hc_keys(n_cells);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int p = 0; p < static_cast<int>(n_cells); p++)
  {
    int offset = points_idxs[p];
    key_t hc_key;
    for (uint8_t i = 0; i < N_DIMS; ++i)
    {
      const size_t coord_index = static_cast<size_t>(offset) * N_DIMS + i;
      hc_key.idx[i] = get_axis_interval_index_unbounded<value_t>(
          points[coord_index],
          this->axes_origin_internal[i],
          this->axes_step_inv_internal[i]);
    }
    hc_keys[p] = hc_key;
  }

  // Collect unique missing hypercube keys
  std::vector<key_t> unique_hc(hc_keys.begin(), hc_keys.end());
  std::sort(unique_hc.begin(), unique_hc.end());
  unique_hc.erase(std::unique(unique_hc.begin(), unique_hc.end()), unique_hc.end());

  std::vector<key_t> missing_hc;
  missing_hc.reserve(unique_hc.size());
  for (const auto &k : unique_hc)
  {
    if (hypercube_data.find(k) == hypercube_data.end())
      missing_hc.push_back(k);
  }

  if (this->timer) this->timer->node["body generation"].node["cache lookup"].stop();

  // Phase 2: materialize missing cache entries
  if (!missing_hc.empty())
  {
    if (this->timer) this->timer->node["body generation"].start();
    materialize_missing_cache(missing_hc);
    if (this->timer) this->timer->node["body generation"].stop();
  }

  // Phase 3: parallel read-only interpolation with thread-local workspace.
  static const uint32_t N_VERTS = (1u << N_DIMS);
  static const size_t workspace_size = (2 * N_VERTS - 1) * N_OPS;

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    std::vector<value_t> workspace(workspace_size);

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int p = 0; p < static_cast<int>(n_cells); p++)
    {
      int offset = points_idxs[p];
      const double *point = points.data() + offset * N_DIMS;
      double *vals = values.data() + offset * N_OPS;
      double *ders = derivatives.data() + offset * N_OPS * N_DIMS;

      value_t axis_low[N_DIMS];
      value_t mult[N_DIMS];
      for (int i = 0; i < N_DIMS; ++i)
      {
        get_axis_interval_index_low_mult_unbounded<value_t>(
            point[i],
            this->axes_origin_internal[i],
            this->axes_step_internal[i],
            this->axes_step_inv_internal[i],
            &axis_low[i], &mult[i]);
      }

      const hypercube_data_t &hc = hypercube_data.at(hc_keys[p]);

      interpolate_point_with_derivatives_ws<value_t, N_DIMS, N_OPS>(
          point, hc.data(), axis_low, mult,
          this->axes_step_inv_internal.data(),
          workspace.data(),
          vals, ders);
    }
  }

  return 0;
}
