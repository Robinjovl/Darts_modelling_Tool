#include <fstream>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <limits>
#include <algorithm>
#include <unordered_set>
#include <utility>

#include "multilinear_adaptive_cpu_interpolator.hpp"

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::
    multilinear_adaptive_cpu_interpolator(operator_set_evaluator_iface *supporting_point_evaluator_,
                                          const std::vector<int> &axes_points_,
                                          const std::vector<double> &axes_min_,
                                          const std::vector<double> &axes_max_)
    : multilinear_interpolator_base<index_t, value_t, N_DIMS, N_OPS>(supporting_point_evaluator_, axes_points_, axes_min_, axes_max_)

{
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
const typename multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::point_data_t &
multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::get_point_data(const index_t point_index)
{
  auto item = point_data.find(point_index);
  typename multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::point_data_t new_point;
  if (item == point_data.end())
  {
    if (this->timer) this->timer->node["body generation"].node["point generation"].start();
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
    // Mark for append-only cache flush after this new point is materialized.
    dirty_point_data.insert(point_index);
    this->n_points_used++;
    if (this->timer) this->timer->node["body generation"].node["point generation"].stop();
    return point_data[point_index];
  }
  else
    return item->second;
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
std::vector<index_t> multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::get_hypercube_indexes() const
{
  std::vector<index_t> keys;
  keys.reserve(hypercube_data.size());
  for (const auto& pair : hypercube_data) {
    keys.push_back(pair.first);
  }
  return keys;
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
const typename multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::hypercube_data_t &
multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::get_hypercube_data(const index_t hypercube_index)
{
  auto item = hypercube_data.find(hypercube_index);
  if (item == hypercube_data.end())
  {
    if (this->timer) this->timer->node["body generation"].start();
    hypercube_points_index_t points;
    typename multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::hypercube_data_t new_hypercube;

    this->get_hypercube_points(hypercube_index, points);

    for (uint32_t i = 0; i < this->N_VERTS; ++i)
    {
      // obtain point data and copy it to hypercube data
      const typename multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::point_data_t &p_data = this->get_point_data(points[i]);
      for (int op = 0; op < N_OPS; op++)
      {
        new_hypercube[i * N_OPS + op] = p_data[op];
      }
    }
    hypercube_data[hypercube_index] = new_hypercube;
    if (this->timer) this->timer->node["body generation"].stop();
    return hypercube_data[hypercube_index];
  }
  else
    return item->second;
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
void multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::materialize_missing_cache(
    const std::vector<index_t> &missing_hc)
{
  // Phase 2a: Collect all unique missing supporting points across all missing hypercubes.
  //           Use a flat sorted vector for deduplication — cheaper than unordered_set for
  //           moderate counts and avoids per-element heap allocation overhead.
  std::vector<index_t> all_point_indices;
  all_point_indices.reserve(missing_hc.size() * this->N_VERTS);

  // Temporary storage for hypercube vertex expansions, reused per hypercube
  hypercube_points_index_t hc_points;

  for (size_t h = 0; h < missing_hc.size(); ++h)
  {
    this->get_hypercube_points(missing_hc[h], hc_points);
    for (uint32_t v = 0; v < this->N_VERTS; ++v)
    {
      if (point_data.find(hc_points[v]) == point_data.end())
      {
        all_point_indices.push_back(hc_points[v]);
      }
    }
  }

  // Deduplicate missing point indices
  std::sort(all_point_indices.begin(), all_point_indices.end());
  all_point_indices.erase(std::unique(all_point_indices.begin(), all_point_indices.end()),
                          all_point_indices.end());

  // Phase 2b: Evaluate all missing points through the evaluator using a single batch call.
  //           The evaluator's evaluate_batch() may dispatch to a multiprocessing pool
  //           (ParallelEvaluator) or fall back to serial per-point evaluate() (default).
  if (!all_point_indices.empty())
  {
    if (this->timer) this->timer->node["body generation"].node["point generation"].start();

    const size_t n_missing = all_point_indices.size();

    // Pack flat coordinate array for all missing points
    std::vector<double> batch_coords(n_missing * N_DIMS);
    point_coordinates_t local_coords(N_DIMS);
    for (size_t i = 0; i < n_missing; ++i)
    {
      this->get_point_coordinates(all_point_indices[i], local_coords);
      std::copy(local_coords.begin(), local_coords.end(),
                batch_coords.begin() + i * N_DIMS);
    }

    // Single batch call — one GIL acquire, one Python method invocation
    std::vector<double> batch_values(n_missing * N_OPS);
    this->supporting_point_evaluator->evaluate_batch(
        batch_coords, static_cast<int>(n_missing), batch_values, N_OPS);

    // Unpack results into point_data map
    point_data.reserve(point_data.size() + n_missing);
    for (size_t i = 0; i < n_missing; ++i)
    {
      const index_t pt_idx = all_point_indices[i];
      point_data_t new_point;
      for (int op = 0; op < N_OPS; op++)
      {
        double val = batch_values[i * N_OPS + op];
        new_point[op] = val;
        if (isnan(val))
        {
          printf("OBL generation warning: nan operator detected! Operator %d for point (", op);
          for (int a = 0; a < N_DIMS; a++)
          {
            printf("%lf, ", batch_coords[i * N_DIMS + a]);
          }
          printf(") is %lf\n", val);
        }
      }
      point_data[pt_idx] = new_point;
      // Mark for append-only cache flush after this new point is materialized.
      dirty_point_data.insert(pt_idx);
      this->n_points_used++;
    }

    if (this->timer) this->timer->node["body generation"].node["point generation"].stop();
  }

  // Phase 2c: Assemble missing hypercube payloads in parallel from the now-complete point cache.
  //           Each thread writes to its own slot in a dense temporary vector — no shared map mutation.
  if (this->timer) this->timer->node["body generation"].node["hypercube generation"].start();

  // Build dense temporary vector of (index, payload) pairs
  std::vector<std::pair<index_t, hypercube_data_t>> new_hc_entries(missing_hc.size());

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int h = 0; h < static_cast<int>(missing_hc.size()); ++h)
  {
    hypercube_points_index_t pts;
    this->get_hypercube_points(missing_hc[h], pts);

    hypercube_data_t payload;
    for (uint32_t v = 0; v < this->N_VERTS; ++v)
    {
      // point_data is read-only here — all needed points were materialized above
      const point_data_t &p_data = point_data.at(pts[v]);
      for (int op = 0; op < N_OPS; op++)
      {
        payload[v * N_OPS + op] = p_data[op];
      }
    }
    new_hc_entries[h] = std::make_pair(missing_hc[h], std::move(payload));
  }

  // Serial commit into hypercube_data — avoids concurrent map mutation
  hypercube_data.reserve(hypercube_data.size() + missing_hc.size());
  for (auto &entry : new_hc_entries)
  {
    hypercube_data[entry.first] = std::move(entry.second);
  }

  if (this->timer) this->timer->node["body generation"].node["hypercube generation"].stop();
}

template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
int multilinear_adaptive_cpu_interpolator<index_t, value_t, N_DIMS, N_OPS>::interpolate_with_derivatives(
    const std::vector<double> &points, const std::vector<int> &points_idxs,
    std::vector<double> &values, std::vector<double> &derivatives)
{
  const size_t n_cells = points_idxs.size();
  if (n_cells == 0)
    return 0;

  // ════════════════════════════════════════════════════════════════════════
  // Phase 1: Parallel computation of hypercube index for every requested cell.
  //          Pure arithmetic on read-only axis parameters — no shared state mutation.
  // ════════════════════════════════════════════════════════════════════════
  if (this->timer) this->timer->node["body generation"].node["cache lookup"].start();

  std::vector<index_t> hc_idxs(n_cells);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int p = 0; p < static_cast<int>(n_cells); p++)
  {
    index_t offset = points_idxs[p];
    index_t hc_idx = 0;
    for (uint8_t i = 0; i < N_DIMS; ++i)
    {
      index_t index = offset * static_cast<index_t>(N_DIMS) + static_cast<index_t>(i);
      int axis_idx = get_axis_interval_index<value_t>(points[index],
                                                      this->axes_min_internal[i], this->axes_max_internal[i],
                                                      this->axes_step_inv_internal[i], this->axes_points[i]);
      hc_idx += static_cast<index_t>(axis_idx) * this->axis_hypercube_mult[i];
    }
    hc_idxs[p] = hc_idx;
  }

  // Collect unique hypercube indices and filter to missing ones (serial — just set operations)
  std::vector<index_t> unique_hc(hc_idxs.begin(), hc_idxs.end());
  std::sort(unique_hc.begin(), unique_hc.end());
  unique_hc.erase(std::unique(unique_hc.begin(), unique_hc.end()), unique_hc.end());

  std::vector<index_t> missing_hc;
  missing_hc.reserve(unique_hc.size());
  for (const auto &idx : unique_hc)
  {
    if (hypercube_data.find(idx) == hypercube_data.end())
      missing_hc.push_back(idx);
  }

  if (this->timer) this->timer->node["body generation"].node["cache lookup"].stop();

  // ════════════════════════════════════════════════════════════════════════
  // Phase 2: Materialize missing cache entries.
  //          - Missing supporting points are evaluated serially (Python GIL-safe).
  //          - Missing hypercube payloads are assembled in parallel from point cache.
  //          After this phase, point_data and hypercube_data are read-only.
  // ════════════════════════════════════════════════════════════════════════
  if (!missing_hc.empty())
  {
    if (this->timer) this->timer->node["body generation"].start();
    materialize_missing_cache(missing_hc);
    if (this->timer) this->timer->node["body generation"].stop();
  }

  // ════════════════════════════════════════════════════════════════════════
  // Phase 3: Parallel read-only interpolation with thread-local workspace.
  //          No mutation of point_data or hypercube_data — safe concurrent reads.
  //          Each thread allocates its workspace once and reuses it across all cells.
  // ════════════════════════════════════════════════════════════════════════
  static const uint32_t N_VERTS = (1 << N_DIMS);
  static const size_t workspace_size = (2 * N_VERTS - 1) * N_OPS;

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    // Thread-local workspace: allocated once per thread, reused across all cells
    std::vector<value_t> workspace(workspace_size);

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int p = 0; p < static_cast<int>(n_cells); p++)
    {
      index_t offset = points_idxs[p];
      const double *point = points.data() + offset * N_DIMS;
      double *vals = values.data() + offset * N_OPS;
      double *ders = derivatives.data() + offset * N_OPS * N_DIMS;

      // Compute per-axis interpolation weights
      value_t axis_low[N_DIMS];
      value_t mult[N_DIMS];
      for (int i = 0; i < N_DIMS; ++i)
      {
        get_axis_interval_index_low_mult<value_t>(point[i],
                                                  this->axes_min_internal[i], this->axes_max_internal[i],
                                                  this->axes_step_internal[i], this->axes_step_inv_internal[i],
                                                  this->axes_points[i], &axis_low[i], &mult[i]);
      }

      // Read-only access to precomputed hypercube data
      const hypercube_data_t &hc = hypercube_data.at(hc_idxs[p]);

      // Interpolate using external workspace — no heap allocation
      interpolate_point_with_derivatives_ws<value_t, N_DIMS, N_OPS>(
          point, hc.data(), axis_low, mult,
          this->axes_step_inv_internal.data(),
          workspace.data(),
          vals, ders);
    }
  }

  return 0;
}
