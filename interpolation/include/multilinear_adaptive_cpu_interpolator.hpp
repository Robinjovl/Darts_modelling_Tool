#ifndef BFD055ED_0AA6_4F1C_A5A5_9157EE0F34FF
#define BFD055ED_0AA6_4F1C_A5A5_9157EE0F34FF

#include <vector>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

#include "multi_index_key.hpp"
#include "multilinear_interpolator_base.hpp"

/**
 * @brief  Piecewise mulitlinear interpolator with adaptive storage
 *
 * Adaptive evaluation is split into three explicit phases:
 *   Phase 1 – parallel computation of hypercube indices for every requested cell
 *   Phase 2 – serial materialization of missing supporting points (Python callback)
 *             and parallel assembly of missing hypercube payloads
 *   Phase 3 – parallel read-only interpolation with thread-local workspace
 *
 * Storage is now keyed on a signed multi-index `cell_key_t<N_DIMS>` instead of a
 * packed integer; the adaptive cache therefore no longer relies on axes_min/axes_max
 * to bound the key space. Cells outside the user-prescribed (axes_min, axes_max)
 * window are evaluated and cached on demand, letting the solver explore state space
 * freely without pre-allocated bookkeeping.
 *
 * Python cache I/O is exposed via the tuple-keyed `point_data_full` property
 * (signed multi-index → operator tuple). The legacy integer-keyed `point_data`
 * shim has been retired on adaptive interpolators since the unbounded grid has
 * no integer-key packing; legacy caches written by older builds are detected on
 * load and skipped (see physics_base.py).
 *
 * @tparam index_t type used for legacy packed-integer indexing (backward compat only)
 * @tparam value_t value type used for supporting point storage, hypercube storage and interpolation
 * @tparam N_DIMS The number of dimensions in paramter space
 * @tparam N_OPS The number of operators to be interpolated
 */
// N_OPS widened to uint16_t — must match base class.
template <typename index_t, typename value_t, uint8_t N_DIMS, uint16_t N_OPS>
class multilinear_adaptive_cpu_interpolator : public multilinear_interpolator_base<index_t, value_t, N_DIMS, N_OPS>
{
public:
   using typename multilinear_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::point_coordinates_t;
   using typename multilinear_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::point_data_t;
   using typename multilinear_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::hypercube_data_t;
   using typename multilinear_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::hypercube_points_index_t;

   typedef cell_key_t<N_DIMS> key_t;
   typedef cell_key_hash<N_DIMS> key_hash_t;
   typedef std::array<key_t, (1 << N_DIMS)> hypercube_vertex_keys_t;

   /**
    * @brief Construct the interpolator parametrized by (origin, step).
    *
    * The grid is unbounded: cells are enumerated on demand via signed multi-index
    * keys (cell_key_t), so no axes_max / axes_points is needed.
    *
    * @param[in] supporting_point_evaluator    Object used to compute operators values at supporting points
    * @param[in] axes_origin              Grid origin (lower corner) for each axis
    * @param[in] axes_step                Cell size for each axis
    */
   multilinear_adaptive_cpu_interpolator(operator_set_evaluator_iface *supporting_point_evaluator,
                                         const std::vector<double> &axes_origin,
                                         const std::vector<double> &axes_step);

   /**
    * @brief adaptive point storage: the values of operators at requested supporting points.
    *
    * Keyed on signed multi-index (cell_key_t). Storage grows dynamically as the solver
    * explores state space; cells beyond the user-prescribed bounds are accepted without
    * complaint.
    */
   std::unordered_map<key_t, point_data_t, key_hash_t> point_data;

   /**
    * @brief adaptive hypercube storage: values of operators at every vertex of requested hypercubes.
    *
    * Keyed on signed multi-index — same conventions as point_data.
    */
   std::unordered_map<key_t, hypercube_data_t, key_hash_t> hypercube_data;

   /**
    * @brief Optional cap on the number of cached hypercube payloads (0 = unbounded; default).
    *
    * hypercube_data is a PURE DERIVED cache — every entry is rebuilt from point_data
    * with NO supporting-point (flash) evaluation, and it is NEVER persisted (the on-disk
    * OBL cache stores only point_data). When > 0, the map is held to ~hypercube_cap
    * most-recently-used entries (LRU by batch epoch), capping peak RAM. The per-batch
    * live working set is bounded by the number of requested cells (one hypercube key
    * per cell), not by the cumulative number of explored cells, so eviction is lossless
    * on the hot path and never re-triggers flash.
    */
   size_t hypercube_cap = 0;

   /// Last batch epoch (eval_index) at which each cached hypercube was used; LRU side table.
   std::unordered_map<key_t, uint64_t, key_hash_t> hc_last_used;

   /**
    * @brief Get multi-index keys of all evaluated hypercubes.
    *
    * Each key is the signed lower-corner multi-index of a generated hypercube. This
    * is the canonical (unbounded) view used for cache export and body-path output;
    * there is no integer-key packing any more (the grid is unbounded).
    */
   std::vector<key_t> get_hypercube_keys() const;

   /**
    * @brief Number of cached supporting points / hypercubes (in-memory).
    */
   size_t get_n_cached_points() const { return point_data.size(); }
   size_t get_n_cached_hypercubes() const { return hypercube_data.size(); }

   /**
    * @brief Bound the in-memory derived hypercube cache to ~cap most-recently-used
    *        entries (0 = unbounded). Does NOT affect point_data or the cache file format.
    */
   void set_hypercube_cap(size_t cap) { hypercube_cap = cap; }
   size_t get_hypercube_cap() const { return hypercube_cap; }

   /**
    * @brief Drop all cached hypercube payloads and release their memory. They are
    *        rebuilt on demand from point_data (no flash). point_data and the on-disk
    *        cache are untouched. Call only between evaluate() calls (not thread-safe
    *        against an in-flight interpolate_with_derivatives).
    */
   void clear_hypercube_data();

   /**
    * @brief Multi-index keys of supporting points materialized since the last external
    * cache flush. Mirrors the development-branch tracker but in the cell_key_t key space
    * so the OBL cache writer can persist only newly evaluated points.
    */
   std::unordered_set<key_t, key_hash_t> dirty_point_data;

   /**
    * @brief Evaluation epoch of each dirty (unflushed) supporting point: maps the
    * point multi-index key to the batch-interpolation index (≈ time step /
    * nonlinear iteration) at which it was first materialized. Persisted next to the
    * append-only delta so the OBL cache file records when each point entered the
    * sampling, allowing offline reconstruction of how the active hypercubes evolve.
    * Cleared together with dirty_point_data on every external cache flush.
    */
   std::unordered_map<key_t, uint64_t, key_hash_t> dirty_point_epochs;

   /**
    * @brief Number of batch interpolation calls processed so far. Incremented once
    * at the start of every interpolate_with_derivatives() (one nonlinear-iteration
    * assembly of this operator set) and used to stamp newly materialized points.
    */
   uint64_t eval_index = 0;

   /**
    * @brief Single-point interpolation; overrides base to use multi-index path.
    */
   int interpolate(const std::vector<double> &point, std::vector<double> &values) override;

protected:
   // Bring the base's bounded (index_t) get_hypercube_data into scope so the cell-key
   // overload below *overloads* rather than *hides* it (silences -Wxxx #997-D). The
   // bounded overload is never called on this adaptive path; it only exists for the
   // shared static-storage machinery in the base.
   using multilinear_interpolator_base<index_t, value_t, N_DIMS, N_OPS>::get_hypercube_data;

   /**
    * @brief Cell-key-driven supporting-point access (creates on miss).
    */
   const point_data_t &get_point_data(const key_t &point_key);

   /**
    * @brief Cell-key-driven hypercube access (creates on miss, recursively materializing vertices).
    */
   const hypercube_data_t &get_hypercube_data(const key_t &hypercube_key);

   /**
    * @brief Compute physical coordinates of a supporting point from its multi-index.
    */
   void get_point_coordinates_from_key(const key_t &k, point_coordinates_t &coordinates) const;

   /**
    * @brief Produce the N_VERTS vertex keys of a hypercube from its lower-corner key.
    */
   void get_hypercube_vertex_keys(const key_t &hc_key, hypercube_vertex_keys_t &vertex_keys) const;

   /**
    * @brief Materialize missing supporting points + hypercube payloads in batch.
    *        Missing points are evaluated through a single batched evaluator call
    *        (multiprocessing-aware); hypercube payloads are assembled in parallel.
    */
   void materialize_missing_cache(const std::vector<key_t> &missing_hc);

   /**
    * @brief Evict least-recently-used hypercubes down to ~0.9*hypercube_cap when the
    *        cache exceeds hypercube_cap. No-op when hypercube_cap == 0. Never evicts
    *        entries used in the current batch (eval_index). Erases only derived data.
    */
   void evict_hypercubes();

   /**
    * @brief Batch interpolation; overrides base to use multi-index path throughout.
    */
   int interpolate_with_derivatives(const std::vector<double> &points, const std::vector<int> &points_idxs,
                                    std::vector<double> &values, std::vector<double> &derivatives) override;
};

// now include implementation of the templated class from tpp file
#include "multilinear_adaptive_cpu_interpolator.tpp"

#endif /* BFD055ED_0AA6_4F1C_A5A5_9157EE0F34FF */
