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
 * Backward-compat: the Python-visible `point_data` property is still presented as a
 * dict with integer keys (mixed-radix packing of the multi-index using axes_points).
 * Cells whose multi-index is within [0, axes_points[i]-1] for every axis are exported
 * in this legacy format; out-of-bounds cells are kept in memory but skipped on export.
 *
 * @tparam index_t type used for legacy packed-integer indexing (backward compat only)
 * @tparam value_t value type used for supporting point storage, hypercube storage and interpolation
 * @tparam N_DIMS The number of dimensions in paramter space
 * @tparam N_OPS The number of operators to be interpolated
 */
template <typename index_t, typename value_t, uint8_t N_DIMS, uint8_t N_OPS>
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
    * @brief Construct the interpolator with specified parametrization space
    *
    * @param[in] supporting_point_evaluator    Object used to compute operators values at supporting points
    * @param[in] axes_points               Number of supporting points (minimum 2) along axes
    * @param[in] axes_min                  Minimum value for each axis (interpreted as origin offset)
    * @param[in] axes_max                  Maximum for each axis (advisory; defines axes_step together with axes_points)
    */
   multilinear_adaptive_cpu_interpolator(operator_set_evaluator_iface *supporting_point_evaluator,
                                         const std::vector<int> &axes_points,
                                         const std::vector<double> &axes_min,
                                         const std::vector<double> &axes_max);

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
    * @brief Get indexes of all evaluated hypercubes, packed as legacy integer indices.
    *
    * In-bounds cells only — out-of-bounds cells are silently dropped from this list.
    * Use `get_hypercube_keys()` to retrieve the full multi-index list.
    */
   std::vector<index_t> get_hypercube_indexes() const;

   /**
    * @brief Get multi-index keys of all evaluated hypercubes (no bounds filtering).
    */
   std::vector<key_t> get_hypercube_keys() const;

   /**
    * @brief Translate a multi-index into the legacy packed integer key.
    *
    * Used by the Python-binding compatibility shim. Returns the packed index in
    * mixed-radix form using axes_points as bases. Caller must ensure all components
    * are non-negative; out-of-bounds packing is undefined.
    */
   index_t to_int_key_point(const key_t &k) const;
   index_t to_int_key_hypercube(const key_t &k) const;

   /**
    * @brief Inverse of to_int_key_point — decode a packed integer back into a multi-index.
    */
   key_t from_int_key_point(index_t int_key) const;
   key_t from_int_key_hypercube(index_t int_key) const;

   /**
    * @brief True iff every component of the multi-index is within [0, axes_points[i]-1]
    * (point key) or [0, axes_points[i]-2] (hypercube key).
    */
   bool is_in_bounds_point(const key_t &k) const;
   bool is_in_bounds_hypercube(const key_t &k) const;

   /**
    * @brief Number of cached supporting points / hypercubes (in-memory).
    */
   size_t get_n_cached_points() const { return point_data.size(); }
   size_t get_n_cached_hypercubes() const { return hypercube_data.size(); }

   /**
    * @brief Single-point interpolation; overrides base to use multi-index path.
    */
   int interpolate(const std::vector<double> &point, std::vector<double> &values) override;

protected:
   /**
    * @brief Cell-key-driven supporting-point access (creates on miss).
    */
   const point_data_t &get_point_data(const key_t &point_key);

   /**
    * @brief Cell-key-driven hypercube access (creates on miss, recursively materializing vertices).
    */
   const hypercube_data_t &get_hypercube_data(const key_t &hypercube_key);

   /**
    * @brief Backward-compatible integer-key API required by the base class.
    *
    * Decodes the packed integer back to a multi-index and forwards. Only ever
    * called via paths that originate from the bounded base class machinery; the
    * adaptive batch path bypasses this entirely.
    */
   const hypercube_data_t &get_hypercube_data(const index_t hypercube_index) override;

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
    * @brief Batch interpolation; overrides base to use multi-index path throughout.
    */
   int interpolate_with_derivatives(const std::vector<double> &points, const std::vector<int> &points_idxs,
                                    std::vector<double> &values, std::vector<double> &derivatives) override;
};

// now include implementation of the templated class from tpp file
#include "multilinear_adaptive_cpu_interpolator.tpp"

#endif /* BFD055ED_0AA6_4F1C_A5A5_9157EE0F34FF */
