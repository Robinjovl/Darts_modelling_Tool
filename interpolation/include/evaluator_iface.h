#ifndef B7CB6645_948A_4B50_A7D5_980BEFD16090
#define B7CB6645_948A_4B50_A7D5_980BEFD16090

#include <vector>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include "interpolation_config.h"


/**
 * @brief Virtual interface class for evaluation of physical properties values
 *        Implemented mainly by different C++ physical kernels from darts.physics
 *        However, pure Python implementation is also possible through inheritance
 *
 */
class property_evaluator_iface
{
public:
   property_evaluator_iface(){};

   /**
   * @brief Compute property values for specified state
   *
   * @param state Coordinates in parameter space, where operators to be evaluated
   * @return double property value
   */
   virtual double evaluate(const std::vector<double> &state) = 0;

   /**
   * @brief Compute property values for all specified states
   *        A surrogate for vectorized evaluate function
   *
   * @param[in] states array of states
   * @param[in] n_blocks the number of states
   * @param[out] values  evaluated property values
   * @return int
   */
   int evaluate(const std::vector<double> &states, int n_blocks, std::vector<double> &values)
   {
      size_t state_len = states.size() / n_blocks;
      std::vector<double> state(state_len);
      values.resize(n_blocks);
      for (int i = 0; i < n_blocks; i++)
      {
         std::copy(states.begin() + i * state_len, states.begin() + (i + 1) * state_len, state.begin());
         values[i] = evaluate(state);
      }
      return 0;
   }
};

/**
 * @brief Virtual interface class for evaluation of operators values
 *        Implemented mainly by different C++ physical kernels from darts.physics
 *        However, pure Python implementation is also possible through inheritance
 *
 */
class operator_set_evaluator_iface
{
public:
   /**
   * @brief Construct a new operator set evaluator iface object
   *
   */
   operator_set_evaluator_iface() { timer = 0; };

   /**
   * @brief Initialize timer node to provide timing for operator set evaluation
   *
   * @param[in] timer_ Timer node of global timer.
   */
   void init_timer_node(timer_node *timer_) { timer = timer_; };

   /**
   * @brief Compute operators values for specified state
   *
   * @param state Coordinates in parameter space, where operators to be evaluated
   * @param values Evaluated operators values
   * @return int 0 if evaluation is successful
   */
   virtual int evaluate(const std::vector<double> &state, std::vector<double> &values) = 0;

   /**
   * @brief Compute operators values for a batch of states at once.
   *        Default implementation calls evaluate() per point serially.
   *        Override in Python (e.g. ParallelEvaluator) to dispatch to multiprocessing pool.
   *
   * @param[in]  states     Flat array of coordinates [n_points * n_dims]
   * @param[in]  n_points   Number of points to evaluate
   * @param[out] values     Flat array of operator values [n_points * n_ops], pre-allocated
   * @param[in]  n_ops      Number of operators per point
   * @return int 0 if evaluation is successful
   */
   virtual int evaluate_batch(const std::vector<double> &states, int n_points,
                              std::vector<double> &values, int n_ops)
   {
      int n_dims = (n_points > 0) ? static_cast<int>(states.size()) / n_points : 0;
      std::vector<double> single_state(n_dims);
      std::vector<double> single_values(n_ops);
      for (int i = 0; i < n_points; i++)
      {
         std::copy(states.begin() + i * n_dims,
                   states.begin() + (i + 1) * n_dims, single_state.begin());
         evaluate(single_state, single_values);
         std::copy(single_values.begin(), single_values.end(),
                   values.begin() + i * n_ops);
      }
      return 0;
   }

  //  virtual int extrapolate(const std::vector<double> &state, std::vector<double> &values);

   timer_node *timer;
};

/**
 * @brief Virtual interface class for evaluation of operators values and their gradients
 *        Mainly implemented by a variety of interpolators following OBL method
 *        But also can be derived by, for instance, AD-compatible physics kernel
 *        For GPU-compatible compilation, the interface provides two options: for computations on host and on device.
 *
 *
 */
class operator_set_gradient_evaluator_iface : public operator_set_evaluator_iface
{
public:
   /**
     * @brief Initialize evaluator, default empty implementation
     * Adds a possibility to initialize the object using properties assigned after construction
     *
     * @return int 0 if successful
     */
   virtual int init() { return 0; };

   /**
     * @brief Get the number of supporting points for the given axis, default empty implementation
     *
     * @param axis index of axis in question
     */
	   virtual int get_axis_n_points(int /*axis*/) const { return 0; };
   /**
     * @brief Get the minimum value for given axis, default empty implementation
     *
     * @param axis index of axis in question
     */
	   virtual double get_axis_min(int /*axis*/) const { return 0; };

   /**
     * @brief Get the maximum value for given axis, default empty implementation
     *
     * @param axis index of axis in question
     */
   virtual double get_axis_max(int /*axis*/) const { return 0; };

   /**
     * @brief Get the number of interpolations that took place, default empty implementation
     *
     */
   virtual uint64_t get_n_interpolations() const { return 0; };

   /**
     * @brief Get the number of supporting points used, default empty implementation
     *
     * @return the number of supporting points used
     */
   virtual uint64_t get_n_points_used() const { return 0; };

   /**
     * @brief Get the total number of supporting points in parameter space, default empty implementation
     *
     * @return the total number of supporting points
     */
   virtual uint64_t get_n_points_total() const { return 0; };

   virtual int get_n_ops() const { return 0; };

   /**
     * @brief Cumulative count of per-axis cell indices that overflowed int32 and were
     *        saturation-clamped, across the whole run (all interpolation batches).
     *
     * Nonzero means at least one queried state fell so far outside the (unbounded) OBL
     * grid that its signed multi-index would not fit in int32_t — typically a diverging
     * or oscillating Newton step. Interpolation still proceeds on the clamped index, so
     * affected cells may be inaccurate. The state lives on this shared base so both the
     * CPU (interpolator_base) and GPU (multilinear_gpu_interpolator_base) interpolator
     * hierarchies get one identical counter, getter and warning. Exposed for
     * diagnostics/tests; the human-facing warning is emitted (once) by
     * report_axis_index_overflows() in every build mode.
     */
   uint64_t get_axis_overflow_count() const { return axis_overflow_count; }

   /**
     * @brief Fold one interpolation batch's per-axis-index overflow tally into the
     *        cumulative counter and, on the first nonzero batch, emit a single host
     *        warning (identical in Debug and Release, CPU and GPU). Call once per batch
     *        (per Newton iteration) from the derived interpolate routine AFTER the hot
     *        loop — never per cell — so detection stays branchless on the hot path and
     *        reporting stays off it. axes_origin/axes_step are passed in because this
     *        shared base does not own the grid (each hierarchy keeps its own copy).
     *
     * @param batch_overflows number of clamped per-axis overflows in the just-finished batch
     * @param axes_origin     per-axis grid origin, for the diagnostic message
     * @param axes_step       per-axis cell size, for the diagnostic message
     */
   void report_axis_index_overflows(uint64_t batch_overflows,
                                    const std::vector<double> &axes_origin,
                                    const std::vector<double> &axes_step);

   /**
     * @brief Compute operators values and their gradients for every specified state
     *
     * @param[in]   states        Array of coordinates in parameter space, where operators to be evaluated
     * @param[in]   states_idxs   Indexes of states in the input array which are marked for evaluation
     * @param[out]  values        Evaluated operators values
     * @param[out]  derivatives   Evaluated operators gradients
     * @return 0 if evaluation is successful
     */
   virtual int evaluate_with_derivatives(const std::vector<double> &states, const std::vector<int> &states_idxs,
                                         std::vector<double> &values, std::vector<double> &derivatives) = 0;

#ifdef WITH_GPU
   /**
   * @brief Compute operators values for specified state on device
   *
   * @param state Coordinates in parameter space, where operators to be evaluated
   * @param values Evaluated operators values
   * @return int 0 if evaluation is successful
   */
   virtual int evaluate_d(double *state_d, double *values_d) = 0;
   /**
     * @brief Compute operators values and their gradients for every specified state on GPU device.
     *
     * @param[in]   n_states_idxs      Number of states marked for interpolation, the length of states_idxs_d array
     * @param[in]   state_d       Array of coordinates in parameter space, where operators to be evaluated, device pointer
     * @param[in]   states_idxs_d Indexes of states in the input array which are marked for evaluation, device pointer
     * @param[out]  values_d      Evaluated operators values, device pointer
     * @param[out]  derivatives_d Evaluated operators gradients, device pointer
     * @return 0 if evaluation is successful
     */
   virtual int evaluate_with_derivatives_d(int n_states_idxs, double *state_d, int *states_idxs_d,
                                           double *values_d, double *derivatives_d) = 0;
#endif

protected:
   uint64_t axis_overflow_count = 0; ///< Cumulative per-axis int32 cell-index overflows (saturation-clamped), all batches
   bool axis_overflow_warned = false; ///< Whether the once-per-interpolator overflow warning has already been emitted
};

// Defined inline in the header (not a .cpp) on purpose: the CPU and GPU interpolator
// modules are separate link targets with disjoint base hierarchies, so no single
// translation unit is guaranteed to be linked into both. Inlining puts the (host-only)
// definition in whichever TU instantiates an interpolator, sidestepping that. It is
// only ever called from host code, so nvcc treats it as a plain host function.
inline void operator_set_gradient_evaluator_iface::report_axis_index_overflows(
    uint64_t batch_overflows,
    const std::vector<double> &axes_origin,
    const std::vector<double> &axes_step)
{
   // Off the hot path: called once per interpolation batch (Newton iteration) with the
   // count accumulated branchlessly across that batch's cells (OpenMP reduction on CPU,
   // per-thread register + atomicAdd on GPU). This is what informs the user of overflow
   // in Release, which the former Debug-only per-call warner did not.
   if (batch_overflows == 0)
      return;

   axis_overflow_count += batch_overflows;

   // Throttle to a single detailed warning per interpolator lifetime to avoid flooding
   // stderr when a run diverges over many iterations; the running total stays available
   // via get_axis_overflow_count() for callers that want to keep watching.
   if (axis_overflow_warned)
      return;
   axis_overflow_warned = true;

   fprintf(stderr,
           "OBL warning: %llu per-axis cell index(es) fell outside int32_t this batch and were\n"
           "  clamped to [INT32_MIN, INT32_MAX]. This usually means a diverging/oscillating Newton\n"
           "  step drove the state far outside the OBL grid, or axes_step is too small. Interpolation\n"
           "  continues on the saturated (clamped) index, so affected cells may be inaccurate.\n",
           static_cast<unsigned long long>(batch_overflows));

   fprintf(stderr, "  axes_origin = [");
   for (std::size_t d = 0; d < axes_origin.size(); ++d)
      fprintf(stderr, "%s%g", d ? ", " : "", axes_origin[d]);
   fprintf(stderr, "]\n  axes_step   = [");
   for (std::size_t d = 0; d < axes_step.size(); ++d)
      fprintf(stderr, "%s%g", d ? ", " : "", axes_step[d]);
   fprintf(stderr, "]\n"
           "  Move axes_origin closer to the operating point or widen axes_step; if the model\n"
           "  genuinely needs >2.1e9 cells along one axis, widen cell_key_t::idx to int64_t.\n"
           "  (Further overflow warnings suppressed; query get_axis_overflow_count() for the total.)\n");
}

/**
 * @brief A class for evaluation of operators values and their gradients on host
 *        Provides straightforward device implementation by sending input data to host, calling the host function,
 *        and sending the output back to device
 *
 *
 */
class operator_set_gradient_evaluator_cpu : public operator_set_gradient_evaluator_iface
{
public:
#ifdef WITH_GPU

   /**
   * @brief Compute operators values for specified state on device
   *
   * @param state Coordinates in parameter space, where operators to be evaluated
   * @param values Evaluated operators values
   * @return int 0 if evaluation is successful
   */
   virtual int evaluate_d(double *state_d, double *values_d) final;

   /**
     * @brief Compute operators values and their gradients for every specified state on GPU device.
     *
     * @param[in]   n_states_idxs      Number of states marked for interpolation, the length of states_idxs_d array
     * @param[in]   state_d       Array of coordinates in parameter space, where operators to be evaluated, device pointer
     * @param[in]   states_idxs_d Indexes of states in the input array which are marked for evaluation, device pointer
     * @param[out]  values_d      Evaluated operators values, device pointer
     * @param[out]  derivatives_d Evaluated operators gradients, device pointer
     * @return 0 if evaluation is successful
     */
   virtual int evaluate_with_derivatives_d(int n_states_idxs, double *state_d, int *states_idxs_d,
                                           double *values_d, double *derivatives_d) final;
#endif
};

#ifdef WITH_GPU
/**
 * @brief A class for evaluation of operators values and their gradients on device
 *        Provides straightforward host implementation by sending input data to device, calling the device function,
 *        and sending the output back to host
 *
 */
class operator_set_gradient_evaluator_gpu : public operator_set_gradient_evaluator_iface
{
public:
   /**
   * @brief Compute operators values for specified state
   *
   * @param state Coordinates in parameter space, where operators to be evaluated
   * @param values Evaluated operators values
   * @return int 0 if evaluation is successful
   */
   virtual int evaluate(const std::vector<double> &state, std::vector<double> &values) final;
   /**
     * @brief Compute operators values and their gradients for every specified state
     *
     * @param[in]   states        Array of coordinates in parameter space, where operators to be evaluated
     * @param[in]   states_idxs   Indexes of states in the input array which are marked for evaluation
     * @param[out]  values        Evaluated operators values
     * @param[out]  derivatives   Evaluated operators gradients
     * @return 0 if evaluation is successful
     */
   virtual int evaluate_with_derivatives(const std::vector<double> &states, const std::vector<int> &states_idxs,
                                         std::vector<double> &values, std::vector<double> &derivatives) final;
};
#endif

#endif /* B7CB6645_948A_4B50_A7D5_980BEFD16090 */
