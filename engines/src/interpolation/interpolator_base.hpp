#ifndef BB750132_A4A2_43EF_AB87_90364057EDC8
#define BB750132_A4A2_43EF_AB87_90364057EDC8

#include <array>
#include <limits>
#include <vector>
#include <cstdint>

#include "evaluator_iface.h"

/**
 * Interpolator base class
 */
class interpolator_base : public operator_set_gradient_evaluator_cpu
{
public:
    /**
     * @brief Construct an interpolator with predefined parametrization space
     *
     * @param[in] supporting_point_evaluator    Object used to compute operator values at supporting points
     * @param[in] axes_points                   Number of supporting points (minimum 2) along axes
     * @param[in] axes_min                      Minimum value for each axis
     * @param[in] axes_max                      Maximum for each axis
     * @param[in] axis_nodes                    Optional explicit coordinates of supporting points along each axis.
     *                                          When nullptr, a uniform mesh defined by axes_min/axes_max/axes_points
     *                                          is assumed.  When provided, every axis_nodes[i] vector must contain
     *                                          monotonically increasing coordinates and its length must be equal
     *                                          to axes_points[i].
     */
    interpolator_base(operator_set_evaluator_iface *supporting_point_evaluator,
                      const std::vector<int> &axes_points,
                      const std::vector<double> &axes_min,
                      const std::vector<double> &axes_max,
                      const std::vector<std::vector<double>> *axis_nodes = nullptr);

    /**
     * @brief Initialize interpolator, perform internal sanity checks unavailable at construction time
     *
     * @return int 0 if successful
     */
    virtual int init();

    /**
     * @brief Evaluate all operators at the given state (point in parametrization space)
     *        Runs timer and calls virtual interpolate routine
     *
     * @param[in]   state   Coordinates in parametrization space
     * @param[out]  values  Interpolated values
     * @return 0 if evaluation is successful
     */
    int evaluate(const std::vector<double> &state, std::vector<double> &values);

    /**
     * @brief Evaluate operators and their gradient for every specified state (point in parametrization space)
     *
     * @param[in]   states        Array of coordinates in parametrization space
     * @param[in]   states_idxs   Indexes of states in the input array which are marked for evaluation
     * @param[out]  values        Interpolated values
     * @param[out]  derivatives   Interpolation gradients
     * @return 0 if evaluation is successful
     */
    int evaluate_with_derivatives(const std::vector<double> &states, const std::vector<int> &states_idxs,
                                  std::vector<double> &values, std::vector<double> &derivatives);

    /**
     * @brief Compute interpolation for all operators at the given point
     *
     * @param[in]   point   Coordinates in parametrization space
     * @param[out]  values  Interpolated values
     * @return 0 if interpolation is successful
     */
    virtual int interpolate(const std::vector<double> &point, std::vector<double> &values) = 0;

    /**
     * @brief Compute interpolation and its gradient for all operators at every specified point
     *
     * @param[in]   points        Array of coordinates in parametrization space
     * @param[in]   points_idxs   Indexes of points in the points array which are marked for interpolation
     * @param[out]  values        Interpolated values
     * @param[out]  derivatives   Interpolation gradients
     * @return 0 if interpolation is successful
     */
    virtual int interpolate_with_derivatives(const std::vector<double> &points, const std::vector<int> &points_idxs,
                                             std::vector<double> &values, std::vector<double> &derivatives) = 0;

     /**
      * @brief Write interpolator data to file
      *
      * @param filename name of the file
      * @return int error code
      */
    virtual int write_to_file(const std::string filename)
    {
        printf ("Not implemented!\n");
        return -1;
    };

    /**
     * @brief Get the number of dimensions in interpolation space
     *        Virtual, to be overriden by a child class
     *
     */
    virtual int get_n_dims() const = 0;

    /**
     * @brief Get the number of operators
     *        Virtual, to be overriden by a child class
     *
     */
    virtual int get_n_ops() const = 0;

    /**
     * @brief Get the number of supporting points for the given axis
     *
     * @param axis index of axis in question
     */
    int get_axis_n_points(int axis) const;

    /**
     * @brief Get the parametrization minimum value for given axis
     *
     * @param axis index of axis in question
     */
    double get_axis_min(int axis) const;

    /**
     * @brief Get the parametrization maximum value for given axis
     *
     * @param axis index of axis in question
     */
    double get_axis_max(int axis) const;

    /**
     * @brief Get the number of interpolations that took place
     *
     */
    uint64_t get_n_interpolations() const;

    /**
     * @brief Get the total number of supporting points in parameter space
     *
     * @return the total number of supporting points
     */
    uint64_t get_n_points_total() const;

    /**
     * @brief Get the number of supporting points used (evaluated through supporting_point_evaluator)
     *        The number is equal to n_points_total for static interpolation methods
     *
     * @return the number of supporting points used
     */
    uint64_t get_n_points_used() const;

protected:
    // The binning helper limits the number of linear scans required during interval lookup.
    // A small constant number of bins per axis proved sufficient in profiling, therefore
    // we keep memory footprint predictable by capping the number of bins.
    static constexpr int MAX_AXIS_BINS = 1024;
    const std::vector<int> axes_points;                       ///< number of supporting points along each axis
    const std::vector<double> axes_min;                       ///< minimum at each axis
    const std::vector<double> axes_max;                       ///< maximum of each axis
    operator_set_evaluator_iface *supporting_point_evaluator; ///< object which computes operator values for supporting points

    std::vector<double> axes_step;     ///< the distance between neighbor supporting points for each axis
    std::vector<double> axes_step_inv; ///< inverse of step (to avoid division)

    std::vector<double> axis_nodes_flat;        ///< Flattened coordinates of all axis nodes (size = sum axes_points).
    std::vector<double> axis_inv_dx_flat;       ///< Flattened inverse spacing (per cell) for each axis.
    std::vector<size_t> axis_nodes_offset;      ///< Prefix sums that map an axis index to its slice inside axis_nodes_flat.
    std::vector<size_t> axis_cells_offset;      ///< Prefix sums for axis_inv_dx_flat (there are axes_points[i]-1 cells per axis).
    std::vector<uint32_t> axis_bin_left_idx_flat; ///< Coarse bin -> candidate cell index lookup tables for every axis.
    std::vector<size_t> axis_bin_offset;        ///< Prefix sums pointing to the start of each axis' bin table.
    std::vector<int> axis_bin_count;            ///< Number of coarse bins created for every axis.
    std::vector<double> axis_bin_inv_width;     ///< Inverse bin width used to map coordinates into bin indices.
    bool nonuniform_axes_enabled;               ///< True when explicit axis nodes are provided.

    uint64_t n_interpolations; ///< Number of interpolations that took place
    __uint128_t n_points_total;   ///< Total number of parametrization points
    double n_points_total_fp;  ///< Total number of parametrization points in floating point format, to detect index overflow in derived classes
    __uint128_t n_points_used;    ///< Number of parametrization points which were used (equal to n_points_total for static interpolators)

    std::vector<double> new_point_coords;    ///< intermediate storage for supporting point generation
    std::vector<double> new_operator_values; ///< intermediate storage for supporting point generation

    bool has_nonuniform_axes() const { return nonuniform_axes_enabled; }

private:
    int n_dims; ///< number of dimensions in parameter space
    int n_ops;  ///< number of oparators to be interpolated for every
};

#endif /* BB750132_A4A2_43EF_AB87_90364057EDC8 */
