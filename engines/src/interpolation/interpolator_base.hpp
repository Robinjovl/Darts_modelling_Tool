#ifndef BB750132_A4A2_43EF_AB87_90364057EDC8
#define BB750132_A4A2_43EF_AB87_90364057EDC8

#include <array>
#include <limits>

#include "evaluator_iface.h"

#ifdef _MSC_VER
#include <__msvc_int128.hpp>

struct __uint128_t : std::_Unsigned128 
{
  // Inherit constructors
  using std::_Unsigned128::_Unsigned128;

  // Define a constructor from int to handle assignment from integer literals
  constexpr __uint128_t(int x) : _Unsigned128(x) {};
  constexpr __uint128_t(double x) : _Unsigned128(static_cast<int>(x)) {};
  constexpr __uint128_t(const std::_Unsigned128& x) : _Unsigned128(x) {}

  // Add operator= to handle assignment from other types
  __uint128_t& operator=(int x)
  {
    this->_Word[0] = x;
    this->_Word[1] = 0;
    return *this;
  };

  __uint128_t& operator=(double x)
  {
    this->_Word[0] = static_cast<int>(x);
    this->_Word[1] = 0;
    return *this;
  };

  template <typename T>
  __uint128_t& operator=(const T& other)
  {
    this->_Word[0] = static_cast<int>(other);
    this->_Word[1] = 0;
    return *this;
  };

  template <typename T>
  operator T() const
  {
    return static_cast<T>(this->_Word[0]) + static_cast<T>(this->_Word[1] * std::pow(2, 64));
  };

  __uint128_t operator*(int x) const
  {
    return *this * static_cast<__uint128_t>(x);
  };

  __uint128_t operator*(uint64_t x) const
  {
    return *this * static_cast<__uint128_t>(x);
  };

  __uint128_t operator*(const __uint128_t& other) const
  {
    return __uint128_t(static_cast<const std::_Unsigned128&>(*this) * static_cast<const std::_Unsigned128&>(other));
  };
};

namespace std
{
  template<>
  class numeric_limits<__uint128_t>
  {
  public:
    static constexpr bool is_specialized = true;
    static constexpr __uint128_t min() noexcept { return __uint128_t(0); }
    static constexpr __uint128_t max() noexcept { return __uint128_t(~uint64_t(0), ~uint64_t(0)); }
    static constexpr __uint128_t lowest() noexcept { return min(); }
    static constexpr int digits = 128;
    static constexpr int digits10 = 38; // ceil(log10(2^128))
    static constexpr int max_digits10 = 0;
    static constexpr bool is_signed = false;
    static constexpr bool is_integer = true;
    static constexpr bool is_exact = true;
    static constexpr int radix = 2;
    static constexpr __uint128_t epsilon() noexcept { return __uint128_t(0); }
    static constexpr __uint128_t round_error() noexcept { return __uint128_t(0); }
    static constexpr int min_exponent = 0;
    static constexpr int min_exponent10 = 0;
    static constexpr int max_exponent = 0;
    static constexpr int max_exponent10 = 0;
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = false;
    static constexpr bool has_signaling_NaN = false;
    static constexpr bool has_denorm_loss = false;
    static constexpr bool has_denorm = false;
    static constexpr float_denorm_style has_denorm_style = std::denorm_absent;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = true;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;
    static constexpr float_round_style round_style = std::round_toward_zero;
  };
};

#elif defined(__GNUC__)
#endif

namespace std
{
  template <>
  struct hash<__uint128_t>
  {
    size_t operator()(const __uint128_t& x) const noexcept
    {
#ifdef _MSC_VER
      size_t h1 = std::hash<uint64_t>{}(x._Word[0]);
      size_t h2 = std::hash<uint64_t>{}(x._Word[1]);
#elif defined(__GNUC__)
      size_t h1 = std::hash<uint64_t>{}(static_cast<uint64_t>(x));
      size_t h2 = std::hash<uint64_t>{}(static_cast<uint64_t>(x >> 64));
#endif
      return h1 ^ (h2 * 0x9e3779b97f4a7c15 + 0x7f4a7c15);  // Use a large prime multiplier and a random offset
    }
  };

  // Custom to_string for __uint128_t
  std::string to_string(const __uint128_t& value);
};

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
     */
    interpolator_base(operator_set_evaluator_iface *supporting_point_evaluator,
                      const std::vector<int> &axes_points,
                      const std::vector<double> &axes_min,
                      const std::vector<double> &axes_max);

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
    const std::vector<int> axes_points;                       ///< number of supporting points along each axis
    const std::vector<double> axes_min;                       ///< minimum at each axis
    const std::vector<double> axes_max;                       ///< maximum of each axis
    operator_set_evaluator_iface *supporting_point_evaluator; ///< object which computes operator values for supporting points

    std::vector<double> axes_step;     ///< the distance between neighbor supporting points for each axis
    std::vector<double> axes_step_inv; ///< inverse of step (to avoid division)

    uint64_t n_interpolations; ///< Number of interpolations that took place
    __uint128_t n_points_total;   ///< Total number of parametrization points
    double n_points_total_fp;  ///< Total number of parametrization points in floating point format, to detect index overflow in derived classes
    __uint128_t n_points_used;    ///< Number of parametrization points which were used (equal to n_points_total for static interpolators)

    std::vector<double> new_point_coords;    ///< intermediate storage for supporting point generation
    std::vector<double> new_operator_values; ///< intermediate storage for supporting point generation

private:
    int n_dims; ///< number of dimensions in parameter space
    int n_ops;  ///< number of oparators to be interpolated for every
};

#endif /* BB750132_A4A2_43EF_AB87_90364057EDC8 */
