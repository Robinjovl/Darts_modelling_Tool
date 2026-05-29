#ifndef INTERPOLATION_CONFIG_H
#define INTERPOLATION_CONFIG_H

// Minimal shared configuration for the interpolation library.
// Provides timer_node, basic typedefs, and platform helpers
// without depending on engines/src/globals.h.

#include "timer_node.h"

#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <functional>
#include <string>
#include <limits>

typedef int index_t;
typedef double value_t;
typedef int interp_index_t;
typedef double interp_value_t;

// Maximum number of parameter-space dimensions for interpolator template instantiation.
// Keep this at least as high as the largest OBL table dimensionality needed by models.
#define MAX_DIMS 12

// workaround for vscode grammar checker
#ifdef __INTELLISENSE__
#define __global__
#define __constant__
#endif

#ifdef WITH_GPU
extern int device_num;
#endif

// --- MSVC __uint128_t emulation ---
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

  __uint128_t& operator=(const _Unsigned128& other) {
    this->_Word[0] = other._Word[0];
    this->_Word[1] = other._Word[1];
    return *this;
  }

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

  // Custom to_string for __uint128_t (inline to avoid link dependency)
  inline std::string to_string(const __uint128_t& value)
  {
    return std::to_string(static_cast<double>(value));
  }
};

#endif /* INTERPOLATION_CONFIG_H */
