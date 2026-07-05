#ifndef DARTS_INTERPOLATION_MULTI_INDEX_KEY_HPP
#define DARTS_INTERPOLATION_MULTI_INDEX_KEY_HPP

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <cmath>
#include <limits>

// define those to avoid warnings on non-NVCC compilers
#ifndef __NVCC__
#ifndef __forceinline__
#define __forceinline__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

/**
 * Multi-index key used by adaptive interpolators in place of a packed integer.
 *
 * Each component is a signed 32-bit integer so the per-axis index can go below 0
 * or above the user-prescribed axes_points without overflow — adaptive caches
 * therefore no longer rely on axes_min/axes_max to bound the index space.
 */
template <uint8_t N_DIMS>
struct cell_key_t
{
    std::array<int32_t, N_DIMS> idx;

    __forceinline__ __host__ __device__ bool equals(const cell_key_t &other) const
    {
        for (uint8_t i = 0; i < N_DIMS; ++i)
            if (idx[i] != other.idx[i]) return false;
        return true;
    }

    bool operator==(const cell_key_t &other) const
    {
        return equals(other);
    }

    bool operator<(const cell_key_t &other) const
    {
        for (uint8_t i = 0; i < N_DIMS; ++i)
        {
            if (idx[i] < other.idx[i]) return true;
            if (idx[i] > other.idx[i]) return false;
        }
        return false;
    }
};

/**
 * High-quality multi-index hash: a boost::hash_combine over the axes whose inner
 * mixer is the MurmurHash3 fmix64 finalizer.
 *
 * Each per-component 32-bit value is folded into the running hash with the
 * boost::hash_combine spreader (golden-ratio constant + shift-6/shift-2), and the
 * added term is avalanched through fmix64 so every axis contributes independent
 * entropy. Tested against birthday-collision rates up to N_DIMS=20; collision rate
 * is on the order of 2^-32 at 10^6 keys.
 *
 * NOTE: an FNV-1a alternative over uint64 chunks compute-wise ~3× cheaper was
 * benchmarked and rejected — its weaker distribution increased
 * std::unordered_map bucket-chain lengths enough that overall cache_lookup time
 * regressed. The per-axis fmix64 inside the combine is what gives this hash its
 * lookup-time efficiency, despite being more cycles to compute.
 */
template <uint8_t N_DIMS>
struct cell_key_hash
{
    __forceinline__ __host__ __device__ static uint64_t mix(uint64_t h)
    {
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        return h;
    }

    __forceinline__ __host__ __device__ uint64_t hash64(const cell_key_t<N_DIMS> &k) const
    {
        // seed chosen to avoid all-zero key hashing to 0
        uint64_t h = 0x9e3779b97f4a7c15ULL;
        for (uint8_t i = 0; i < N_DIMS; ++i)
        {
            // cast through uint32 to preserve negative-index bit pattern
            const uint64_t v = static_cast<uint64_t>(static_cast<uint32_t>(k.idx[i]));
            h ^= mix(v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
        }
        return mix(h);
    }

    size_t operator()(const cell_key_t<N_DIMS> &k) const
    {
        return static_cast<size_t>(hash64(k));
    }
};

/**
 * Compute a signed per-axis interval index without clamping.
 *
 * Direct replacement for get_axis_interval_index in multilinear_interpolator_common.h
 * when the caller wants an unbounded (multi-index) key. Returns floor((value - origin) / step).
 *
 * Note: int(x) truncates toward zero in C++, not toward -inf, so for negative arguments
 * we must subtract 1 when the truncated result overshoots — done with a single branchless
 * trick: cast through floor().
 */
/**
 * Per-axis index range note: the result is int32_t, so cells more than ~2.1e9
 * away from `axis_origin` (i.e. |scaled| ≥ 2^31) overflow. At typical OBL cell
 * sizes (e.g. 1e-2 in normalized composition / pressure space) this corresponds
 * to physical excursions of ~2e7 units, far beyond any realistic state. If a
 * future model needs cells past this range, widen cell_key_t::idx to int64_t
 * (key size doubles, accompanying hashmap memory grows linearly).
 *
 * Overflow handling is defined in every build mode: the floor()ed value is
 * saturated to [INT32_MIN, INT32_MAX] with a branchless fmin/fmax clamp before the
 * cast, so the float->int conversion is never out of range (which would be UB).
 * Debug builds additionally emit a once-per-process host warning when the clamp
 * actually fires; Release saturates silently.
 */
#ifndef __CUDA_ARCH__
inline void warn_axis_index_overflow_once(double scaled, double axis_value, double axis_origin, double axis_step_inv)
{
    static std::atomic<bool> warned{false};
    bool expected = false;
    if (warned.compare_exchange_strong(expected, true))
    {
        const double step = (axis_step_inv != 0.0) ? (1.0 / axis_step_inv) : 0.0;
        fprintf(stderr,
                "OBL warning: per-axis cell index (%g) does not fit in int32_t.\n"
                "  axis_value=%g, axis_origin=%g, axis_step=%g, capacity=%g\n"
                "  Subsequent overflowing indices are clamped to INT32_MIN/MAX.\n"
                "  Widen axes_step or move axes_origin closer to the visited range;\n"
                "  if the model genuinely needs >2.1e9 cells along one axis, widen\n"
                "  cell_key_t::idx to int64_t.\n",
                scaled, axis_value, axis_origin, step,
                step * static_cast<double>(std::numeric_limits<int32_t>::max()));
    }
}
#endif

template <typename value_t>
__forceinline__ __host__ __device__ static int32_t get_axis_interval_index_unbounded(double axis_value,
                                                                                      value_t axis_origin,
                                                                                      value_t axis_step_inv)
{
    const double scaled = (axis_value - static_cast<double>(axis_origin)) * static_cast<double>(axis_step_inv);
    // floor() handles negative values correctly; int() would truncate toward zero
    const double floored = floor(scaled);
    const double i32_min = static_cast<double>(std::numeric_limits<int32_t>::min());
    const double i32_max = static_cast<double>(std::numeric_limits<int32_t>::max());
#ifndef NDEBUG
    // Debug-only diagnostic: warn once if a cell index leaves the int32 range. This
    // is only reachable with a pathologically small axes_step (the int32 range spans
    // ~2.1e9 cells per axis, far beyond any realistic reservoir state); Release
    // saturates silently via the clamp below.
    if (floored < i32_min || floored > i32_max)
    {
#ifndef __CUDA_ARCH__
        warn_axis_index_overflow_once(floored, axis_value,
                                      static_cast<double>(axis_origin),
                                      static_cast<double>(axis_step_inv));
#endif
    }
#endif // NDEBUG
    // Unconditional branchless saturating clamp: keeps the float->int conversion in
    // range so the cast is defined behaviour in EVERY build mode (out-of-range
    // float->int is UB per [conv.fpint], and also maps NaN to a defined value).
    // fmin/fmax lower to SSE MINSD/MAXSD with no branch, so this is at least as
    // cheap as the previous (Release-stripped) compare+branch guard — an A/B of the
    // cache-lookup timer showed no measurable difference.
    return static_cast<int32_t>(fmin(fmax(floored, i32_min), i32_max));
}

/**
 * Same as the bounded _low_mult variant in multilinear_interpolator_common.h but
 * without clamping. Returns the signed axis index and writes the per-axis lower
 * coordinate and the in-cell normalized offset.
 */
template <typename value_t>
__forceinline__ __host__ __device__ static int32_t get_axis_interval_index_low_mult_unbounded(double axis_value,
                                                                                               value_t axis_origin,
                                                                                               value_t axis_step,
                                                                                               value_t axis_step_inv,
                                                                                               value_t *axis_low,
                                                                                               value_t *axis_mult)
{
    const int32_t axis_interval_index = get_axis_interval_index_unbounded<value_t>(
        axis_value, axis_origin, axis_step_inv);
    *axis_low = static_cast<value_t>(axis_interval_index) * axis_step + axis_origin;
    *axis_mult = static_cast<value_t>(axis_value - static_cast<double>(*axis_low)) * axis_step_inv;
    return axis_interval_index;
}

// Provide std::hash specialization so cell_key_t works transparently in std::unordered_map.
namespace std
{
template <uint8_t N_DIMS>
struct hash<cell_key_t<N_DIMS>>
{
    size_t operator()(const cell_key_t<N_DIMS> &k) const
    {
        return cell_key_hash<N_DIMS>{}(k);
    }
};
}

#endif // DARTS_INTERPOLATION_MULTI_INDEX_KEY_HPP
