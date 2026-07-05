#ifndef DARTS_INTERPOLATION_MULTI_INDEX_KEY_HPP
#define DARTS_INTERPOLATION_MULTI_INDEX_KEY_HPP

#include <array>
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
 * Multi-index hash: a boost::hash_combine over the axes whose inner
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
 * to physical excursions of ~2e7 units, far beyond any realistic state. It is not
 * reachable at construction time — a diverging/oscillating Newton path can drive a
 * state arbitrarily far outside the grid mid-simulation — so the guard has to live
 * on the (hot) lookup path, not in a one-off setup check. If a future model needs
 * cells past this range, widen cell_key_t::idx to int64_t (key size doubles,
 * accompanying hashmap memory grows linearly).
 *
 * Overflow handling is IDENTICAL in Debug and Release (no #ifdef split): the
 * floor()ed value is saturated to [INT32_MIN, INT32_MAX] with a branchless fmin/fmax
 * clamp before the cast, so the float->int conversion is never out of range (which
 * would be UB) and NaN maps to a defined value. Detection is likewise always-on and
 * branchless: when the caller passes a non-null `overflow_accum`, this fn adds
 * `(clamped != floored)` (0 or 1) into it — no branch, no atomic, no I/O on the hot
 * path. Callers accumulate per Newton iteration (OpenMP reduction on CPU, per-thread
 * register + atomicAdd on GPU) and hand the batch total to
 * operator_set_gradient_evaluator_iface::report_axis_index_overflows() — the shared
 * base of both the CPU and GPU interpolator hierarchies — which emits a single
 * throttled host warning off the hot path. This replaced the former Debug-only
 * per-call fprintf warner, so Release builds now inform the user of overflow too.
 */
template <typename value_t>
__forceinline__ __host__ __device__ static int32_t get_axis_interval_index_unbounded(double axis_value,
                                                                                      value_t axis_origin,
                                                                                      value_t axis_step_inv,
                                                                                      int *overflow_accum = nullptr)
{
    const double scaled = (axis_value - static_cast<double>(axis_origin)) * static_cast<double>(axis_step_inv);
    // floor() handles negative values correctly; int() would truncate toward zero
    const double floored = floor(scaled);
    const double i32_min = static_cast<double>(std::numeric_limits<int32_t>::min());
    const double i32_max = static_cast<double>(std::numeric_limits<int32_t>::max());
    // Unconditional branchless saturating clamp: keeps the float->int conversion in
    // range so the cast is defined behaviour in EVERY build mode (out-of-range
    // float->int is UB per [conv.fpint], and also maps NaN to a defined value).
    // fmin/fmax lower to SSE MINSD/MAXSD with no branch, so this is at least as
    // cheap as the previous (Release-stripped) compare+branch guard — an A/B of the
    // cache-lookup timer showed no measurable difference.
    const double clamped = fmin(fmax(floored, i32_min), i32_max);
    // Always-on, branchless overflow tally. `overflow_accum` is a stack local (CPU)
    // or per-thread register (GPU) at every hot call site, so after inlining the
    // compiler proves it non-null and folds this to a single compare + add with no
    // branch; call sites that must not count (e.g. re-computation passes) pass the
    // default nullptr and the whole statement is elided.
    if (overflow_accum)
        *overflow_accum += static_cast<int>(clamped != floored);
    return static_cast<int32_t>(clamped);
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
                                                                                               value_t *axis_mult,
                                                                                               int *overflow_accum = nullptr)
{
    const int32_t axis_interval_index = get_axis_interval_index_unbounded<value_t>(
        axis_value, axis_origin, axis_step_inv, overflow_accum);
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
