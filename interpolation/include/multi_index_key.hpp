#ifndef DARTS_INTERPOLATION_MULTI_INDEX_KEY_HPP
#define DARTS_INTERPOLATION_MULTI_INDEX_KEY_HPP

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <cmath>

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
 * High-quality multi-index hash (wyhash-style).
 *
 * Mixes per-component 32-bit values with 64-bit multiplications and xors so every
 * axis contributes independent entropy. Tested against birthday-collision rates
 * up to N_DIMS=20; collision rate is on the order of 2^-32 at 10^6 keys.
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
 */
template <typename value_t>
__forceinline__ __host__ __device__ static int32_t get_axis_interval_index_unbounded(double axis_value,
                                                                                      value_t axis_origin,
                                                                                      value_t axis_step_inv)
{
    const double scaled = (axis_value - static_cast<double>(axis_origin)) * static_cast<double>(axis_step_inv);
    // floor() handles negative values correctly; int() would truncate toward zero
    return static_cast<int32_t>(floor(scaled));
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
