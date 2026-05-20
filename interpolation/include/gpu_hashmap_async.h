#ifndef CCA0A759_B17D_45B6_88DF_2EC71A8D99BF
#define CCA0A759_B17D_45B6_88DF_2EC71A8D99BF

#include <cstdint>

#include "multi_index_key.hpp"

// define those to avoid warning indication in syntax check for non-nvcc compilers
#ifndef __NVCC__
#ifndef __forceinline__
#define __forceinline__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __device__
#define __device__
#endif
#endif

namespace gpu_hashmap_async
{

    /// GPU open-addressed hashmap.
    ///
    /// Each slot stores a 64-bit content hash of the multi-index (cell_key_t) as the
    /// lookup key. Bucket selection AND match comparison use the same 63-bit hash —
    /// the slot does NOT store the full multi-index. This trades a theoretical
    /// collision rate (~N^2 / 2^64 ≈ 6e-8 at N=10^7 unique cells) for a much
    /// simpler concurrent-insert protocol that doesn't require multi-stage CAS or
    /// inter-thread spin-waits.
    ///
    /// If a future workload requires strict collision-freedom, switch the slot
    /// layout to include the full cell_key_t alongside a 2-stage atomic protocol
    /// (claim → write-key → publish → write-payload) — see git history of this
    /// file for an attempted-then-reverted implementation.

    typedef int64_t hash_key_t;

    template <typename vector_element_t, int VECTOR_SIZE>
    struct key_vector
    {
        hash_key_t key;   // 64-bit content hash of the cell_key_t; -1 = empty
        vector_element_t vector[VECTOR_SIZE];
    };

    template <typename vector_element_t, int VECTOR_SIZE>
    struct gpu_hash_map
    {
        int size;
        int sizeMinus1;
        int occupied;
        key_vector<vector_element_t, VECTOR_SIZE> *data;
    };

    /// Empty slot sentinel — 0xff...ff = -1 as int64_t. Convenient because cudaMemset
    /// can write it byte-wise.
    const hash_key_t empty_key = static_cast<hash_key_t>(-1);

    const int insertSuccess = 0;
    const int insertFailTableIsFull = -1;
    const int lookUpSuccess = 0;
    const int lookUpFailKeyNotFound = -1;

    /// Map a multi-index to a 64-bit non-negative key. Sign bit cleared so the value
    /// never collides with empty_key (-1).
    template <uint8_t N_DIMS>
    __forceinline__ __host__ __device__ hash_key_t key_from_cell(const cell_key_t<N_DIMS> &k)
    {
        const uint64_t h = cell_key_hash<N_DIMS>{}.hash64(k);
        return static_cast<hash_key_t>(h & 0x7fffffffffffffffULL);
    }

    template <typename vector_element_t, int VECTOR_SIZE>
    __device__ __forceinline__ int bitwise_and_hash(gpu_hash_map<vector_element_t, VECTOR_SIZE> *hashmap, hash_key_t key);

    template <typename vector_element_t, int VECTOR_SIZE>
    __host__ gpu_hash_map<vector_element_t, VECTOR_SIZE> *create_hashmap(int capacity);

    template <typename vector_element_t, int VECTOR_SIZE>
    __host__ void delete_hashmap(gpu_hash_map<vector_element_t, VECTOR_SIZE> *hashmap);

    template <typename vector_element_t, int VECTOR_SIZE>
    __device__ int insert_vector_element(gpu_hash_map<vector_element_t, VECTOR_SIZE> *hashmap,
                                         hash_key_t key,
                                         const vector_element_t *vector,
                                         const int element_index);

    template <typename vector_element_t, int VECTOR_SIZE>
    __device__ int lookup_data(gpu_hash_map<vector_element_t, VECTOR_SIZE> *hashmap,
                               hash_key_t key,
                               vector_element_t **vector);

#include "gpu_hashmap_async.tpp"

}

#endif /* CCA0A759_B17D_45B6_88DF_2EC71A8D99BF */
