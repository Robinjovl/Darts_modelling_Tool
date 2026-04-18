#ifndef CCA0A759_B17D_45B6_88DF_2EC71A8D99BF
#define CCA0A759_B17D_45B6_88DF_2EC71A8D99BF

// define those to avoid warning indication in syntax check for non-nvcc compilers
#ifndef __NVCC__
#define __forceinline__
#define __host__
#define __global__
#define __device__
#endif

namespace gpu_hashmap_async
{

    /// GPU hashmap based on simple bitwise operation with the size of 2^n-1
    /// Uses async memory operations to achieve higher performance
    /// Only device interface is provided

    template <typename vector_element_t, int VECTOR_SIZE>
    struct key_vector
    {
        int key;
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

    /// set -1 as empty key, since all expected input keys are nonnegative
    /// the byte 0xff makes it easier to initialize hashmap (see cudaMemset in create_hashmap)
    const int empty_key = 0xffffffff;

    const int insertSuccess = 0;
    const int insertFailTableIsFull = -1;
    const int lookUpSuccess = 0;
    const int lookUpFailKeyNotFound = -1;

    template <typename vector_element_t, int VECTOR_SIZE>
    __device__ __forceinline__ int bitwise_and_hash(gpu_hash_map<vector_element_t, VECTOR_SIZE> *hashmap, int key);

    template <typename vector_element_t, int VECTOR_SIZE>
    __host__ gpu_hash_map<vector_element_t, VECTOR_SIZE> *create_hashmap(int capacity);

    template <typename vector_element_t, int VECTOR_SIZE>
    __host__ void delete_hashmap(gpu_hash_map<vector_element_t, VECTOR_SIZE> *hashmap);

    template <typename vector_element_t, int VECTOR_SIZE>
    __device__ int insert_vector_element(gpu_hash_map<vector_element_t, VECTOR_SIZE> *hashmap, const int key, const vector_element_t *vector, const int element_index);

    template <typename vector_element_t, int VECTOR_SIZE>
    __device__ int lookup_data(gpu_hash_map<vector_element_t, VECTOR_SIZE> *hashmap, const int key, double &value);

#include "gpu_hashmap_async.tpp"

}

#endif /* CCA0A759_B17D_45B6_88DF_2EC71A8D99BF */
