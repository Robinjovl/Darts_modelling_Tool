#include "gpu_hashmap_async.h"
#include <cuda_runtime_api.h>
#include <stdio.h>

template <typename vector_element_t, int VECTOR_SIZE>
__device__ __forceinline__ int bitwise_and_hash(gpu_hash_map<vector_element_t, VECTOR_SIZE> *hashmap, hash_key_t key)
{
    // hash_key_t is already a hashed 63-bit value; just AND with sizeMinus1 to derive slot.
    return static_cast<int>(static_cast<uint64_t>(key) & static_cast<uint64_t>(hashmap->sizeMinus1));
}

template <typename vector_element_t, int VECTOR_SIZE>
__global__ void copy_hashmap_data(gpu_hash_map<vector_element_t, VECTOR_SIZE> *src_hashmap,
                                  gpu_hash_map<vector_element_t, VECTOR_SIZE> *dst_hashmap);

template <typename vector_element_t, int VECTOR_SIZE>
__host__ gpu_hash_map<vector_element_t, VECTOR_SIZE> *create_hashmap(int capacity)
{
    /// size is set as a power of 2 in order to use bitwise operations and to avoid expensive modulo

    gpu_hash_map<vector_element_t, VECTOR_SIZE> host_dummy;
    host_dummy.size = 2;
    while (capacity >>= 1)
        host_dummy.size <<= 1;
    host_dummy.sizeMinus1 = host_dummy.size - 1;
    host_dummy.occupied = 0;

    gpu_hash_map<vector_element_t, VECTOR_SIZE> *hashmap;

    cudaMalloc(&hashmap, sizeof(gpu_hash_map<vector_element_t, VECTOR_SIZE>));

    cudaMalloc(&host_dummy.data, sizeof(key_vector<vector_element_t, VECTOR_SIZE>) * host_dummy.size);
    cudaMemset(host_dummy.data, 0xff, sizeof(key_vector<vector_element_t, VECTOR_SIZE>) * host_dummy.size); ///< 0xff fills key with -1 (empty sentinel)

    cudaMemcpy(hashmap, &host_dummy, sizeof(gpu_hash_map<vector_element_t, VECTOR_SIZE>), cudaMemcpyHostToDevice);

    return hashmap;
}

template <typename vector_element_t, int VECTOR_SIZE>
__host__ gpu_hash_map<vector_element_t, VECTOR_SIZE> *expand_hashmap(gpu_hash_map<vector_element_t, VECTOR_SIZE> *old_hashmap, int expand_factor)
{
    gpu_hash_map<vector_element_t, VECTOR_SIZE> new_hashmap_h;
    cudaMemcpy(&new_hashmap_h, old_hashmap, sizeof(gpu_hash_map<vector_element_t, VECTOR_SIZE>), cudaMemcpyDeviceToHost);

    int old_hashmap_size = new_hashmap_h.size;

    while (expand_factor >>= 1)
        new_hashmap_h.size <<= 1;
    new_hashmap_h.sizeMinus1 = new_hashmap_h.size - 1;

    gpu_hash_map<vector_element_t, VECTOR_SIZE> *new_hashmap;

    cudaMalloc(&new_hashmap, sizeof(gpu_hash_map<vector_element_t, VECTOR_SIZE>));

    cudaMalloc(&new_hashmap_h.data, sizeof(key_vector<vector_element_t, VECTOR_SIZE>) * new_hashmap_h.size);

    cudaMemset(new_hashmap_h.data, 0xff, sizeof(key_vector<vector_element_t, VECTOR_SIZE>) * new_hashmap_h.size);

    cudaMemcpy(new_hashmap, &new_hashmap_h, sizeof(gpu_hash_map<vector_element_t, VECTOR_SIZE>), cudaMemcpyHostToDevice);

    copy_hashmap_data<vector_element_t, VECTOR_SIZE> KERNEL_1D_THREAD(old_hashmap_size * VECTOR_SIZE, 128)(old_hashmap, new_hashmap);

    delete_hashmap(old_hashmap);
    return new_hashmap;
}

template <typename vector_element_t, int VECTOR_SIZE>
__host__ void delete_hashmap(gpu_hash_map<vector_element_t, VECTOR_SIZE> *hashmap)
{
    gpu_hash_map<vector_element_t, VECTOR_SIZE> host_dummy;
    cudaMemcpy(&host_dummy, hashmap, sizeof(gpu_hash_map<vector_element_t, VECTOR_SIZE>), cudaMemcpyDeviceToHost);
    cudaFree(host_dummy.data);
    cudaFree(hashmap);
}

// Concurrent insert of vector_element is allowed
template <typename vector_element_t, int VECTOR_SIZE>
__device__ int insert_vector_element(gpu_hash_map<vector_element_t, VECTOR_SIZE> *hashmap,
                                     hash_key_t key,
                                     const vector_element_t *vector,
                                     const int element_index)
{
    if (hashmap->occupied >= hashmap->size)
        return insertFailTableIsFull;

    int slot = bitwise_and_hash(hashmap, key);

    for (int i = 0; i < hashmap->size; i++)
    {
#ifdef __CUDA_ARCH__
        unsigned long long current_key_ull = atomicCAS(
            reinterpret_cast<unsigned long long *>(&hashmap->data[slot].key),
            static_cast<unsigned long long>(empty_key),
            static_cast<unsigned long long>(key));
        hash_key_t current_key = static_cast<hash_key_t>(current_key_ull);
#else
        hash_key_t current_key = __sync_val_compare_and_swap(&hashmap->data[slot].key, empty_key, key);
#endif
        if (current_key == empty_key || current_key == key)
        {
            hashmap->data[slot].vector[element_index] = vector[element_index];
            return insertSuccess;
        }
        slot = (slot + 1) & hashmap->sizeMinus1;
    }
    return insertFailTableIsFull;
}

template <typename vector_element_t, int VECTOR_SIZE>
__device__ int lookup_data(gpu_hash_map<vector_element_t, VECTOR_SIZE> *hashmap,
                           hash_key_t key,
                           vector_element_t **vector)
{
    int slot = bitwise_and_hash(hashmap, key);

    for (int i = 0; i < hashmap->size; i++)
    {
        if (hashmap->data[slot].key == key)
        {
            *vector = hashmap->data[slot].vector;
            return lookUpSuccess;
        }
        if (hashmap->data[slot].key == empty_key)
            return lookUpFailKeyNotFound;
        slot = (slot + 1) & hashmap->sizeMinus1;
    }
    return lookUpFailKeyNotFound;
}

template <typename vector_element_t, int VECTOR_SIZE>
__global__ void copy_hashmap_data(gpu_hash_map<vector_element_t, VECTOR_SIZE> *src_hashmap,
                                  gpu_hash_map<vector_element_t, VECTOR_SIZE> *dst_hashmap)
{
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned src_slot = tid / VECTOR_SIZE;
    const unsigned vector_index = tid % VECTOR_SIZE;

    if (src_slot >= static_cast<unsigned>(src_hashmap->size))
        return;

    const hash_key_t key = src_hashmap->data[src_slot].key;
    if (key == empty_key)
        return;

    int dst_slot = bitwise_and_hash(dst_hashmap, key);

    for (int i = 0; i < dst_hashmap->size; i++)
    {
#ifdef __CUDA_ARCH__
        unsigned long long current_key_ull = atomicCAS(
            reinterpret_cast<unsigned long long *>(&dst_hashmap->data[dst_slot].key),
            static_cast<unsigned long long>(empty_key),
            static_cast<unsigned long long>(key));
        hash_key_t current_key = static_cast<hash_key_t>(current_key_ull);
#else
        hash_key_t current_key = __sync_val_compare_and_swap(&dst_hashmap->data[dst_slot].key, empty_key, key);
#endif
        if (current_key == empty_key || current_key == key)
        {
            dst_hashmap->data[dst_slot].vector[vector_index] = src_hashmap->data[src_slot].vector[vector_index];
            return;
        }
        dst_slot = (dst_slot + 1) & dst_hashmap->sizeMinus1;
    }
}
