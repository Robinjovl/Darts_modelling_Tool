//*************************************************************************
//    Copyright (c) 2021
//            Bowen Zhu         B.Zhu-3@student.tudelft.nl
//            Mark Khait         M.Khait@tudelft.nl
//            Denis Voskov    D.V.Voskov@tudelft.nl
//    Delft University of Technology, the Netherlands
//
//    This file is part of the Delft Advanced Research Terra Simulator (DARTS)
//
//    DARTS is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as
//    published by the Free Software Foundation, either version 3 of the
//    License, or (at your option) any later version.
//
//    DARTS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public
//    License along with DARTS. If not, see <http://www.gnu.org/licenses/>.
// *************************************************************************

#include "gpu_hashmap_async.h"
#include <cuda_runtime_api.h>
#include <stdio.h>

template <typename vector_element_t, int VECTOR_SIZE>
__device__ __forceinline__ int bitwise_and_hash(gpu_hash_map<vector_element_t, VECTOR_SIZE> *hashmap, int key)
{
    return key & hashmap->sizeMinus1;
}

template <typename vector_element_t, int VECTOR_SIZE>
__global__ void copy_hashmap_data(gpu_hash_map<vector_element_t, VECTOR_SIZE> *src_hashmap, gpu_hash_map<vector_element_t, VECTOR_SIZE> *dst_hashmap);

template <typename vector_element_t, int VECTOR_SIZE>
__host__ gpu_hash_map<vector_element_t, VECTOR_SIZE> *create_hashmap(int capacity)
{
    /// size is set as a power of 2 in order to use bitwise operations and to avoid expensive modulo

    // create host struct and fill it up
    gpu_hash_map<vector_element_t, VECTOR_SIZE> host_dummy;
    host_dummy.size = 2;
    while (capacity >>= 1)
        host_dummy.size <<= 1;
    host_dummy.sizeMinus1 = host_dummy.size - 1;
    host_dummy.occupied = 0;

    gpu_hash_map<vector_element_t, VECTOR_SIZE> *hashmap;

    cudaMalloc(&hashmap, sizeof(gpu_hash_map<vector_element_t, VECTOR_SIZE>));

    cudaMalloc(&host_dummy.data, sizeof(key_vector<vector_element_t, VECTOR_SIZE>) * host_dummy.size);
    cudaMemset(host_dummy.data, 0xff, sizeof(key_vector<vector_element_t, VECTOR_SIZE>) * host_dummy.size); ///< initialize all to empty key 0xffffffff
    // copy host dummy struct to device
    cudaMemcpy(hashmap, &host_dummy, sizeof(gpu_hash_map<vector_element_t, VECTOR_SIZE>), cudaMemcpyHostToDevice);

    return hashmap;
}

template <typename vector_element_t, int VECTOR_SIZE>
__host__ gpu_hash_map<vector_element_t, VECTOR_SIZE> *expand_hashmap(gpu_hash_map<vector_element_t, VECTOR_SIZE> *old_hashmap, int expand_factor)
{
    // create host struct and fill it up from old hashmap device data
    gpu_hash_map<vector_element_t, VECTOR_SIZE> new_hashmap_h;
    cudaMemcpy(&new_hashmap_h, old_hashmap, sizeof(gpu_hash_map<vector_element_t, VECTOR_SIZE>), cudaMemcpyDeviceToHost);
    // save the pointer to old data and its size to use it for copy
    auto old_hashmap_data = new_hashmap_h.data;
    int old_hashmap_size = new_hashmap_h.size;

    while (expand_factor >>= 1)
        new_hashmap_h.size <<= 1;
    new_hashmap_h.sizeMinus1 = new_hashmap_h.size - 1;

    gpu_hash_map<vector_element_t, VECTOR_SIZE> *new_hashmap;

    cudaMalloc(&new_hashmap, sizeof(gpu_hash_map<vector_element_t, VECTOR_SIZE>));

    cudaMalloc(&new_hashmap_h.data, sizeof(key_vector<vector_element_t, VECTOR_SIZE>) * new_hashmap_h.size); ///< allocate data storage for new hashmap

    cudaMemset(new_hashmap_h.data, 0xff, sizeof(key_vector<vector_element_t, VECTOR_SIZE>) * new_hashmap_h.size); ///< initialize all to empty key 0xffffffff

    cudaMemcpy(new_hashmap, &new_hashmap_h, sizeof(gpu_hash_map<vector_element_t, VECTOR_SIZE>), cudaMemcpyHostToDevice);

    copy_hashmap_data<vector_element_t, VECTOR_SIZE> KERNEL_1D_THREAD(old_hashmap_size * VECTOR_SIZE, 128)(old_hashmap, new_hashmap); ///< move data

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
__device__ int insert_vector_element(gpu_hash_map<vector_element_t, VECTOR_SIZE> *hashmap, const int key, const vector_element_t *vector, const int element_index)
{
    if (hashmap->occupied >= hashmap->size)
        return insertFailTableIsFull;

    int slot = bitwise_and_hash(hashmap, key);
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
    const unsigned vector_index = tid % VECTOR_SIZE;

    for (int i = 0; i < hashmap->size; i++)
    {
#ifdef __CUDA_ARCH__
        int current_key = atomicCAS_system(&hashmap->data[slot].key, empty_key, key);
#else
        int current_key = __sync_val_compare_and_swap(&hashmap->data[slot].key, empty_key, key);
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
__device__ int lookup_data(gpu_hash_map<vector_element_t, VECTOR_SIZE> *hashmap, const int key, vector_element_t **vector)
{
    int slot = bitwise_and_hash(hashmap, key);
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
    // if (tid == 0)
    // {
    //     printf("Requesting key %d slot %d\n", key, slot);
    // }
    for (int i = 0; i < hashmap->size; i++)
    {
        if (hashmap->data[slot].key == key)
        {
            *vector = hashmap->data[slot].vector;
            // if (tid == 0)
            // {
            //     printf("Found key %d int slot %d\n", key, slot);
            // }
            return lookUpSuccess;
        }
        if (hashmap->data[slot].key == empty_key)
            return lookUpFailKeyNotFound;
        slot = (slot + 1) & hashmap->sizeMinus1;
    }
    return lookUpFailKeyNotFound;
}

template <typename vector_element_t, int VECTOR_SIZE>
__global__ void copy_hashmap_data(gpu_hash_map<vector_element_t, VECTOR_SIZE> *src_hashmap, gpu_hash_map<vector_element_t, VECTOR_SIZE> *dst_hashmap)
{
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
    const unsigned src_slot = tid / VECTOR_SIZE;
    const unsigned vector_index = tid % VECTOR_SIZE;

    if (src_slot >= src_hashmap->size)
    {
        return;
    }

    int key = src_hashmap->data[src_slot].key;

    // move only non-empty keys
    if (key != empty_key)
    {
        int dst_slot = bitwise_and_hash(dst_hashmap, key);
        //printf("11 tid %d, moving key %d from slot %d \n", tid, key, src_slot);

        for (int i = 0; i < dst_hashmap->size; i++)
        {
#ifdef __CUDA_ARCH__
            int current_key = atomicCAS_system(&dst_hashmap->data[dst_slot].key, empty_key, key);
#else
            int current_key = __sync_val_compare_and_swap(&dst_hashmap->data[dst_slot].key, empty_key, key);
#endif
            if (current_key == empty_key || current_key == key)
            {
                dst_hashmap->data[dst_slot].vector[vector_index] = src_hashmap->data[src_slot].vector[vector_index];
                //printf("22 tid %d, moving key %d from slot %d to slot %d with val[-1]=%e\n", tid, key, src_slot, dst_slot, dst_hashmap->data[dst_slot].vector[vector_index]);
                return;
            }
            dst_slot = (dst_slot + 1) & dst_hashmap->sizeMinus1;
        }
    }
}