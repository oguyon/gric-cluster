/**
 * @file cuda_common.cu
 * @brief Common CUDA utility routines and device management implementation.
 */

#include "cuda_common.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <string.h>

/**
 * gric_cuda_is_available() - Check if at least one CUDA device is available.
 *
 * Return: 1 if CUDA device is present and functional, 0 otherwise.
 */
int gric_cuda_is_available(void)
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0)
    {
        return 0;
    }
    return 1;
}

/**
 * gric_cuda_init() - Initialize CUDA runtime on the specified device.
 * @device_id: GPU device index (0 for default device).
 *
 * Return: 0 on success, -1 on error.
 */
int gric_cuda_init(
    int device_id)
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0)
    {
        fprintf(stderr, "CUDA error: No CUDA-capable devices found.\n");
        return -1;
    }

    if (device_id < 0 || device_id >= device_count)
    {
        fprintf(stderr, "CUDA error: Invalid device ID %d (system has %d devices).\n",
                device_id, device_count);
        return -1;
    }

    err = cudaSetDevice(device_id);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: cudaSetDevice(%d) failed: %s\n",
                device_id, cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

/**
 * gric_cuda_get_device_info() - Query device name and memory capacity.
 * @device_id: GPU device index.
 * @name_buf:  Buffer to receive device name string (can be NULL).
 * @name_len:  Length of name_buf.
 * @total_mem: Pointer to receive total global memory in bytes (can be NULL).
 * @free_mem:  Pointer to receive free global memory in bytes (can be NULL).
 *
 * Return: 0 on success, -1 on error.
 */
int gric_cuda_get_device_info(
    int     device_id,
    char   *name_buf,
    size_t  name_len,
    size_t *total_mem,
    size_t *free_mem)
{
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess)
    {
        return -1;
    }

    if (name_buf != NULL && name_len > 0)
    {
        strncpy(name_buf, prop.name, name_len - 1);
        name_buf[name_len - 1] = '\0';
    }

    if (total_mem != NULL || free_mem != NULL)
    {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        err = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (err == cudaSuccess)
        {
            if (total_mem != NULL)
            {
                *total_mem = total_bytes;
            }
            if (free_mem != NULL)
            {
                *free_mem = free_bytes;
            }
        }
        else
        {
            if (total_mem != NULL)
            {
                *total_mem = prop.totalGlobalMem;
            }
            if (free_mem != NULL)
            {
                *free_mem = 0;
            }
        }
    }

    return 0;
}
