/**
 * @file cuda_common.h
 * @brief Common CUDA utility routines and device management for GRIC.
 */

#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <stddef.h>

#ifdef __cplusplus
#ifndef restrict
#define restrict __restrict__
#endif
extern "C" {
#endif

/**
 * @brief Check if at least one CUDA-capable GPU is present in the system.
 *
 * @return 1 if CUDA GPU is available, 0 otherwise.
 */
int gric_cuda_is_available(void);

/**
 * @brief Initialize CUDA runtime on the specified device.
 *
 * @param device_id GPU device index (0 for default device).
 *
 * @return 0 on success, -1 on error.
 */
int gric_cuda_init(
    int device_id);

/**
 * @brief Query device name and memory capacity.
 *
 * @param device_id GPU device index.
 * @param name_buf  Buffer to receive device name string (can be NULL).
 * @param name_len  Length of name_buf.
 * @param total_mem Pointer to receive total global memory in bytes (can be NULL).
 * @param free_mem  Pointer to receive free global memory in bytes (can be NULL).
 *
 * @return 0 on success, -1 on error.
 */
int gric_cuda_get_device_info(
    int     device_id,
    char   *name_buf,
    size_t  name_len,
    size_t *total_mem,
    size_t *free_mem);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_COMMON_H */
