/**
 * @file cluster_cuda.h
 * @brief GPU-accelerated Pass 2 clustering routines for gric-cluster.
 */

#ifndef CLUSTER_CUDA_H
#define CLUSTER_CUDA_H

#include "cluster_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Check if CUDA GPU acceleration is available for gric-cluster.
 *
 * @return 1 if GPU is available, 0 otherwise.
 */
int cluster_cuda_is_available(void);

/**
 * @brief Reassign all frames to closest cluster anchor using GPU cuBLAS GEMM.
 *
 * @param config Pointer to active ClusterConfig.
 * @param state  Pointer to active ClusterState.
 *
 * @return Number of frames reassigned, or -1 on error.
 */
long cluster_cuda_run_pass2(
    ClusterConfig *config,
    ClusterState  *state);

/**
 * @brief Execute GPU brute-force online clustering (Pass 1).
 *
 * Evaluates streaming frame batches against GPU-resident cluster anchors,
 * assigning matching frames and dynamically spawning new clusters directly
 * in GPU memory.
 *
 * @param config Pointer to active ClusterConfig.
 * @param state  Pointer to active ClusterState.
 *
 * @return 0 on success, -1 on error.
 */
int cluster_cuda_run_pass1_bruteforce(
    ClusterConfig *config,
    ClusterState  *state);

#ifdef __cplusplus
}
#endif

#endif /* CLUSTER_CUDA_H */
