/**
 * @file cluster_dcc.h
 * @brief Accessor functions and lower-triangular storage abstraction layer for
 *        pairwise inter-cluster distance (DCC) matrix and bounds cache.
 */

#ifndef CLUSTER_DCC_H
#define CLUSTER_DCC_H

#include "cluster_defs.h"
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/**
 * dcc_alloc_row() - Allocate a row buffer with 64-byte alignment for SIMD vector loads.
 * @bytes: Number of bytes to allocate.
 *
 * Return: Pointer to aligned buffer, or NULL on failure.
 */
static inline void *dcc_alloc_row(size_t bytes)
{
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, bytes) != 0)
    {
        ptr = malloc(bytes);
    }
    return ptr;
}

/**
 * dcc_ensure_row() - Ensure lower-triangular row @r is allocated with padding.
 * @state:           Running clustering state.
 * @r:               Row index to ensure.
 * @sparse_dcc_mode: Non-zero if sparse bounds mode is active.
 */
static inline void dcc_ensure_row(
    ClusterState *state,
    int           r,
    int           sparse_dcc_mode)
{
    if (r < 0 || state->scratch.dcc_min_rows == NULL)
    {
        return;
    }
    if ((size_t)r >= state->scratch.dcc_capacity)
    {
        return;
    }
    if (state->scratch.dcc_min_rows[r] != NULL)
    {
        return;
    }

    size_t alloc_elem = ((size_t)r + 15) & ~7ULL;
    if (alloc_elem < 8)
    {
        alloc_elem = 8;
    }

    double *min_row = (double *)dcc_alloc_row(alloc_elem * sizeof(double));
    if (min_row != NULL)
    {
        double init_val = sparse_dcc_mode ? 0.0 : -1.0;
        for (size_t k = 0; k < alloc_elem; k++)
        {
            min_row[k] = init_val;
        }
        state->scratch.dcc_min_rows[r] = min_row;
    }

    if (sparse_dcc_mode && state->scratch.dcc_max_rows != NULL)
    {
        double *max_row = (double *)dcc_alloc_row(alloc_elem * sizeof(double));
        if (max_row != NULL)
        {
            for (size_t k = 0; k < alloc_elem; k++)
            {
                max_row[k] = 1e19;
            }
            state->scratch.dcc_max_rows[r] = max_row;
        }
    }

    if (sparse_dcc_mode && state->scratch.dcc_measured_rows != NULL)
    {
        char *meas_row = (char *)dcc_alloc_row(alloc_elem * sizeof(char));
        if (meas_row != NULL)
        {
            memset(meas_row, 0, alloc_elem * sizeof(char));
            state->scratch.dcc_measured_rows[r] = meas_row;
        }
    }

    if (state->scratch.dcc_sq16_rows != NULL)
    {
        uint16_t *sq16_row = (uint16_t *)dcc_alloc_row(alloc_elem * sizeof(uint16_t));
        if (sq16_row != NULL)
        {
            for (size_t k = 0; k < alloc_elem; k++)
            {
                sq16_row[k] = DCC_SQ16_UNMEASURED;
            }
            state->scratch.dcc_sq16_rows[r] = sq16_row;
        }
    }
}

/**
 * dcc_init_matrix() - Initialize DCC matrix row pointer tables at startup.
 * @state:           Running clustering state.
 * @max_clusters:    Maximum number of clusters.
 * @sparse_dcc_mode: Non-zero if sparse DCC bounds mode is active.
 * @use_sq16:        Non-zero if SQ16 cache rows are allocated.
 */
static inline void dcc_init_matrix(
    ClusterState *state,
    size_t        max_clusters,
    int           sparse_dcc_mode,
    int           use_sq16)
{
    state->scratch.dcc_capacity = max_clusters;
    state->scratch.dcc_min_rows = (double **)calloc(max_clusters, sizeof(double *));
    if (sparse_dcc_mode)
    {
        state->scratch.dcc_max_rows = (double **)calloc(max_clusters, sizeof(double *));
        state->scratch.dcc_measured_rows = (char **)calloc(max_clusters, sizeof(char *));
    }
    else
    {
        state->scratch.dcc_max_rows = NULL;
        state->scratch.dcc_measured_rows = NULL;
    }

    if (use_sq16)
    {
        state->scratch.dcc_sq16_rows = (uint16_t **)calloc(max_clusters, sizeof(uint16_t *));
    }
    else
    {
        state->scratch.dcc_sq16_rows = NULL;
    }

    /* Allocate dummy row 0 so row 0 pointers are non-NULL */
    dcc_ensure_row(state, 0, sparse_dcc_mode);
}

/**
 * dcc_grow_capacity() - Reallocate row pointer tables for expanded capacity.
 * @state:           Running clustering state.
 * @new_capacity:    New maximum cluster capacity.
 * @sparse_dcc_mode: Non-zero if sparse DCC bounds mode is active.
 *
 * Return: 0 on success, or -1 on allocation failure.
 */
static inline int dcc_grow_capacity(
    ClusterState *state,
    size_t        new_capacity,
    int           sparse_dcc_mode)
{
    size_t old_cap = state->scratch.dcc_capacity;
    if (new_capacity <= old_cap)
    {
        return 0;
    }

    double **new_min = (double **)realloc(
        state->scratch.dcc_min_rows, new_capacity * sizeof(double *));
    if (!new_min)
    {
        return -1;
    }
    memset(new_min + old_cap, 0, (new_capacity - old_cap) * sizeof(double *));
    state->scratch.dcc_min_rows = new_min;

    if (sparse_dcc_mode || state->scratch.dcc_max_rows != NULL)
    {
        double **new_max = (double **)realloc(
            state->scratch.dcc_max_rows, new_capacity * sizeof(double *));
        if (!new_max)
        {
            return -1;
        }
        memset(new_max + old_cap, 0, (new_capacity - old_cap) * sizeof(double *));
        state->scratch.dcc_max_rows = new_max;

        char **new_meas = (char **)realloc(
            state->scratch.dcc_measured_rows, new_capacity * sizeof(char *));
        if (!new_meas)
        {
            return -1;
        }
        memset(new_meas + old_cap, 0, (new_capacity - old_cap) * sizeof(char *));
        state->scratch.dcc_measured_rows = new_meas;
    }

    if (state->scratch.dcc_sq16_rows != NULL)
    {
        uint16_t **new_sq16 = (uint16_t **)realloc(
            state->scratch.dcc_sq16_rows, new_capacity * sizeof(uint16_t *));
        if (!new_sq16)
        {
            return -1;
        }
        memset(new_sq16 + old_cap, 0, (new_capacity - old_cap) * sizeof(uint16_t *));
        state->scratch.dcc_sq16_rows = new_sq16;
    }

    state->scratch.dcc_capacity = new_capacity;
    return 0;
}

/**
 * dcc_free_matrix() - Free all allocated row buffers and pointer tables.
 * @state: Running clustering state.
 */
static inline void dcc_free_matrix(ClusterState *state)
{
    if (state->scratch.dcc_min_rows != NULL)
    {
        for (size_t r = 0; r < state->scratch.dcc_capacity; r++)
        {
            if (state->scratch.dcc_min_rows[r] != NULL)
            {
                free(state->scratch.dcc_min_rows[r]);
                state->scratch.dcc_min_rows[r] = NULL;
            }
        }
        free(state->scratch.dcc_min_rows);
        state->scratch.dcc_min_rows = NULL;
    }

    if (state->scratch.dcc_max_rows != NULL)
    {
        for (size_t r = 0; r < state->scratch.dcc_capacity; r++)
        {
            if (state->scratch.dcc_max_rows[r] != NULL)
            {
                free(state->scratch.dcc_max_rows[r]);
                state->scratch.dcc_max_rows[r] = NULL;
            }
        }
        free(state->scratch.dcc_max_rows);
        state->scratch.dcc_max_rows = NULL;
    }

    if (state->scratch.dcc_measured_rows != NULL)
    {
        for (size_t r = 0; r < state->scratch.dcc_capacity; r++)
        {
            if (state->scratch.dcc_measured_rows[r] != NULL)
            {
                free(state->scratch.dcc_measured_rows[r]);
                state->scratch.dcc_measured_rows[r] = NULL;
            }
        }
        free(state->scratch.dcc_measured_rows);
        state->scratch.dcc_measured_rows = NULL;
    }

    if (state->scratch.dcc_sq16_rows != NULL)
    {
        for (size_t r = 0; r < state->scratch.dcc_capacity; r++)
        {
            if (state->scratch.dcc_sq16_rows[r] != NULL)
            {
                free(state->scratch.dcc_sq16_rows[r]);
                state->scratch.dcc_sq16_rows[r] = NULL;
            }
        }
        free(state->scratch.dcc_sq16_rows);
        state->scratch.dcc_sq16_rows = NULL;
    }

    state->scratch.dcc_capacity = 0;
}

/**
 * dcc_reset_matrix() - Free rows > 0 and reset row 0 for reusable sessions.
 * @state:           Running clustering state.
 * @sparse_dcc_mode: Non-zero if sparse DCC bounds mode is active.
 */
static inline void dcc_reset_matrix(
    ClusterState *state,
    int           sparse_dcc_mode)
{
    if (state->scratch.dcc_min_rows == NULL)
    {
        return;
    }
    for (size_t r = 1; r < state->scratch.dcc_capacity; r++)
    {
        if (state->scratch.dcc_min_rows[r] != NULL)
        {
            free(state->scratch.dcc_min_rows[r]);
            state->scratch.dcc_min_rows[r] = NULL;
        }
        if (state->scratch.dcc_max_rows != NULL && state->scratch.dcc_max_rows[r] != NULL)
        {
            free(state->scratch.dcc_max_rows[r]);
            state->scratch.dcc_max_rows[r] = NULL;
        }
        if (state->scratch.dcc_measured_rows != NULL && state->scratch.dcc_measured_rows[r] != NULL)
        {
            free(state->scratch.dcc_measured_rows[r]);
            state->scratch.dcc_measured_rows[r] = NULL;
        }
        if (state->scratch.dcc_sq16_rows != NULL && state->scratch.dcc_sq16_rows[r] != NULL)
        {
            free(state->scratch.dcc_sq16_rows[r]);
            state->scratch.dcc_sq16_rows[r] = NULL;
        }
    }
    dcc_ensure_row(state, 0, sparse_dcc_mode);
}

/**
 * dcc_get_dist() - Retrieve pairwise distance between cluster @i and cluster @j.
 * @state: Running clustering state.
 * @i:     First cluster index.
 * @j:     Second cluster index.
 *
 * Return: Pairwise Euclidean distance between anchors @i and @j.
 */
static inline double dcc_get_dist(
    const ClusterState *state,
    int                 i,
    int                 j)
{
    if (i == j)
    {
        return 0.0;
    }
    int r = (i > j) ? i : j;
    int c = (i > j) ? j : i;
    if (state->scratch.dcc_min_rows == NULL || state->scratch.dcc_min_rows[r] == NULL)
    {
        return -1.0;
    }
    return state->scratch.dcc_min_rows[r][c];
}

/**
 * dcc_get_min() - Retrieve minimum distance bound between cluster @i and cluster @j.
 * @state: Running clustering state.
 * @i:     First cluster index.
 * @j:     Second cluster index.
 *
 * Return: Minimum distance bound.
 */
static inline double dcc_get_min(
    const ClusterState *state,
    int                 i,
    int                 j)
{
    if (i == j)
    {
        return 0.0;
    }
    int r = (i > j) ? i : j;
    int c = (i > j) ? j : i;
    if (state->scratch.dcc_min_rows == NULL || state->scratch.dcc_min_rows[r] == NULL)
    {
        return 0.0;
    }
    return state->scratch.dcc_min_rows[r][c];
}

/**
 * dcc_get_max() - Retrieve maximum distance bound between cluster @i and cluster @j.
 * @state: Running clustering state.
 * @i:     First cluster index.
 * @j:     Second cluster index.
 *
 * Return: Maximum distance bound.
 */
static inline double dcc_get_max(
    const ClusterState *state,
    int                 i,
    int                 j)
{
    if (i == j)
    {
        return 0.0;
    }
    int r = (i > j) ? i : j;
    int c = (i > j) ? j : i;
    if (state->scratch.dcc_max_rows != NULL && state->scratch.dcc_max_rows[r] != NULL)
    {
        return state->scratch.dcc_max_rows[r][c];
    }
    if (state->scratch.dcc_min_rows != NULL && state->scratch.dcc_min_rows[r] != NULL)
    {
        return state->scratch.dcc_min_rows[r][c];
    }
    return 1e19;
}

/**
 * dcc_is_measured() - Check if distance between cluster @i and @j is exactly measured.
 * @state: Running clustering state.
 * @i:     First cluster index.
 * @j:     Second cluster index.
 *
 * Return: Non-zero if exactly measured, 0 if unmeasured bound.
 */
static inline int dcc_is_measured(
    const ClusterState *state,
    int                 i,
    int                 j)
{
    if (i == j)
    {
        return 1;
    }
    int r = (i > j) ? i : j;
    int c = (i > j) ? j : i;
    if (state->scratch.dcc_measured_rows != NULL && state->scratch.dcc_measured_rows[r] != NULL)
    {
        return (int)state->scratch.dcc_measured_rows[r][c];
    }
    if (state->scratch.dcc_min_rows != NULL && state->scratch.dcc_min_rows[r] != NULL)
    {
        return (state->scratch.dcc_min_rows[r][c] >= 0.0) ? 1 : 0;
    }
    return 0;
}

/**
 * dcc_get_sq16() - Retrieve 16-bit quantized distance between cluster @i and cluster @j.
 * @state: Running clustering state.
 * @i:     First cluster index.
 * @j:     Second cluster index.
 *
 * Return: 16-bit quantized distance or DCC_SQ16_UNMEASURED.
 */
static inline uint16_t dcc_get_sq16(
    const ClusterState *state,
    int                 i,
    int                 j)
{
    if (i == j)
    {
        return 0;
    }
    if (state->scratch.dcc_sq16_rows == NULL)
    {
        return DCC_SQ16_UNMEASURED;
    }
    int r = (i > j) ? i : j;
    int c = (i > j) ? j : i;
    if (state->scratch.dcc_sq16_rows[r] == NULL)
    {
        return DCC_SQ16_UNMEASURED;
    }
    return state->scratch.dcc_sq16_rows[r][c];
}

/**
 * dcc_row_dist() - Get pointer to read-only distance row for cluster @i.
 * @state: Running clustering state.
 * @i:     Cluster index.
 *
 * Return: Const pointer to distance row @i (valid for columns 0..i-1).
 */
static inline const double *dcc_row_dist(
    const ClusterState *state,
    int                 i)
{
    return state->scratch.dcc_min_rows[i];
}

/**
 * dcc_row_dist_mut() - Get pointer to mutable distance row for cluster @i.
 * @state: Running clustering state.
 * @i:     Cluster index.
 *
 * Return: Mutable pointer to distance row @i (valid for columns 0..i-1).
 */
static inline double *dcc_row_dist_mut(
    ClusterState *state,
    int           i)
{
    return state->scratch.dcc_min_rows[i];
}

/**
 * dcc_row_max() - Get pointer to read-only maximum bound row for cluster @i.
 * @state: Running clustering state.
 * @i:     Cluster index.
 *
 * Return: Const pointer to max bound row @i, or NULL in dense mode.
 */
static inline const double *dcc_row_max(
    const ClusterState *state,
    int                 i)
{
    return state->scratch.dcc_max_rows ? state->scratch.dcc_max_rows[i] : NULL;
}

/**
 * dcc_row_max_mut() - Get pointer to mutable maximum bound row for cluster @i.
 * @state: Running clustering state.
 * @i:     Cluster index.
 *
 * Return: Mutable pointer to max bound row @i, or NULL in dense mode.
 */
static inline double *dcc_row_max_mut(
    ClusterState *state,
    int           i)
{
    return state->scratch.dcc_max_rows ? state->scratch.dcc_max_rows[i] : NULL;
}

/**
 * dcc_row_measured() - Get pointer to read-only measured flags row for cluster @i.
 * @state: Running clustering state.
 * @i:     Cluster index.
 *
 * Return: Const pointer to measured flags row @i, or NULL in dense mode.
 */
static inline const char *dcc_row_measured(
    const ClusterState *state,
    int                 i)
{
    return state->scratch.dcc_measured_rows ? state->scratch.dcc_measured_rows[i] : NULL;
}

/**
 * dcc_row_measured_mut() - Get pointer to mutable measured flags row for cluster @i.
 * @state: Running clustering state.
 * @i:     Cluster index.
 *
 * Return: Mutable pointer to measured flags row @i, or NULL in dense mode.
 */
static inline char *dcc_row_measured_mut(
    ClusterState *state,
    int           i)
{
    return state->scratch.dcc_measured_rows ? state->scratch.dcc_measured_rows[i] : NULL;
}

/**
 * dcc_row_sq16() - Get pointer to read-only SQ16 distance row for cluster @i.
 * @state: Running clustering state.
 * @i:     Cluster index.
 *
 * Return: Const pointer to SQ16 distance row @i, or NULL if SQ16 is disabled.
 */
static inline const uint16_t *dcc_row_sq16(
    const ClusterState *state,
    int                 i)
{
    return state->scratch.dcc_sq16_rows ? state->scratch.dcc_sq16_rows[i] : NULL;
}

/**
 * dcc_row_sq16_mut() - Get pointer to mutable SQ16 distance row for cluster @i.
 * @state: Running clustering state.
 * @i:     Cluster index.
 *
 * Return: Mutable pointer to SQ16 distance row @i, or NULL if SQ16 is disabled.
 */
static inline uint16_t *dcc_row_sq16_mut(
    ClusterState *state,
    int           i)
{
    return state->scratch.dcc_sq16_rows ? state->scratch.dcc_sq16_rows[i] : NULL;
}

/**
 * dcc_set_pair() - Update pairwise distance cache entry in float and SQ16.
 * @state: Running clustering state.
 * @c1:    First cluster index.
 * @c2:    Second cluster index.
 * @d:     Pairwise Euclidean distance.
 */
static inline void dcc_set_pair(
    ClusterState *state,
    int           c1,
    int           c2,
    double        d)
{
    if (c1 == c2)
    {
        return;
    }
    int r = (c1 > c2) ? c1 : c2;
    int c = (c1 > c2) ? c2 : c1;

    state->scratch.dcc_min_rows[r][c] = d;

    if (state->scratch.dcc_max_rows != NULL && state->scratch.dcc_max_rows[r] != NULL)
    {
        state->scratch.dcc_max_rows[r][c] = d;
    }
    if (state->scratch.dcc_measured_rows != NULL && state->scratch.dcc_measured_rows[r] != NULL)
    {
        state->scratch.dcc_measured_rows[r][c] = 1;
    }
    if (state->scratch.dcc_sq16_rows != NULL && state->scratch.dcc_sq16_rows[r] != NULL)
    {
        double s = state->scratch.dcc_sq16_scale;
        uint16_t q = (d <= 0.0) ? 0 :
            ((d * s >= 65534.0) ? 65534 : (uint16_t)(d * s + 0.5));
        state->scratch.dcc_sq16_rows[r][c] = q;
    }
}

/**
 * dcc_set_bounds() - Set bound entries for cluster pair (@i, @j).
 * @state:    Running clustering state.
 * @i:        First cluster index.
 * @j:        Second cluster index.
 * @min_d:    Minimum distance bound.
 * @max_d:    Maximum distance bound.
 * @measured: Non-zero if exactly measured, 0 otherwise.
 */
static inline void dcc_set_bounds(
    ClusterState *state,
    int           i,
    int           j,
    double        min_d,
    double        max_d,
    char          measured)
{
    if (i == j)
    {
        return;
    }
    int r = (i > j) ? i : j;
    int c = (i > j) ? j : i;

    state->scratch.dcc_min_rows[r][c] = min_d;
    if (state->scratch.dcc_max_rows != NULL && state->scratch.dcc_max_rows[r] != NULL)
    {
        state->scratch.dcc_max_rows[r][c] = max_d;
    }
    if (state->scratch.dcc_measured_rows != NULL && state->scratch.dcc_measured_rows[r] != NULL)
    {
        state->scratch.dcc_measured_rows[r][c] = measured;
    }
}

/**
 * dcc_set_min_pair() - Update minimum distance bound for pair (@i, @j).
 * @state: Running clustering state.
 * @i:     First cluster index.
 * @j:     Second cluster index.
 * @min_d: Minimum distance bound.
 */
static inline void dcc_set_min_pair(
    ClusterState *state,
    int           i,
    int           j,
    double        min_d)
{
    if (i == j)
    {
        return;
    }
    int r = (i > j) ? i : j;
    int c = (i > j) ? j : i;
    state->scratch.dcc_min_rows[r][c] = min_d;
}

/**
 * dcc_set_max_pair() - Update maximum distance bound for pair (@i, @j).
 * @state: Running clustering state.
 * @i:     First cluster index.
 * @j:     Second cluster index.
 * @max_d: Maximum distance bound.
 */
static inline void dcc_set_max_pair(
    ClusterState *state,
    int           i,
    int           j,
    double        max_d)
{
    if (i == j)
    {
        return;
    }
    int r = (i > j) ? i : j;
    int c = (i > j) ? j : i;
    if (state->scratch.dcc_max_rows != NULL && state->scratch.dcc_max_rows[r] != NULL)
    {
        state->scratch.dcc_max_rows[r][c] = max_d;
    }
}

/**
 * dcc_init_sparse_unmeasured_pair() - Initialize pair as unmeasured [0, 1e19].
 * @state: Running clustering state.
 * @c1:    First cluster index.
 * @c2:    Second cluster index.
 */
static inline void dcc_init_sparse_unmeasured_pair(
    ClusterState *state,
    int           c1,
    int           c2)
{
    if (c1 == c2)
    {
        return;
    }
    int r = (c1 > c2) ? c1 : c2;
    int c = (c1 > c2) ? c2 : c1;

    state->scratch.dcc_min_rows[r][c] = 0.0;
    if (state->scratch.dcc_max_rows != NULL && state->scratch.dcc_max_rows[r] != NULL)
    {
        state->scratch.dcc_max_rows[r][c] = 1e19;
    }
    if (state->scratch.dcc_measured_rows != NULL && state->scratch.dcc_measured_rows[r] != NULL)
    {
        state->scratch.dcc_measured_rows[r][c] = 0;
    }
    if (state->scratch.dcc_sq16_rows != NULL && state->scratch.dcc_sq16_rows[r] != NULL)
    {
        state->scratch.dcc_sq16_rows[r][c] = DCC_SQ16_UNMEASURED;
    }
}

/**
 * dcc_sync_symmetric_new_cluster() - Symmetrize row into columns (no-op in lower-triangular).
 * @state:  Running clustering state.
 * @new_cl: Index of the newly added cluster.
 */
static inline void dcc_sync_symmetric_new_cluster(
    ClusterState *state,
    int           new_cl)
{
    (void)state;
    (void)new_cl;
}

/**
 * dcc_remove_cluster_swap() - Swap-remove cluster from DCC dynamic rows in O(K).
 * @state: Running clustering state.
 * @u:     Index of cluster being removed.
 */
static inline void dcc_remove_cluster_swap(
    ClusterState *state,
    int           u)
{
    int last = state->num_clusters - 1;
    if (u < 0 || u > last)
    {
        return;
    }

    if (u == last)
    {
        if (state->scratch.dcc_min_rows[last] != NULL)
        {
            free(state->scratch.dcc_min_rows[last]);
            state->scratch.dcc_min_rows[last] = NULL;
        }
        if (state->scratch.dcc_max_rows != NULL && state->scratch.dcc_max_rows[last] != NULL)
        {
            free(state->scratch.dcc_max_rows[last]);
            state->scratch.dcc_max_rows[last] = NULL;
        }
        if (state->scratch.dcc_measured_rows != NULL &&
            state->scratch.dcc_measured_rows[last] != NULL)
        {
            free(state->scratch.dcc_measured_rows[last]);
            state->scratch.dcc_measured_rows[last] = NULL;
        }
        if (state->scratch.dcc_sq16_rows != NULL && state->scratch.dcc_sq16_rows[last] != NULL)
        {
            free(state->scratch.dcc_sq16_rows[last]);
            state->scratch.dcc_sq16_rows[last] = NULL;
        }
        return;
    }

    /* u < last: copy cluster 'last' distances into slot 'u' */

    /* 1. Row u gets distances between cluster 'last' and c < u */
    if (u > 0)
    {
        if (state->scratch.dcc_min_rows[u] != NULL && state->scratch.dcc_min_rows[last] != NULL)
        {
            memcpy(state->scratch.dcc_min_rows[u],
                   state->scratch.dcc_min_rows[last],
                   (size_t)u * sizeof(double));
        }
        if (state->scratch.dcc_max_rows != NULL &&
            state->scratch.dcc_max_rows[u] != NULL &&
            state->scratch.dcc_max_rows[last] != NULL)
        {
            memcpy(state->scratch.dcc_max_rows[u],
                   state->scratch.dcc_max_rows[last],
                   (size_t)u * sizeof(double));
        }
        if (state->scratch.dcc_measured_rows != NULL &&
            state->scratch.dcc_measured_rows[u] != NULL &&
            state->scratch.dcc_measured_rows[last] != NULL)
        {
            memcpy(state->scratch.dcc_measured_rows[u],
                   state->scratch.dcc_measured_rows[last],
                   (size_t)u * sizeof(char));
        }
        if (state->scratch.dcc_sq16_rows != NULL &&
            state->scratch.dcc_sq16_rows[u] != NULL &&
            state->scratch.dcc_sq16_rows[last] != NULL)
        {
            memcpy(state->scratch.dcc_sq16_rows[u],
                   state->scratch.dcc_sq16_rows[last],
                   (size_t)u * sizeof(uint16_t));
        }
    }

    /* 2. For rows r between u + 1 and last - 1, column u receives d(r, last) */
    for (int r = u + 1; r < last; r++)
    {
        if (state->scratch.dcc_min_rows[r] != NULL && state->scratch.dcc_min_rows[last] != NULL)
        {
            state->scratch.dcc_min_rows[r][u] = state->scratch.dcc_min_rows[last][r];
        }
        if (state->scratch.dcc_max_rows != NULL &&
            state->scratch.dcc_max_rows[r] != NULL &&
            state->scratch.dcc_max_rows[last] != NULL)
        {
            state->scratch.dcc_max_rows[r][u] = state->scratch.dcc_max_rows[last][r];
        }
        if (state->scratch.dcc_measured_rows != NULL &&
            state->scratch.dcc_measured_rows[r] != NULL &&
            state->scratch.dcc_measured_rows[last] != NULL)
        {
            state->scratch.dcc_measured_rows[r][u] = state->scratch.dcc_measured_rows[last][r];
        }
        if (state->scratch.dcc_sq16_rows != NULL &&
            state->scratch.dcc_sq16_rows[r] != NULL &&
            state->scratch.dcc_sq16_rows[last] != NULL)
        {
            state->scratch.dcc_sq16_rows[r][u] = state->scratch.dcc_sq16_rows[last][r];
        }
    }

    /* 3. Free row 'last' */
    if (state->scratch.dcc_min_rows[last] != NULL)
    {
        free(state->scratch.dcc_min_rows[last]);
        state->scratch.dcc_min_rows[last] = NULL;
    }
    if (state->scratch.dcc_max_rows != NULL && state->scratch.dcc_max_rows[last] != NULL)
    {
        free(state->scratch.dcc_max_rows[last]);
        state->scratch.dcc_max_rows[last] = NULL;
    }
    if (state->scratch.dcc_measured_rows != NULL &&
        state->scratch.dcc_measured_rows[last] != NULL)
    {
        free(state->scratch.dcc_measured_rows[last]);
        state->scratch.dcc_measured_rows[last] = NULL;
    }
    if (state->scratch.dcc_sq16_rows != NULL && state->scratch.dcc_sq16_rows[last] != NULL)
    {
        free(state->scratch.dcc_sq16_rows[last]);
        state->scratch.dcc_sq16_rows[last] = NULL;
    }
}

/**
 * dcc_remove_cluster() - Shift DCC rows and columns when removing a cluster.
 * @state:           Running clustering state.
 * @index_to_remove: Index of cluster being removed.
 * @sparse_dcc_mode: Non-zero if sparse DCC bounds mode is active.
 */
static inline void dcc_remove_cluster(
    ClusterState *state,
    int           index_to_remove,
    int           sparse_dcc_mode)
{
    (void)sparse_dcc_mode;
    int num_clusters = state->num_clusters;
    int u = index_to_remove;

    /* Free the removed cluster's row */
    if (state->scratch.dcc_min_rows[u] != NULL)
    {
        free(state->scratch.dcc_min_rows[u]);
        state->scratch.dcc_min_rows[u] = NULL;
    }
    if (state->scratch.dcc_max_rows != NULL && state->scratch.dcc_max_rows[u] != NULL)
    {
        free(state->scratch.dcc_max_rows[u]);
        state->scratch.dcc_max_rows[u] = NULL;
    }
    if (state->scratch.dcc_measured_rows != NULL && state->scratch.dcc_measured_rows[u] != NULL)
    {
        free(state->scratch.dcc_measured_rows[u]);
        state->scratch.dcc_measured_rows[u] = NULL;
    }
    if (state->scratch.dcc_sq16_rows != NULL && state->scratch.dcc_sq16_rows[u] != NULL)
    {
        free(state->scratch.dcc_sq16_rows[u]);
        state->scratch.dcc_sq16_rows[u] = NULL;
    }

    /* Shift rows r from u + 1 to num_clusters - 1 down to r - 1 */
    for (int r = u + 1; r < num_clusters; r++)
    {
        size_t count_after_u = (size_t)(r - 1 - u);
        if (count_after_u > 0)
        {
            if (state->scratch.dcc_min_rows[r] != NULL)
            {
                memmove(&state->scratch.dcc_min_rows[r][u],
                        &state->scratch.dcc_min_rows[r][u + 1],
                        count_after_u * sizeof(double));
            }
            if (state->scratch.dcc_max_rows != NULL && state->scratch.dcc_max_rows[r] != NULL)
            {
                memmove(&state->scratch.dcc_max_rows[r][u],
                        &state->scratch.dcc_max_rows[r][u + 1],
                        count_after_u * sizeof(double));
            }
            if (state->scratch.dcc_measured_rows != NULL &&
                state->scratch.dcc_measured_rows[r] != NULL)
            {
                memmove(&state->scratch.dcc_measured_rows[r][u],
                        &state->scratch.dcc_measured_rows[r][u + 1],
                        count_after_u * sizeof(char));
            }
            if (state->scratch.dcc_sq16_rows != NULL && state->scratch.dcc_sq16_rows[r] != NULL)
            {
                memmove(&state->scratch.dcc_sq16_rows[r][u],
                        &state->scratch.dcc_sq16_rows[r][u + 1],
                        count_after_u * sizeof(uint16_t));
            }
        }

        /* Move row pointer to r - 1 */
        state->scratch.dcc_min_rows[r - 1] = state->scratch.dcc_min_rows[r];
        if (state->scratch.dcc_max_rows != NULL)
        {
            state->scratch.dcc_max_rows[r - 1] = state->scratch.dcc_max_rows[r];
        }
        if (state->scratch.dcc_measured_rows != NULL)
        {
            state->scratch.dcc_measured_rows[r - 1] = state->scratch.dcc_measured_rows[r];
        }
        if (state->scratch.dcc_sq16_rows != NULL)
        {
            state->scratch.dcc_sq16_rows[r - 1] = state->scratch.dcc_sq16_rows[r];
        }
    }

    /* Clear the vacated last pointer */
    int last = num_clusters - 1;
    state->scratch.dcc_min_rows[last] = NULL;
    if (state->scratch.dcc_max_rows != NULL)
    {
        state->scratch.dcc_max_rows[last] = NULL;
    }
    if (state->scratch.dcc_measured_rows != NULL)
    {
        state->scratch.dcc_measured_rows[last] = NULL;
    }
    if (state->scratch.dcc_sq16_rows != NULL)
    {
        state->scratch.dcc_sq16_rows[last] = NULL;
    }
}

/**
 * dcc_count_populated_pairs() - Count measured cluster pairs.
 * @state: Running clustering state.
 *
 * Return: Number of unique pairs (i, j) with i > j having measured distance.
 */
static inline uint64_t dcc_count_populated_pairs(const ClusterState *state)
{
    uint64_t count = 0;
    for (int i = 1; i < state->num_clusters; i++)
    {
        for (int j = 0; j < i; j++)
        {
            if (dcc_is_measured(state, i, j))
            {
                count++;
            }
        }
    }
    return count;
}

#endif // CLUSTER_DCC_H
