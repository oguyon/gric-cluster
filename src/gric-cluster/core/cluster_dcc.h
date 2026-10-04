/**
 * @file cluster_dcc.h
 * @brief Accessor functions and abstraction layer for the pairwise inter-cluster
 *        distance (DCC) matrix and bounds cache.
 */

#ifndef CLUSTER_DCC_H
#define CLUSTER_DCC_H

#include "cluster_defs.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

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
    return state->scratch.dcc_min[(size_t)i * state->scratch.dcc_stride + (size_t)j];
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
    return state->scratch.dcc_min[(size_t)i * state->scratch.dcc_stride + (size_t)j];
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
    return state->scratch.dcc_max[(size_t)i * state->scratch.dcc_stride + (size_t)j];
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
    return state->scratch.dcc_measured[(size_t)i * state->scratch.dcc_stride + (size_t)j];
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
    if (state->scratch.dcc_sq16 == NULL)
    {
        return DCC_SQ16_UNMEASURED;
    }
    return state->scratch.dcc_sq16[(size_t)i * state->scratch.dcc_stride + (size_t)j];
}

/**
 * dcc_row_dist() - Get pointer to read-only distance row for cluster @i.
 * @state: Running clustering state.
 * @i:     Cluster index.
 *
 * Return: Const pointer to distance row @i.
 */
static inline const double *dcc_row_dist(
    const ClusterState *state,
    int                 i)
{
    return &state->scratch.dcc_min[(size_t)i * state->scratch.dcc_stride];
}

/**
 * dcc_row_dist_mut() - Get pointer to mutable distance row for cluster @i.
 * @state: Running clustering state.
 * @i:     Cluster index.
 *
 * Return: Mutable pointer to distance row @i.
 */
static inline double *dcc_row_dist_mut(
    ClusterState *state,
    int           i)
{
    return &state->scratch.dcc_min[(size_t)i * state->scratch.dcc_stride];
}

/**
 * dcc_row_max() - Get pointer to read-only maximum bound row for cluster @i.
 * @state: Running clustering state.
 * @i:     Cluster index.
 *
 * Return: Const pointer to max bound row @i.
 */
static inline const double *dcc_row_max(
    const ClusterState *state,
    int                 i)
{
    return &state->scratch.dcc_max[(size_t)i * state->scratch.dcc_stride];
}

/**
 * dcc_row_max_mut() - Get pointer to mutable maximum bound row for cluster @i.
 * @state: Running clustering state.
 * @i:     Cluster index.
 *
 * Return: Mutable pointer to max bound row @i.
 */
static inline double *dcc_row_max_mut(
    ClusterState *state,
    int           i)
{
    return &state->scratch.dcc_max[(size_t)i * state->scratch.dcc_stride];
}

/**
 * dcc_row_measured() - Get pointer to read-only measured flags row for cluster @i.
 * @state: Running clustering state.
 * @i:     Cluster index.
 *
 * Return: Const pointer to measured flags row @i.
 */
static inline const char *dcc_row_measured(
    const ClusterState *state,
    int                 i)
{
    return &state->scratch.dcc_measured[(size_t)i * state->scratch.dcc_stride];
}

/**
 * dcc_row_measured_mut() - Get pointer to mutable measured flags row for cluster @i.
 * @state: Running clustering state.
 * @i:     Cluster index.
 *
 * Return: Mutable pointer to measured flags row @i.
 */
static inline char *dcc_row_measured_mut(
    ClusterState *state,
    int           i)
{
    return &state->scratch.dcc_measured[(size_t)i * state->scratch.dcc_stride];
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
    if (state->scratch.dcc_sq16 == NULL)
    {
        return NULL;
    }
    return &state->scratch.dcc_sq16[(size_t)i * state->scratch.dcc_stride];
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
    if (state->scratch.dcc_sq16 == NULL)
    {
        return NULL;
    }
    return &state->scratch.dcc_sq16[(size_t)i * state->scratch.dcc_stride];
}

/**
 * dcc_set_pair() - Update symmetric pairwise distance cache entries in float and SQ16.
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
    size_t stride = state->scratch.dcc_stride;
    size_t idx12 = (size_t)c1 * stride + (size_t)c2;
    size_t idx21 = (size_t)c2 * stride + (size_t)c1;

    state->scratch.dcc_min[idx12] = d;
    state->scratch.dcc_min[idx21] = d;
    state->scratch.dcc_max[idx12] = d;
    state->scratch.dcc_max[idx21] = d;
    state->scratch.dcc_measured[idx12] = 1;
    state->scratch.dcc_measured[idx21] = 1;

    if (state->scratch.dcc_sq16 != NULL)
    {
        double s = state->scratch.dcc_sq16_scale;
        uint16_t q = (d * s >= 65534.0) ? 65534 : (uint16_t)(d * s + 0.5);
        state->scratch.dcc_sq16[idx12] = q;
        state->scratch.dcc_sq16[idx21] = q;
    }
}

/**
 * dcc_set_bounds() - Set directional bound entries for cluster pair (@i, @j).
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
    size_t stride = state->scratch.dcc_stride;
    size_t idx = (size_t)i * stride + (size_t)j;

    state->scratch.dcc_min[idx] = min_d;
    state->scratch.dcc_max[idx] = max_d;
    state->scratch.dcc_measured[idx] = measured;
}

/**
 * dcc_set_min_pair() - Update symmetric minimum distance bounds for pair (@i, @j).
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
    size_t stride = state->scratch.dcc_stride;
    state->scratch.dcc_min[(size_t)i * stride + (size_t)j] = min_d;
    state->scratch.dcc_min[(size_t)j * stride + (size_t)i] = min_d;
}

/**
 * dcc_set_max_pair() - Update symmetric maximum distance bounds for pair (@i, @j).
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
    size_t stride = state->scratch.dcc_stride;
    state->scratch.dcc_max[(size_t)i * stride + (size_t)j] = max_d;
    state->scratch.dcc_max[(size_t)j * stride + (size_t)i] = max_d;
}

/**
 * dcc_init_sparse_unmeasured_pair() - Initialize symmetric pair as unmeasured [0, 1e19].
 * @state: Running clustering state.
 * @c1:    First cluster index.
 * @c2:    Second cluster index.
 */
static inline void dcc_init_sparse_unmeasured_pair(
    ClusterState *state,
    int           c1,
    int           c2)
{
    dcc_set_bounds(state, c1, c2, 0.0, 1e19, 0);
    dcc_set_bounds(state, c2, c1, 0.0, 1e19, 0);
    if (state->scratch.dcc_sq16 != NULL)
    {
        size_t stride = state->scratch.dcc_stride;
        state->scratch.dcc_sq16[(size_t)c1 * stride + (size_t)c2] = DCC_SQ16_UNMEASURED;
        state->scratch.dcc_sq16[(size_t)c2 * stride + (size_t)c1] = DCC_SQ16_UNMEASURED;
    }
}

/**
 * dcc_sync_symmetric_new_cluster() - Scatter newly populated row @new_cl into symmetric columns.
 * @state:  Running clustering state.
 * @new_cl: Index of the newly added cluster.
 */
static inline void dcc_sync_symmetric_new_cluster(
    ClusterState *state,
    int           new_cl)
{
    size_t stride = state->scratch.dcc_stride;
    double   *dcc_min_row = &state->scratch.dcc_min[(size_t)new_cl * stride];
    double   *dcc_min = state->scratch.dcc_min;
    double   *dcc_max = state->scratch.dcc_max;
    char     *dcc_meas = state->scratch.dcc_measured;
    uint16_t *dcc_sq16 = state->scratch.dcc_sq16;
    uint16_t *dcc_sq16_row = (dcc_sq16 != NULL)
        ? &dcc_sq16[(size_t)new_cl * stride]
        : NULL;

    if (dcc_sq16 != NULL)
    {
        for (int k = 0; k < new_cl; k++)
        {
            size_t col_idx = (size_t)k * stride + (size_t)new_cl;
            double d = dcc_min_row[k];
            dcc_min[col_idx] = d;
            dcc_max[col_idx] = d;
            dcc_meas[col_idx] = 1;
            dcc_sq16[col_idx] = dcc_sq16_row[k];
        }
    }
    else
    {
        for (int k = 0; k < new_cl; k++)
        {
            size_t col_idx = (size_t)k * stride + (size_t)new_cl;
            double d = dcc_min_row[k];
            dcc_min[col_idx] = d;
            dcc_max[col_idx] = d;
            dcc_meas[col_idx] = 1;
        }
    }
}

/**
 * dcc_init_matrix() - Initialize DCC matrix buffers at startup or reset.
 * @state:           Running clustering state.
 * @max_clusters:    Maximum number of clusters.
 * @sparse_dcc_mode: Non-zero if sparse DCC bounds mode is active.
 */
static inline void dcc_init_matrix(
    ClusterState *state,
    size_t        max_clusters,
    int           sparse_dcc_mode)
{
    state->scratch.dcc_stride = max_clusters;
    size_t total = max_clusters * max_clusters;

    if (sparse_dcc_mode)
    {
        for (size_t ii = 0; ii < total; ii++)
        {
            state->scratch.dcc_min[ii] = 0.0;
            state->scratch.dcc_max[ii] = 1e19;
            state->scratch.dcc_measured[ii] = 0;
            if (state->scratch.dcc_sq16 != NULL)
            {
                state->scratch.dcc_sq16[ii] = DCC_SQ16_UNMEASURED;
            }
        }
        for (size_t r = 0; r < max_clusters; r++)
        {
            size_t idx = r * max_clusters + r;
            state->scratch.dcc_min[idx] = 0.0;
            state->scratch.dcc_max[idx] = 0.0;
            state->scratch.dcc_measured[idx] = 1;
            if (state->scratch.dcc_sq16 != NULL)
            {
                state->scratch.dcc_sq16[idx] = 0;
            }
        }
    }
    else
    {
        for (size_t ii = 0; ii < total; ii++)
        {
            state->scratch.dcc_min[ii] = -1.0;
            state->scratch.dcc_max[ii] = -1.0;
            state->scratch.dcc_measured[ii] = 0;
            if (state->scratch.dcc_sq16 != NULL)
            {
                state->scratch.dcc_sq16[ii] = DCC_SQ16_UNMEASURED;
            }
        }
        for (size_t r = 0; r < max_clusters; r++)
        {
            if (state->scratch.dcc_sq16 != NULL)
            {
                state->scratch.dcc_sq16[r * max_clusters + r] = 0;
            }
        }
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
    size_t N = state->scratch.dcc_stride;
    int num_clusters = state->num_clusters;

    // Shift Rows up
    for (int r = index_to_remove; r < num_clusters - 1; r++)
    {
        memcpy(&state->scratch.dcc_min[(size_t)r * N],
               &state->scratch.dcc_min[(size_t)(r + 1) * N],
               N * sizeof(double));
        memcpy(&state->scratch.dcc_max[(size_t)r * N],
               &state->scratch.dcc_max[(size_t)(r + 1) * N],
               N * sizeof(double));
        memcpy(&state->scratch.dcc_measured[(size_t)r * N],
               &state->scratch.dcc_measured[(size_t)(r + 1) * N],
               N * sizeof(char));
        if (state->scratch.dcc_sq16 != NULL)
        {
            memcpy(&state->scratch.dcc_sq16[(size_t)r * N],
                   &state->scratch.dcc_sq16[(size_t)(r + 1) * N],
                   N * sizeof(uint16_t));
        }
    }

    // Shift Columns left for ALL rows
    for (int r = 0; r < num_clusters - 1; r++)
    {
        size_t dest_idx = (size_t)r * N + (size_t)index_to_remove;
        size_t src_idx = (size_t)r * N + (size_t)index_to_remove + 1;
        size_t count = N - 1 - (size_t)index_to_remove;
        if (count > 0)
        {
            memmove(&state->scratch.dcc_min[dest_idx],
                    &state->scratch.dcc_min[src_idx],
                    count * sizeof(double));
            memmove(&state->scratch.dcc_max[dest_idx],
                    &state->scratch.dcc_max[src_idx],
                    count * sizeof(double));
            memmove(&state->scratch.dcc_measured[dest_idx],
                    &state->scratch.dcc_measured[src_idx],
                    count * sizeof(char));
            if (state->scratch.dcc_sq16 != NULL)
            {
                memmove(&state->scratch.dcc_sq16[dest_idx],
                        &state->scratch.dcc_sq16[src_idx],
                        count * sizeof(uint16_t));
            }
        }
    }

    // Clear the now-unused last row/col
    int last = num_clusters - 1;
    for (size_t r = 0; r < N; r++)
    {
        if (state->scratch.dcc_sq16 != NULL)
        {
            state->scratch.dcc_sq16[(size_t)last * N + r] = DCC_SQ16_UNMEASURED;
            state->scratch.dcc_sq16[r * N + (size_t)last] = DCC_SQ16_UNMEASURED;
        }
        if (sparse_dcc_mode)
        {
            state->scratch.dcc_min[(size_t)last * N + r] = 0.0;
            state->scratch.dcc_min[r * N + (size_t)last] = 0.0;
            state->scratch.dcc_max[(size_t)last * N + r] = 1e19;
            state->scratch.dcc_max[r * N + (size_t)last] = 1e19;
            state->scratch.dcc_measured[(size_t)last * N + r] = 0;
            state->scratch.dcc_measured[r * N + (size_t)last] = 0;
        }
        else
        {
            state->scratch.dcc_min[(size_t)last * N + r] = -1.0;
            state->scratch.dcc_min[r * N + (size_t)last] = -1.0;
            state->scratch.dcc_max[(size_t)last * N + r] = -1.0;
            state->scratch.dcc_max[r * N + (size_t)last] = -1.0;
            state->scratch.dcc_measured[(size_t)last * N + r] = 0;
            state->scratch.dcc_measured[r * N + (size_t)last] = 0;
        }
    }
    state->scratch.dcc_min[(size_t)last * N + (size_t)last] = 0.0;
    state->scratch.dcc_max[(size_t)last * N + (size_t)last] = 0.0;
    state->scratch.dcc_measured[(size_t)last * N + (size_t)last] = 1;
    if (state->scratch.dcc_sq16 != NULL)
    {
        state->scratch.dcc_sq16[(size_t)last * N + (size_t)last] = 0;
    }
}

/**
 * dcc_count_populated_pairs() - Count measured cluster pairs.
 * @state: Running clustering state.
 *
 * Return: Number of unique pairs (i, j) with i < j having measured distance.
 */
static inline uint64_t dcc_count_populated_pairs(const ClusterState *state)
{
    uint64_t count = 0;
    for (int i = 0; i < state->num_clusters; i++)
    {
        const char *row = dcc_row_measured(state, i);
        for (int j = i + 1; j < state->num_clusters; j++)
        {
            if (row[j])
            {
                count++;
            }
        }
    }
    return count;
}

#endif // CLUSTER_DCC_H
