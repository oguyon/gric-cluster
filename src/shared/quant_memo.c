/**
 * @file quant_memo.c
 * @brief Implementation of bounded open-addressing memoization hash table.
 */

#include "quant_memo.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define QUANT_MEMO_MAX_PROBE 64

/**
 * quant_memo_init() - Allocate and initialize a bounded quantized memoization table.
 * @table:      Pointer to QuantizedMemoTable structure.
 * @capacity:   Desired slot capacity (will be rounded up to power of two).
 * @err_radius: Quantization error radius: sqrt(D) * scale * 0.5.
 * @rlim:       Clustering distance threshold.
 *
 * Return: 0 on success, -1 on allocation failure or invalid arguments.
 */
int quant_memo_init(
    QuantizedMemoTable *table,
    size_t              capacity,
    double              err_radius,
    double              rlim)
{
    if (table == NULL)
    {
        return -1;
    }

    /* Enforce minimum capacity of 1024 and round up to next power of two */
    if (capacity < 1024)
    {
        capacity = 1024;
    }

    size_t pow2_cap = 1;
    while (pow2_cap < capacity)
    {
        pow2_cap <<= 1;
    }
    capacity = pow2_cap;

    table->entries = (QuantizedMemoEntry *)calloc(capacity, sizeof(QuantizedMemoEntry));
    if (table->entries == NULL)
    {
        return -1;
    }

    table->capacity = capacity;
    table->mask = capacity - 1;
    table->entry_count = 0;
    table->lookup_count = 0;
    table->hit_count = 0;
    table->eviction_count = 0;
    table->err_radius = err_radius;
    table->rlim = rlim;
    table->allocated_bytes = capacity * sizeof(QuantizedMemoEntry);

    return 0;
}

/**
 * quant_memo_free() - Release allocated memory for the memoization table.
 * @table: Pointer to QuantizedMemoTable to release.
 */
void quant_memo_free(
    QuantizedMemoTable *table)
{
    if (table == NULL)
    {
        return;
    }

    if (table->entries != NULL)
    {
        free(table->entries);
        table->entries = NULL;
    }

    table->capacity = 0;
    table->mask = 0;
    table->entry_count = 0;
    table->allocated_bytes = 0;
}

/**
 * quant_memo_clear() - Reset table contents without reallocating entries.
 * @table: Pointer to QuantizedMemoTable to clear.
 */
void quant_memo_clear(
    QuantizedMemoTable *table)
{
    if (table == NULL || table->entries == NULL)
    {
        return;
    }

    memset(table->entries, 0, table->capacity * sizeof(QuantizedMemoEntry));
    table->entry_count = 0;
    table->lookup_count = 0;
    table->hit_count = 0;
    table->eviction_count = 0;
}

/**
 * quant_memo_lookup() - Probe the table for a matching quantized cell.
 * @table:           Active memoization table.
 * @hash:            Primary 64-bit hash.
 * @hash_aux:        Auxiliary 64-bit hash.
 * @out_cluster_id:  Receives matched cluster index on hit.
 * @out_anchor_dist: Receives cached distance to cluster anchor.
 * @out_needs_verify: Receives 1 if near boundary, 0 if guaranteed inside.
 *
 * Return: 1 on cache hit (sample is duplicate/cell-identical), 0 on cache miss.
 */
int quant_memo_lookup(
    QuantizedMemoTable *table,
    uint64_t            hash,
    uint64_t            hash_aux,
    int                *out_cluster_id,
    float              *out_anchor_dist,
    int                *out_needs_verify)
{
    if (table == NULL || table->entries == NULL || table->entry_count == 0)
    {
        return 0;
    }

    table->lookup_count++;
    size_t idx = (size_t)(hash & table->mask);

    for (size_t step = 0; step < QUANT_MEMO_MAX_PROBE; step++)
    {
        QuantizedMemoEntry *entry = &table->entries[idx];
        if (!entry->occupied)
        {
            return 0;
        }

        if (entry->hash == hash && entry->hash_aux == hash_aux)
        {
            entry->hit_count++;
            table->hit_count++;

            if (out_cluster_id != NULL)
            {
                *out_cluster_id = entry->cluster_id;
            }
            if (out_anchor_dist != NULL)
            {
                *out_anchor_dist = entry->anchor_dist;
            }

            /*
             * Metric Verification Invariant:
             * Maximum distance distortion between continuous space and quantized space
             * across 2 samples in the same quantized cell is bounded by 2 * err_radius.
             * If anchor_dist <= rlim - (2 * err_radius), the sample is strictly inside
             * the cluster covering ball. If greater, verification is required.
             */
            if (out_needs_verify != NULL)
            {
                double safe_cutoff = table->rlim - (2.0 * table->err_radius);
                *out_needs_verify = (entry->anchor_dist > (float)safe_cutoff) ? 1 : 0;
            }

            return 1;
        }

        idx = (idx + 1) & table->mask;
    } // for (size_t step = 0; step < QUANT_MEMO_MAX_PROBE; step++)

    return 0;
}

/**
 * quant_memo_insert() - Insert or update a quantized cell assignment in the table.
 * @table:         Active memoization table.
 * @hash:          Primary 64-bit hash.
 * @hash_aux:      Auxiliary 64-bit hash.
 * @cluster_id:    Assigned cluster index.
 * @anchor_dist:   Measured Euclidean distance to cluster anchor.
 * @current_epoch: Frame index / epoch timestamp.
 *
 * Return: 0 on success, -1 on table error.
 */
int quant_memo_insert(
    QuantizedMemoTable *table,
    uint64_t            hash,
    uint64_t            hash_aux,
    int                 cluster_id,
    float               anchor_dist,
    uint32_t            current_epoch)
{
    if (table == NULL || table->entries == NULL)
    {
        return -1;
    }

    size_t idx = (size_t)(hash & table->mask);
    size_t oldest_idx = idx;
    uint32_t oldest_epoch = UINT32_MAX;

    for (size_t step = 0; step < QUANT_MEMO_MAX_PROBE; step++)
    {
        QuantizedMemoEntry *entry = &table->entries[idx];

        if (!entry->occupied)
        {
            /* Insert into free slot */
            entry->hash = hash;
            entry->hash_aux = hash_aux;
            entry->cluster_id = cluster_id;
            entry->anchor_dist = anchor_dist;
            entry->hit_count = 0;
            entry->last_epoch = current_epoch;
            entry->occupied = 1;
            table->entry_count++;
            return 0;
        }

        if (entry->hash == hash && entry->hash_aux == hash_aux)
        {
            /* Update existing entry */
            entry->cluster_id = cluster_id;
            entry->anchor_dist = anchor_dist;
            entry->last_epoch = current_epoch;
            return 0;
        }

        if (entry->last_epoch < oldest_epoch)
        {
            oldest_epoch = entry->last_epoch;
            oldest_idx = idx;
        }

        idx = (idx + 1) & table->mask;
    } // for (size_t step = 0; step < QUANT_MEMO_MAX_PROBE; step++)

    /* Evict oldest probed entry if probe chain is saturated */
    QuantizedMemoEntry *evict = &table->entries[oldest_idx];
    evict->hash = hash;
    evict->hash_aux = hash_aux;
    evict->cluster_id = cluster_id;
    evict->anchor_dist = anchor_dist;
    evict->hit_count = 0;
    evict->last_epoch = current_epoch;
    evict->occupied = 1;
    table->eviction_count++;

    return 0;
}
