#ifndef QUANT_MEMO_H
#define QUANT_MEMO_H

/**
 * @file quant_memo.h
 * @brief Fast open-addressing memoization table for quantized sample vectors.
 */

#include <stddef.h>
#include <stdint.h>

/** Default power-of-two capacity for memoization cache */
#define QUANT_MEMO_DEFAULT_CAPACITY 65536

/**
 * @brief Single slot in the Quantized Memoization Table.
 */
typedef struct
{
    uint64_t hash;        /**< Primary 64-bit hash */
    uint64_t hash_aux;    /**< Auxiliary 64-bit hash for 128-bit collision immunity */
    int      cluster_id;  /**< Assigned cluster identifier */
    float    anchor_dist; /**< Distance from sample to assigned cluster anchor */
    uint32_t hit_count;   /**< Number of sample collisions on this quantized cell */
    uint32_t last_epoch;  /**< Frame epoch when this entry was last accessed */
    uint8_t  occupied;    /**< 1 if slot holds active entry, 0 if empty */
} QuantizedMemoEntry;

/**
 * @brief Bounded open-addressing memoization hash table.
 */
typedef struct
{
    QuantizedMemoEntry *entries;         /**< Pre-allocated flat entry array */
    size_t              capacity;        /**< Power-of-two table capacity */
    size_t              mask;            /**< Bitmask (capacity - 1) */
    size_t              entry_count;     /**< Number of active entries */
    uint64_t            lookup_count;    /**< Total probe queries */
    uint64_t            hit_count;       /**< Total cache hits (duplicate samples) */
    uint64_t            eviction_count;  /**< Displaced entries due to capacity */
    double              err_radius;      /**< Quantization error radius (sqrt(D)*scale*0.5) */
    double              rlim;            /**< Active clustering radius cutoff */
    size_t              allocated_bytes; /**< Total memory in bytes */
} QuantizedMemoTable;

/**
 * @brief Initialize a quantized memoization table.
 */
int quant_memo_init(
    QuantizedMemoTable *table,
    size_t              capacity,
    double              err_radius,
    double              rlim);

/**
 * @brief Release memory associated with a quantized memoization table.
 */
void quant_memo_free(
    QuantizedMemoTable *table);

/**
 * @brief Clear all entries and reset counters in the memoization table.
 */
void quant_memo_clear(
    QuantizedMemoTable *table);

/**
 * @brief Query the memoization table for a matching quantized cell.
 */
int quant_memo_lookup(
    QuantizedMemoTable *table,
    uint64_t            hash,
    uint64_t            hash_aux,
    int                *out_cluster_id,
    float              *out_anchor_dist,
    int                *out_needs_verify);

/**
 * @brief Insert or update a quantized cell assignment in the memoization table.
 */
int quant_memo_insert(
    QuantizedMemoTable *table,
    uint64_t            hash,
    uint64_t            hash_aux,
    int                 cluster_id,
    float               anchor_dist,
    uint32_t            current_epoch);

#endif // QUANT_MEMO_H
