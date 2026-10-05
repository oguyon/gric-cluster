/**
 * @file cluster_trace.h
 * @brief Structured circular trace buffer for
 *        step-by-step clustering diagnostics.
 *
 * Provides a fixed-size ring buffer of TraceEvent
 * structs that the C clustering engine writes into
 * at each decision point. The WASM API exposes the
 * buffer for zero-copy reads by the JS explain UI.
 *
 * When the TraceBuffer pointer in ClusterState is
 * NULL (the default), no events are emitted and there
 * is zero overhead on the hot path.
 */

#ifndef CLUSTER_TRACE_H
#define CLUSTER_TRACE_H

#include <stdint.h>

/**
 * enum trace_event_type - Decision point categories.
 *
 * Each value maps to a JS explain step type string
 * for rendering in the Narrative Inspector panel.
 */
enum trace_event_type
{
    TRACE_FRAME_INGEST = 0,
    TRACE_INITIAL_CLUSTER,
    TRACE_TARGET_SELECTED,
    TRACE_MATCH,
    TRACE_MISMATCH,
    TRACE_PRUNE_3P,
    TRACE_PRUNE_4P,
    TRACE_PRUNE_5P,
    TRACE_NEW_CLUSTER,
    TRACE_EVICT_STOP,
    TRACE_EVICT_DISCARD,
    TRACE_EVICT_MERGE,
    TRACE_ENTROPY_GATE,
    TRACE_PRIOR_MIXING,
};

/**
 * enum trace_selection_reason - Why a target was
 * chosen for measurement.
 */
enum trace_selection_reason
{
    REASON_GREEDY_STATIC = 0,
    REASON_GREEDY_DYNAMIC,
    REASON_PREDICTION,
    REASON_ENTROPY_FULL,
    REASON_ENTROPY_FAST,
    REASON_ENTROPY_GATED,
    REASON_LEADER_SHORTCUT,
};

/** Max entropy candidates stored per event. */
#define TRACE_MAX_CANDIDATES 8

/**
 * struct TraceCandidateEntry - Compact entropy
 * candidate ranking entry.
 * @id:         Cluster index.
 * @prob:       Prior probability.
 * @expected_h: Expected residual entropy (bits).
 * @info_gain:  Information gain (bits).
 */
typedef struct
{
    int    id;
    double prob;
    double expected_h;
    double info_gain;
} TraceCandidateEntry;

/**
 * struct TraceEvent - Fixed-size trace event.
 * @type:             enum trace_event_type.
 * @reason:           enum trace_selection_reason.
 * @frame_id:         Monotonic frame counter.
 * @cluster_id:       Primary cluster involved.
 * @distance:         Measured Euclidean distance.
 * @rlim:             Active radius threshold.
 * @lower_bound:      Pruning or gating bound value.
 * @entropy_h:        Current Shannon entropy (bits).
 * @active_remaining: Active candidates after event.
 * @pruned_count:     Candidates pruned by this event.
 * @num_candidates:   Entries in candidates[].
 * @candidates:       Top entropy candidate rankings.
 */
typedef struct
{
    uint16_t type;
    uint16_t reason;
    int      frame_id;
    int      cluster_id;
    double   distance;
    double   rlim;
    double   lower_bound;
    double   entropy_h;
    int      active_remaining;
    int      pruned_count;
    int      num_candidates;
    TraceCandidateEntry candidates[TRACE_MAX_CANDIDATES];
} TraceEvent;

/**
 * struct TraceBuffer - Circular ring buffer of
 * TraceEvent entries.
 * @events:      Contiguous array of capacity entries.
 * @capacity:    Maximum number of entries.
 * @head:        Next write index (modulo capacity).
 * @count:       Total events written (capped).
 * @frame_start: Head index at last begin_frame().
 */
typedef struct TraceBuffer
{
    TraceEvent *events;
    int         capacity;
    int         head;
    int         count;
    int         frame_start;
} TraceBuffer;

/**
 * trace_buffer_create() - Allocate a trace buffer.
 * @capacity: Number of event slots.
 *
 * Return: Pointer to new buffer, or NULL on failure.
 */
TraceBuffer *trace_buffer_create(int capacity);

/**
 * trace_buffer_destroy() - Free a trace buffer.
 * @tb: Buffer to destroy (may be NULL).
 */
void trace_buffer_destroy(TraceBuffer *tb);

/**
 * trace_buffer_clear() - Reset buffer to empty.
 * @tb: Buffer to clear.
 */
void trace_buffer_clear(TraceBuffer *tb);

/**
 * trace_buffer_begin_frame() - Mark the start of a
 * new frame's events. Saves head as frame_start.
 * @tb: Buffer (may be NULL).
 */
void trace_buffer_begin_frame(TraceBuffer *tb);

/**
 * trace_emit() - Write a new event to the buffer.
 * @tb:   Buffer (may be NULL; returns NULL).
 * @type: Event type enum value.
 *
 * Zeroes the slot, sets the type field, advances
 * head. Caller fills remaining fields on the
 * returned pointer.
 *
 * Return: Pointer to the new event slot, or NULL.
 */
TraceEvent *trace_emit(
    TraceBuffer *tb,
    int          type);

#endif /* CLUSTER_TRACE_H */
