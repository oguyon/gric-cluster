/**
 * @file cluster_trace.c
 * @brief Circular ring buffer implementation for
 *        structured clustering trace events.
 *
 * The buffer is a flat array of TraceEvent structs.
 * New events overwrite the oldest when full.
 */

#include "cluster_trace.h"

#include <stdlib.h>
#include <string.h>

/**
 * trace_buffer_create() - Allocate a trace buffer.
 * @capacity: Number of event slots (must be > 0).
 *
 * Return: Pointer to new buffer, or NULL on failure.
 */
TraceBuffer *trace_buffer_create(int capacity)
{
    if (capacity <= 0)
    {
        return NULL;
    }

    TraceBuffer *tb = calloc(1, sizeof(*tb));
    if (!tb)
    {
        return NULL;
    }

    tb->events = calloc((size_t)capacity,
                         sizeof(TraceEvent));
    if (!tb->events)
    {
        free(tb);
        return NULL;
    }

    tb->capacity    = capacity;
    tb->head        = 0;
    tb->count       = 0;
    tb->frame_start = 0;

    return tb;
}

/**
 * trace_buffer_destroy() - Free a trace buffer.
 * @tb: Buffer to destroy (may be NULL).
 */
void trace_buffer_destroy(TraceBuffer *tb)
{
    if (!tb)
    {
        return;
    }
    free(tb->events);
    free(tb);
}

/**
 * trace_buffer_clear() - Reset buffer to empty.
 * @tb: Buffer to clear (may be NULL).
 */
void trace_buffer_clear(TraceBuffer *tb)
{
    if (!tb)
    {
        return;
    }
    tb->head        = 0;
    tb->count       = 0;
    tb->frame_start = 0;
}

/**
 * trace_buffer_begin_frame() - Mark the start of
 * a new frame's trace events.
 * @tb: Buffer (may be NULL).
 *
 * Saves the current head index as frame_start so
 * that consumers can extract only the events for
 * the most recent frame.
 */
void trace_buffer_begin_frame(TraceBuffer *tb)
{
    if (!tb)
    {
        return;
    }
    tb->frame_start = tb->head;
}

/**
 * trace_emit() - Write a new event to the buffer.
 * @tb:   Buffer (may be NULL; returns NULL).
 * @type: Event type enum value.
 *
 * Zeroes the slot, sets the type field, advances
 * head modulo capacity. The caller fills remaining
 * fields on the returned pointer.
 *
 * Return: Pointer to the new event slot, or NULL
 *         if tb is NULL.
 */
TraceEvent *trace_emit(
    TraceBuffer *tb,
    int          type)
{
    if (!tb)
    {
        return NULL;
    }

    TraceEvent *ev = &tb->events[tb->head];
    memset(ev, 0, sizeof(*ev));
    ev->type = (uint16_t)type;

    tb->head = (tb->head + 1) % tb->capacity;
    if (tb->count < tb->capacity)
    {
        tb->count++;
    }

    return ev;
}
