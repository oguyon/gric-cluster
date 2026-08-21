/**
 * @file trace_buffer_test.c
 * @brief Unit tests for the circular trace buffer.
 *
 * Verifies ring buffer wrap-around, capacity
 * capping, begin_frame index tracking, and clear.
 */

#include "cluster_trace.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static void test_create_destroy(void)
{
    TraceBuffer *tb = trace_buffer_create(16);
    assert(tb != NULL);
    assert(tb->capacity == 16);
    assert(tb->head == 0);
    assert(tb->count == 0);
    trace_buffer_destroy(tb);

    /* NULL capacity */
    tb = trace_buffer_create(0);
    assert(tb == NULL);

    /* Negative capacity */
    tb = trace_buffer_create(-1);
    assert(tb == NULL);

    /* Destroy NULL is safe */
    trace_buffer_destroy(NULL);

    printf("  PASS: create_destroy\n");
}

static void test_emit_basic(void)
{
    TraceBuffer *tb = trace_buffer_create(4);

    TraceEvent *ev = trace_emit(tb, TRACE_MATCH);
    assert(ev != NULL);
    assert(ev->type == TRACE_MATCH);
    assert(ev->reason == 0);
    assert(ev->distance == 0.0);
    assert(tb->head == 1);
    assert(tb->count == 1);

    ev->cluster_id = 42;
    ev->distance = 1.5;
    assert(tb->events[0].cluster_id == 42);
    assert(tb->events[0].distance == 1.5);

    printf("  PASS: emit_basic\n");
    trace_buffer_destroy(tb);
}

static void test_wraparound(void)
{
    TraceBuffer *tb = trace_buffer_create(4);

    for (int i = 0; i < 6; i++)
    {
        TraceEvent *ev = trace_emit(tb, TRACE_MISMATCH);
        ev->frame_id = i;
    }

    /* After 6 writes into capacity-4 buffer:
     *   head = 6 % 4 = 2
     *   count = 4 (capped)
     *   slots: [4, 5, 2, 3] (oldest at head) */
    assert(tb->head == 2);
    assert(tb->count == 4);
    assert(tb->events[0].frame_id == 4);
    assert(tb->events[1].frame_id == 5);
    assert(tb->events[2].frame_id == 2);
    assert(tb->events[3].frame_id == 3);

    printf("  PASS: wraparound\n");
    trace_buffer_destroy(tb);
}

static void test_begin_frame(void)
{
    TraceBuffer *tb = trace_buffer_create(16);

    trace_emit(tb, TRACE_MATCH);
    trace_emit(tb, TRACE_PRUNE_3P);
    assert(tb->frame_start == 0);

    trace_buffer_begin_frame(tb);
    assert(tb->frame_start == 2);

    trace_emit(tb, TRACE_TARGET_SELECTED);
    trace_emit(tb, TRACE_MISMATCH);
    trace_emit(tb, TRACE_PRUNE_3P);

    /* frame_start should still be 2 */
    assert(tb->frame_start == 2);
    assert(tb->head == 5);

    /* Current frame events: indices 2, 3, 4 */
    assert(tb->events[2].type == TRACE_TARGET_SELECTED);
    assert(tb->events[3].type == TRACE_MISMATCH);
    assert(tb->events[4].type == TRACE_PRUNE_3P);

    printf("  PASS: begin_frame\n");
    trace_buffer_destroy(tb);
}

static void test_clear(void)
{
    TraceBuffer *tb = trace_buffer_create(8);

    trace_emit(tb, TRACE_MATCH);
    trace_emit(tb, TRACE_MISMATCH);
    trace_buffer_begin_frame(tb);
    trace_emit(tb, TRACE_NEW_CLUSTER);

    assert(tb->count == 3);
    assert(tb->head == 3);

    trace_buffer_clear(tb);
    assert(tb->count == 0);
    assert(tb->head == 0);
    assert(tb->frame_start == 0);

    /* Clear NULL is safe */
    trace_buffer_clear(NULL);

    printf("  PASS: clear\n");
    trace_buffer_destroy(tb);
}

static void test_null_safety(void)
{
    /* All functions should handle NULL gracefully */
    trace_buffer_begin_frame(NULL);
    TraceEvent *ev = trace_emit(NULL, TRACE_MATCH);
    assert(ev == NULL);

    printf("  PASS: null_safety\n");
}

static void test_candidate_entries(void)
{
    TraceBuffer *tb = trace_buffer_create(4);

    TraceEvent *ev = trace_emit(tb, TRACE_TARGET_SELECTED);
    ev->num_candidates = 3;
    ev->candidates[0].id = 5;
    ev->candidates[0].prob = 0.6;
    ev->candidates[0].expected_h = 1.2;
    ev->candidates[0].info_gain = 0.8;
    ev->candidates[1].id = 2;
    ev->candidates[1].prob = 0.3;
    ev->candidates[2].id = 7;
    ev->candidates[2].prob = 0.1;

    assert(tb->events[0].num_candidates == 3);
    assert(tb->events[0].candidates[0].id == 5);
    assert(tb->events[0].candidates[0].prob == 0.6);
    assert(tb->events[0].candidates[2].id == 7);

    printf("  PASS: candidate_entries\n");
    trace_buffer_destroy(tb);
}

int main(void)
{
    printf("trace_buffer_test:\n");
    test_create_destroy();
    test_emit_basic();
    test_wraparound();
    test_begin_frame();
    test_clear();
    test_null_safety();
    test_candidate_entries();
    printf("All tests passed.\n");
    return 0;
}
