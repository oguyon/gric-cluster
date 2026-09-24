/**
 * @file test_frame_info_arena.c
 * @brief Unit tests for FrameInfoArena chunked linear memory arena.
 */

#include "frame_info_arena.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static void test_arena_lifecycle(void)
{
    printf("Testing arena lifecycle...\n");

    FrameInfoArena arena;
    frame_info_arena_init(&arena, 4096);
    assert(arena.head == NULL);
    assert(arena.current == NULL);
    assert(arena.default_chunk_size == 4096);

    /* Allocate small buffer */
    void *ptr1 = frame_info_arena_alloc(&arena, 64, 16);
    assert(ptr1 != NULL);
    assert(((uintptr_t)ptr1 % 16) == 0);
    assert(arena.head != NULL);
    assert(arena.head == arena.current);

    /* Allocate another small buffer from same chunk */
    void *ptr2 = frame_info_arena_alloc(&arena, 128, 8);
    assert(ptr2 != NULL);
    assert(((uintptr_t)ptr2 % 8) == 0);
    assert(arena.head == arena.current);

    frame_info_arena_destroy(&arena);
    assert(arena.head == NULL);
    assert(arena.current == NULL);
}

static void test_arena_chunk_growth(void)
{
    printf("Testing arena multi-chunk growth...\n");

    FrameInfoArena arena;
    /* Use tiny chunk size 512 bytes to force frequent chunk allocations */
    frame_info_arena_init(&arena, 512);

    int *prev_indices = NULL;
    double *prev_dists = NULL;

    for (int iter = 0; iter < 100; iter++)
    {
        int count = 10;
        int *indices = NULL;
        double *dists = NULL;

        int rc = frame_info_arena_alloc_records(&arena, count, &indices, &dists);
        assert(rc == 0);
        assert(indices != NULL);
        assert(dists != NULL);
        assert(((uintptr_t)indices % sizeof(int)) == 0);
        assert(((uintptr_t)dists % sizeof(double)) == 0);

        for (int i = 0; i < count; i++)
        {
            indices[i] = iter * 100 + i;
            dists[i] = (double)(iter * 100 + i) * 0.125;
        }

        if (iter > 0)
        {
            /* Verify previous iteration was not corrupted */
            assert(prev_indices[0] == (iter - 1) * 100);
            assert(prev_dists[0] == (double)((iter - 1) * 100) * 0.125);
        }

        prev_indices = indices;
        prev_dists = dists;
    } // for (int iter = 0; iter < 100; iter++)

    /* Verify multiple chunks were allocated */
    assert(arena.head != NULL);
    assert(arena.head != arena.current);

    frame_info_arena_destroy(&arena);
}

static void test_arena_edge_cases(void)
{
    printf("Testing arena edge cases...\n");

    FrameInfoArena arena;
    frame_info_arena_init(&arena, 0);
    assert(arena.default_chunk_size == FRAME_INFO_ARENA_DEFAULT_CHUNK_SIZE);

    /* Zero count should succeed and return NULL */
    int *indices = (int *)0x123;
    double *dists = (double *)0x456;
    int rc = frame_info_arena_alloc_records(&arena, 0, &indices, &dists);
    assert(rc == 0);
    assert(indices == NULL);
    assert(dists == NULL);

    /* Allocation larger than default chunk size */
    int huge_count = 500000;
    rc = frame_info_arena_alloc_records(&arena, huge_count, &indices, &dists);
    assert(rc == 0);
    assert(indices != NULL);
    assert(dists != NULL);

    indices[0] = 42;
    indices[huge_count - 1] = 999;
    dists[0] = 3.14159;
    dists[huge_count - 1] = 2.71828;

    assert(indices[0] == 42);
    assert(indices[huge_count - 1] == 999);
    assert(dists[0] == 3.14159);
    assert(dists[huge_count - 1] == 2.71828);

    /* NULL arena safety */
    assert(frame_info_arena_alloc(NULL, 10, 8) == NULL);
    assert(frame_info_arena_alloc_records(NULL, 10, &indices, &dists) == -1);
    frame_info_arena_destroy(NULL);

    frame_info_arena_destroy(&arena);
}

int main(void)
{
    printf("=========================================\n");
    printf("   Running FrameInfoArena Unit Tests\n");
    printf("=========================================\n");

    test_arena_lifecycle();
    test_arena_chunk_growth();
    test_arena_edge_cases();

    printf("All FrameInfoArena unit tests passed!\n");
    return 0;
}
