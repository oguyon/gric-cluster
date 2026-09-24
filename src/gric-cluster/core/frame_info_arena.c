/**
 * @file frame_info_arena.c
 * @brief Implementation of chunked linear memory arena for frame logging.
 */

#define _POSIX_C_SOURCE 200809L
#include "frame_info_arena.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/**
 * frame_info_arena_init() - Initialize arena with default chunk size.
 * @arena:      Pointer to arena structure to initialize.
 * @chunk_size: Desired chunk size in bytes (0 for default 1 MiB).
 */
void frame_info_arena_init(
    FrameInfoArena *arena,
    size_t          chunk_size)
{
    if (arena == NULL)
    {
        return;
    }

    arena->head = NULL;
    arena->current = NULL;
    arena->default_chunk_size = (chunk_size > 0) ? chunk_size
                                                 : FRAME_INFO_ARENA_DEFAULT_CHUNK_SIZE;
}

/**
 * frame_info_arena_allocate_chunk() - Allocate and link a new chunk into the arena.
 * @arena:        Pointer to arena structure.
 * @min_capacity: Minimum capacity required for this chunk.
 *
 * Return: Pointer to newly allocated chunk, or NULL on allocation failure.
 */
static FrameInfoChunk *frame_info_arena_allocate_chunk(
    FrameInfoArena *arena,
    size_t          min_capacity)
{
    size_t cap = arena->default_chunk_size;
    if (min_capacity > cap)
    {
        cap = min_capacity;
    }

    FrameInfoChunk *chunk = (FrameInfoChunk *)malloc(sizeof(FrameInfoChunk));
    if (chunk == NULL)
    {
        return NULL;
    }

    chunk->data = NULL;
    if (posix_memalign((void **)&chunk->data, 64, cap) != 0)
    {
        chunk->data = (char *)malloc(cap);
        if (chunk->data == NULL)
        {
            free(chunk);
            return NULL;
        }
    }

    chunk->used = 0;
    chunk->capacity = cap;
    chunk->next = NULL;

    if (arena->head == NULL)
    {
        arena->head = chunk;
    }
    else if (arena->current != NULL)
    {
        arena->current->next = chunk;
    }
    arena->current = chunk;

    return chunk;
}

/**
 * frame_info_arena_alloc() - Allocate aligned memory block from arena.
 * @arena:     Pointer to initialized arena.
 * @bytes:     Number of bytes to allocate.
 * @alignment: Alignment requirement in bytes (e.g. 4, 8, 64).
 *
 * Return: Aligned pointer to allocated memory, or NULL on error.
 */
void *frame_info_arena_alloc(
    FrameInfoArena *arena,
    size_t          bytes,
    size_t          alignment)
{
    if (arena == NULL || bytes == 0)
    {
        return NULL;
    }

    if (alignment < sizeof(void *))
    {
        alignment = sizeof(void *);
    }

    if (arena->default_chunk_size == 0)
    {
        arena->default_chunk_size = FRAME_INFO_ARENA_DEFAULT_CHUNK_SIZE;
    }

    FrameInfoChunk *chunk = arena->current;
    if (chunk != NULL)
    {
        uintptr_t curr_addr = (uintptr_t)(chunk->data + chunk->used);
        uintptr_t aligned_addr = (curr_addr + (alignment - 1)) & ~(alignment - 1);
        size_t offset = (size_t)(aligned_addr - (uintptr_t)chunk->data);

        if (offset + bytes <= chunk->capacity)
        {
            chunk->used = offset + bytes;
            return (void *)aligned_addr;
        }
    }

    /* Current chunk full or absent; allocate new chunk */
    chunk = frame_info_arena_allocate_chunk(arena, bytes + alignment);
    if (chunk == NULL)
    {
        return NULL;
    }

    uintptr_t curr_addr = (uintptr_t)chunk->data;
    uintptr_t aligned_addr = (curr_addr + (alignment - 1)) & ~(alignment - 1);
    size_t offset = (size_t)(aligned_addr - curr_addr);
    chunk->used = offset + bytes;

    return (void *)aligned_addr;
}

/**
 * frame_info_arena_alloc_records() - Allocate matched indices and distances arrays.
 * @arena:       Pointer to initialized arena.
 * @count:       Number of records (indices and distances).
 * @out_indices: Output receiving pointer to int array of size count.
 * @out_dists:   Output receiving pointer to double array of size count.
 *
 * Return: 0 on success, -1 on allocation failure.
 */
int frame_info_arena_alloc_records(
    FrameInfoArena *arena,
    int             count,
    int           **out_indices,
    double        **out_dists)
{
    if (arena == NULL || out_indices == NULL || out_dists == NULL)
    {
        return -1;
    }

    if (count <= 0)
    {
        *out_indices = NULL;
        *out_dists = NULL;
        return 0;
    }

    if (arena->default_chunk_size == 0)
    {
        arena->default_chunk_size = FRAME_INFO_ARENA_DEFAULT_CHUNK_SIZE;
    }

    size_t bytes_indices = (size_t)count * sizeof(int);
    size_t bytes_dists = (size_t)count * sizeof(double);
    size_t align_double = sizeof(double);

    FrameInfoChunk *chunk = arena->current;
    if (chunk != NULL)
    {
        uintptr_t base_addr = (uintptr_t)chunk->data;
        uintptr_t curr_addr = base_addr + chunk->used;
        uintptr_t idx_addr = (curr_addr + (sizeof(int) - 1)) & ~(sizeof(int) - 1);
        uintptr_t dist_addr = (idx_addr + bytes_indices + (align_double - 1)) &
                              ~(align_double - 1);
        size_t end_offset = (size_t)((dist_addr + bytes_dists) - base_addr);

        if (end_offset <= chunk->capacity)
        {
            chunk->used = end_offset;
            *out_indices = (int *)idx_addr;
            *out_dists = (double *)dist_addr;
            return 0;
        }
    }

    /* Allocate in new chunk */
    size_t total_needed = bytes_indices + bytes_dists + 2 * align_double;
    chunk = frame_info_arena_allocate_chunk(arena, total_needed);
    if (chunk == NULL)
    {
        *out_indices = NULL;
        *out_dists = NULL;
        return -1;
    }

    uintptr_t base_addr = (uintptr_t)chunk->data;
    uintptr_t idx_addr = (base_addr + (sizeof(int) - 1)) & ~(sizeof(int) - 1);
    uintptr_t dist_addr = (idx_addr + bytes_indices + (align_double - 1)) &
                          ~(align_double - 1);
    size_t end_offset = (size_t)((dist_addr + bytes_dists) - base_addr);

    chunk->used = end_offset;
    *out_indices = (int *)idx_addr;
    *out_dists = (double *)dist_addr;

    return 0;
}

/**
 * frame_info_arena_destroy() - Free all chunks associated with arena.
 * @arena: Pointer to arena structure to release.
 */
void frame_info_arena_destroy(
    FrameInfoArena *arena)
{
    if (arena == NULL)
    {
        return;
    }

    FrameInfoChunk *chunk = arena->head;
    while (chunk != NULL)
    {
        FrameInfoChunk *next = chunk->next;
        if (chunk->data != NULL)
        {
            free(chunk->data);
            chunk->data = NULL;
        }
        free(chunk);
        chunk = next;
    } // while (chunk != NULL)

    arena->head = NULL;
    arena->current = NULL;
}
