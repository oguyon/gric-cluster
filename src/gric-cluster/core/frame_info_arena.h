/**
 * @file frame_info_arena.h
 * @brief Chunked linear memory arena for per-frame distance and index tracking.
 */

#ifndef FRAME_INFO_ARENA_H
#define FRAME_INFO_ARENA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FRAME_INFO_ARENA_DEFAULT_CHUNK_SIZE (1024 * 1024)

/**
 * struct FrameInfoChunk - Contiguous memory block within the arena.
 * @next:     Pointer to the next chunk in the linked list.
 * @used:     Number of bytes allocated from this chunk.
 * @capacity: Total usable capacity in bytes.
 * @data:     64-byte aligned data buffer.
 */
typedef struct FrameInfoChunk
{
    struct FrameInfoChunk *next;
    size_t                 used;
    size_t                 capacity;
    char                  *data;
} FrameInfoChunk;

/**
 * struct FrameInfoArena - Linked-list chunk arena for frame evaluation logging.
 * @head:               Pointer to the first allocated chunk.
 * @current:            Pointer to the active chunk receiving allocations.
 * @default_chunk_size: Size in bytes for newly allocated chunks.
 */
typedef struct
{
    FrameInfoChunk *head;
    FrameInfoChunk *current;
    size_t          default_chunk_size;
} FrameInfoArena;

/**
 * frame_info_arena_init() - Initialize arena with default chunk size.
 * @arena:      Pointer to arena structure to initialize.
 * @chunk_size: Desired chunk size in bytes (0 for default 1 MiB).
 */
void frame_info_arena_init(
    FrameInfoArena *arena,
    size_t          chunk_size);

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
    size_t          alignment);

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
    double        **out_dists);

/**
 * frame_info_arena_destroy() - Free all chunks associated with arena.
 * @arena: Pointer to arena structure to release.
 */
void frame_info_arena_destroy(
    FrameInfoArena *arena);

#ifdef __cplusplus
}
#endif

#endif // FRAME_INFO_ARENA_H
