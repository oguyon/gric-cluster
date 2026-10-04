/**
 * @file gric_mem.h
 * @brief Memory allocation utilities for large contiguous buffers and huge pages.
 */

#ifndef GRIC_MEM_H
#define GRIC_MEM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Huge page alignment and size threshold (2 MB on Linux x86_64). */
#define GRIC_HUGEPAGE_SIZE (2UL * 1024UL * 1024UL)

/**
 * gric_alloc_large - Allocate a memory buffer, backed by huge pages if large enough.
 * @size: Number of bytes to allocate.
 *
 * For allocations >= 2 MB, aligns to 2 MB boundaries using posix_memalign
 * and issues madvise(MADV_HUGEPAGE) to advise transparent huge page backing.
 * Smaller allocations fall back to standard malloc().
 *
 * Return: Pointer to allocated memory, or NULL on failure or if size is 0.
 */
void *gric_alloc_large(
    size_t size);

/**
 * gric_free_large - Free a buffer previously allocated with gric_alloc_large().
 * @ptr: Pointer returned by gric_alloc_large() (may be NULL).
 *
 * Standard free() may also be called directly on pointers returned by
 * gric_alloc_large().
 */
void gric_free_large(
    void *ptr);

#ifdef __cplusplus
}
#endif

#endif /* GRIC_MEM_H */
