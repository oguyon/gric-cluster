/**
 * @file gric_mem.c
 * @brief Memory allocation utilities for large contiguous buffers and huge pages.
 */

#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "gric_mem.h"

#include <stdlib.h>
#include <sys/mman.h>

/**
 * gric_alloc_large - Allocate a memory buffer, backed by huge pages if large enough.
 * @size: Number of bytes to allocate.
 *
 * Allocates 2 MB-aligned memory via posix_memalign() and advises the kernel
 * with MADV_HUGEPAGE if size is at least GRIC_HUGEPAGE_SIZE (2 MB).
 * Falls back to standard malloc() for smaller sizes or if alignment fails.
 *
 * Return: Pointer to allocated memory, or NULL on failure or if size is 0.
 */
void *gric_alloc_large(
    size_t size)
{
    if (size == 0)
    {
        return NULL;
    }

    if (size >= GRIC_HUGEPAGE_SIZE)
    {
        void *ptr = NULL;
        if (posix_memalign(&ptr, GRIC_HUGEPAGE_SIZE, size) == 0)
        {
#if defined(MADV_HUGEPAGE)
            madvise(ptr, size, MADV_HUGEPAGE);
#endif
            return ptr;
        }
    }

    return malloc(size);
}

/**
 * gric_free_large - Free a buffer previously allocated with gric_alloc_large().
 * @ptr: Pointer returned by gric_alloc_large() (may be NULL).
 */
void gric_free_large(
    void *ptr)
{
    free(ptr);
}
