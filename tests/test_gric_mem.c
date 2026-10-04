/**
 * @file test_gric_mem.c
 * @brief Unit tests for gric_alloc_large() and gric_free_large().
 */

#include "gric_mem.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    printf("Testing gric_alloc_large with size 0...\n");
    void *ptr_zero = gric_alloc_large(0);
    assert(ptr_zero == NULL);

    printf("Testing gric_alloc_large with small size (1024 bytes)...\n");
    size_t small_sz = 1024;
    char *ptr_small = (char *)gric_alloc_large(small_sz);
    assert(ptr_small != NULL);
    memset(ptr_small, 0x5A, small_sz);
    for (size_t ii = 0; ii < small_sz; ii++)
    {
        assert(ptr_small[ii] == 0x5A);
    }
    gric_free_large(ptr_small);

    printf("Testing gric_alloc_large with huge page size (4 MB)...\n");
    size_t huge_sz = 4UL * 1024UL * 1024UL;
    uint8_t *ptr_huge = (uint8_t *)gric_alloc_large(huge_sz);
    assert(ptr_huge != NULL);
    assert(((uintptr_t)ptr_huge % GRIC_HUGEPAGE_SIZE) == 0);

    /* Touch first and last page to ensure memory is writable */
    ptr_huge[0] = 0xAA;
    ptr_huge[huge_sz - 1] = 0x55;
    assert(ptr_huge[0] == 0xAA);
    assert(ptr_huge[huge_sz - 1] == 0x55);

    gric_free_large(ptr_huge);

    printf("Testing standard free() on gric_alloc_large allocation...\n");
    void *ptr_free = gric_alloc_large(2UL * 1024UL * 1024UL);
    assert(ptr_free != NULL);
    free(ptr_free);

    printf("Testing gric_free_large(NULL)...\n");
    gric_free_large(NULL);

    printf("All gric_mem tests passed!\n");
    return 0;
}
