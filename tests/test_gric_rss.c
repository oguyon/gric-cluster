/**
 * @file test_gric_rss.c
 * @brief Unit tests for rate-limited RSS sampling API.
 */

#include "gric_rss.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

int main(void)
{
    printf("Testing gric_rss...\n");

    /* 1. Basic query returns non-zero RSS */
    uint64_t rss_kb1 = gric_rss_kb_sampled();
    printf("Initial RSS: %lu KB\n", (unsigned long)rss_kb1);
    assert(rss_kb1 > 0);

    /* 2. MB conversion matches KB */
    double rss_mb1 = gric_rss_mb_sampled();
    printf("Initial RSS: %.2f MB\n", rss_mb1);
    assert(rss_mb1 > 0.0);
    assert(fabs(rss_mb1 - ((double)rss_kb1 / 1024.0)) < 1e-3);

    /* 3. Fast cached path in tight loop */
    struct timespec t0;
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < 10000; i++)
    {
        uint64_t cached_kb = gric_rss_kb_sampled();
        assert(cached_kb == rss_kb1);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_us = (double)(t1.tv_sec - t0.tv_sec) * 1e6 +
                        (double)(t1.tv_nsec - t0.tv_nsec) / 1e3;
    printf("10,000 cached RSS calls took %.2f us (%.2f ns/call)\n",
           elapsed_us, (elapsed_us * 1000.0) / 10000.0);

    /* 4. Refresh after sampling interval (110 ms) */
    usleep(110000);
    uint64_t rss_kb2 = gric_rss_kb_sampled();
    assert(rss_kb2 > 0);

    /* 5. Reopen after explicit close */
    gric_rss_close();
    uint64_t rss_kb3 = gric_rss_kb_sampled();
    assert(rss_kb3 > 0);

    printf("gric_rss tests passed successfully.\n");
    return 0;
}
