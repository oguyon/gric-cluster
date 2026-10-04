/**
 * @file gric_rss.c
 * @brief Implementation of rate-limited process resident set size (RSS) query API.
 */

#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "gric_rss.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#ifndef CLOCK_MONOTONIC_COARSE
#define CLOCK_MONOTONIC_COARSE CLOCK_MONOTONIC
#endif

#define GRIC_RSS_SAMPLE_INTERVAL_NS 100000000ULL /* 100 ms */

static int      s_statm_fd       = -1;
static uint64_t s_last_time_ns   = 0;
static uint64_t s_cached_rss_kb  = 0;
static long     s_page_size_kb   = 0;

/**
 * read_statm_rss_kb() - Read resident pages from /proc/self/statm and convert to KB.
 *
 * Return: Memory resident set size in KB, or 0 on error.
 */
static uint64_t read_statm_rss_kb(void)
{
    int fd = __atomic_load_n(&s_statm_fd, __ATOMIC_ACQUIRE);
    if (fd < 0)
    {
        int new_fd = open("/proc/self/statm", O_RDONLY | O_CLOEXEC);
        if (new_fd < 0)
        {
            struct rusage usage;
            if (getrusage(RUSAGE_SELF, &usage) == 0)
            {
                return (uint64_t)usage.ru_maxrss;
            }
            return 0;
        }

        int expected = -1;
        if (__atomic_compare_exchange_n(&s_statm_fd, &expected, new_fd, 0,
                                        __ATOMIC_RELEASE, __ATOMIC_ACQUIRE))
        {
            fd = new_fd;
        }
        else
        {
            close(new_fd);
            fd = __atomic_load_n(&s_statm_fd, __ATOMIC_ACQUIRE);
        }
    } // if (fd < 0)

    char buf[128];
    ssize_t bytes_read = pread(fd, buf, sizeof(buf) - 1, 0);
    if (bytes_read <= 0)
    {
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage) == 0)
        {
            return (uint64_t)usage.ru_maxrss;
        }
        return 0;
    }

    buf[bytes_read] = '\0';
    const char *p = buf;

    /* Skip total program size */
    while (*p >= '0' && *p <= '9')
    {
        p++;
    }
    while (*p == ' ' || *p == '\t')
    {
        p++;
    }

    unsigned long resident_pages = strtoul(p, NULL, 10);

    if (s_page_size_kb <= 0)
    {
        long page_size = sysconf(_SC_PAGESIZE);
        if (page_size <= 0)
        {
            page_size = 4096;
        }
        s_page_size_kb = page_size / 1024;
        if (s_page_size_kb <= 0)
        {
            s_page_size_kb = 4;
        }
    }

    return (uint64_t)resident_pages * (uint64_t)s_page_size_kb;
}

/**
 * gric_rss_kb_sampled() - Query process resident set size in kilobytes with rate-limiting.
 *
 * Checks if 100 ms has elapsed since last measurement using CLOCK_MONOTONIC_COARSE.
 * Returns cached value if within sampling window; otherwise reads /proc/self/statm.
 *
 * Return: Memory resident set size in kilobytes, or 0 on error.
 */
uint64_t gric_rss_kb_sampled(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_COARSE, &ts) == 0)
    {
        uint64_t now_ns = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
        uint64_t last_ns = __atomic_load_n(&s_last_time_ns, __ATOMIC_RELAXED);

        if (now_ns >= last_ns && (now_ns - last_ns) < GRIC_RSS_SAMPLE_INTERVAL_NS)
        {
            uint64_t cached = __atomic_load_n(&s_cached_rss_kb, __ATOMIC_RELAXED);
            if (cached > 0)
            {
                return cached;
            }
        }

        uint64_t rss_kb = read_statm_rss_kb();
        __atomic_store_n(&s_cached_rss_kb, rss_kb, __ATOMIC_RELAXED);
        __atomic_store_n(&s_last_time_ns, now_ns, __ATOMIC_RELAXED);
        return rss_kb;
    } // if (clock_gettime == 0)

    return read_statm_rss_kb();
}

/**
 * gric_rss_mb_sampled() - Query process resident set size in megabytes with rate-limiting.
 *
 * Return: Memory resident set size in megabytes, or 0.0 on error.
 */
double gric_rss_mb_sampled(void)
{
    uint64_t kb = gric_rss_kb_sampled();
    return (double)kb / 1024.0;
}

/**
 * gric_rss_close() - Close cached file descriptor if open.
 */
void gric_rss_close(void)
{
    int fd = __atomic_exchange_n(&s_statm_fd, -1, __ATOMIC_SEQ_CST);
    if (fd >= 0)
    {
        close(fd);
    }
}

#if defined(__GNUC__) || defined(__clang__)
/**
 * gric_rss_destructor() - Cleanup file descriptor at library unload.
 */
__attribute__((destructor))
static void gric_rss_destructor(void)
{
    gric_rss_close();
}
#endif
