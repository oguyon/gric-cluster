/**
 * @file gric_rss.h
 * @brief Rate-limited process resident set size (RSS) query API.
 */

#ifndef GRIC_RSS_H
#define GRIC_RSS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * gric_rss_kb_sampled() - Query process resident set size in kilobytes with rate-limiting.
 *
 * Return: Memory resident set size in kilobytes, or 0 on error.
 */
uint64_t gric_rss_kb_sampled(void);

/**
 * gric_rss_mb_sampled() - Query process resident set size in megabytes with rate-limiting.
 *
 * Return: Memory resident set size in megabytes, or 0.0 on error.
 */
double gric_rss_mb_sampled(void);

/**
 * gric_rss_close() - Close cached file descriptor if open.
 */
void gric_rss_close(void);

#ifdef __cplusplus
}
#endif

#endif /* GRIC_RSS_H */
