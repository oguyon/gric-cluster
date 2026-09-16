#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    void           *data;
    int             is_double;
    long            width;
    long            height;
    int             id;
    uint64_t        cnt0;
    struct timespec atime;
} Frame;

typedef struct
{
    Frame    anchor;     /**< Frame serving as the cluster anchor point */
    int      id;         /**< Unique cluster index identifier */
    double   prob;        /**< Prior frequency probability distribution (CFPD/DFPD) */
    uint8_t *anchor_sq8;  /**< Optional 8-bit quantized anchor buffer */
    int16_t *anchor_sq16; /**< Optional 16-bit quantized anchor buffer */
} Cluster;

typedef struct
{
    int assignment;
    int num_dists;
    int *cluster_indices;
    double *distances;
} FrameInfo;

/**
 * is_ascii_input_mode() - Check if input mode is configured for ASCII text stream.
 *
 * Return: 1 if ASCII input mode is active, 0 otherwise.
 */
int is_ascii_input_mode(void);

#ifdef __cplusplus
}
#endif

#endif // COMMON_H
