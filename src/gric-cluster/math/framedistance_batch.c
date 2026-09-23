/**
 * @file framedistance_batch.c
 * @brief Top-level dispatcher for batched 1-query to N-anchor Euclidean distance calculations.
 */

#include "framedistance.h"
#include "common.h"

/**
 * framedist_batch() - Computes Euclidean distances from query Frame to N candidate Frames.
 * @q:         Pointer to query Frame.
 * @anchors:   Array of pointers to candidate Frames.
 * @n_anchors: Number of candidate Frames.
 * @out_dists: Array of N doubles to receive computed distances.
 */
void framedist_batch(
    const Frame  *q,
    const Frame **anchors,
    int           n_anchors,
    double       *out_dists)
{
    if (n_anchors <= 0)
    {
        return;
    }

    long size = q->width * q->height;
    const void *ptrs_stack[256];
    const void **ptrs = ptrs_stack;

    if (n_anchors > 256)
    {
        ptrs = (const void **)malloc((size_t)n_anchors * sizeof(const void *));
        if (!ptrs)
        {
            return;
        }
    }

    for (int i = 0; i < n_anchors; i++)
    {
        ptrs[i] = anchors[i]->data;
    }

    if (q->is_double)
    {
        framedist_batch_double(
            (const double *)q->data,
            (const double *const *)ptrs,
            n_anchors,
            out_dists,
            size);
    }
    else
    {
        framedist_batch_float(
            (const float *)q->data,
            (const float *const *)ptrs,
            n_anchors,
            out_dists,
            size);
    }

    if (ptrs != ptrs_stack)
    {
        free(ptrs);
    }
}

