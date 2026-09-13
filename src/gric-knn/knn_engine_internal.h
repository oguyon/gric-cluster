/**
 * @file knn_engine_internal.h
 * @brief Internal data structures and prototypes shared across k-NN search modules.
 */

#ifndef KNN_ENGINE_INTERNAL_H
#define KNN_ENGINE_INTERNAL_H

#define _POSIX_C_SOURCE 200809L
#include "knn_defs.h"
#include "knn_heap.h"
#include "knn_reader.h"
#include "knn_tree.h"
#include "cluster_locator.h"
#include "framedistance.h"
#include "scalar_quant.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define GRIC_PREFETCH_T0(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T0)
#elif defined(__GNUC__) || defined(__clang__)
#define GRIC_PREFETCH_T0(addr) __builtin_prefetch((const void *)(addr), 0, 3)
#else
#define GRIC_PREFETCH_T0(addr) ((void)0)
#endif

#define MAX_MEASURED_PIVOTS 16
#define GRAPH_FRONTIER_MAX 256
#define KNN_NUM_BUCKET_LOCKS 4096
#define KNN_BUCKET_LOCK_MASK (KNN_NUM_BUCKET_LOCKS - 1)

/** Cluster candidate record for sorting by ascending lower bound and center proximity */
typedef struct
{
    int    id;
    double lb;
    double dcc;
} ClusterScore;

/** Measured anchor pivot record for Multi-Anchor Pivot Bounding (AESA) */
typedef struct
{
    int    cluster_id;
    double d_anchor;
} MeasuredPivot;

/** Graph frontier node for approximate dynamic search */
typedef struct
{
    long    frame_id;
    double  dist;
    long    parent_id;
    double  parent_dist;
    uint8_t expanded;
} FrontierNode;

/** Priority queue node for cluster graph routing */
typedef struct
{
    int    cluster_id;
    double dist;
} ClusterPqNode;

/** Thread-local scratch buffers for cluster graph routing */
typedef struct
{
    ClusterPqNode *pq;
    uint32_t      *enqueued_tags;
    uint32_t       enqueued_epoch;
    double        *anchor_dists;
    uint8_t       *anchor_is_sq16;
} KnnClusterGraphScratch;

#endif // KNN_ENGINE_INTERNAL_H
