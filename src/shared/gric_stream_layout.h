/**
 * @file gric_stream_layout.h
 * @brief Canonical stream layout and telemetry packet definitions for ImageStreamIO streams.
 */

#ifndef GRIC_STREAM_LAYOUT_H
#define GRIC_STREAM_LAYOUT_H

#include <stddef.h>

/**
 * GRIC_ASSIGN_FIELDS: Canonical field list for the 1D assignment telemetry stream.
 * Columns:
 * 1. NAME:  Identifier suffix for enum gric_assign_field
 * 2. IDX:   Array index in the float vector
 * 3. KEY:   Canonical JSON key name
 * 4. DESCR: Human-readable description
 */
#define GRIC_ASSIGN_FIELDS(X) \
    X(FRAME_INDEX,    0, "frame_index",    "Input frame counter") \
    X(CLUSTER_ID,     1, "cluster_id",     "Assigned cluster index") \
    X(DIST,           2, "distance",       "Distance to assigned anchor") \
    X(IS_NEW,         3, "is_new_anchor",  "1 if a new cluster was created") \
    X(TOTAL_CLUSTERS, 4, "total_clusters", "Clusters after this frame") \
    X(LATENCY_US,     5, "latency_us",     "Per-frame processing latency (us)") \
    X(WITHIN_RLIM,    6, "within_rlim",    "1 if distance <= rlim") \
    X(QUERY_MODE,     7, "query_mode",     "1 if FPS runs in query mode")

#define GRIC_ASSIGN_ENUM_X(NAME, IDX, KEY, DESCR) GRIC_ASSIGN_##NAME = IDX,

enum gric_assign_field
{
    GRIC_ASSIGN_FIELDS(GRIC_ASSIGN_ENUM_X)
    GRIC_ASSIGN_NFIELDS
};

#undef GRIC_ASSIGN_ENUM_X

struct gric_assign_sample
{
    float frame_index;
    float cluster_id;
    float distance;
    float is_new_anchor;
    float total_clusters;
    float latency_us;
    float within_rlim;
    float query_mode;
};

/**
 * gric_assign_pack() - Pack sample struct into float telemetry array.
 * @dst: Destination float array of at least GRIC_ASSIGN_NFIELDS elements.
 * @s:   Source sample structure.
 */
static inline void gric_assign_pack(
    float                           *dst,
    const struct gric_assign_sample *s)
{
    if (dst == NULL || s == NULL)
    {
        return;
    }
    dst[GRIC_ASSIGN_FRAME_INDEX]    = s->frame_index;
    dst[GRIC_ASSIGN_CLUSTER_ID]     = s->cluster_id;
    dst[GRIC_ASSIGN_DIST]           = s->distance;
    dst[GRIC_ASSIGN_IS_NEW]         = s->is_new_anchor;
    dst[GRIC_ASSIGN_TOTAL_CLUSTERS] = s->total_clusters;
    dst[GRIC_ASSIGN_LATENCY_US]     = s->latency_us;
    dst[GRIC_ASSIGN_WITHIN_RLIM]    = s->within_rlim;
    dst[GRIC_ASSIGN_QUERY_MODE]     = s->query_mode;
}

#endif /* GRIC_STREAM_LAYOUT_H */
