/**
 * @file knn_parser.h
 * @brief Parsers for Pass 1 clustering artifacts.
 *
 * Declares parsing routines for extracting cluster structure and metric metadata from
 * Pass 1 clustering artifacts. Covers binary and ASCII membership files, cluster radii,
 * execution logs, and distance-to-cluster-center (DCC) tables.
 */

#ifndef KNN_PARSER_H
#define KNN_PARSER_H

#include "knn_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * knn_compare_member_meta_radii() - Sort member records by ascending anchor distance.
 * @a: Pointer to first MemberMeta.
 * @b: Pointer to second MemberMeta.
 *
 * Return: -1 if a < b, 1 if a > b, 0 if equal.
 */
int knn_compare_member_meta_radii(
    const void *a,
    const void *b);

/**
 * knn_parse_membership_file() - Load frame membership and anchor distance metadata.
 * @path:  Path to frame_membership.txt or binary equivalent.
 * @model: Pointer to KnnModel to populate.
 *
 * Return: 0 on success, -1 on error.
 */
int knn_parse_membership_file(
    const char *path,
    KnnModel   *model);

/**
 * knn_parse_radii_file() - Parse cluster boundary radii from cluster_radii.txt.
 * @path:  Path to cluster_radii.txt.
 * @model: Pointer to KnnModel to populate.
 */
void knn_parse_radii_file(
    const char *path,
    KnnModel   *model);

/**
 * knn_parse_cluster_log() - Extract dataset size and dimension from cluster_log.txt.
 * @path:  Path to cluster_log.txt.
 * @model: Pointer to KnnModel to populate.
 */
void knn_parse_cluster_log(
    const char *path,
    KnnModel   *model);

/**
 * knn_parse_dcc_file() - Load inter-cluster center distance matrix DCC.
 * @path:  Path to cluster_dcc.txt or binary equivalent.
 * @model: Pointer to KnnModel to populate.
 *
 * Return: 0 on success, -1 on error.
 */
int knn_parse_dcc_file(
    const char *path,
    KnnModel   *model);

#ifdef __cplusplus
}
#endif

#endif // KNN_PARSER_H
