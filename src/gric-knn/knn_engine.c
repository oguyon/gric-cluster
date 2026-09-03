/**
 * @file knn_engine.c
 * @brief High-performance metric-pruned k-NN solver engine.
 */

#define _POSIX_C_SOURCE 200809L
#include "knn_engine.h"
#include "knn_heap.h"
#include "knn_reader.h"
#include "knn_tree.h"
#include "cluster_locator.h"
#include <alloca.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

/** Cluster candidate record for sorting by ascending lower bound and center proximity */
typedef struct
{
    int    id;
    double lb;
    double dcc;
} ClusterScore;

#define MAX_MEASURED_PIVOTS 8
#define GRAPH_FRONTIER_MAX 256

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
    uint8_t expanded;
} FrontierNode;

/**
 * compare_cluster_scores() - Sort cluster candidates by ascending lower bound.
 * @a: Pointer to first ClusterScore.
 * @b: Pointer to second ClusterScore.
 *
 * Return: -1 if a < b, 1 if a > b, 0 if equal.
 */
static int compare_cluster_scores(
    const void *a,
    const void *b)
{
    const ClusterScore *ca = (const ClusterScore *)a;
    const ClusterScore *cb = (const ClusterScore *)b;
    if (ca->lb < cb->lb)
    {
        return -1;
    }
    if (ca->lb > cb->lb)
    {
        return 1;
    }
    if (ca->dcc < cb->dcc)
    {
        return -1;
    }
    if (ca->dcc > cb->dcc)
    {
        return 1;
    }
    return 0;
}

/**
 * find_member_lower_bound() - Binary search for first member with r_anchor >= val.
 * @members: Sorted array of MemberMeta records.
 * @n:       Total number of members.
 * @val:     Threshold radius.
 *
 * Return: Index in [0, n] of first element >= val.
 */
static inline int find_member_lower_bound(
    const MemberMeta *members,
    int               n,
    float             val)
{
    int low = 0;
    int high = n;
    while (low < high)
    {
        int mid = low + ((high - low) >> 1);
        if (members[mid].r_anchor < val)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
    }
    return low;
}

/**
 * find_member_upper_bound() - Binary search for first member with r_anchor > val.
 * @members: Sorted array of MemberMeta records.
 * @n:       Total number of members.
 * @val:     Threshold radius.
 *
 * Return: Index in [0, n] of first element > val.
 */
static inline int find_member_upper_bound(
    const MemberMeta *members,
    int               n,
    float             val)
{
    int low = 0;
    int high = n;
    while (low < high)
    {
        int mid = low + ((high - low) >> 1);
        if (members[mid].r_anchor <= val)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
    }
    return low;
}

/**
 * compute_euclidean_distance() - Vectorized Euclidean distance between frames.
 * @da:   Pointer to first pixel array.
 * @db:   Pointer to second pixel array.
 * @size: Number of elements in frame.
 *
 * Return: Euclidean L2 distance.
 */
static inline double compute_euclidean_distance(
    const double *restrict da,
    const double *restrict db,
    long                   size)
{
    double sum = 0.0;
    long i = 0;

#if defined(__AVX__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    if (size >= 4)
    {
        __m256d sum_vec = _mm256_setzero_pd();
        for (; i <= size - 4; i += 4)
        {
            __m256d va = _mm256_loadu_pd(&da[i]);
            __m256d vb = _mm256_loadu_pd(&db[i]);
            __m256d diff = _mm256_sub_pd(va, vb);
#ifdef __FMA__
            sum_vec = _mm256_fmadd_pd(diff, diff, sum_vec);
#else
            sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(diff, diff));
#endif
        }
        __m256d hsum = _mm256_hadd_pd(sum_vec, sum_vec);
        sum += ((double *)&hsum)[0] + ((double *)&hsum)[2];
    }
#endif

    for (; i < size; i++)
    {
        double diff = da[i] - db[i];
        sum += diff * diff;
    }

    return sqrt(sum);
}

/**
 * check_temporal_separation() - Verify if candidate satisfies temporal criteria.
 * @query_id:     Frame ID of query.
 * @candidate_id: Frame ID of candidate.
 * @config:       Active KnnConfig.
 *
 * Return: 1 if candidate is temporally valid, 0 if excluded.
 */
static inline int check_temporal_separation(
    long             query_id,
    long             candidate_id,
    const KnnConfig *config)
{
    if (config->query_data_path != NULL)
    {
        return 1; // In cross-dataset mode, queries have no temporal continuity with candidates
    }

    if (query_id == candidate_id)
    {
        return 0; // Self-match exclusion
    }

    if (config->past_only && candidate_id > query_id)
    {
        return 0;
    }

    if (config->future_only && candidate_id < query_id)
    {
        return 0;
    }

    long diff = query_id - candidate_id;
    if (diff < 0)
    {
        diff = -diff;
    }

    if (diff < (long)config->min_temporal_sep)
    {
        return 0;
    }

    return 1;
}

/**
 * is_candidate_angular_pruned() - Prune candidate neighbor using angular geometry.
 * @d0:        Distance from query to central node: d(b, a0).
 * @ri:        Distance from central node to evaluated neighbor: d(a0, ai).
 * @rj:        Distance from central node to un-evaluated neighbor: d(a0, aj).
 * @di:        Evaluated distance from query to neighbor: d(b, ai).
 * @d_A_ij:    Precomputed mutual distance between ai and aj: d(ai, aj).
 * @tau_eff:   Effective search threshold radius (tau / eps_factor).
 * @rlim:      Cutoff radius cutoff (if > 0).
 *
 * Return: 1 if candidate is pruned, 0 if candidate remains viable.
 */
static inline int is_candidate_angular_pruned(
    double d0,
    double ri,
    double rj,
    double di,
    double d_A_ij,
    double tau_eff,
    double rlim)
{
    // Fast path: 1D metric pivot bound check
    double lb_met = fabs(di - d_A_ij);
    if (lb_met >= tau_eff || (rlim > 0.0 && lb_met >= rlim))
    {
        return 1;
    }

    if (d0 <= 1e-12 || ri <= 1e-12 || rj <= 1e-12)
    {
        return 0;
    }

    double cos_th_i = (d0 * d0 + ri * ri - di * di) / (2.0 * d0 * ri);
    if (cos_th_i > 1.0)
    {
        cos_th_i = 1.0;
    }
    else if (cos_th_i < -1.0)
    {
        cos_th_i = -1.0;
    }

    double cos_phi = (ri * ri + rj * rj - d_A_ij * d_A_ij) / (2.0 * ri * rj);
    if (cos_phi > 1.0)
    {
        cos_phi = 1.0;
    }
    else if (cos_phi < -1.0)
    {
        cos_phi = -1.0;
    }

    double sin_th_i = sqrt(fmax(0.0, 1.0 - cos_th_i * cos_th_i));
    double sin_phi = sqrt(fmax(0.0, 1.0 - cos_phi * cos_phi));

    double cos_th_j_max = cos_th_i * cos_phi + sin_th_i * sin_phi;
    if (cos_th_j_max > 1.0)
    {
        cos_th_j_max = 1.0;
    }

    double lb2 = d0 * d0 + rj * rj - 2.0 * d0 * rj * cos_th_j_max;
    double tau_target = (rlim > 0.0 && rlim < tau_eff) ? rlim : tau_eff;

    if (lb2 >= tau_target * tau_target)
    {
        return 1;
    }

    return 0;
}

/**
 * compute_angular_lower_bound_sq() - Compute squared lower bound from angular geometry.
 * @d0:     Distance from query to central node: d(b, a0).
 * @ri:     Distance from central node to evaluated neighbor: d(a0, ai).
 * @rj:     Distance from central node to un-evaluated neighbor: d(a0, aj).
 * @di:     Evaluated distance from query to neighbor: d(b, ai).
 * @d_A_ij: Precomputed mutual distance between ai and aj: d(ai, aj).
 *
 * Return: Maximum squared lower bound from metric pivot and angular cosine bounds.
 */
static inline double compute_angular_lower_bound_sq(
    double d0,
    double ri,
    double rj,
    double di,
    double d_A_ij)
{
    double diff = fabs(di - d_A_ij);
    double lb2_met = diff * diff;

    if (d0 <= 1e-12 || ri <= 1e-12 || rj <= 1e-12)
    {
        return lb2_met;
    }

    double cos_th_i = (d0 * d0 + ri * ri - di * di) / (2.0 * d0 * ri);
    if (cos_th_i > 1.0)
    {
        cos_th_i = 1.0;
    }
    else if (cos_th_i < -1.0)
    {
        cos_th_i = -1.0;
    }

    double cos_phi = (ri * ri + rj * rj - d_A_ij * d_A_ij) / (2.0 * ri * rj);
    if (cos_phi > 1.0)
    {
        cos_phi = 1.0;
    }
    else if (cos_phi < -1.0)
    {
        cos_phi = -1.0;
    }

    double sin_th_i = sqrt(fmax(0.0, 1.0 - cos_th_i * cos_th_i));
    double sin_phi = sqrt(fmax(0.0, 1.0 - cos_phi * cos_phi));

    double cos_th_j_max = cos_th_i * cos_phi + sin_th_i * sin_phi;
    if (cos_th_j_max > 1.0)
    {
        cos_th_j_max = 1.0;
    }

    double lb2_ang = d0 * d0 + rj * rj - 2.0 * d0 * rj * cos_th_j_max;
    return fmax(lb2_met, lb2_ang);
}

/**
 * is_member_pruned_by_pointwise_pivots() - Multi-Pivot Pointwise Member Bounding.
 * @cand_id:          Candidate frame index.
 * @d_anchor:         Distance from query to candidate cluster anchor.
 * @r_cand:           Distance from anchor to member candidate.
 * @dcc_best:         Distance from closest anchor to candidate cluster anchor.
 * @min_d_anchor:     Distance from query to closest anchor.
 * @num_seed_pivots:  Number of evaluated graph seed pivots [0..8].
 * @seed_pivot_ids:   Array of evaluated graph seed frame IDs.
 * @seed_pivot_dists: Array of computed distances from query to seeds.
 * @model:            KnnModel pointer.
 * @tau_eff:          Effective search radius bound.
 * @rlim_cutoff:      Radius cutoff (if any).
 * @telem:            Telemetry structure to record pruning counters.
 *
 * Return: 1 if candidate is pruned, 0 if candidate must be evaluated.
 */
static inline int is_member_pruned_by_pointwise_pivots(
    long            cand_id,
    double          d_anchor,
    double          r_cand,
    double          dcc_best,
    double          min_d_anchor,
    int             num_seed_pivots,
    const long     *seed_pivot_ids,
    const double   *seed_pivot_dists,
    const KnnModel *model,
    double          tau_eff,
    double          rlim_cutoff,
    KnnTelemetry   *telem)
{
    // Pivot 1: Own cluster anchor
    double lb1 = fabs(d_anchor - r_cand);
    if (lb1 >= tau_eff || (rlim_cutoff > 0.0 && lb1 >= rlim_cutoff))
    {
        telem->level3_annular_pruned++;
        return 1;
    }

    // Pivot 2: Closest cluster anchor (if distinct from own anchor)
    if (dcc_best > 0.0)
    {
        double lb2_1 = dcc_best - r_cand - min_d_anchor;
        double lb2_2 = min_d_anchor - dcc_best - r_cand;
        double lb2 = (lb2_1 > lb2_2) ? lb2_1 : lb2_2;
        if (lb2 >= tau_eff || (rlim_cutoff > 0.0 && lb2 >= rlim_cutoff))
        {
            telem->multi_pivot_pruned++;
            return 1;
        }
    }

    // Pivots 3..H: Evaluated Graph Seed Hops
    if (num_seed_pivots > 0 && model->has_knn_graph && model->graph_distances != NULL)
    {
        int graph_k = model->graph_k;
        for (int p = 0; p < num_seed_pivots; p++)
        {
            long   s_id = seed_pivot_ids[p];
            double s_dist = seed_pivot_dists[p];

            double r_gball =
                (double)model->graph_distances[s_id * (long)graph_k + (graph_k - 1)];
            double lb_gball = r_gball - s_dist;

            const uint32_t *snbrs = &model->graph_indices[s_id * (long)graph_k];
            const float    *sdists = &model->graph_distances[s_id * (long)graph_k];

            double edge_d = -1.0;
            for (int kn = 0; kn < graph_k; kn++)
            {
                if ((long)snbrs[kn] == cand_id)
                {
                    edge_d = (double)sdists[kn];
                    break;
                }
            }

            if (edge_d >= 0.0)
            {
                double lb_edge = fabs(s_dist - edge_d);
                if (lb_edge >= tau_eff || (rlim_cutoff > 0.0 && lb_edge >= rlim_cutoff))
                {
                    telem->graph_edges_pruned++;
                    return 1;
                }
            }
            else if (lb_gball >= tau_eff || (rlim_cutoff > 0.0 && lb_gball >= rlim_cutoff))
            {
                telem->graph_edges_pruned++;
                return 1;
            }
        } // for (int p = 0; ...)
    }

    return 0;
}

/**
 * record_neighbor_and_reciprocal() - Insert candidate into heap and update reciprocal neighbor.
 * @query_id:     Query frame index.
 * @cand_id:      Candidate frame index.
 * @dist:         Calculated distance.
 * @config:       Active KnnConfig.
 * @model:        Active KnnModel.
 * @heap:         Heap for query frame.
 * @all_heaps:    All heaps array.
 * @bucket_locks: OpenMP bucket locks (if multithreaded).
 */
static inline void record_neighbor_and_reciprocal(
    long             query_id,
    long             cand_id,
    double           dist,
    const KnnConfig *config,
    const KnnModel  *model,
    KnnMaxHeap      *heap,
    KnnMaxHeap      *all_heaps
#ifdef _OPENMP
    , omp_lock_t    *bucket_locks
#endif
)
{
    if (config->rlim_cutoff > 0.0 && dist > config->rlim_cutoff)
    {
        return;
    }

#ifdef _OPENMP
    if (bucket_locks != NULL)
    {
        omp_set_lock(&bucket_locks[query_id & 0xFF]);
        knn_heap_push(heap, (int)cand_id, dist);
        omp_unset_lock(&bucket_locks[query_id & 0xFF]);
    }
    else
    {
        knn_heap_push(heap, (int)cand_id, dist);
    }
#else
    knn_heap_push(heap, (int)cand_id, dist);
#endif

    if (config->use_reciprocal && cand_id > query_id &&
        cand_id < model->total_dataset_frames && all_heaps != NULL)
    {
        KnnMaxHeap *target_heap = &all_heaps[cand_id];
        if (target_heap->count < target_heap->k || dist < target_heap->data[0].dist)
        {
#ifdef _OPENMP
            if (bucket_locks != NULL)
            {
                omp_set_lock(&bucket_locks[cand_id & 0xFF]);
                knn_heap_push(target_heap, (int)query_id, dist);
                omp_unset_lock(&bucket_locks[cand_id & 0xFF]);
            }
            else
            {
                knn_heap_push(target_heap, (int)query_id, dist);
            }
#else
            knn_heap_push(target_heap, (int)query_id, dist);
#endif
        }
    }
}

/**
 * knn_search_intra_cluster() - Search members of the query's home cluster.
 * @query_id:        Index of query frame.
 * @query_data:      Query frame pixel data.
 * @home_cluster_id: Home cluster index.
 * @r_home:          Distance from query to home anchor.
 * @model:           Active KnnModel.
 * @config:          Active KnnConfig.
 * @reader:          KnnFrameReader context.
 * @cand_buffer:     Candidate pixel buffer.
 * @heap:            Max-heap for current query.
 * @all_heaps:       Array of all frame heaps.
 * @bucket_locks:    OpenMP locks.
 * @visited:         Per-query frame visited tracker.
 * @telem:           Telemetry record.
 */
static void knn_search_intra_cluster(
    long                   query_id,
    const double *restrict query_data,
    int                    home_cluster_id,
    double                 r_home,
    const KnnModel        *model,
    const KnnConfig       *config,
    KnnFrameReader        *reader,
    double        *restrict cand_buffer,
    KnnMaxHeap            *heap,
    KnnMaxHeap            *all_heaps,
#ifdef _OPENMP
    omp_lock_t            *bucket_locks,
#endif
    KnnVisitedTracker     *visited,
    KnnTelemetry  *restrict telem)
{
    const KnnCluster *home_cl = &model->clusters[home_cluster_id];
    telem->total_candidates_considered += (uint64_t)home_cl->num_members;
    double eps_factor = 1.0 + config->epsilon;

    int start_m = 0;
    int end_m = home_cl->num_members;
    double tau_init = knn_heap_peek_max_dist(heap);
    if (tau_init < 1e20)
    {
        float r_min = (float)fmax(0.0, r_home - tau_init / eps_factor);
        float r_max = (float)(r_home + tau_init / eps_factor);
        start_m = find_member_lower_bound(home_cl->members, home_cl->num_members, r_min);
        end_m = find_member_upper_bound(home_cl->members, home_cl->num_members, r_max);
        telem->level3_annular_pruned +=
            (uint64_t)start_m + (uint64_t)(home_cl->num_members - end_m);
    }

    for (int m = start_m; m < end_m; m++)
    {
        long cand_id = (long)home_cl->members[m].frame_id;

        if (knn_visited_check_and_mark(visited, cand_id))
        {
            continue;
        }

        if (!check_temporal_separation(query_id, cand_id, config))
        {
            telem->temporal_pruned++;
            continue;
        }

        double current_tau = knn_heap_peek_max_dist(heap);
        double r_cand = (double)home_cl->members[m].r_anchor;
        double lb_annular = fabs(r_home - r_cand);

        if (lb_annular >= current_tau / eps_factor)
        {
            telem->level3_annular_pruned++;
            continue;
        }

        if (config->use_reciprocal && knn_heap_contains(heap, (int)cand_id))
        {
            telem->reciprocal_reused++;
            continue;
        }

        if (knn_reader_read_frame(reader, cand_id, cand_buffer) == 0)
        {
            telem->framedist_calls++;
            double d = compute_euclidean_distance(
                query_data, cand_buffer, model->frame_elements
            );
            record_neighbor_and_reciprocal(
                query_id, cand_id, d, config, model, heap, all_heaps
#ifdef _OPENMP
                , bucket_locks
#endif
            );
        }
    } // for (int m = start_m; ...)
}

/**
 * knn_warm_start_nearest_cluster() - Search nearest neighboring cluster if heap is not full.
 * @query_id:        Index of query frame.
 * @query_data:      Query frame pixel data.
 * @home_cluster_id: Home cluster index.
 * @model:           Active KnnModel.
 * @config:          Active KnnConfig.
 * @reader:          KnnFrameReader context.
 * @cand_buffer:     Candidate pixel buffer.
 * @heap:            Max-heap for current query.
 * @all_heaps:       Array of all frame heaps.
 * @bucket_locks:    OpenMP locks.
 * @pivots:          Output pivot array.
 * @num_pivots:      Output pivot count.
 * @visited:         Per-query frame visited tracker.
 * @telem:           Telemetry record.
 *
 * Return: Cluster ID of nearest cluster used for warm start, or -1.
 */
static int knn_warm_start_nearest_cluster(
    long                   query_id,
    const double *restrict query_data,
    int                    home_cluster_id,
    const KnnModel        *model,
    const KnnConfig       *config,
    KnnFrameReader        *reader,
    double        *restrict cand_buffer,
    KnnMaxHeap            *heap,
    KnnMaxHeap            *all_heaps,
#ifdef _OPENMP
    omp_lock_t            *bucket_locks,
#endif
    MeasuredPivot         *pivots,
    int                   *num_pivots,
    KnnVisitedTracker     *visited,
    KnnTelemetry  *restrict telem)
{
    if (heap->count >= heap->k || home_cluster_id < 0 || home_cluster_id >= model->num_clusters)
    {
        return -1;
    }

    int M = model->num_clusters;
    int best_c = -1;
    double min_dcc = 1e19;

    for (int c = 0; c < M; c++)
    {
        if (c == home_cluster_id || model->clusters[c].num_members == 0)
        {
            continue;
        }
        double dcc = model->dcc_matrix[home_cluster_id * M + c];
        if (dcc > 0.0 && dcc < min_dcc)
        {
            min_dcc = dcc;
            best_c = c;
        }
    }

    if (best_c < 0)
    {
        return -1;
    }

    const KnnCluster *warm_cl = &model->clusters[best_c];
    telem->total_candidates_considered += (uint64_t)warm_cl->num_members;
    telem->framedist_calls++;

    double d_anchor = compute_euclidean_distance(
        query_data, warm_cl->anchor_data, model->frame_elements
    );
    double eps_factor = 1.0 + config->epsilon;

    for (int m = 0; m < warm_cl->num_members; m++)
    {
        long cand_id = (long)warm_cl->members[m].frame_id;

        if (knn_visited_check_and_mark(visited, cand_id))
        {
            continue;
        }

        if (!check_temporal_separation(query_id, cand_id, config))
        {
            telem->temporal_pruned++;
            continue;
        }
        double r_cand = (double)warm_cl->members[m].r_anchor;
        double lb1 = fabs(d_anchor - r_cand);
        double current_tau = knn_heap_peek_max_dist(heap);
        if (lb1 >= current_tau / eps_factor)
        {
            telem->level3_annular_pruned++;
            continue;
        }
        if (config->use_reciprocal && knn_heap_contains(heap, (int)cand_id))
        {
            telem->reciprocal_reused++;
            continue;
        }
        if (knn_reader_read_frame(reader, cand_id, cand_buffer) == 0)
        {
            telem->framedist_calls++;
            double d = compute_euclidean_distance(
                query_data, cand_buffer, model->frame_elements
            );
            record_neighbor_and_reciprocal(
                query_id, cand_id, d, config, model, heap, all_heaps
#ifdef _OPENMP
                , bucket_locks
#endif
            );
        }
        if (heap->count >= heap->k)
        {
            break;
        }
    } // for (int m = 0; ...)

    if (config->use_multi_pivot && *num_pivots < MAX_MEASURED_PIVOTS)
    {
        pivots[*num_pivots].cluster_id = best_c;
        pivots[*num_pivots].d_anchor = d_anchor;
        (*num_pivots)++;
    }

    return best_c;
}

/**
 * knn_score_candidate_clusters() - Filter super-clusters and score candidate clusters.
 * @home_cluster_id: Home cluster index.
 * @warm_cluster_id: Warm-started cluster index (or -1).
 * @r_home:          Distance to home anchor.
 * @model:           Active KnnModel.
 * @config:          Active KnnConfig.
 * @heap:            Max-heap for current query.
 * @scores_buffer:   Output array for scored candidate clusters.
 * @telem:           Telemetry record.
 *
 * Return: Number of candidate clusters populated in scores_buffer.
 */
static int knn_score_candidate_clusters(
    int                   home_cluster_id,
    int                   warm_cluster_id,
    double                r_home,
    const KnnModel       *model,
    const KnnConfig      *config,
    const KnnMaxHeap     *heap,
    ClusterScore *restrict scores_buffer,
    KnnTelemetry *restrict telem)
{
    int M = model->num_clusters;
    int num_cand_clusters = 0;
    double eps_factor = 1.0 + config->epsilon;
    double current_tau = knn_heap_peek_max_dist(heap);

    if (model->num_super_clusters > 1 && home_cluster_id >= 0 &&
        model->cluster_super_map != NULL)
    {
        int home_super_id = model->cluster_super_map[home_cluster_id];
        double r_home_super = model->super_clusters[home_super_id].radius;
        int K = model->num_super_clusters;

        for (int s = 0; s < K; s++)
        {
            if (s != home_super_id)
            {
                double dss = model->dss_matrix[home_super_id * K + s];
                double r_s = model->super_clusters[s].radius;
                double lb_super = dss - r_home_super - r_s;
                if (lb_super < 0.0)
                {
                    lb_super = 0.0;
                }

                // Level 0: Super-Cluster Pruning
                if (lb_super >= current_tau / eps_factor)
                {
                    telem->level0_super_clusters_pruned++;
                    telem->level1_clusters_pruned +=
                        (uint64_t)model->super_clusters[s].num_clusters;
                    continue; // Skip all child clusters in super-cluster s
                }
            }

            // Populate child clusters of surviving super-clusters
            const KnnSuperCluster *sc = &model->super_clusters[s];
            for (int ci = 0; ci < sc->num_clusters; ci++)
            {
                int q = sc->cluster_ids[ci];
                if (q == home_cluster_id || q == warm_cluster_id ||
                    model->clusters[q].num_members == 0)
                {
                    continue;
                }

                double dcc = model->dcc_matrix[home_cluster_id * M + q];
                double r_q = model->clusters[q].radius;
                double lb = dcc - r_home - r_q;
                if (lb < 0.0)
                {
                    lb = 0.0;
                }

                scores_buffer[num_cand_clusters].id = q;
                scores_buffer[num_cand_clusters].lb = lb;
                scores_buffer[num_cand_clusters].dcc = dcc;
                num_cand_clusters++;
            } // for (int ci = 0; ...)
        } // for (int s = 0; ...)
    }
    else
    {
        for (int q = 0; q < M; q++)
        {
            if (q == home_cluster_id || q == warm_cluster_id ||
                model->clusters[q].num_members == 0)
            {
                continue;
            }

            double dcc = model->dcc_matrix[home_cluster_id * M + q];
            double r_q = model->clusters[q].radius;
            double lb = dcc - r_home - r_q;
            if (lb < 0.0)
            {
                lb = 0.0;
            }

            scores_buffer[num_cand_clusters].id = q;
            scores_buffer[num_cand_clusters].lb = lb;
            scores_buffer[num_cand_clusters].dcc = dcc;
            num_cand_clusters++;
        } // for (int q = 0; ...)
    }

    qsort(scores_buffer, (size_t)num_cand_clusters, sizeof(ClusterScore),
          compare_cluster_scores);

    return num_cand_clusters;
}

/**
 * knn_search_inter_clusters() - Evaluate inter-cluster candidate members.
 * @query_id:          Index of query frame.
 * @query_data:        Query frame pixel data.
 * @home_cluster_id:   Home cluster index.
 * @r_home:            Distance to home anchor.
 * @num_cand_clusters: Number of candidate clusters.
 * @model:             Active KnnModel.
 * @config:            Active KnnConfig.
 * @reader:            KnnFrameReader context.
 * @cand_buffer:       Candidate pixel buffer.
 * @scores_buffer:     Candidate cluster records.
 * @heap:              Max-heap for current query.
 * @all_heaps:         Array of all heaps.
 * @bucket_locks:      OpenMP bucket locks.
 * @pivots:            Pivot array.
 * @num_pivots:        Pointer to pivot count.
 * @visited:           Per-query frame visited tracker.
 * @telem:             Telemetry record.
 */
static void knn_search_inter_clusters(
    long                   query_id,
    const double *restrict query_data,
    int                    home_cluster_id,
    double                 r_home,
    int                    num_cand_clusters,
    const KnnModel        *model,
    const KnnConfig       *config,
    KnnFrameReader        *reader,
    double        *restrict cand_buffer,
    ClusterScore  *restrict scores_buffer,
    KnnMaxHeap            *heap,
    KnnMaxHeap            *all_heaps,
#ifdef _OPENMP
    omp_lock_t            *bucket_locks,
#endif
    MeasuredPivot         *pivots,
    int                   *num_pivots,
    KnnVisitedTracker     *visited,
    KnnTelemetry  *restrict telem)
{
    int M = model->num_clusters;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;

    for (int idx = 0; idx < num_cand_clusters; idx++)
    {
        int q = scores_buffer[idx].id;
        double lb_cluster = scores_buffer[idx].lb;
        double current_tau = knn_heap_peek_max_dist(heap);

        // Level 1: Cluster-level DCC bound
        if (lb_cluster >= current_tau / eps_factor)
        {
            telem->level1_clusters_pruned++;
            continue;
        }

        // Level 2: Query-to-Anchor evaluation
        const KnnCluster *cl = &model->clusters[q];
        telem->framedist_calls++;
        double d_anchor = compute_euclidean_distance(
            query_data, cl->anchor_data, frame_elem
        );

        // Dynamic Bound Tightening (Multi-Pivot)
        if (config->use_multi_pivot)
        {
            for (int j = idx + 1; j < num_cand_clusters; j++)
            {
                int other_q = scores_buffer[j].id;
                double dcc_pivot = model->dcc_matrix[q * M + other_q];
                if (dcc_pivot > 0.0)
                {
                    double r_other = model->clusters[other_q].radius;
                    double lb_pivot = dcc_pivot - d_anchor - r_other;
                    if (lb_pivot > scores_buffer[j].lb)
                    {
                        scores_buffer[j].lb = lb_pivot;
                    }
                }
            } // for (int j = idx + 1; ...)
        }

        // Record measured anchor pivot for Multi-Anchor Pivot Bounding (AESA)
        if (config->use_multi_pivot && *num_pivots < MAX_MEASURED_PIVOTS)
        {
            int already_present = 0;
            for (int p = 0; p < *num_pivots; p++)
            {
                if (pivots[p].cluster_id == q)
                {
                    already_present = 1;
                    break;
                }
            }
            if (!already_present)
            {
                pivots[*num_pivots].cluster_id = q;
                pivots[*num_pivots].d_anchor = d_anchor;
                (*num_pivots)++;
            }
        }

        double lb_anchor = d_anchor - cl->radius;
        if (lb_anchor < 0.0)
        {
            lb_anchor = 0.0;
        }

        if (lb_anchor >= current_tau / eps_factor)
        {
            telem->level2_anchors_pruned++;
            continue;
        }

        // Level 3: Radially-Sorted Annular Window Slicing & Multi-Pivot Filter
        telem->total_candidates_considered += (uint64_t)cl->num_members;

        double tau_eff = current_tau / eps_factor;
        if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_eff)
        {
            tau_eff = config->rlim_cutoff;
        }

        float r_min = (float)fmax(0.0, d_anchor - tau_eff);
        float r_max = (float)(d_anchor + tau_eff);

        int start_m = find_member_lower_bound(cl->members, cl->num_members, r_min);
        int end_m = find_member_upper_bound(cl->members, cl->num_members, r_max);

        telem->level3_annular_pruned +=
            (uint64_t)start_m + (uint64_t)(cl->num_members - end_m);

        double dcc_home = model->dcc_matrix[home_cluster_id * M + q];

        for (int m = start_m; m < end_m; m++)
        {
            long cand_id = (long)cl->members[m].frame_id;

            if (knn_visited_check_and_mark(visited, cand_id))
            {
                continue;
            }

            if (!check_temporal_separation(query_id, cand_id, config))
            {
                telem->temporal_pruned++;
                continue;
            }

            current_tau = knn_heap_peek_max_dist(heap);
            double r_cand = (double)cl->members[m].r_anchor;

            // Primary pivot lower bound: anchor A_q
            double lb1 = fabs(d_anchor - r_cand);
            if (lb1 >= current_tau / eps_factor)
            {
                telem->level3_annular_pruned++;
                continue;
            }

            // Secondary pivot lower bound: home anchor A_home
            double lb_home = dcc_home - r_cand - r_home;
            if (lb_home >= current_tau / eps_factor)
            {
                telem->level3_annular_pruned++;
                continue;
            }

            // Multi-Anchor Pivot Bounding (AESA / LAESA Indexing)
            if (config->use_multi_pivot && *num_pivots > 0)
            {
                int pruned_by_pivot = 0;
                for (int p = 0; p < *num_pivots; p++)
                {
                    int p_cl = pivots[p].cluster_id;
                    if (p_cl == q || p_cl == home_cluster_id)
                    {
                        continue;
                    }
                    double d_qp = pivots[p].d_anchor;
                    double dcc_pq = model->dcc_matrix[p_cl * M + q];
                    if (dcc_pq > 0.0)
                    {
                        double lb_p1 = dcc_pq - d_qp - r_cand;
                        double lb_p2 = d_qp - (dcc_pq + r_cand);
                        double max_lb_p = (lb_p1 > lb_p2) ? lb_p1 : lb_p2;
                        if (max_lb_p >= current_tau / eps_factor ||
                            (config->rlim_cutoff > 0.0 && max_lb_p >= config->rlim_cutoff))
                        {
                            pruned_by_pivot = 1;
                            break;
                        }
                    }
                } // for (int p = 0; ...)

                if (pruned_by_pivot)
                {
                    telem->level3_annular_pruned++;
                    continue;
                }
            }

            if (config->rlim_cutoff > 0.0 &&
                (lb1 >= config->rlim_cutoff || lb_home >= config->rlim_cutoff))
            {
                telem->level3_annular_pruned++;
                continue;
            }

            if (config->use_reciprocal && knn_heap_contains(heap, (int)cand_id))
            {
                telem->reciprocal_reused++;
                continue;
            }

            // Level 4: Exact Distance Evaluation
            if (knn_reader_read_frame(reader, cand_id, cand_buffer) == 0)
            {
                telem->framedist_calls++;
                double d = compute_euclidean_distance(
                    query_data, cand_buffer, frame_elem
                );
                record_neighbor_and_reciprocal(
                    query_id, cand_id, d, config, model, heap, all_heaps
#ifdef _OPENMP
                    , bucket_locks
#endif
                );
            }
        } // for (int m = start_m; ...)
    } // for (int idx = 0; ...)
}

/**
 * knn_search_single_frame() - Execute multi-level metric pruned search for one query frame.
 * @query_id:       Index of query frame.
 * @query_data:     Pixel buffer of query frame.
 * @model:          Active KnnModel.
 * @config:         Active KnnConfig.
 * @reader:         Thread-local KnnFrameReader.
 * @cand_buffer:    Scratch buffer for candidate frame pixels.
 * @scores_buffer:  Scratch buffer for cluster sorting.
 * @all_heaps:      Array of KnnMaxHeap structures for all frames.
 * @bucket_locks:   Array of OpenMP bucket locks (if OpenMP enabled).
 * @visited:        Per-query frame visited tracker.
 * @telem:          Thread-local KnnTelemetry.
 */
static void knn_search_single_frame(
    long                   query_id,
    const double *restrict query_data,
    const KnnModel        *model,
    const KnnConfig       *config,
    KnnFrameReader        *reader,
    double        *restrict cand_buffer,
    ClusterScore  *restrict scores_buffer,
    KnnMaxHeap            *all_heaps,
#ifdef _OPENMP
    omp_lock_t            *bucket_locks,
#endif
    KnnVisitedTracker     *visited,
    KnnTelemetry  *restrict telem)
{
    KnnMaxHeap *heap = &all_heaps[query_id];
    int home_cluster_id = model->frame_cluster_map[query_id];
    double r_home = (double)model->frame_r_anchor[query_id];
    int M = model->num_clusters;

    MeasuredPivot pivots[MAX_MEASURED_PIVOTS];
    int           num_pivots = 0;
    if (config->use_multi_pivot && home_cluster_id >= 0 && home_cluster_id < M)
    {
        pivots[num_pivots].cluster_id = home_cluster_id;
        pivots[num_pivots].d_anchor = r_home;
        num_pivots++;
    }

    // Step 1: Intra-Cluster Search (Home cluster c_p)
    if (home_cluster_id >= 0 && home_cluster_id < M)
    {
        knn_search_intra_cluster(
            query_id, query_data, home_cluster_id, r_home, model, config,
            reader, cand_buffer, heap, all_heaps,
#ifdef _OPENMP
            bucket_locks,
#endif
            visited, telem
        );
    }

    // Fast tau-Contraction: Warm-start tau by searching closest neighbor cluster
    int warm_cluster_id = knn_warm_start_nearest_cluster(
        query_id, query_data, home_cluster_id, model, config, reader,
        cand_buffer, heap, all_heaps,
#ifdef _OPENMP
        bucket_locks,
#endif
        pivots, &num_pivots, visited, telem
    );

    if (config->approx_mode)
    {
        telem->level1_clusters_pruned += (uint64_t)M;
        return;
    }

    // Step 2: Rank and Sort candidate clusters
    int num_cand_clusters = knn_score_candidate_clusters(
        home_cluster_id, warm_cluster_id, r_home, model, config,
        heap, scores_buffer, telem
    );

    // Step 3 & 4: Inter-Cluster Pruning and Verification
    knn_search_inter_clusters(
        query_id, query_data, home_cluster_id, r_home, num_cand_clusters,
        model, config, reader, cand_buffer, scores_buffer, heap, all_heaps,
#ifdef _OPENMP
        bucket_locks,
#endif
        pivots, &num_pivots, visited, telem
    );
}

/**
 * knn_cross_seed_frontier() - Populate dynamic search frontier with nearest evaluated anchor seeds.
 * @loc_res:         Cluster locator result with evaluated anchors.
 * @best_c:          Best anchor cluster ID.
 * @query_data:      Query frame pixel data.
 * @model:           Active KnnModel.
 * @config:          Active KnnConfig.
 * @cand_reader:     Candidate frame reader.
 * @cand_buffer:     Candidate frame pixel buffer.
 * @heap:            Max-heap for current query.
 * @frontier:        Output frontier node array.
 * @frontier_count:  Pointer to frontier count.
 * @best_seed_id:    Pointer to best seed frame ID.
 * @best_seed_dist:  Pointer to best seed distance.
 * @visited:         Per-query frame visited tracker.
 * @telem:           Telemetry record.
 */
static void knn_cross_seed_frontier(
    const ClusterLocatorResult *loc_res,
    int                         best_c,
    const double *restrict      query_data,
    const KnnModel             *model,
    const KnnConfig            *config,
    KnnFrameReader             *cand_reader,
    double *restrict            cand_buffer,
    KnnMaxHeap                 *heap,
    FrontierNode               *frontier,
    int                        *frontier_count,
    long                       *best_seed_id,
    double                     *best_seed_dist,
    KnnVisitedTracker          *visited,
    KnnTelemetry *restrict      telem)
{
    int sorted_eval_clusters[32];
    double sorted_eval_dists[32];
    int num_eval = loc_res->num_evaluated_anchors;
    if (num_eval > 32)
    {
        num_eval = 32;
    }
    for (int e = 0; e < num_eval; e++)
    {
        sorted_eval_clusters[e] = loc_res->evaluated_clusters[e];
        sorted_eval_dists[e] = loc_res->evaluated_dists[e];
    }
    for (int i = 0; i < num_eval - 1; i++)
    {
        for (int j = i + 1; j < num_eval; j++)
        {
            if (sorted_eval_dists[j] < sorted_eval_dists[i])
            {
                double tmp_d = sorted_eval_dists[i];
                sorted_eval_dists[i] = sorted_eval_dists[j];
                sorted_eval_dists[j] = tmp_d;
                int tmp_c = sorted_eval_clusters[i];
                sorted_eval_clusters[i] = sorted_eval_clusters[j];
                sorted_eval_clusters[j] = tmp_c;
            }
        }
    }

    int M = model->num_clusters;
    long frame_elem = model->frame_elements;

    int max_frontier_seeds = (config->approx_mode) ? 2 : 8;
    for (int e = 0; e < num_eval && *frontier_count < max_frontier_seeds; e++)
    {
        int anc_c = sorted_eval_clusters[e];
        if (anc_c >= 0 && anc_c < M && anc_c != best_c)
        {
            const KnnCluster *anc_cl = &model->clusters[anc_c];
            if (anc_cl->num_members > 0)
            {
                for (int mi = 0; mi < anc_cl->num_members && mi < 2; mi++)
                {
                    long s_frame = (long)anc_cl->members[mi].frame_id;
                    if (knn_visited_check_and_mark(visited, s_frame))
                    {
                        continue;
                    }
                    if (!knn_heap_contains(heap, (int)s_frame))
                    {
                        if (knn_reader_read_frame(cand_reader, s_frame, cand_buffer) == 0)
                        {
                            telem->framedist_calls++;
                            telem->graph_seeds_evaluated++;
                            double d = compute_euclidean_distance(
                                query_data, cand_buffer, frame_elem
                            );
                            if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
                            {
                                knn_heap_push(heap, (int)s_frame, d);
                            }
                            if (d < *best_seed_dist)
                            {
                                *best_seed_id = s_frame;
                                *best_seed_dist = d;
                            }
                            if (frontier_count != NULL && *frontier_count < GRAPH_FRONTIER_MAX)
                            {
                                frontier[*frontier_count].frame_id = s_frame;
                                frontier[*frontier_count].dist = d;
                                frontier[*frontier_count].expanded = 0;
                                (*frontier_count)++;
                            }
                        }
                    }
                } // for (int mi = 0; ...)
            }
        }
    } // for (int e = 0; ...)
}

/**
 * knn_greedy_route_to_basin() - Fast Phase 1 greedy 1-NN routing along graph.
 * @query_data:       Pixel buffer of query frame.
 * @model:            Active KnnModel.
 * @config:           Active KnnConfig.
 * @cand_reader:      Candidate frame reader.
 * @cand_buffer:      Scratch buffer for candidate frame pixels.
 * @heap:             Max-heap for current query.
 * @best_seed_id:     Pointer to best seed frame ID.
 * @best_seed_dist:   Pointer to best seed distance.
 * @seed_pivot_ids:   Array of evaluated seed frame IDs.
 * @seed_pivot_dists: Array of computed distances from query to seed pivots.
 * @num_seed_pivots:  Pointer to seed pivot count.
 * @visited:          Per-query frame visited tracker.
 * @telem:            Telemetry record.
 */
static void knn_greedy_route_to_basin(
    const double *restrict query_data,
    const KnnModel        *model,
    const KnnConfig       *config,
    KnnFrameReader        *cand_reader,
    double        *restrict cand_buffer,
    KnnMaxHeap            *heap,
    long                  *best_seed_id,
    double                *best_seed_dist,
    long                  *seed_pivot_ids,
    double                *seed_pivot_dists,
    int                   *num_seed_pivots,
    KnnVisitedTracker     *visited,
    KnnTelemetry  *restrict telem)
{
    long curr_u = *best_seed_id;
    double curr_d = *best_seed_dist;
    int graph_k = model->graph_k;
    long frame_elem = model->frame_elements;
    int max_hops = 16;
    int check_k = (graph_k < 16) ? graph_k : 16;

    for (int hop = 0; hop < max_hops; hop++)
    {
        const uint32_t *neighbors = &model->graph_indices[curr_u * (long)graph_k];
        const float    *n_dists = &model->graph_distances[curr_u * (long)graph_k];
        int improved = 0;
        long next_u = curr_u;
        double next_d = curr_d;

        for (int k_idx = 0; k_idx < check_k; k_idx++)
        {
            long nb_id = (long)neighbors[k_idx];
            if (nb_id < 0 || nb_id >= model->total_dataset_frames || nb_id == curr_u)
            {
                continue;
            }

            if (knn_visited_check_and_mark(visited, nb_id))
            {
                continue;
            }

            double d_edge = (double)n_dists[k_idx];
            double lb_edge = fabs(curr_d - d_edge);
            if (lb_edge >= curr_d)
            {
                telem->graph_edges_pruned++;
                continue;
            }

            if (knn_reader_read_frame(cand_reader, nb_id, cand_buffer) == 0)
            {
                telem->framedist_calls++;
                telem->graph_seeds_evaluated++;
                double d = compute_euclidean_distance(
                    query_data, cand_buffer, frame_elem
                );
                if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
                {
                    knn_heap_push(heap, (int)nb_id, d);
                }
                if (d < *best_seed_dist)
                {
                    *best_seed_dist = d;
                    *best_seed_id = nb_id;
                }
                if (*num_seed_pivots < 8)
                {
                    seed_pivot_ids[*num_seed_pivots] = nb_id;
                    seed_pivot_dists[*num_seed_pivots] = d;
                    (*num_seed_pivots)++;
                }

                if (d < next_d)
                {
                    next_u = nb_id;
                    next_d = d;
                    improved = 1;
                    break;
                }
            }
        } // for (int k_idx = 0; ...)

        if (!improved)
        {
            break;
        }

        curr_u = next_u;
        curr_d = next_d;
    } // for (int hop = 0; ...)
}

/**
 * knn_direct_basin_expansion() - Direct Option B1 neighbor expansion around 1-NN basin.
 * @query_data:       Pixel buffer of query frame.
 * @model:            Active KnnModel.
 * @config:           Active KnnConfig.
 * @cand_reader:      Candidate frame reader.
 * @cand_buffer:      Scratch buffer for candidate frame pixels.
 * @heap:             Max-heap for current query.
 * @best_seed_id:     Pointer to best seed frame ID (u*).
 * @best_seed_dist:   Pointer to best seed distance.
 * @seed_pivot_ids:   Array of evaluated seed frame IDs.
 * @seed_pivot_dists: Array of computed distances from query to seed pivots.
 * @num_seed_pivots:  Pointer to seed pivot count.
 * @visited:          Per-query frame visited tracker.
 * @telem:            Telemetry record.
 */
static void knn_direct_basin_expansion(
    const double *restrict query_data,
    const KnnModel        *model,
    const KnnConfig       *config,
    KnnFrameReader        *cand_reader,
    double        *restrict cand_buffer,
    KnnMaxHeap            *heap,
    long                  *best_seed_id,
    double                *best_seed_dist,
    long                  *seed_pivot_ids,
    double                *seed_pivot_dists,
    int                   *num_seed_pivots,
    KnnVisitedTracker     *visited,
    KnnTelemetry  *restrict telem)
{
    long center_u = *best_seed_id;
    double center_d = *best_seed_dist;
    int graph_k = model->graph_k;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;

    const uint32_t *neighbors = &model->graph_indices[center_u * (long)graph_k];
    const float    *n_dists = &model->graph_distances[center_u * (long)graph_k];

    float local_lb2[64];
    int cand_order[64];
    float cand_alignment[64];
    int eval_lim = (graph_k < 64) ? graph_k : 64;

    for (int i = 0; i < eval_lim; i++)
    {
        cand_order[i] = i;
        cand_alignment[i] = 0.0f;
        local_lb2[i] = 0.0f;
    }

    int first_eval_done = 0;
    long improving_node = -1;
    int non_improving_streak = 0;
    const int max_streak = 8;

    for (int step = 0; step < eval_lim; step++)
    {
        int k_idx = cand_order[step];
        long nb_id = (long)neighbors[k_idx];
        if (nb_id < 0 || nb_id >= model->total_dataset_frames || nb_id == center_u)
        {
            continue;
        }

        double current_tau = knn_heap_peek_max_dist(heap);
        double tau_eff = current_tau / eps_factor;
        double tau_eff2 = tau_eff * tau_eff;
        if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_eff)
        {
            tau_eff2 = config->rlim_cutoff * config->rlim_cutoff;
        }

        if (k_idx < 64 && (heap->count >= heap->k || config->rlim_cutoff > 0.0) &&
            (double)local_lb2[k_idx] >= tau_eff2)
        {
            telem->angular_pruned++;
            if (heap->count >= heap->k)
            {
                non_improving_streak++;
                if (non_improving_streak >= max_streak)
                {
                    break;
                }
            }
            continue;
        }

        if (knn_visited_check_and_mark(visited, nb_id))
        {
            continue;
        }

        double d_edge = (double)n_dists[k_idx];
        double lb_edge = fabs(center_d - d_edge);
        if (heap->count >= heap->k &&
            (lb_edge >= current_tau / eps_factor ||
             (config->rlim_cutoff > 0.0 && lb_edge >= config->rlim_cutoff)))
        {
            telem->graph_edges_pruned++;
            non_improving_streak++;
            if (non_improving_streak >= max_streak)
            {
                break;
            }
            continue;
        }

        if (knn_reader_read_frame(cand_reader, nb_id, cand_buffer) == 0)
        {
            telem->framedist_calls++;
            telem->graph_seeds_evaluated++;
            double d = compute_euclidean_distance(
                query_data, cand_buffer, frame_elem
            );
            if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
            {
                knn_heap_push(heap, (int)nb_id, d);
            }
            if (heap->count >= heap->k)
            {
                if (d < current_tau)
                {
                    non_improving_streak = 0;
                }
                else
                {
                    non_improving_streak++;
                    if (non_improving_streak >= max_streak)
                    {
                        break;
                    }
                }
            }
            if (d < *best_seed_dist)
            {
                *best_seed_dist = d;
                *best_seed_id = nb_id;
                improving_node = nb_id;
            }
            if (*num_seed_pivots < 8)
            {
                seed_pivot_ids[*num_seed_pivots] = nb_id;
                seed_pivot_dists[*num_seed_pivots] = d;
                (*num_seed_pivots)++;
            }

            if (model->graph_mutual_dists != NULL)
            {
                for (int rem = step + 1; rem < eval_lim; rem++)
                {
                    int m_idx = cand_order[rem];
                    if (m_idx < 64)
                    {
                        double r_m = (double)n_dists[m_idx];
                        float d_mut = knn_get_mutual_dist(
                            model, center_u, k_idx, m_idx
                        );
                        if (d_mut > 0.0f)
                        {
                            // Opportunity 3: Mutual distance triangle lower bound
                            double lb_tri = fabs(d - (double)d_mut);
                            double lb2_tri = lb_tri * lb_tri;
                            if (lb2_tri > (double)local_lb2[m_idx])
                            {
                                local_lb2[m_idx] = (float)lb2_tri;
                            }

                            if (config->use_angular_bound)
                            {
                                double lb2_m = compute_angular_lower_bound_sq(
                                    center_d, d_edge, r_m, d, (double)d_mut
                                );
                                if (lb2_m > (double)local_lb2[m_idx])
                                {
                                    local_lb2[m_idx] = (float)lb2_m;
                                }
                            }

                            if (!first_eval_done && config->use_angular_bound &&
                                center_d > 1e-9 && d_edge > 1e-9 && r_m > 1e-9)
                            {
                                double cos_th = (center_d * center_d +
                                                 d_edge * d_edge - d * d) /
                                                (2.0 * center_d * d_edge);
                                cos_th = fmax(-1.0, fmin(1.0, cos_th));
                                double sin_th = sqrt(fmax(0.0, 1.0 - cos_th * cos_th));

                                double cos_phi = (d_edge * d_edge + r_m * r_m -
                                                  (double)d_mut * d_mut) /
                                                 (2.0 * d_edge * r_m);
                                cos_phi = fmax(-1.0, fmin(1.0, cos_phi));
                                double sin_phi = sqrt(fmax(0.0, 1.0 - cos_phi * cos_phi));

                                cand_alignment[m_idx] = (float)(cos_th * cos_phi +
                                                                sin_th * sin_phi);
                            }
                        }
                    }
                } // for (int rem = step + 1; ...)

                if (!first_eval_done)
                {
                    first_eval_done = 1;
                    for (int i = step + 1; i < eval_lim - 1; i++)
                    {
                        int best_p = i;
                        float max_a = cand_alignment[cand_order[i]];
                        for (int j = i + 1; j < eval_lim; j++)
                        {
                            if (cand_alignment[cand_order[j]] > max_a)
                            {
                                max_a = cand_alignment[cand_order[j]];
                                best_p = j;
                            }
                        }
                        if (best_p != i)
                        {
                            int tmp = cand_order[i];
                            cand_order[i] = cand_order[best_p];
                            cand_order[best_p] = tmp;
                        }
                    }
                }
            } // if (config->use_angular_bound ...)
        }
    } // for (int step = 0; ...)

    // Phase 2b: Also check top neighbors of the closest discovered nodes in heap
    int top_seeds[3];
    int num_top_seeds = 0;
    if (improving_node >= 0)
    {
        top_seeds[num_top_seeds++] = (int)improving_node;
    }

    for (int i = 0; i < heap->count && num_top_seeds < 3; i++)
    {
        int cand = heap->data[i].frame_id;
        if (cand != (int)center_u && (num_top_seeds == 0 || cand != top_seeds[0]))
        {
            top_seeds[num_top_seeds++] = cand;
        }
    }

    for (int ts = 0; ts < num_top_seeds; ts++)
    {
        long s_node = (long)top_seeds[ts];
        if (s_node >= 0 && s_node < model->total_dataset_frames)
        {
            const uint32_t *s_nb = &model->graph_indices[s_node * (long)graph_k];
            const float    *s_dst = &model->graph_distances[s_node * (long)graph_k];
            int check_k = (graph_k < 12) ? graph_k : 12;

            for (int i = 0; i < check_k; i++)
            {
                long nb2 = (long)s_nb[i];
                if (nb2 >= 0 && nb2 < model->total_dataset_frames)
                {
                    if (knn_visited_check_and_mark(visited, nb2))
                    {
                        continue;
                    }

                    double current_tau = knn_heap_peek_max_dist(heap);
                    double tau_eff = current_tau / eps_factor;
                    double d_edge = (double)s_dst[i];
                    if (heap->count >= heap->k && d_edge >= tau_eff)
                    {
                        telem->graph_edges_pruned++;
                        continue;
                    }

                    if (knn_reader_read_frame(cand_reader, nb2, cand_buffer) == 0)
                    {
                        telem->framedist_calls++;
                        telem->graph_seeds_evaluated++;
                        double d2 = compute_euclidean_distance(
                            query_data, cand_buffer, frame_elem
                        );
                        if (config->rlim_cutoff <= 0.0 || d2 <= config->rlim_cutoff)
                        {
                            knn_heap_push(heap, (int)nb2, d2);
                        }
                        if (d2 < *best_seed_dist)
                        {
                            *best_seed_dist = d2;
                            *best_seed_id = nb2;
                        }
                    }
                }
            } // for (int i = 0; ...)
        }
    } // for (int ts = 0; ...)
}

/**
 * knn_cross_explore_graph_frontier() - Explore KNN graph frontier for cross-dataset query.
 * @query_data:       Query frame pixel data.
 * @model:            Active KnnModel.
 * @config:           Active KnnConfig.
 * @cand_reader:      Candidate frame reader.
 * @cand_buffer:      Candidate frame pixel buffer.
 * @heap:             Max-heap for current query.
 * @frontier:         Frontier node array.
 * @frontier_count:   Frontier node count.
 * @best_seed_id:     Pointer to best seed ID.
 * @best_seed_dist:   Pointer to best seed distance.
 * @seed_pivot_ids:   Array of evaluated seed pivot IDs.
 * @seed_pivot_dists: Array of computed distances from query to seed pivots.
 * @num_seed_pivots:  Pointer to seed pivot count.
 * @visited:          Per-query frame visited tracker.
 * @telem:            Telemetry record.
 *
 * Return: 1 if global containment criterion was satisfied, 0 otherwise.
 */
static int knn_cross_explore_graph_frontier(
    const double *restrict query_data,
    const KnnModel        *model,
    const KnnConfig       *config,
    KnnFrameReader        *cand_reader,
    double        *restrict cand_buffer,
    KnnMaxHeap            *heap,
    FrontierNode          *frontier,
    int                    frontier_count,
    long                  *best_seed_id,
    double                *best_seed_dist,
    long                  *seed_pivot_ids,
    double                *seed_pivot_dists,
    int                   *num_seed_pivots,
    KnnVisitedTracker     *visited,
    KnnTelemetry  *restrict telem)
{
    int graph_k = model->graph_k;
    int target_k = heap->k;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;

    int max_expansions = (config->approx_mode) ? 4 : 16;
    int max_eval_k = (config->approx_mode) ? ((graph_k < 12) ? graph_k : 12) : graph_k;
    int expansions_done = 0;
    int stagnation_count = 0;
    double prev_tau = 1e20;

    while ((expansions_done < max_expansions || heap->count < target_k) &&
           expansions_done < GRAPH_FRONTIER_MAX)
    {
        int best_idx = -1;
        double min_f_dist = 1e20;
        for (int fi = 0; fi < frontier_count; fi++)
        {
            if (!frontier[fi].expanded && frontier[fi].dist < min_f_dist)
            {
                min_f_dist = frontier[fi].dist;
                best_idx = fi;
            }
        }

        if (best_idx < 0)
        {
            break;
        }

        double current_tau = knn_heap_peek_max_dist(heap);
        if (heap->count >= target_k && min_f_dist > current_tau / eps_factor)
        {
            break;
        }

        if (config->approx_mode && heap->count >= target_k)
        {
            if (current_tau >= prev_tau * 0.9999)
            {
                stagnation_count++;
                if (stagnation_count >= 3)
                {
                    break;
                }
            }
            else
            {
                stagnation_count = 0;
                prev_tau = current_tau;
            }
        }

        frontier[best_idx].expanded = 1;
        expansions_done++;

        long curr_u = frontier[best_idx].frame_id;
        double curr_u_dist = frontier[best_idx].dist;
        if (curr_u_dist < *best_seed_dist)
        {
            *best_seed_id = curr_u;
            *best_seed_dist = curr_u_dist;
        }
        uint32_t const *neighbors = &model->graph_indices[curr_u * (long)graph_k];
        const float    *n_dists = &model->graph_distances[curr_u * (long)graph_k];

        float local_lb2[64];
        int cand_order[64];
        float cand_alignment[64];
        int eval_lim = (max_eval_k < 64) ? max_eval_k : 64;
        for (int i = 0; i < eval_lim; i++)
        {
            cand_order[i] = i;
            cand_alignment[i] = 0.0f;
            local_lb2[i] = 0.0f;
        }

        int first_eval_done = 0;

        for (int step = 0; step < eval_lim; step++)
        {
            int k_idx = cand_order[step];
            long nb_id = (long)neighbors[k_idx];
            if (nb_id < 0 || nb_id >= model->total_dataset_frames || nb_id == curr_u)
            {
                continue;
            }

            current_tau = knn_heap_peek_max_dist(heap);
            double tau_eff = current_tau / eps_factor;
            double tau_eff2 = tau_eff * tau_eff;
            if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_eff)
            {
                tau_eff2 = config->rlim_cutoff * config->rlim_cutoff;
            }

            if (k_idx < 64 && (heap->count >= heap->k || config->rlim_cutoff > 0.0) &&
                (double)local_lb2[k_idx] >= tau_eff2)
            {
                telem->angular_pruned++;
                continue;
            }

            if (knn_visited_check_and_mark(visited, nb_id))
            {
                continue;
            }

            int already_in_frontier = 0;
            for (int fi = 0; fi < frontier_count; fi++)
            {
                if (frontier[fi].frame_id == nb_id)
                {
                    already_in_frontier = 1;
                    break;
                }
            }
            if (already_in_frontier)
            {
                continue;
            }

            double d_edge = (double)n_dists[k_idx];
            double lb_edge = fabs(curr_u_dist - d_edge);
            current_tau = knn_heap_peek_max_dist(heap);

            if (heap->count >= heap->k &&
                (lb_edge >= current_tau / eps_factor ||
                 (config->rlim_cutoff > 0.0 && lb_edge >= config->rlim_cutoff)))
            {
                telem->graph_edges_pruned++;
                continue;
            }

            if (knn_reader_read_frame(cand_reader, nb_id, cand_buffer) == 0)
            {
                telem->framedist_calls++;
                telem->graph_seeds_evaluated++;
                double d = compute_euclidean_distance(
                    query_data, cand_buffer, frame_elem
                );
                if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
                {
                    knn_heap_push(heap, (int)nb_id, d);
                }
                if (d < *best_seed_dist)
                {
                    *best_seed_id = nb_id;
                    *best_seed_dist = d;
                }
                if (*num_seed_pivots < 8)
                {
                    seed_pivot_ids[*num_seed_pivots] = nb_id;
                    seed_pivot_dists[*num_seed_pivots] = d;
                    (*num_seed_pivots)++;
                }

                if (config->use_angular_bound && model->graph_mutual_dists != NULL)
                {
                    for (int rem = step + 1; rem < eval_lim; rem++)
                    {
                        int m_idx = cand_order[rem];
                        if (m_idx < 64)
                        {
                            double r_m = (double)n_dists[m_idx];
                            float d_mut = knn_get_mutual_dist(
                                model, curr_u, k_idx, m_idx
                            );
                            if (d_mut > 0.0f)
                            {
                                double lb2_m = compute_angular_lower_bound_sq(
                                    curr_u_dist, d_edge, r_m, d, (double)d_mut
                                );
                                if (lb2_m > (double)local_lb2[m_idx])
                                {
                                    local_lb2[m_idx] = (float)lb2_m;
                                }

                                if (!first_eval_done && curr_u_dist > 1e-9 &&
                                    d_edge > 1e-9 && r_m > 1e-9)
                                {
                                    double cos_th = (curr_u_dist * curr_u_dist +
                                                     d_edge * d_edge - d * d) /
                                                    (2.0 * curr_u_dist * d_edge);
                                    cos_th = fmax(-1.0, fmin(1.0, cos_th));
                                    double sin_th = sqrt(fmax(0.0, 1.0 - cos_th * cos_th));

                                    double cos_phi = (d_edge * d_edge + r_m * r_m -
                                                      (double)d_mut * d_mut) /
                                                     (2.0 * d_edge * r_m);
                                    cos_phi = fmax(-1.0, fmin(1.0, cos_phi));
                                    double sin_phi = sqrt(fmax(0.0, 1.0 - cos_phi * cos_phi));

                                    cand_alignment[m_idx] = (float)(cos_th * cos_phi +
                                                                    sin_th * sin_phi);
                                }
                            }
                        }
                    } // for (int rem = step + 1; ...)

                    if (!first_eval_done)
                    {
                        first_eval_done = 1;
                        for (int i = step + 1; i < eval_lim - 1; i++)
                        {
                            int best_p = i;
                            float max_a = cand_alignment[cand_order[i]];
                            for (int j = i + 1; j < eval_lim; j++)
                            {
                                if (cand_alignment[cand_order[j]] > max_a)
                                {
                                    max_a = cand_alignment[cand_order[j]];
                                    best_p = j;
                                }
                            }
                            if (best_p != i)
                            {
                                int tmp = cand_order[i];
                                cand_order[i] = cand_order[best_p];
                                cand_order[best_p] = tmp;
                            }
                        }
                    }
                }

                current_tau = knn_heap_peek_max_dist(heap);
                if ((d < current_tau / eps_factor || heap->count < heap->k) &&
                    frontier_count < GRAPH_FRONTIER_MAX)
                {
                    frontier[frontier_count].frame_id = nb_id;
                    frontier[frontier_count].dist = d;
                    frontier[frontier_count].expanded = 0;
                    frontier_count++;
                }
            }
        } // for (int step = 0; ...)
    } // while (expansions_done ...)

    // Global Containment Criterion via Local k-Ball Radius R_k(best_seed_id)
    if (*best_seed_id >= 0 && heap->count >= heap->k)
    {
        double current_tau = knn_heap_peek_max_dist(heap);
        double r_graph_ball =
            (double)model->graph_distances[*best_seed_id * (long)graph_k + (graph_k - 1)];

        if (r_graph_ball > 0.0 && (*best_seed_dist + current_tau / eps_factor) <= r_graph_ball)
        {
            const uint32_t *neighbors = &model->graph_indices[*best_seed_id * (long)graph_k];
            const float *n_dists = &model->graph_distances[*best_seed_id * (long)graph_k];

            float local_lb2_gc[64];
            int eval_gc_lim = (graph_k < 64) ? graph_k : 64;
            for (int i = 0; i < eval_gc_lim; i++)
            {
                local_lb2_gc[i] = 0.0f;
            }

            for (int k_idx = 0; k_idx < graph_k; k_idx++)
            {
                long nb_id = (long)neighbors[k_idx];
                if (nb_id < 0 || nb_id >= model->total_dataset_frames || nb_id == *best_seed_id)
                {
                    continue;
                }

                current_tau = knn_heap_peek_max_dist(heap);
                double tau_eff = current_tau / eps_factor;
                double tau_eff2 = tau_eff * tau_eff;
                if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_eff)
                {
                    tau_eff2 = config->rlim_cutoff * config->rlim_cutoff;
                }

                if (k_idx < 64 && (double)local_lb2_gc[k_idx] >= tau_eff2)
                {
                    telem->angular_pruned++;
                    continue;
                }

                if (knn_visited_check_and_mark(visited, nb_id))
                {
                    continue;
                }
                if (knn_heap_contains(heap, (int)nb_id))
                {
                    continue;
                }

                double d_edge = (double)n_dists[k_idx];
                double lb_edge = fabs(*best_seed_dist - d_edge);
                current_tau = knn_heap_peek_max_dist(heap);

                if (lb_edge >= current_tau / eps_factor ||
                    (config->rlim_cutoff > 0.0 && lb_edge >= config->rlim_cutoff))
                {
                    telem->graph_edges_pruned++;
                    continue;
                }

                if (knn_reader_read_frame(cand_reader, nb_id, cand_buffer) == 0)
                {
                    telem->framedist_calls++;
                    telem->graph_seeds_evaluated++;
                    double d = compute_euclidean_distance(
                        query_data, cand_buffer, frame_elem
                    );
                    if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
                    {
                        knn_heap_push(heap, (int)nb_id, d);
                    }

                    if (config->use_angular_bound && model->graph_mutual_dists != NULL)
                    {
                        for (int m_idx = k_idx + 1; m_idx < graph_k; m_idx++)
                        {
                            if (m_idx < 64)
                            {
                                double r_m = (double)n_dists[m_idx];
                                float d_mut = knn_get_mutual_dist(
                                    model, *best_seed_id, k_idx, m_idx
                                );
                                if (d_mut > 0.0f)
                                {
                                    double lb2_m = compute_angular_lower_bound_sq(
                                        *best_seed_dist, d_edge, r_m, d, (double)d_mut
                                    );
                                    if (lb2_m > (double)local_lb2_gc[m_idx])
                                    {
                                        local_lb2_gc[m_idx] = (float)lb2_m;
                                    }
                                }
                            }
                        } // for (int m_idx = ...)
                    }
                }
            } // for (int k_idx = 0; ...)

            telem->level0_super_clusters_pruned += (uint64_t)model->num_super_clusters;
            telem->level1_clusters_pruned += (uint64_t)model->num_clusters;
            telem->global_containment_hits++;
            return 1;
        } // if contained
    } // Global Containment

    return 0;
}

/**
 * knn_cross_eval_intra_cluster() - Search target cluster containing best seed for cross dataset.
 * @best_c:           Initial best anchor cluster.
 * @best_seed_id:     Best seed frame ID.
 * @min_d_anchor:     Distance to closest anchor.
 * @num_seed_pivots:  Number of evaluated seed pivots.
 * @seed_pivot_ids:   Evaluated seed pivot frame IDs.
 * @seed_pivot_dists: Computed distances to seed pivots.
 * @query_data:       Query frame pixel data.
 * @model:            Active KnnModel.
 * @config:           Active KnnConfig.
 * @cand_reader:      Candidate frame reader.
 * @cand_buffer:      Candidate frame pixel buffer.
 * @anchor_dists:     Query-to-anchor distance cache.
 * @heap:             Max-heap for current query.
 * @visited:          Per-query frame visited tracker.
 * @telem:            Telemetry record.
 */
static void knn_cross_eval_intra_cluster(
    int                    best_c,
    long                   best_seed_id,
    double                 min_d_anchor,
    int                    num_seed_pivots,
    const long            *seed_pivot_ids,
    const double          *seed_pivot_dists,
    const double *restrict query_data,
    const KnnModel        *model,
    const KnnConfig       *config,
    KnnFrameReader        *cand_reader,
    double        *restrict cand_buffer,
    double        *restrict anchor_dists,
    KnnMaxHeap            *heap,
    KnnVisitedTracker     *visited,
    KnnTelemetry  *restrict telem)
{
    int M = model->num_clusters;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;

    int target_c = best_c;
    if (best_seed_id >= 0 && model->frame_cluster_map != NULL)
    {
        int mapped_c = (int)model->frame_cluster_map[best_seed_id];
        if (mapped_c >= 0 && mapped_c < M)
        {
            target_c = mapped_c;
        }
    }

    if (target_c >= 0 && target_c < M)
    {
        const KnnCluster *target_cl = &model->clusters[target_c];
        telem->total_candidates_considered += (uint64_t)target_cl->num_members;

        double d_anchor_target = anchor_dists[target_c];
        if (d_anchor_target < 0.0)
        {
            telem->framedist_calls++;
            d_anchor_target = compute_euclidean_distance(
                query_data, target_cl->anchor_data, frame_elem
            );
            anchor_dists[target_c] = d_anchor_target;
        }

        int start_m = 0;
        int end_m = target_cl->num_members;
        double tau_init = knn_heap_peek_max_dist(heap);
        if (tau_init < 1e20)
        {
            float r_min = (float)fmax(0.0, d_anchor_target - tau_init / eps_factor);
            float r_max = (float)(d_anchor_target + tau_init / eps_factor);
            start_m = find_member_lower_bound(
                target_cl->members, target_cl->num_members, r_min
            );
            end_m = find_member_upper_bound(
                target_cl->members, target_cl->num_members, r_max
            );
            telem->level3_annular_pruned +=
                (uint64_t)start_m + (uint64_t)(target_cl->num_members - end_m);
        }

        for (int m = start_m; m < end_m; m++)
        {
            long cand_id = (long)target_cl->members[m].frame_id;
            if (knn_visited_check_and_mark(visited, cand_id))
            {
                continue;
            }
            if (knn_heap_contains(heap, (int)cand_id))
            {
                continue;
            }

            double current_tau = knn_heap_peek_max_dist(heap);
            double tau_eff = current_tau / eps_factor;
            double r_cand = (double)target_cl->members[m].r_anchor;

            double r_max_allowed = (heap->count >= heap->k) ?
                (d_anchor_target + tau_eff) : 1e30;
            if (config->rlim_cutoff > 0.0)
            {
                double r_cutoff = d_anchor_target + config->rlim_cutoff;
                if (r_cutoff < r_max_allowed)
                {
                    r_max_allowed = r_cutoff;
                }
            }

            if (r_cand > r_max_allowed)
            {
                telem->level3_annular_pruned += (uint64_t)(end_m - m);
                break;
            }

            if (is_member_pruned_by_pointwise_pivots(
                    cand_id, d_anchor_target, r_cand, 0.0, min_d_anchor,
                    num_seed_pivots, seed_pivot_ids, seed_pivot_dists,
                    model, tau_eff, config->rlim_cutoff, telem))
            {
                continue;
            }

            if (knn_reader_read_frame(cand_reader, cand_id, cand_buffer) == 0)
            {
                telem->framedist_calls++;
                double d = compute_euclidean_distance(
                    query_data, cand_buffer, frame_elem
                );
                if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
                {
                    knn_heap_push(heap, (int)cand_id, d);
                }
            }
        } // for (int m = start_m; ...)
    } // if (target_c >= 0 ...)
}

/**
 * knn_cross_eval_inter_clusters() - Evaluate inter-cluster candidate members for cross dataset.
 * @loc_res:          Cluster locator results.
 * @best_c:           Best anchor cluster index.
 * @min_d_anchor:     Distance to closest anchor.
 * @num_seed_pivots:  Number of evaluated seed pivots.
 * @seed_pivot_ids:   Evaluated seed pivot frame IDs.
 * @seed_pivot_dists: Distances to evaluated seed pivots.
 * @query_data:       Query frame pixel data.
 * @model:            Active KnnModel.
 * @config:           Active KnnConfig.
 * @cand_reader:      Candidate frame reader.
 * @cand_buffer:      Candidate frame pixel buffer.
 * @anchor_dists:     Query-to-anchor distance cache.
 * @scores_buffer:    Scratch buffer for candidate cluster sorting.
 * @active_mask:      Mask of surviving clusters.
 * @heap:             Max-heap for current query.
 * @visited:          Per-query frame visited tracker.
 * @telem:            Telemetry record.
 */
static void knn_cross_eval_inter_clusters(
    const ClusterLocatorResult *loc_res,
    int                         best_c,
    double                      min_d_anchor,
    int                         num_seed_pivots,
    const long                 *seed_pivot_ids,
    const double               *seed_pivot_dists,
    const double *restrict      query_data,
    const KnnModel             *model,
    const KnnConfig            *config,
    KnnFrameReader             *cand_reader,
    double *restrict            cand_buffer,
    double *restrict            anchor_dists,
    ClusterScore *restrict      scores_buffer,
    uint8_t                    *active_mask,
    KnnMaxHeap                 *heap,
    KnnVisitedTracker          *visited,
    KnnTelemetry *restrict      telem)
{
    int M = model->num_clusters;
    long frame_elem = model->frame_elements;
    double eps_factor = 1.0 + config->epsilon;
    double current_tau = knn_heap_peek_max_dist(heap);
    int num_cand_clusters = 0;

    for (int q = 0; q < M; q++)
    {
        if (q == best_c || model->clusters[q].num_members == 0)
        {
            continue;
        }

        if (active_mask[q] == 0)
        {
            telem->level1_clusters_pruned++;
            continue;
        }

        // 3P test against all evaluated anchor pivots using tightened tau_k
        int pruned_3p = 0;
        double r_q = model->cluster_radii[q];
        for (int e = 0; e < loc_res->num_evaluated_anchors; e++)
        {
            int p = loc_res->evaluated_clusters[e];
            double d_p = loc_res->evaluated_dists[e];
            double dcc = model->dcc_matrix[p * M + q];
            double lb1 = dcc - r_q - d_p;
            double lb2 = d_p - dcc - r_q;
            double lb = (lb1 > lb2) ? lb1 : lb2;

            if (lb >= current_tau / eps_factor ||
                (config->rlim_cutoff > 0.0 && lb >= config->rlim_cutoff))
            {
                active_mask[q] = 0;
                pruned_3p = 1;
                break;
            }
        }

        if (pruned_3p)
        {
            telem->level1_clusters_pruned++;
            continue;
        }

        if (anchor_dists[q] < 0.0)
        {
            telem->framedist_calls++;
            anchor_dists[q] = compute_euclidean_distance(
                query_data, model->clusters[q].anchor_data, frame_elem
            );
        }

        double d_a = anchor_dists[q];
        double lb = d_a - r_q;
        if (lb < 0.0)
        {
            lb = 0.0;
        }

        if (lb >= current_tau / eps_factor ||
            (config->rlim_cutoff > 0.0 && lb >= config->rlim_cutoff))
        {
            telem->level1_clusters_pruned++;
            continue;
        }

        scores_buffer[num_cand_clusters].id = q;
        scores_buffer[num_cand_clusters].lb = lb;
        scores_buffer[num_cand_clusters].dcc = d_a;
        num_cand_clusters++;
    } // for (int q = 0; ...)

    qsort(scores_buffer, (size_t)num_cand_clusters, sizeof(ClusterScore),
          compare_cluster_scores);

    // Inter-Cluster Search on surviving candidate clusters
    for (int idx = 0; idx < num_cand_clusters; idx++)
    {
        int q = scores_buffer[idx].id;
        double lb_cluster = scores_buffer[idx].lb;
        current_tau = knn_heap_peek_max_dist(heap);

        if (lb_cluster >= current_tau / eps_factor)
        {
            telem->level1_clusters_pruned++;
            continue;
        }

        if (config->rlim_cutoff > 0.0 && lb_cluster >= config->rlim_cutoff)
        {
            telem->level1_clusters_pruned++;
            continue;
        }

        const KnnCluster *cl = &model->clusters[q];
        double d_anchor = anchor_dists[q];
        telem->total_candidates_considered += (uint64_t)cl->num_members;

        double tau_eff = current_tau / eps_factor;
        if (config->rlim_cutoff > 0.0 && config->rlim_cutoff < tau_eff)
        {
            tau_eff = config->rlim_cutoff;
        }

        float r_min = (float)fmax(0.0, d_anchor - tau_eff);
        float r_max = (float)(d_anchor + tau_eff);

        int start_m = find_member_lower_bound(cl->members, cl->num_members, r_min);
        int end_m = find_member_upper_bound(cl->members, cl->num_members, r_max);

        telem->level3_annular_pruned +=
            (uint64_t)start_m + (uint64_t)(cl->num_members - end_m);

        double dcc_best = model->dcc_matrix[best_c * M + q];

        for (int m = start_m; m < end_m; m++)
        {
            long cand_id = (long)cl->members[m].frame_id;
            if (knn_visited_check_and_mark(visited, cand_id))
            {
                continue;
            }
            if (knn_heap_contains(heap, (int)cand_id))
            {
                telem->graph_edges_pruned++;
                continue;
            }

            current_tau = knn_heap_peek_max_dist(heap);
            tau_eff = current_tau / eps_factor;
            double r_cand = (double)cl->members[m].r_anchor;

            double r_max_allowed = (heap->count >= heap->k) ?
                (d_anchor + tau_eff) : 1e30;
            if (config->rlim_cutoff > 0.0)
            {
                double r_cutoff = d_anchor + config->rlim_cutoff;
                if (r_cutoff < r_max_allowed)
                {
                    r_max_allowed = r_cutoff;
                }
            }

            if (r_cand > r_max_allowed)
            {
                telem->level3_annular_pruned += (uint64_t)(end_m - m);
                break;
            }

            if (is_member_pruned_by_pointwise_pivots(
                    cand_id, d_anchor, r_cand, dcc_best, min_d_anchor,
                    num_seed_pivots, seed_pivot_ids, seed_pivot_dists,
                    model, tau_eff, config->rlim_cutoff, telem))
            {
                continue;
            }

            if (knn_reader_read_frame(cand_reader, cand_id, cand_buffer) == 0)
            {
                telem->framedist_calls++;
                double d = compute_euclidean_distance(
                    query_data, cand_buffer, frame_elem
                );
                if (config->rlim_cutoff <= 0.0 || d <= config->rlim_cutoff)
                {
                    knn_heap_push(heap, (int)cand_id, d);
                }
            }
        } // for (int m = start_m; ...)
    } // for (int idx = 0; ...)
}

/**
 * knn_search_cross_dataset_frame() - Metric-pruned k-NN search for external query frame.
 * @query_id:        Index of query frame.
 * @query_data:      Pixel buffer of query frame.
 * @model:           Active KnnModel.
 * @config:          Active KnnConfig.
 * @cand_reader:     Thread-local candidate frame reader.
 * @cand_buffer:     Scratch buffer for candidate frame pixels.
 * @anchor_dists:    Scratch buffer for query-to-anchor distances [num_clusters].
 * @scores_buffer:   Scratch buffer for cluster sorting [num_clusters].
 * @heap:            KnnMaxHeap structure for this query frame.
 * @prev_cluster_id: Pointer to previously selected cluster ID (for temporal tracking).
 * @visited:         Per-query frame visited tracker.
 * @telem:           Thread-local KnnTelemetry.
 */
static void knn_search_cross_dataset_frame(
    long                   query_id,
    const double *restrict query_data,
    const KnnModel        *model,
    const KnnConfig       *config,
    KnnFrameReader        *cand_reader,
    double        *restrict cand_buffer,
    double        *restrict anchor_dists,
    ClusterScore  *restrict scores_buffer,
    KnnMaxHeap            *heap,
    int                   *prev_cluster_id,
    KnnVisitedTracker     *visited,
    KnnTelemetry  *restrict telem)
{
    (void)query_id;
    int M = model->num_clusters;
    long frame_elem = model->frame_elements;

    int best_c = -1;
    double min_d_anchor = 1e20;
    long best_seed_id = -1;
    double best_seed_dist = 1e20;
    int num_seed_pivots = 0;
    long seed_pivot_ids[8];
    double seed_pivot_dists[8];

    for (int c = 0; c < M; c++)
    {
        anchor_dists[c] = -1.0;
    }

    uint8_t *active_mask = (uint8_t *)alloca((size_t)M);
    ClusterLocatorConfig loc_cfg;
    memset(&loc_cfg, 0, sizeof(loc_cfg));
    loc_cfg.max_targets = (config->approx_mode) ? 24 : 32;
    loc_cfg.te4_mode = 1;
    loc_cfg.rlim = config->rlim_cutoff;
    loc_cfg.tau_max = knn_heap_peek_max_dist(heap);
    loc_cfg.epsilon = config->epsilon;
    loc_cfg.prev_cluster_id = (prev_cluster_id != NULL) ? *prev_cluster_id : -1;

    ClusterLocatorResult loc_res;
    loc_res.active_cluster_mask = active_mask;

    if (cluster_locate_sample(query_data, frame_elem, M, model->anchor_ptrs,
                              model->cluster_radii, model->dcc_matrix, &loc_cfg,
                              &loc_res) == 0)
    {
        telem->framedist_calls += (uint64_t)loc_res.num_evaluated_anchors;
        for (int e = 0; e < loc_res.num_evaluated_anchors; e++)
        {
            anchor_dists[loc_res.evaluated_clusters[e]] = loc_res.evaluated_dists[e];
        }
        best_c = loc_res.best_cluster_id;
        min_d_anchor = loc_res.best_anchor_dist;
        if (prev_cluster_id != NULL && best_c >= 0)
        {
            *prev_cluster_id = best_c;
        }
    }

    if (best_c < 0)
    {
        best_c = 0;
    }

    // Select initial seed in best_c
    if (best_c >= 0 && best_c < M)
    {
        const KnnCluster *best_cl = &model->clusters[best_c];
        double d_anchor_best = (anchor_dists[best_c] >= 0.0) ?
            anchor_dists[best_c] : min_d_anchor;

        int mid = find_member_lower_bound(
            best_cl->members, best_cl->num_members, (float)d_anchor_best
        );
        if (mid >= best_cl->num_members)
        {
            mid = best_cl->num_members - 1;
        }
        if (mid >= 0)
        {
            long seed_0 = (long)best_cl->members[mid].frame_id;
            knn_visited_check_and_mark(visited, seed_0);
            if (knn_reader_read_frame(cand_reader, seed_0, cand_buffer) == 0)
            {
                telem->framedist_calls++;
                double d0 = compute_euclidean_distance(
                    query_data, cand_buffer, frame_elem
                );
                if (config->rlim_cutoff <= 0.0 || d0 <= config->rlim_cutoff)
                {
                    knn_heap_push(heap, (int)seed_0, d0);
                }
                best_seed_id = seed_0;
                best_seed_dist = d0;
                seed_pivot_ids[0] = seed_0;
                seed_pivot_dists[0] = d0;
                num_seed_pivots = 1;
            }
        }
    }

    // Graph Search: Dynamic Frontier Search across candidate seeds
    if (model->has_knn_graph && best_seed_id >= 0 && model->graph_indices != NULL &&
        model->graph_distances != NULL)
    {
        // Phase 1 & 2 (Approx mode): Fast Greedy 1-NN Routing + Direct Basin Expansion
        if (config->approx_mode)
        {
            knn_greedy_route_to_basin(
                query_data, model, config, cand_reader, cand_buffer, heap,
                &best_seed_id, &best_seed_dist, seed_pivot_ids, seed_pivot_dists,
                &num_seed_pivots, visited, telem
            );

            knn_direct_basin_expansion(
                query_data, model, config, cand_reader, cand_buffer, heap,
                &best_seed_id, &best_seed_dist, seed_pivot_ids, seed_pivot_dists,
                &num_seed_pivots, visited, telem
            );

        }

        // Opportunity 1: In approx mode, if basin expansion has saturated the top-k heap,
        // verify whether any other evaluated candidate cluster could overlap the search ball.
        int skip_frontier = 0;
        if (config->approx_mode && heap->count >= heap->k)
        {
            double current_tau = knn_heap_peek_max_dist(heap);
            double eps_factor = 1.0 + config->epsilon;
            double tau_eff = current_tau / eps_factor;
            double min_other_lb = 1e30;
            for (int e = 0; e < loc_res.num_evaluated_anchors; e++)
            {
                int c = loc_res.evaluated_clusters[e];
                if (c != best_c && c >= 0 && c < M)
                {
                    double d_anc = loc_res.evaluated_dists[e];
                    double r_c = model->cluster_radii[c];
                    double lb = d_anc - r_c;
                    if (lb < min_other_lb)
                    {
                        min_other_lb = lb;
                    }
                }
            }
            if (tau_eff <= min_other_lb)
            {
                skip_frontier = 1;
            }
        }

        if (!skip_frontier)
        {
            FrontierNode frontier[GRAPH_FRONTIER_MAX];
            int frontier_count = 0;

            frontier[0].frame_id = best_seed_id;
            frontier[0].dist = best_seed_dist;
            frontier[0].expanded = 0;
            frontier_count = 1;

            knn_cross_seed_frontier(
                &loc_res, best_c, query_data, model, config, cand_reader,
                cand_buffer, heap, frontier, &frontier_count,
                &best_seed_id, &best_seed_dist, visited, telem
            );

            // Phase 2: Local Basin Top-k Collection with Angular Prioritization
            int contained = knn_cross_explore_graph_frontier(
                query_data, model, config, cand_reader, cand_buffer, heap,
                frontier, frontier_count, &best_seed_id, &best_seed_dist,
                seed_pivot_ids, seed_pivot_dists, &num_seed_pivots, visited, telem
            );

            if (contained)
            {
                return;
            }
        }
    }

    // Intra-Cluster Member Search in target cluster (containing best_seed_id)
    knn_cross_eval_intra_cluster(
        best_c, best_seed_id, min_d_anchor, num_seed_pivots,
        seed_pivot_ids, seed_pivot_dists, query_data, model,
        config, cand_reader, cand_buffer, anchor_dists, heap, visited, telem
    );

    if (config->approx_mode)
    {
        telem->level1_clusters_pruned += (uint64_t)M;
        return;
    }

    // Inter-Cluster Search on surviving candidate clusters
    knn_cross_eval_inter_clusters(
        &loc_res, best_c, min_d_anchor, num_seed_pivots,
        seed_pivot_ids, seed_pivot_dists, query_data, model,
        config, cand_reader, cand_buffer, anchor_dists,
        scores_buffer, active_mask, heap, visited, telem
    );
}

/**
 * knn_run_search() - Multi-threaded driver executing k-NN search across all frames.
 * @config:    Active KnnConfig.
 * @model:     Active KnnModel.
 * @results:   Output KnnResults structure to populate.
 * @telemetry: Output aggregated KnnTelemetry structure.
 *
 * Return: 0 on success, -1 on error.
 */
int knn_run_search(
    const KnnConfig *config,
    const KnnModel  *model,
    KnnResults      *results,
    KnnTelemetry    *telemetry)
{
    if (config == NULL || model == NULL || results == NULL || telemetry == NULL)
    {
        return -1;
    }

    memset(telemetry, 0, sizeof(KnnTelemetry));

    int is_cross_dataset = (config->query_data_path != NULL) ? 1 : 0;
    long N_query = model->total_dataset_frames;
    long N_cand = model->total_dataset_frames;
    long q_w = model->frame_width;
    long q_h = model->frame_height;

    if (is_cross_dataset)
    {
        if (knn_reader_inspect(config->query_data_path, &N_query, &q_w, &q_h) != 0 ||
            N_query <= 0)
        {
            fprintf(stderr, "Error: Could not inspect query dataset '%s'\n",
                    config->query_data_path);
            return -1;
        }

        if (q_w * q_h != model->frame_elements)
        {
            fprintf(stderr,
                    "Error: Query frame dim (%ld) does not match model (%ld)\n",
                    q_w * q_h, model->frame_elements);
            return -1;
        }
    }

    int k = config->k;

    results->num_queries = N_query;
    results->indices = (int *)malloc((size_t)N_query * (size_t)k * sizeof(int));
    results->distances = (double *)malloc((size_t)N_query * (size_t)k * sizeof(double));

    if (results->indices == NULL || results->distances == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed for results buffer\n");
        return -1;
    }

    KnnMaxHeap *all_heaps = (KnnMaxHeap *)malloc((size_t)N_query * sizeof(KnnMaxHeap));
    if (all_heaps == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed for heaps array\n");
        knn_results_free(results);
        return -1;
    }

    int search_k = (config->ef_search > 0) ? config->ef_search :
                   (config->approx_mode ? 2 * k : k);
    if (search_k < k)
    {
        search_k = k;
    }

    for (long i = 0; i < N_query; i++)
    {
        if (knn_heap_init(&all_heaps[i], search_k) != 0)
        {
            fprintf(stderr, "Error: Failed to init heap %ld\n", i);
            for (long j = 0; j < i; j++)
            {
                knn_heap_free(&all_heaps[j]);
            }
            free(all_heaps);
            knn_results_free(results);
            return -1;
        }
    } // for (long i = 0; ...)

#ifdef _OPENMP
    omp_lock_t bucket_locks[256];
    if (!is_cross_dataset)
    {
        for (int b = 0; b < 256; b++)
        {
            omp_init_lock(&bucket_locks[b]);
        }
    }
#endif

    KnnFrameReader master_cand_reader;
    KnnFrameReader master_query_reader;

    if (config->memory_data != NULL)
    {
        if (knn_reader_open_memory(&master_cand_reader, config->memory_data, N_cand,
                                   model->frame_elements) != 0)
        {
            knn_results_free(results);
            return -1;
        }
    }
    else if (knn_reader_open(&master_cand_reader, config->input_data_path, N_cand,
                             model->frame_width, model->frame_height) != 0)
    {
        knn_results_free(results);
        return -1;
    }

    if (is_cross_dataset)
    {
        if (knn_reader_open(&master_query_reader, config->query_data_path, N_query,
                            q_w, q_h) != 0)
        {
            knn_reader_close(&master_cand_reader);
            knn_results_free(results);
            return -1;
        }
    }

    int nthreads = config->nthreads;
#ifdef _OPENMP
    if (nthreads > 0)
    {
        omp_set_num_threads(nthreads);
    }
    else
    {
        nthreads = omp_get_max_threads();
    }
#else
    nthreads = 1;
#endif

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    long progress_step = N_query / 100;
    if (progress_step < 1)
    {
        progress_step = 1;
    }

    uint64_t global_telem_calls = 0;
    uint64_t global_telem_l0 = 0;
    uint64_t global_telem_l1 = 0;
    uint64_t global_telem_l2 = 0;
    uint64_t global_telem_l3 = 0;
    uint64_t global_telem_temp = 0;
    uint64_t global_telem_recip = 0;
    uint64_t global_telem_graph_seeds = 0;
    uint64_t global_telem_graph_edges = 0;
    uint64_t global_telem_multi_pivot = 0;
    uint64_t global_telem_angular = 0;
    uint64_t global_telem_containment = 0;
    uint64_t global_telem_cand = 0;

#ifdef _OPENMP
#pragma omp parallel reduction(+:global_telem_calls, global_telem_l0, global_telem_l1, \
                                 global_telem_l2, global_telem_l3, global_telem_temp,   \
                                 global_telem_recip, global_telem_graph_seeds,          \
                                 global_telem_graph_edges, global_telem_multi_pivot,    \
                                 global_telem_angular, global_telem_containment,        \
                                 global_telem_cand)
#endif
    {
        KnnFrameReader thread_cand_reader;
        KnnFrameReader thread_query_reader;

        knn_reader_clone_thread(&master_cand_reader, &thread_cand_reader);
        if (is_cross_dataset)
        {
            knn_reader_clone_thread(&master_query_reader, &thread_query_reader);
        }

        double *query_buffer =
            (double *)malloc((size_t)model->frame_elements * sizeof(double));
        double *cand_buffer =
            (double *)malloc((size_t)model->frame_elements * sizeof(double));
        double *anchor_dists =
            (double *)malloc((size_t)model->num_clusters * sizeof(double));
        ClusterScore *scores_buf =
            (ClusterScore *)malloc((size_t)model->num_clusters * sizeof(ClusterScore));

        KnnTelemetry thread_telem;
        memset(&thread_telem, 0, sizeof(KnnTelemetry));
        int thread_prev_cluster = -1;

        KnnVisitedTracker visited;
        visited.tags = (uint32_t *)calloc((size_t)N_cand, sizeof(uint32_t));
        visited.epoch = 1;

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 32)
#endif
        for (long i = 0; i < N_query; i++)
        {
            visited.epoch++;
            if (visited.epoch == 0)
            {
                if (visited.tags != NULL)
                {
                    memset(visited.tags, 0, (size_t)N_cand * sizeof(uint32_t));
                }
                visited.epoch = 1;
            }

            KnnFrameReader *active_qreader = is_cross_dataset ? &thread_query_reader :
                                                                &thread_cand_reader;

            if (knn_reader_read_frame(active_qreader, i, query_buffer) == 0)
            {
                if (is_cross_dataset)
                {
                    knn_search_cross_dataset_frame(
                        i, query_buffer, model, config, &thread_cand_reader,
                        cand_buffer, anchor_dists, scores_buf, &all_heaps[i],
                        &thread_prev_cluster, &visited, &thread_telem);
                }
                else
                {
                    knn_search_single_frame(
                        i, query_buffer, model, config, &thread_cand_reader,
                        cand_buffer, scores_buf, all_heaps,
#ifdef _OPENMP
                        bucket_locks,
#endif
                        &visited, &thread_telem);
                }
                thread_telem.total_queries++;
            }

            if (config->progress_mode && i % progress_step == 0)
            {
#ifdef _OPENMP
                if (omp_get_thread_num() == 0)
#endif
                {
                    double pct = 100.0 * (double)i / (double)N_query;
                    int bar_offset = 40 - (int)(pct * 0.4);
                    if (bar_offset < 0)
                    {
                        bar_offset = 0;
                    }
                    if (bar_offset > 40)
                    {
                        bar_offset = 40;
                    }
                    const char *bar = "========================================";
                    printf("\rSearching k-NN: [%-40s] %5.1f%% (%ld / %ld frames)",
                           &bar[bar_offset], pct, i, N_query);
                    fflush(stdout);
                }
            }
        } // for (long i = 0; ...)

        global_telem_calls += thread_telem.framedist_calls;
        global_telem_l0 += thread_telem.level0_super_clusters_pruned;
        global_telem_l1 += thread_telem.level1_clusters_pruned;
        global_telem_l2 += thread_telem.level2_anchors_pruned;
        global_telem_l3 += thread_telem.level3_annular_pruned;
        global_telem_temp += thread_telem.temporal_pruned;
        global_telem_recip += thread_telem.reciprocal_reused;
        global_telem_graph_seeds += thread_telem.graph_seeds_evaluated;
        global_telem_graph_edges += thread_telem.graph_edges_pruned;
        global_telem_multi_pivot += thread_telem.multi_pivot_pruned;
        global_telem_angular += thread_telem.angular_pruned;
        global_telem_containment += thread_telem.global_containment_hits;
        global_telem_cand += thread_telem.total_candidates_considered;

        if (visited.tags != NULL)
        {
            free(visited.tags);
        }

        free(scores_buf);
        free(anchor_dists);
        free(cand_buffer);
        free(query_buffer);

        if (is_cross_dataset)
        {
            knn_reader_close_thread(&thread_query_reader);
        }
        knn_reader_close_thread(&thread_cand_reader);
    } // OpenMP parallel block

    // Extract sorted results in parallel from all heaps
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < N_query; i++)
    {
        knn_heap_extract_sorted(&all_heaps[i], &results->indices[i * k],
                                &results->distances[i * k], k);
        knn_heap_free(&all_heaps[i]);
    }
    free(all_heaps);

#ifdef _OPENMP
    if (!is_cross_dataset)
    {
        for (int b = 0; b < 256; b++)
        {
            omp_destroy_lock(&bucket_locks[b]);
        }
    }
#endif

    clock_gettime(CLOCK_MONOTONIC, &end_time);

    if (is_cross_dataset)
    {
        knn_reader_close(&master_query_reader);
    }
    knn_reader_close(&master_cand_reader);

    if (config->progress_mode)
    {
        printf("\rSearching k-NN: [========================================] "
               "100.0%% (%ld / %ld frames)\n",
               N_query, N_query);
        fflush(stdout);
    }

    telemetry->total_queries = (uint64_t)N_query;
    telemetry->framedist_calls = global_telem_calls;
    telemetry->level0_super_clusters_pruned = global_telem_l0;
    telemetry->level1_clusters_pruned = global_telem_l1;
    telemetry->level2_anchors_pruned = global_telem_l2;
    telemetry->level3_annular_pruned = global_telem_l3;
    telemetry->temporal_pruned = global_telem_temp;
    telemetry->reciprocal_reused = global_telem_recip;
    telemetry->graph_seeds_evaluated = global_telem_graph_seeds;
    telemetry->graph_edges_pruned = global_telem_graph_edges;
    telemetry->multi_pivot_pruned = global_telem_multi_pivot;
    telemetry->angular_pruned = global_telem_angular;
    telemetry->global_containment_hits = global_telem_containment;
    telemetry->total_candidates_considered = global_telem_cand;
    telemetry->time_search_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                                (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;

    return 0;
}

/**
 * knn_results_free() - Clean up KnnResults arrays.
 * @results: Pointer to KnnResults.
 */
void knn_results_free(
    KnnResults *results)
{
    if (results == NULL)
    {
        return;
    }

    if (results->indices != NULL)
    {
        free(results->indices);
        results->indices = NULL;
    }

    if (results->distances != NULL)
    {
        free(results->distances);
        results->distances = NULL;
    }
}

