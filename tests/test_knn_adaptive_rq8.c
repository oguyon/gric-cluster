/**
 * @file test_knn_adaptive_rq8.c
 * @brief Unit test verifying cluster-adaptive RQ8 scaling and tighter lower bounds.
 */

#define _POSIX_C_SOURCE 200809L
#include "knn_cache.h"
#include "knn_pruning.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void free_test_model(
    KnnModel *model)
{
    if (model == NULL)
    {
        return;
    }

    if (model->clusters != NULL)
    {
        for (int c = 0; c < model->num_clusters; c++)
        {
            free(model->clusters[c].anchor_data);
            free(model->clusters[c].members);
        }
    }
    free(model->clusters);
    free(model->cluster_radii);
    free(model->frame_cluster_map);
    free(model->dataset_buffer);
    free(model->rq8_dataset_buffer);
    free(model->rq8_transposed_buffer);
    free(model->frame_to_cluster_pos);
}

static void init_adaptive_test_model(
    KnnModel *model,
    int       with_dataset)
{
    memset(model, 0, sizeof(*model));
    model->frame_width = 3;
    model->frame_height = 1;
    model->frame_elements = 3;
    model->num_clusters = 2;
    model->total_dataset_frames = 4;
    model->model_rlim = 0.50;
    model->is_double = 0;

    model->clusters = (KnnCluster *)calloc(2, sizeof(KnnCluster));
    model->cluster_radii = (double *)calloc(2, sizeof(double));
    model->frame_cluster_map = (int *)calloc(4, sizeof(int));
    assert(model->clusters != NULL);
    assert(model->cluster_radii != NULL);
    assert(model->frame_cluster_map != NULL);

    /* Cluster 0: Compact cluster (r = 0.05) */
    model->clusters[0].cluster_id = 0;
    model->clusters[0].anchor_data = malloc(3 * sizeof(float));
    model->clusters[0].members = (MemberMeta *)calloc(2, sizeof(MemberMeta));
    assert(model->clusters[0].anchor_data != NULL && model->clusters[0].members != NULL);
    float a0[3] = {0.0f, 0.0f, 0.0f};
    memcpy(model->clusters[0].anchor_data, a0, 3 * sizeof(float));
    model->clusters[0].radius = 0.05;
    model->cluster_radii[0] = 0.05;
    model->clusters[0].num_members = 2;
    model->clusters[0].members[0].frame_id = 0;
    model->clusters[0].members[0].r_anchor = 0.0f;
    model->clusters[0].members[1].frame_id = 1;
    model->clusters[0].members[1].r_anchor = 0.05f;

    /* Cluster 1: Wide cluster (r = 0.50) */
    model->clusters[1].cluster_id = 1;
    model->clusters[1].anchor_data = malloc(3 * sizeof(float));
    model->clusters[1].members = (MemberMeta *)calloc(2, sizeof(MemberMeta));
    assert(model->clusters[1].anchor_data != NULL && model->clusters[1].members != NULL);
    float a1[3] = {2.0f, 2.0f, 2.0f};
    memcpy(model->clusters[1].anchor_data, a1, 3 * sizeof(float));
    model->clusters[1].radius = 0.50;
    model->cluster_radii[1] = 0.50;
    model->clusters[1].num_members = 2;
    model->clusters[1].members[0].frame_id = 2;
    model->clusters[1].members[0].r_anchor = 0.0f;
    model->clusters[1].members[1].frame_id = 3;
    model->clusters[1].members[1].r_anchor = 0.50f;

    model->frame_cluster_map[0] = 0;
    model->frame_cluster_map[1] = 0;
    model->frame_cluster_map[2] = 1;
    model->frame_cluster_map[3] = 1;

    if (with_dataset)
    {
        float *buf = (float *)calloc(4 * 3, sizeof(float));
        assert(buf != NULL);
        /* Frame 0: anchor 0 */
        buf[0] = 0.0f; buf[1] = 0.0f; buf[2] = 0.0f;
        /* Frame 1: delta 0.03, 0.04, 0.0 (distance = 0.05) */
        buf[3] = 0.03f; buf[4] = 0.04f; buf[5] = 0.0f;
        /* Frame 2: anchor 1 */
        buf[6] = 2.0f; buf[7] = 2.0f; buf[8] = 2.0f;
        /* Frame 3: delta 0.30, 0.40, 0.0 (distance = 0.50) */
        buf[9] = 2.30f; buf[10] = 2.40f; buf[11] = 2.0f;
        model->dataset_buffer = buf;
    }
}

int main(void)
{
    printf("[TEST] Running adaptive per-cluster RQ8 scaling test...\n");

    KnnModel model;
    KnnConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.use_rq8 = 1;
    const char *tmp_sidecar = "/tmp/test_adaptive_rq8.rq8";
    cfg.rq8_save_path = (char *)tmp_sidecar;

    init_adaptive_test_model(&model, 1);

    int build_res = knn_model_build_or_load_rq8(&model, &cfg);
    assert(build_res == 0);
    assert(model.rq8_dataset_buffer != NULL);

    /* Verify adaptive scale assignment: cluster 0 (0.05) vs cluster 1 (0.50) */
    float scale0 = model.clusters[0].rq8_params.scale;
    float scale1 = model.clusters[1].rq8_params.scale;
    float rlim0 = model.clusters[0].rq8_params.rlim;
    float rlim1 = model.clusters[1].rq8_params.rlim;

    printf("  Cluster 0 (r = %.2f): scale = %.6f\n", rlim0, scale0);
    printf("  Cluster 1 (r = %.2f): scale = %.6f\n", rlim1, scale1);

    assert(fabsf(rlim0 - 0.05f) < 1e-5f);
    assert(fabsf(rlim1 - 0.50f) < 1e-5f);
    assert(fabsf(scale0 - (0.05f / 127.0f)) < 1e-5f);
    assert(fabsf(scale1 - (0.50f / 127.0f)) < 1e-5f);
    assert(scale0 < scale1 * 0.15f); /* Scale 0 should be ~10x tighter than Scale 1 */

    /* Verify cutoff calculation: at search radius 0.04 */
    uint64_t cutoff_cl0 = compute_rq8_cutoff_thresh_cluster(
        0.04, &model.clusters[0].rq8_params, &cfg
    );
    uint64_t cutoff_global = compute_rq8_cutoff_thresh(
        0.04, &model, &cfg
    );

    printf("  Cutoff at tau=0.04: cluster 0 = %lu, global (max r) = %lu\n",
           (unsigned long)cutoff_cl0, (unsigned long)cutoff_global);
    assert(cutoff_cl0 > 0 && cutoff_cl0 != UINT64_MAX);
    assert(cutoff_global > 0 && cutoff_global != UINT64_MAX);

    /* Verify query residual quantization on cluster 0 */
    float query[3] = {0.0f, 0.0f, 0.0f};
    int16_t q_res[3];
    int clip = rq8_quantize_query_residual_float(
        query, (const float *)model.clusters[0].anchor_data, q_res,
        &model.clusters[0].rq8_params
    );
    assert(clip == 0);
    assert(q_res[0] == 0 && q_res[1] == 0 && q_res[2] == 0);

    /* Verify sidecar reload preserves per-cluster configuration */
    KnnModel model_reload;
    KnnConfig cfg_reload;
    memset(&cfg_reload, 0, sizeof(cfg_reload));
    cfg_reload.use_rq8 = 1;
    cfg_reload.rq8_load_path = (char *)tmp_sidecar;

    init_adaptive_test_model(&model_reload, 0);
    int load_res = knn_model_build_or_load_rq8(&model_reload, &cfg_reload);
    assert(load_res == 0);
    assert(model_reload.rq8_dataset_buffer != NULL);
    assert(fabsf(model_reload.clusters[0].rq8_params.scale - scale0) < 1e-6f);
    assert(fabsf(model_reload.clusters[1].rq8_params.scale - scale1) < 1e-6f);

    remove(tmp_sidecar);
    free_test_model(&model);
    free_test_model(&model_reload);

    printf("[TEST] Adaptive per-cluster RQ8 scaling test PASSED successfully.\n");
    return 0;
}
