#include "knn_cache.h"
#include <assert.h>
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
    free(model->frame_cluster_map);
    free(model->dataset_buffer);
    free(model->rq8_dataset_buffer);
    free(model->rq8_transposed_buffer);
}

static void init_test_model(
    KnnModel      *model,
    const float    anchor[3],
    int            with_dataset)
{
    memset(model, 0, sizeof(*model));
    model->frame_width = 3;
    model->frame_height = 1;
    model->frame_elements = 3;
    model->num_clusters = 1;
    model->total_dataset_frames = 1;
    model->model_rlim = 1.0;
    model->is_double = 0;

    model->clusters = (KnnCluster *)calloc(1, sizeof(KnnCluster));
    model->frame_cluster_map = (int *)calloc(1, sizeof(int));
    assert(model->clusters != NULL);
    assert(model->frame_cluster_map != NULL);

    model->clusters[0].anchor_data = malloc(3 * sizeof(float));
    model->clusters[0].members = (MemberMeta *)calloc(1, sizeof(MemberMeta));
    assert(model->clusters[0].anchor_data != NULL);
    assert(model->clusters[0].members != NULL);

    memcpy(model->clusters[0].anchor_data, anchor, 3 * sizeof(float));
    model->clusters[0].radius = 1.0f;
    model->clusters[0].num_members = 1;
    model->clusters[0].members[0].frame_id = 0;
    model->clusters[0].members[0].r_anchor = 0.0f;
    model->frame_cluster_map[0] = 0;

    if (with_dataset)
    {
        model->dataset_buffer = calloc(3, sizeof(float));
        assert(model->dataset_buffer != NULL);
    }
}

int main(void)
{
    const char *tmp_path = "/tmp/test_rq8_sidecar_mismatch.rq8";
    const float anchor_a[3] = {0.0f, 0.0f, 0.0f};
    const float anchor_b[3] = {0.5f, 0.0f, 0.0f};
    KnnConfig cfg;
    KnnModel model_a;
    KnnModel model_b;

    init_test_model(&model_a, anchor_a, 1);
    memset(&cfg, 0, sizeof(cfg));
    cfg.use_rq8 = 1;
    cfg.rq8_save_path = (char *)tmp_path;
    assert(knn_model_build_or_load_rq8(&model_a, &cfg) == 0);

    init_test_model(&model_b, anchor_b, 0);
    memset(&cfg, 0, sizeof(cfg));
    cfg.use_rq8 = 1;
    cfg.rq8_load_path = (char *)tmp_path;
    assert(knn_model_build_or_load_rq8(&model_b, &cfg) == -1);
    assert(model_b.rq8_dataset_buffer == NULL);

    remove(tmp_path);
    free_test_model(&model_a);
    free_test_model(&model_b);
    printf("RQ8 sidecar fingerprint mismatch rejection passed.\n");
    return 0;
}
