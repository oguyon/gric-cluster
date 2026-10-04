/**
 * @file test_libgric_c.c
 * @brief Unit test verifying C API functionality for libgric.
 */

#include "gric/gric.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <unistd.h>

int main(void)
{
    printf("Testing libgric C API (version %s)...\n", gric_version());

    gric_cluster_config_t cfg;
    gric_status_t status = gric_cluster_config_default(&cfg);
    assert(status == GRIC_SUCCESS);
    assert(cfg.rlim == 0.5);

    cfg.rlim = 1.0;
    cfg.maxnbclust = 32;

    const size_t ndim = 4;
    gric_cluster_t *ctx = gric_cluster_create(&cfg, ndim);
    assert(ctx != NULL);
    assert(gric_cluster_get_num_clusters(ctx) == 0);

    // Frame 0: at origin -> should create cluster 0
    double f0[4] = {0.0, 0.0, 0.0, 0.0};
    int64_t c0 = -1;
    status = gric_cluster_feed_frame(ctx, f0, &c0);
    assert(status == GRIC_SUCCESS);
    assert(c0 == 0);
    assert(gric_cluster_get_num_clusters(ctx) == 1);

    // Frame 1: close to origin (dist = 0.2 < rlim) -> should match cluster 0
    double f1[4] = {0.1, 0.1, 0.0, 0.0};
    int64_t c1 = -1;
    status = gric_cluster_feed_frame(ctx, f1, &c1);
    assert(status == GRIC_SUCCESS);
    assert(c1 == 0);
    assert(gric_cluster_get_num_clusters(ctx) == 1);

    // Frame 2: far from origin (dist = 5.0 > rlim) -> should create cluster 1
    double f2[4] = {5.0, 0.0, 0.0, 0.0};
    int64_t c2 = -1;
    status = gric_cluster_feed_frame(ctx, f2, &c2);
    assert(status == GRIC_SUCCESS);
    assert(c2 == 1);
    assert(gric_cluster_get_num_clusters(ctx) == 2);

    // Test batch feed
    double batch[8] = {
        0.05, 0.05, 0.0, 0.0,  // close to cluster 0
        5.1,  0.0,  0.0, 0.0   // close to cluster 1
    };
    int64_t batch_out[2] = {-1, -1};
    status = gric_cluster_feed_batch(ctx, batch, 2, batch_out);
    assert(status == GRIC_SUCCESS);
    assert(batch_out[0] == 0);
    assert(batch_out[1] == 1);

    // Test anchor export
    double anchors[8];
    int members[2];
    int64_t n_anchors = gric_cluster_get_anchors(ctx, anchors, members, 2);
    assert(n_anchors == 2);
    assert(members[0] >= 2);
    assert(members[1] >= 2);

    // Test stats export
    gric_cluster_stats_t stats;
    status = gric_cluster_get_stats(ctx, &stats);
    assert(status == GRIC_SUCCESS);
    assert(stats.total_frames_processed == 5);
    assert(stats.num_clusters == 2);
    assert(stats.num_new_clusters >= 1);
    assert(stats.framedist_calls > 0);

    // Test binary results export
    status = gric_cluster_save_results(ctx, "/tmp/test_libgric_results");
    assert(status == GRIC_SUCCESS);
    FILE *test_fp = fopen("/tmp/test_libgric_results/anchors.bin", "rb");
    assert(test_fp != NULL);
    fclose(test_fp);
    test_fp = fopen("/tmp/test_libgric_results/dcc.bin", "rb");
    assert(test_fp != NULL);
    fclose(test_fp);
    test_fp = fopen("/tmp/test_libgric_results/cluster_counts.bin", "rb");
    assert(test_fp != NULL);
    fclose(test_fp);
    test_fp = fopen("/tmp/test_libgric_results/cluster_radii.bin", "rb");
    assert(test_fp != NULL);
    fclose(test_fp);

    unlink("/tmp/test_libgric_results/anchors.bin");
    unlink("/tmp/test_libgric_results/dcc.bin");
    unlink("/tmp/test_libgric_results/cluster_counts.bin");
    unlink("/tmp/test_libgric_results/cluster_radii.bin");
    rmdir("/tmp/test_libgric_results");

    // Test single-precision float32 feed APIs
    float f_single[4] = {0.08f, 0.08f, 0.0f, 0.0f};
    int64_t c_f32 = -1;
    status = gric_cluster_feed_frame_f32(ctx, f_single, &c_f32);
    assert(status == GRIC_SUCCESS);
    assert(c_f32 == 0);

    float f_batch[8] = {
        0.02f, 0.02f, 0.0f, 0.0f,
        5.05f, 0.0f, 0.0f, 0.0f
    };
    int64_t batch_out_f32[2] = {-1, -1};
    status = gric_cluster_feed_batch_f32(ctx, f_batch, 2, batch_out_f32);
    assert(status == GRIC_SUCCESS);
    assert(batch_out_f32[0] == 0);
    assert(batch_out_f32[1] == 1);

    gric_cluster_destroy(ctx);
    printf("libgric C API test passed successfully.\n");
    return 0;
}
