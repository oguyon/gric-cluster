/**
 * @file test_libgric_c.c
 * @brief Unit test verifying C API functionality for libgric.
 */

#include "gric/gric.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

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

    gric_cluster_destroy(ctx);
    printf("libgric C API test passed successfully.\n");
    return 0;
}
