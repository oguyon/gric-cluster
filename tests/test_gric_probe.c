/**
 * @file test_gric_probe.c
 * @brief Unit tests for dataset probing and .gricprof JSON serialization.
 */

#include "gric_profile.h"
#include "probe_engine.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/**
 * test_profile_json_roundtrip() - Test serialization and parsing of GricProfile.
 */
static void test_profile_json_roundtrip(void)
{
    const char *test_path = "/tmp/test_probe_roundtrip.gricprof";
    GricProfile p1;
    gric_profile_init(&p1, 10);

    snprintf(p1.dataset_path, sizeof(p1.dataset_path), "%s", "test_dataset.fits");
    p1.num_frames = 2000;
    p1.width = 10;
    p1.height = 1;
    p1.rlim_p01 = 0.015;
    p1.rlim_p03 = 0.035;
    p1.rlim_fine = 0.05;
    p1.rlim_balanced = 0.10;
    p1.rlim_coarse = 0.25;
    p1.rlim_recommended = 0.10;
    p1.recommended_maxcl = 2500;
    p1.tiles_x = 2;
    p1.tiles_y = 2;
    p1.pred_enabled = 1;
    p1.pred_len = 3;
    p1.pred_h = 500;
    p1.continuity_ratio = 0.15;
    p1.tm_mixing_coeff = 0.35;
    p1.use_sq8 = 1;
    p1.sq8_params.min_val = -10.0f;
    p1.sq8_params.max_val = 10.0f;
    p1.sq8_params.scale = 20.0f / 255.0f;
    p1.sq8_params.err_radius = 0.05f;
    p1.te4_enabled = 1;
    p1.te5_enabled = 1;
    p1.sparse_dcc_enabled = 1;
    p1.entropy_enabled = 1;
    p1.entropy_gate = 0.25;
    p1.soft_bayesian_enabled = 1;
    p1.soft_bayesian_sigma_coeff = 1.35;
    p1.recommend_double = 1;
    p1.noise_floor_est = 0.045;
    p1.te3_prune_rate = 0.92;
    p1.te4_marginal_rate = 0.04;
    p1.te5_marginal_rate = 0.02;
    strcpy(p1.recommended_prune_mode, "5P");

    int write_res = gric_profile_write_json(test_path, &p1);
    assert(write_res == 0);

    GricProfile p2;
    int read_res = gric_profile_read_json(test_path, &p2);
    assert(read_res == 0);

    assert(p2.num_frames == p1.num_frames);
    assert(p2.dim == p1.dim);
    assert(fabs(p2.rlim_p01 - p1.rlim_p01) < 1e-5);
    assert(fabs(p2.rlim_p03 - p1.rlim_p03) < 1e-5);
    assert(fabs(p2.rlim_fine - p1.rlim_fine) < 1e-5);
    assert(fabs(p2.rlim_balanced - p1.rlim_balanced) < 1e-5);
    assert(fabs(p2.rlim_coarse - p1.rlim_coarse) < 1e-5);
    assert(fabs(p2.rlim_recommended - p1.rlim_recommended) < 1e-5);
    assert(p2.recommended_maxcl == p1.recommended_maxcl);
    assert(p2.pred_enabled == p1.pred_enabled);
    assert(p2.pred_len == p1.pred_len);
    assert(p2.pred_h == p1.pred_h);
    assert(fabs(p2.continuity_ratio - p1.continuity_ratio) < 1e-4);
    assert(fabs(p2.tm_mixing_coeff - p1.tm_mixing_coeff) < 1e-3);
    assert(p2.use_sq8 == p1.use_sq8);
    assert(p2.te4_enabled == p1.te4_enabled);
    assert(p2.te5_enabled == p1.te5_enabled);
    assert(p2.sparse_dcc_enabled == p1.sparse_dcc_enabled);
    assert(p2.entropy_enabled == p1.entropy_enabled);
    assert(fabs(p2.entropy_gate - p1.entropy_gate) < 1e-3);
    assert(p2.soft_bayesian_enabled == p1.soft_bayesian_enabled);
    assert(fabs(p2.soft_bayesian_sigma_coeff - p1.soft_bayesian_sigma_coeff) < 1e-3);
    assert(p2.recommend_double == p1.recommend_double);
    assert(fabs(p2.noise_floor_est - p1.noise_floor_est) < 1e-4);
    assert(strcmp(p2.recommended_prune_mode, "5P") == 0);

    gric_profile_free(&p1);
    gric_profile_free(&p2);
    unlink(test_path);
    printf("test_profile_json_roundtrip passed.\n");
}

/**
 * test_probe_execution_synthetic() - Test probe_run on 2Dspiral dataset.
 */
static void test_probe_execution_synthetic(void)
{
    ProbeConfig cfg;
    memset(&cfg, 0, sizeof(ProbeConfig));
    snprintf(cfg.dataset_path, sizeof(cfg.dataset_path), "benchmarks/2Dspiral.txt");
    cfg.sample_limit = 500;
    cfg.verbose_level = 0;
    cfg.show_progress = 0;

    ProbeResults res;
    int rc = probe_run(&cfg, &res);
    assert(rc == 0);

    assert(res.profile.num_frames == 500);
    assert(res.profile.dim == 2);
    assert(res.profile.rlim_p01 > 0.0);
    assert(res.profile.rlim_p03 >= res.profile.rlim_p01);
    assert(res.profile.rlim_fine >= res.profile.rlim_p03);
    assert(res.profile.rlim_balanced >= res.profile.rlim_fine);
    assert(res.profile.rlim_coarse >= res.profile.rlim_balanced);
    assert(res.profile.continuity_ratio < 0.10);
    assert(res.profile.pred_enabled == 1);
    assert(res.profile.tm_mixing_coeff >= 0.15);
    assert(res.profile.te3_prune_rate > 0.50);
    assert(res.profile.te4_enabled == 0);
    assert(strcmp(res.profile.recommended_prune_mode, "3P") == 0);

    probe_results_free(&res);
    printf("test_probe_execution_synthetic passed.\n");
}

int main(void)
{
    test_profile_json_roundtrip();
    test_probe_execution_synthetic();
    printf("All probe tests passed successfully.\n");
    return 0;
}
