/**
 * @file test_pass2_reassign.c
 * @brief Unit and integration tests for Second Pass closest-anchor clustering.
 */

#include "cluster_core.h"
#include "cluster_defs.h"
#include "cluster_reassign.h"
#include "frameread.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * create_test_spiral_data() - Generate a 2D spiral test file.
 * @filename: Output path.
 * @npoints:  Number of coordinate points to write.
 */
static void create_test_spiral_data(
    const char *filename,
    int         npoints)
{
    FILE *fp = fopen(filename, "w");
    assert(fp != NULL);

    for (int i = 0; i < npoints; i++)
    {
        double theta = (double)i * 0.05;
        double r = 0.01 + 0.0005 * (double)i;
        double x = r * cos(theta);
        double y = r * sin(theta);
        fprintf(fp, "%.6f %.6f\n", x, y);
    }

    fclose(fp);
}

#include <unistd.h>

/**
 * get_binary_path() - Locate the gric-cluster executable.
 */
static const char *get_binary_path(void)
{
    if (access("./gric-cluster", X_OK) == 0)
    {
        return "./gric-cluster";
    }
    if (access("./build/gric-cluster", X_OK) == 0)
    {
        return "./build/gric-cluster";
    }
    return "gric-cluster";
}

/**
 * test_pass2_nearest_accuracy() - Test that Pass 2 assigns every point to its closest anchor.
 */
static void test_pass2_nearest_accuracy(void)
{
    const char *test_file = "/tmp/test_pass2_data.txt";
    const char *out_dir_p1 = "/tmp/test_pass2_out_p1";
    const char *out_dir_p2 = "/tmp/test_pass2_out_p2";
    const char *bin = get_binary_path();
    int npoints = 500;

    create_test_spiral_data(test_file, npoints);

    /* Run Pass 1 alone */
    {
        char cmd[512];
        snprintf(cmd, sizeof(cmd),
                 "%s %s -rlim 0.05 -outdir %s > /dev/null",
                 bin, test_file, out_dir_p1);
        int res = system(cmd);
        assert(res == 0);
    }

    /* Run Pass 1 + Pass 2 */
    {
        char cmd[512];
        snprintf(cmd, sizeof(cmd),
                 "%s %s -rlim 0.05 -pass2nearest -outdir %s > /dev/null",
                 bin, test_file, out_dir_p2);
        int res = system(cmd);
        assert(res == 0);
    }

    /* Verify Pass 2 results: load anchors and points */
    FILE *anchors_fp = fopen("/tmp/test_pass2_out_p2/anchors.txt", "r");
    assert(anchors_fp != NULL);

    double anchors[256][2];
    int num_anchors = 0;
    while (fscanf(anchors_fp, "%lf %lf",
                  &anchors[num_anchors][0],
                  &anchors[num_anchors][1]) == 2)
    {
        num_anchors++;
        if (num_anchors >= 256)
        {
            break;
        }
    }
    fclose(anchors_fp);
    assert(num_anchors > 1);

    /* Load membership from Pass 2 */
    FILE *mem_fp = fopen("/tmp/test_pass2_out_p2/frame_membership.txt", "r");
    assert(mem_fp != NULL);

    FILE *pts_fp = fopen(test_file, "r");
    assert(pts_fp != NULL);

    long frame_idx;
    int assigned_cl;
    double dist_recorded;
    int verified_count = 0;

    while (fscanf(mem_fp, "%ld %d %lf",
                  &frame_idx, &assigned_cl, &dist_recorded) == 3)
    {
        double px, py;
        int read_pts = fscanf(pts_fp, "%lf %lf", &px, &py);
        assert(read_pts == 2);

        /* Compute true closest anchor among all anchors */
        int true_best_k = -1;
        double min_d = 1e30;

        for (int k = 0; k < num_anchors; k++)
        {
            double dx = px - anchors[k][0];
            double dy = py - anchors[k][1];
            double d = sqrt(dx * dx + dy * dy);
            if (d < min_d)
            {
                min_d = d;
                true_best_k = k;
            }
        }

        /* Check that Pass 2 assigned the frame to the globally nearest anchor */
        assert(assigned_cl >= 0 && assigned_cl < num_anchors);
        assert(assigned_cl == true_best_k);
        double actual_dx = px - anchors[assigned_cl][0];
        double actual_dy = py - anchors[assigned_cl][1];
        double actual_d = sqrt(actual_dx * actual_dx + actual_dy * actual_dy);

        /* Distance must match the minimum distance within floating-point tolerance */
        assert(fabs(actual_d - min_d) < 1e-5);
        verified_count++;
    }

    fclose(mem_fp);
    fclose(pts_fp);
    assert(verified_count == npoints);

    printf("  PASS: test_pass2_nearest_accuracy (%d points verified)\n", verified_count);
}

int main(void)
{
    printf("Running Pass 2 Nearest-Anchor Clustering Tests...\n");
    test_pass2_nearest_accuracy();
    printf("All Pass 2 tests passed successfully!\n");
    return 0;
}
