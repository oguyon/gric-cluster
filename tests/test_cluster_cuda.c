/**
 * @file test_cluster_cuda.c
 * @brief Parity tests verifying GPU Pass 2 closest-anchor reassignment vs CPU.
 */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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
 * test_cluster_cuda_parity() - Verify GPU Pass 2 produces exact assignments as CPU.
 */
static void test_cluster_cuda_parity(void)
{
    const char *test_file = "/tmp/test_cluster_cuda_data.txt";
    const char *out_dir_cpu = "/tmp/test_cluster_cuda_cpu";
    const char *out_dir_gpu = "/tmp/test_cluster_cuda_gpu";
    const char *bin = get_binary_path();
    int npoints = 500;

    create_test_spiral_data(test_file, npoints);

    /* Run Pass 1 + Pass 2 on CPU */
    {
        char cmd[512];
        snprintf(cmd, sizeof(cmd),
                 "%s %s -rlim 0.05 -pass2nearest --cpu -outdir %s > /dev/null",
                 bin, test_file, out_dir_cpu);
        int res = system(cmd);
        assert(res == 0);
    }

    /* Run Pass 1 + Pass 2 on GPU */
    {
        char cmd[512];
        snprintf(cmd, sizeof(cmd),
                 "%s %s -rlim 0.05 -pass2nearest --gpu -outdir %s > /dev/null",
                 bin, test_file, out_dir_gpu);
        int res = system(cmd);
        assert(res == 0);
    }

    /* Compare binary frame_membership files */
    {
        char path_cpu[512];
        char path_gpu[512];
        snprintf(path_cpu, sizeof(path_cpu), "%s/frame_membership.bin", out_dir_cpu);
        snprintf(path_gpu, sizeof(path_gpu), "%s/frame_membership.bin", out_dir_gpu);

        FILE *f_cpu = fopen(path_cpu, "rb");
        FILE *f_gpu = fopen(path_gpu, "rb");
        assert(f_cpu != NULL && f_gpu != NULL);

        int *mem_cpu = (int *)malloc((size_t)npoints * sizeof(int));
        int *mem_gpu = (int *)malloc((size_t)npoints * sizeof(int));
        assert(mem_cpu != NULL && mem_gpu != NULL);

        size_t r1 = fread(mem_cpu, sizeof(int), (size_t)npoints, f_cpu);
        size_t r2 = fread(mem_gpu, sizeof(int), (size_t)npoints, f_gpu);
        assert(r1 == (size_t)npoints && r2 == (size_t)npoints);

        fclose(f_cpu);
        fclose(f_gpu);

        int matches = 0;
        for (int i = 0; i < npoints; i++)
        {
            if (mem_cpu[i] == mem_gpu[i])
            {
                matches++;
            }
        }

        free(mem_cpu);
        free(mem_gpu);

        printf("  Pass 2 frame membership parity: %d / %d (%.1f%%)\n",
               matches, npoints, 100.0 * (double)matches / (double)npoints);
        assert(matches == npoints);
    }

    printf("  PASS: test_cluster_cuda_parity\n");
}

int main(void)
{
    printf("Running GPU Clustering Pass 2 Parity Tests...\n");
    test_cluster_cuda_parity();
    printf("All GPU clustering tests passed!\n");
    return 0;
}
