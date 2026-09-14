/**
 * @file test_knn_cuda.c
 * @brief Parity tests verifying GPU-accelerated k-NN results match CPU baseline.
 */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/**
 * create_spiral_dataset() - Generate 2D synthetic dataset for k-NN testing.
 * @path:    Output file path.
 * @nframes: Number of coordinate frames to generate.
 */
static void create_spiral_dataset(
    const char *path,
    int         nframes)
{
    FILE *fp = fopen(path, "w");
    assert(fp != NULL);

    for (int i = 0; i < nframes; i++)
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
 * get_binary_path() - Locate executable binary path.
 * @name: Executable name ("gric-cluster" or "gric-knn").
 * @out:  Buffer to receive path.
 * @sz:   Buffer size.
 */
static void get_binary_path(
    const char *name,
    char       *out,
    size_t      sz)
{
    snprintf(out, sz, "./%s", name);
    if (access(out, X_OK) == 0)
    {
        return;
    }
    snprintf(out, sz, "./build/%s", name);
    if (access(out, X_OK) == 0)
    {
        return;
    }
    snprintf(out, sz, "%s", name);
}

/**
 * compare_binary_indices() - Compare binary index files for parity.
 * @file_cpu: Path to CPU output binary file.
 * @file_gpu: Path to GPU output binary file.
 * @nqueries: Number of query frames.
 * @k:        Number of neighbors.
 *
 * Return: Number of matching index entries.
 */
static int compare_binary_indices(
    const char *file_cpu,
    const char *file_gpu,
    int         nqueries,
    int         k)
{
    FILE *f_cpu = fopen(file_cpu, "rb");
    FILE *f_gpu = fopen(file_gpu, "rb");
    assert(f_cpu != NULL && f_gpu != NULL);

    size_t total_elements = (size_t)nqueries * (size_t)k;
    int *cpu_idx = (int *)malloc(total_elements * sizeof(int));
    int *gpu_idx = (int *)malloc(total_elements * sizeof(int));
    assert(cpu_idx != NULL && gpu_idx != NULL);

    size_t r1 = fread(cpu_idx, sizeof(int), total_elements, f_cpu);
    size_t r2 = fread(gpu_idx, sizeof(int), total_elements, f_gpu);
    assert(r1 == total_elements && r2 == total_elements);

    fclose(f_cpu);
    fclose(f_gpu);

    int matches = 0;
    for (size_t i = 0; i < total_elements; i++)
    {
        if (cpu_idx[i] == gpu_idx[i])
        {
            matches++;
        }
    }

    free(cpu_idx);
    free(gpu_idx);
    return matches;
}

/**
 * test_knn_cuda_parity() - Run end-to-end k-NN on CPU and GPU and verify parity.
 */
static void test_knn_cuda_parity(void)
{
    const char *data_file = "/tmp/test_knn_cuda_data.txt";
    const char *cluster_dir = "/tmp/test_knn_cuda_clusters";
    char cluster_bin[256];
    char knn_bin[256];
    get_binary_path("gric-cluster", cluster_bin, sizeof(cluster_bin));
    get_binary_path("gric-knn", knn_bin, sizeof(knn_bin));

    int nframes = 1000;
    int k = 10;
    char cmd[1024];

    create_spiral_dataset(data_file, nframes);

    /* Generate Pass 1 clusters */
    snprintf(cmd, sizeof(cmd),
             "%s %s -rlim 0.10 -outdir %s > /dev/null 2>&1",
             cluster_bin, data_file, cluster_dir);
    int rc = system(cmd);
    assert(rc == 0);

    /* 1. Bidirectional Parity */
    {
        snprintf(cmd, sizeof(cmd),
                 "%s %s %s -k %d -dtmin 5 --cpu -o /tmp/knn_bidi_cpu.txt > /dev/null 2>&1",
                 knn_bin, data_file, cluster_dir, k);
        rc = system(cmd);
        assert(rc == 0);

        snprintf(cmd, sizeof(cmd),
                 "%s %s %s -k %d -dtmin 5 --gpu -o /tmp/knn_bidi_gpu.txt > /dev/null 2>&1",
                 knn_bin, data_file, cluster_dir, k);
        rc = system(cmd);
        assert(rc == 0);

        int matches = compare_binary_indices(
            "/tmp/knn_bidi_cpu_indices.bin",
            "/tmp/knn_bidi_gpu_indices.bin",
            nframes, k);
        double pct = 100.0 * (double)matches / (double)(nframes * k);
        printf("  Bidirectional k-NN parity: %d / %d (%.2f%%)\n",
               matches, nframes * k, pct);
        assert(pct >= 97.0);
    }

    /* 2. Past-Only Parity */
    {
        snprintf(cmd, sizeof(cmd),
                 "%s %s %s -k %d -dtmin 5 -past --cpu -o /tmp/knn_past_cpu.txt > /dev/null 2>&1",
                 knn_bin, data_file, cluster_dir, k);
        rc = system(cmd);
        assert(rc == 0);

        snprintf(cmd, sizeof(cmd),
                 "%s %s %s -k %d -dtmin 5 -past --gpu -o /tmp/knn_past_gpu.txt > /dev/null 2>&1",
                 knn_bin, data_file, cluster_dir, k);
        rc = system(cmd);
        assert(rc == 0);

        int matches = compare_binary_indices(
            "/tmp/knn_past_cpu_indices.bin",
            "/tmp/knn_past_gpu_indices.bin",
            nframes, k);
        double pct = 100.0 * (double)matches / (double)(nframes * k);
        printf("  Past-Only k-NN parity:     %d / %d (%.2f%%)\n",
               matches, nframes * k, pct);
        assert(pct >= 97.0);
    }

    /* 3. Future-Only Parity */
    {
        snprintf(cmd, sizeof(cmd),
                 "%s %s %s -k %d -dtmin 5 -future --cpu -o /tmp/knn_fut_cpu.txt > /dev/null 2>&1",
                 knn_bin, data_file, cluster_dir, k);
        rc = system(cmd);
        assert(rc == 0);

        snprintf(cmd, sizeof(cmd),
                 "%s %s %s -k %d -dtmin 5 -future --gpu -o /tmp/knn_fut_gpu.txt > /dev/null 2>&1",
                 knn_bin, data_file, cluster_dir, k);
        rc = system(cmd);
        assert(rc == 0);

        int matches = compare_binary_indices(
            "/tmp/knn_fut_cpu_indices.bin",
            "/tmp/knn_fut_gpu_indices.bin",
            nframes, k);
        double pct = 100.0 * (double)matches / (double)(nframes * k);
        printf("  Future-Only k-NN parity:   %d / %d (%.2f%%)\n",
               matches, nframes * k, pct);
        assert(pct >= 97.0);
    }

    /* 4. Radius Cutoff (-rlim) Parity */
    {
        snprintf(cmd, sizeof(cmd),
                 "%s %s %s -k %d -rlim 0.05 --cpu -o /tmp/knn_rlim_cpu.txt > /dev/null 2>&1",
                 knn_bin, data_file, cluster_dir, k);
        rc = system(cmd);
        assert(rc == 0);

        snprintf(cmd, sizeof(cmd),
                 "%s %s %s -k %d -rlim 0.05 --gpu -o /tmp/knn_rlim_gpu.txt > /dev/null 2>&1",
                 knn_bin, data_file, cluster_dir, k);
        rc = system(cmd);
        assert(rc == 0);

        int matches = compare_binary_indices(
            "/tmp/knn_rlim_cpu_indices.bin",
            "/tmp/knn_rlim_gpu_indices.bin",
            nframes, k);
        double pct = 100.0 * (double)matches / (double)(nframes * k);
        printf("  Radius Cutoff k-NN parity: %d / %d (%.2f%%)\n",
               matches, nframes * k, pct);
        assert(pct >= 97.0);
    }

    printf("  PASS: test_knn_cuda_parity\n");
}

int main(void)
{
    printf("Running GPU k-NN Parity Tests...\n");
    test_knn_cuda_parity();
    printf("All GPU k-NN tests passed!\n");
    return 0;
}
