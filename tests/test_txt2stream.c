/**
 * @file test_txt2stream.c
 * @brief Integration tests for gric-txt2stream and gric-cluster streaming mode.
 */

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

/**
 * get_binary_path() - Locate an executable binary in local or build directories.
 * @name: Base name of the binary.
 * @buf:  Output buffer.
 * @size: Size of output buffer.
 */
static const char *get_binary_path(
    const char *name,
    char       *buf,
    size_t      size)
{
    snprintf(buf, size, "./%s", name);
    if (access(buf, X_OK) == 0)
    {
        return buf;
    }
    snprintf(buf, size, "./build/%s", name);
    if (access(buf, X_OK) == 0)
    {
        return buf;
    }
    snprintf(buf, size, "%s", name);
    return buf;
}

/**
 * create_test_dataset() - Generate a simple 2D spiral test file.
 * @filename: Output file path.
 * @npoints:  Number of points.
 */
static void create_test_dataset(
    const char *filename,
    int         npoints)
{
    FILE *fp = fopen(filename, "w");
    assert(fp != NULL);

    for (int i = 0; i < npoints; i++)
    {
        double t = (double)i / (double)npoints;
        double r = t * 0.9;
        double theta = 4.0 * 3.1415926535 * t;
        double x = r * cos(theta);
        double y = r * sin(theta);
        fprintf(fp, "%.6f %.6f\n", x, y);
    }
    fclose(fp);
}

/**
 * test_streaming_pipeline() - Run gric-txt2stream into gric-cluster and verify clustering output.
 */
static void test_streaming_pipeline(void)
{
    const char *test_file = "/tmp/ctest_txt2stream_data.txt";
    const char *out_dir = "/tmp/ctest_txt2stream_out";
    const char *stream_name = "ctest_isio_stream";
    int npoints = 300;

    create_test_dataset(test_file, npoints);

    char txt2stream_bin[256];
    char cluster_bin[256];
    get_binary_path("gric-txt2stream", txt2stream_bin, sizeof(txt2stream_bin));
    get_binary_path("gric-cluster", cluster_bin, sizeof(cluster_bin));

    /* Clean up any old SHM files */
    unlink("/dev/shm/ctest_isio_stream.im.shm");
    unlink("/dev/shm/ctest_isio_stream.sem.shm");

    /* Fork background streamer */
    pid_t spid = fork();
    assert(spid >= 0);

    if (spid == 0)
    {
        char fps_str[32] = "1000";
        char maxfr_str[32];
        snprintf(maxfr_str, sizeof(maxfr_str), "%d", npoints);

        char *s_argv[] = {
            txt2stream_bin,
            (char *)test_file,
            (char *)stream_name,
            "-fps", fps_str,
            "-maxfr", maxfr_str,
            "-cnt2sync",
            NULL
        };
        execv(txt2stream_bin, s_argv);
        _exit(127);
    }

    /* Wait for stream creation */
    usleep(100000);

    /* Run gric-cluster in stream mode */
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "%s 0.08 %s -stream -cnt2sync -maxim %d -outdir %s > /dev/null 2>&1",
             cluster_bin, stream_name, npoints, out_dir);
    int res = system(cmd);
    assert(res == 0);

    /* Wait for streamer to terminate */
    int s_status = 0;
    waitpid(spid, &s_status, 0);

    /* Verify output files */
    char counts_path[512];
    snprintf(counts_path, sizeof(counts_path), "%s/cluster_counts.txt", out_dir);
    FILE *cfp = fopen(counts_path, "r");
    if (!cfp)
    {
        snprintf(counts_path, sizeof(counts_path), "%s.clusterdat/cluster_counts.txt", out_dir);
        cfp = fopen(counts_path, "r");
    }
    assert(cfp != NULL);

    int num_clusters = 0;
    int cl_idx, count;
    while (fscanf(cfp, "Cluster %d: %d frames\n", &cl_idx, &count) == 2)
    {
        num_clusters++;
    }
    fclose(cfp);
    assert(num_clusters > 0);

    /* Clean up */
    unlink("/dev/shm/ctest_isio_stream.im.shm");
    unlink("/dev/shm/ctest_isio_stream.sem.shm");
    unlink(test_file);

    printf("  PASS: test_streaming_pipeline (%d frames clustered into %d clusters)\n",
           npoints, num_clusters);
}

int main(void)
{
    printf("Running gric-txt2stream ImageStreamIO streaming tests...\n");
    test_streaming_pipeline();
    printf("All streaming tests passed successfully!\n");
    return 0;
}
