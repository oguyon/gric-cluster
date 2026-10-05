/**
 * @file test_mcp_jobs.c
 * @brief Unit and integration tests for async background job execution and lifecycle tools.
 */

#include "mcp_tools.h"
#include "mcp_jobs.h"
#include "tests/mcp/mcp_test_util.h"
#include "shared/cjson/cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void ensure_test_dataset(void)
{
    if (access("/tmp/ctest_spiral.txt", R_OK) != 0)
    {
        char root[1024];
        mcp_get_project_root(root, sizeof(root));
        char cmd[2048];
        snprintf(
            cmd, sizeof(cmd),
            "%s/build/gric-mktxtseq 1000 /tmp/ctest_spiral.txt 2Dspiral", root);
        int r = system(cmd);
        (void)r;
    }
} // ensure_test_dataset

static void test_job_cluster_lifecycle(
    char  *saved_outdir,
    size_t outdir_size)
{
    ensure_test_dataset();

    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "dataset", "/tmp/ctest_spiral.txt");
    cJSON_AddNumberToObject(args, "rlim", 0.10);
    cJSON_AddNumberToObject(args, "maxim", 1000);

    cJSON *res = cJSON_CreateObject();
    int ret = mcp_tool_cluster_start(args, res);

    CHECK(ret == 0);

    cJSON *j_item = cJSON_GetObjectItem(res, "job_id");
    CHECK(j_item != NULL && cJSON_IsString(j_item));
    char job_id[64];
    snprintf(job_id, sizeof(job_id), "%s", j_item->valuestring);

    cJSON *s_item = cJSON_GetObjectItem(res, "status");
    CHECK(s_item != NULL && strcmp(s_item->valuestring, "RUNNING") == 0);

    cJSON *o_item = cJSON_GetObjectItem(res, "outdir");
    CHECK(o_item != NULL && cJSON_IsString(o_item));
    if (saved_outdir != NULL && outdir_size > 0)
    {
        snprintf(saved_outdir, outdir_size, "%s", o_item->valuestring);
    }

    cJSON_Delete(args);
    cJSON_Delete(res);

    /* Poll status until completion */
    int completed = 0;
    for (int ii = 0; ii < 50; ii++)
    {
        usleep(100000); /* 100 ms */

        cJSON *st_args = cJSON_CreateObject();
        cJSON_AddStringToObject(st_args, "job_id", job_id);
        cJSON *st_res = cJSON_CreateObject();

        ret = mcp_tool_job_status(st_args, st_res);
        CHECK(ret == 0);

        cJSON *stat = cJSON_GetObjectItem(st_res, "status");
        CHECK(stat != NULL);

        if (strcmp(stat->valuestring, "COMPLETED") == 0)
        {
            completed = 1;
            cJSON_Delete(st_args);
            cJSON_Delete(st_res);
            break;
        }

        cJSON_Delete(st_args);
        cJSON_Delete(st_res);
    }

    CHECK_MSG(completed, "Async clustering job did not complete in time");

    /* Retrieve structured result */
    cJSON *res_args = cJSON_CreateObject();
    cJSON_AddStringToObject(res_args, "job_id", job_id);
    cJSON *res_out = cJSON_CreateObject();

    ret = mcp_tool_job_result(res_args, res_out);
    CHECK(ret == 0);

    cJSON *res_stat = cJSON_GetObjectItem(res_out, "status");
    CHECK(res_stat != NULL && strcmp(res_stat->valuestring, "COMPLETED") == 0);

    cJSON *clusters = cJSON_GetObjectItem(res_out, "num_clusters");
    CHECK(clusters != NULL && clusters->valueint == 62);

    cJSON *frames = cJSON_GetObjectItem(res_out, "num_frames");
    CHECK(frames != NULL && frames->valueint == 1000);

    cJSON *dists = cJSON_GetObjectItem(res_out, "num_distances_evaluated");
    CHECK(dists != NULL && dists->valuedouble >= 2828.0 && dists->valuedouble <= 2830.0);

    cJSON_Delete(res_args);
    cJSON_Delete(res_out);
} // test_job_cluster_lifecycle

static void test_job_cancel(void)
{
    ensure_test_dataset();

    char root[1024];
    mcp_get_project_root(root, sizeof(root));

    /* Create large 50000-frame dataset to ensure job runs long enough to cancel */
    const char *large_ds = "/tmp/ctest_large_spiral.txt";
    if (access(large_ds, R_OK) != 0)
    {
        char cmd[2048];
        snprintf(
            cmd, sizeof(cmd),
            "%s/build/gric-mktxtseq 50000 %s 2Dspiral", root, large_ds);
        int r = system(cmd);
        (void)r;
    }

    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "dataset", large_ds);
    cJSON_AddNumberToObject(args, "rlim", 0.05);
    cJSON_AddNumberToObject(args, "maxim", 50000);

    cJSON *res = cJSON_CreateObject();
    int ret = mcp_tool_cluster_start(args, res);
    CHECK(ret == 0);

    cJSON *j_item = cJSON_GetObjectItem(res, "job_id");
    CHECK(j_item != NULL && cJSON_IsString(j_item));
    char job_id[64];
    snprintf(job_id, sizeof(job_id), "%s", j_item->valuestring);

    cJSON_Delete(args);
    cJSON_Delete(res);

    /* Cancel job */
    cJSON *c_args = cJSON_CreateObject();
    cJSON_AddStringToObject(c_args, "job_id", job_id);
    cJSON_AddBoolToObject(c_args, "clean_output", 1);
    cJSON *c_res = cJSON_CreateObject();

    ret = mcp_tool_job_cancel(c_args, c_res);
    CHECK(ret == 0);

    cJSON *c_stat = cJSON_GetObjectItem(c_res, "status");
    CHECK(c_stat != NULL && strcmp(c_stat->valuestring, "CANCELLED") == 0);

    cJSON_Delete(c_args);
    cJSON_Delete(c_res);

    /* Verify status reports CANCELLED */
    cJSON *st_args = cJSON_CreateObject();
    cJSON_AddStringToObject(st_args, "job_id", job_id);
    cJSON *st_res = cJSON_CreateObject();

    ret = mcp_tool_job_status(st_args, st_res);
    CHECK(ret == 0);

    cJSON *stat = cJSON_GetObjectItem(st_res, "status");
    CHECK(stat != NULL && strcmp(stat->valuestring, "CANCELLED") == 0);

    cJSON_Delete(st_args);
    cJSON_Delete(st_res);
} // test_job_cancel

static void test_job_knn_lifecycle(
    const char *cluster_dir)
{
    ensure_test_dataset();

    const char *knn_out = "/tmp/ctest_async_knn.txt";
    unlink(knn_out);

    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "dataset", "/tmp/ctest_spiral.txt");
    cJSON_AddStringToObject(args, "cluster_dir", cluster_dir);
    cJSON_AddStringToObject(args, "output", knn_out);
    cJSON_AddNumberToObject(args, "k", 10);
    cJSON_AddNumberToObject(args, "dtmin", 5);

    cJSON *res = cJSON_CreateObject();
    int ret = mcp_tool_knn_start(args, res);
    CHECK(ret == 0);

    cJSON *j_item = cJSON_GetObjectItem(res, "job_id");
    CHECK(j_item != NULL && cJSON_IsString(j_item));
    char job_id[64];
    snprintf(job_id, sizeof(job_id), "%s", j_item->valuestring);

    cJSON_Delete(args);
    cJSON_Delete(res);

    /* Poll status until completion */
    int completed = 0;
    for (int ii = 0; ii < 50; ii++)
    {
        usleep(100000); /* 100 ms */

        cJSON *st_args = cJSON_CreateObject();
        cJSON_AddStringToObject(st_args, "job_id", job_id);
        cJSON *st_res = cJSON_CreateObject();

        ret = mcp_tool_job_status(st_args, st_res);
        CHECK(ret == 0);

        cJSON *stat = cJSON_GetObjectItem(st_res, "status");
        CHECK(stat != NULL);

        if (strcmp(stat->valuestring, "COMPLETED") == 0)
        {
            completed = 1;
            cJSON_Delete(st_args);
            cJSON_Delete(st_res);
            break;
        }

        cJSON_Delete(st_args);
        cJSON_Delete(st_res);
    }

    CHECK_MSG(completed, "Async k-NN job did not complete in time");
    CHECK(access(knn_out, R_OK) == 0);
    unlink(knn_out);
} // test_job_knn_lifecycle

int main(void)
{
    printf("Running test_job_cluster_lifecycle...\n");
    char cluster_out[1024] = "/tmp/ctest_spiral_out";
    test_job_cluster_lifecycle(cluster_out, sizeof(cluster_out));

    printf("Running test_job_knn_lifecycle...\n");
    test_job_knn_lifecycle(cluster_out);

    printf("Running test_job_cancel...\n");
    test_job_cancel();

    printf("All test_mcp_jobs tests PASSED!\n");
    return 0;
}
