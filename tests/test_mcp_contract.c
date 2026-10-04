/**
 * @file test_mcp_contract.c
 * @brief Contract verification tests for MCP analysis tools and output parsers.
 */

#include "mcp_tools.h"
#include "tests/mcp/mcp_test_util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void ensure_test_fixtures(void)
{
    char root[512];
    mcp_get_project_root(root, sizeof(root));

    if (access("/tmp/ctest_spiral.txt", F_OK) != 0)
    {
        char cmd[1024];
        snprintf(
            cmd, sizeof(cmd),
            "%s/build/gric-mktxtseq 1000 /tmp/ctest_spiral.txt 2Dspiral",
            root);
        int ret = system(cmd);
        (void)ret;
    }

    if (access("/tmp/ctest_spiral_out/frame_membership.txt", F_OK) != 0)
    {
        char cmd[1024];
        snprintf(
            cmd, sizeof(cmd),
            "%s/build/gric-cluster 0.1 /tmp/ctest_spiral.txt "
            "-maxim 1000 -outdir /tmp/ctest_spiral_out -txt",
            root);
        int ret = system(cmd);
        (void)ret;
    }
}

/**
 * test_contract_inspect_run() - Verify gric_inspect_run contract on valid clusterdir.
 */
static void test_contract_inspect_run(void)
{
    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "run_dir", "/tmp/ctest_spiral_out");

    cJSON *res = cJSON_CreateObject();
    int ret = mcp_tool_inspect_run(args, res);
    CHECK(ret == 0);

    cJSON *rundir = cJSON_GetObjectItemCaseSensitive(res, "run_directory");
    CHECK(rundir != NULL && cJSON_IsString(rundir));

    cJSON *clusters = cJSON_GetObjectItemCaseSensitive(res, "num_clusters");
    CHECK(clusters != NULL && cJSON_IsNumber(clusters));
    CHECK(clusters->valuedouble > 0.0);

    cJSON *frames = cJSON_GetObjectItemCaseSensitive(res, "num_frames");
    CHECK(frames != NULL && cJSON_IsNumber(frames));
    CHECK(frames->valuedouble > 0.0);

    cJSON *entropy = cJSON_GetObjectItemCaseSensitive(res, "shannon_entropy");
    CHECK(entropy != NULL && cJSON_IsNumber(entropy));

    cJSON_Delete(args);
    cJSON_Delete(res);
    printf("PASS: test_contract_inspect_run\n");
} // test_contract_inspect_run

/**
 * test_contract_verify_invariants() - Verify invariant checker contract on valid clusterdir.
 */
static void test_contract_verify_invariants(void)
{
    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "run_dir", "/tmp/ctest_spiral_out");
    cJSON_AddStringToObject(args, "dataset", "/tmp/ctest_spiral.txt");

    cJSON *res = cJSON_CreateObject();
    int ret = mcp_tool_verify_invariants(args, res);
    CHECK(ret == 0);

    cJSON *status = cJSON_GetObjectItemCaseSensitive(res, "status");
    CHECK(status != NULL && cJSON_IsString(status));
    CHECK(strcmp(status->valuestring, "PASS") == 0);

    cJSON *violations = cJSON_GetObjectItemCaseSensitive(res, "violations_count");
    CHECK(violations != NULL && cJSON_IsNumber(violations));
    CHECK(violations->valuedouble == 0.0);

    cJSON_Delete(args);
    cJSON_Delete(res);
    printf("PASS: test_contract_verify_invariants\n");
} // test_contract_verify_invariants

/**
 * test_contract_probe_dataset() - Verify dataset probe contract on coordinate data.
 */
static void test_contract_probe_dataset(void)
{
    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "dataset_path", "/tmp/ctest_spiral.txt");
    cJSON_AddNumberToObject(args, "sample_limit", 500);

    cJSON *res = cJSON_CreateObject();
    int ret = mcp_tool_probe_dataset(args, res);
    CHECK(ret == 0);

    cJSON *dim = cJSON_GetObjectItemCaseSensitive(res, "dimension");
    CHECK(dim != NULL && cJSON_IsNumber(dim));
    CHECK(dim->valuedouble > 0.0);

    cJSON *frames = cJSON_GetObjectItemCaseSensitive(res, "num_frames");
    CHECK(frames != NULL && cJSON_IsNumber(frames));
    CHECK(frames->valuedouble > 0.0);

    cJSON *presets = cJSON_GetObjectItemCaseSensitive(res, "radii_presets");
    CHECK(presets != NULL && cJSON_IsObject(presets));

    cJSON *rec_rlim = cJSON_GetObjectItemCaseSensitive(presets, "recommended_rlim");
    CHECK(rec_rlim != NULL && cJSON_IsNumber(rec_rlim));
    CHECK(rec_rlim->valuedouble > 0.0);

    cJSON_Delete(args);
    cJSON_Delete(res);
    printf("PASS: test_contract_probe_dataset\n");
} // test_contract_probe_dataset

int main(void)
{
    printf("=== Running MCP Analysis Tool Contract Tests ===\n");
    ensure_test_fixtures();
    test_contract_inspect_run();
    test_contract_verify_invariants();
    test_contract_probe_dataset();
    printf("=== All MCP Contract Tests Passed ===\n");
    return 0;
}
