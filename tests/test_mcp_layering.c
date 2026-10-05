/**
 * @file test_mcp_layering.c
 * @brief Unit tests for gric_dev_check_layering architectural linter.
 */

#include "mcp_tools.h"
#include "mcp_registry.h"
#include "shared/cjson/cJSON.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/**
 * test_registry_lookup() - Verify gric_dev_check_layering registration in MCP registry.
 */
static void test_registry_lookup(void)
{
    printf("[test_mcp_layering] Running test_registry_lookup...\n");

    struct mcp_server_config cfg = {
        .toolsets        = MCP_TS_ALL,
        .read_only       = 0,
        .has_source_tree = 1,
    };
    mcp_registry_configure(&cfg);

    const struct mcp_tool_def *tool = mcp_registry_find("gric_dev_check_layering");
    assert(tool != NULL);
    assert(strcmp(tool->name, "gric_dev_check_layering") == 0);
    assert(tool->toolset == MCP_TS_DEV);
    assert(tool->side_effects == 0);
    assert(tool->fn != NULL);
    assert(tool->input_schema != NULL);
} // test_registry_lookup

/**
 * test_error_handling() - Verify handling of invalid directory paths.
 */
static void test_error_handling(void)
{
    printf("[test_mcp_layering] Running test_error_handling...\n");

    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "src_dir", "/path/that/does/not/exist/for/gric/test");

    cJSON *res = cJSON_CreateObject();
    int rc = mcp_tool_dev_check_layering(args, res);
    assert(rc != 0);

    cJSON *err = cJSON_GetObjectItemCaseSensitive(res, "error");
    assert(err != NULL && cJSON_IsString(err));

    cJSON_Delete(args);
    cJSON_Delete(res);
} // test_error_handling

/**
 * test_clean_source_scan() - Verify that src/ directory has zero architectural violations.
 */
static void test_clean_source_scan(void)
{
    printf("[test_mcp_layering] Running test_clean_source_scan...\n");

    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "src_dir", "src");

    cJSON *res = cJSON_CreateObject();
    int rc = mcp_tool_dev_check_layering(args, res);
    assert(rc == 0);

    cJSON *status = cJSON_GetObjectItemCaseSensitive(res, "status");
    assert(status != NULL && cJSON_IsString(status));
    assert(strcmp(status->valuestring, "PASS") == 0);

    cJSON *v_count = cJSON_GetObjectItemCaseSensitive(res, "violations_count");
    assert(v_count != NULL && cJSON_IsNumber(v_count));
    assert(v_count->valueint == 0);

    cJSON *f_scanned = cJSON_GetObjectItemCaseSensitive(res, "files_scanned");
    assert(f_scanned != NULL && cJSON_IsNumber(f_scanned));
    assert(f_scanned->valueint > 50);

    cJSON_Delete(args);
    cJSON_Delete(res);
} // test_clean_source_scan

/**
 * test_synthetic_violation() - Verify detection of illegal cross-layer includes.
 */
static void test_synthetic_violation(void)
{
    printf("[test_mcp_layering] Running test_synthetic_violation...\n");

    char temp_dir[] = "/tmp/gric_layering_test_XXXXXX";
    char *res_dir = mkdtemp(temp_dir);
    assert(res_dir != NULL);

    char shared_dir[512];
    snprintf(shared_dir, sizeof(shared_dir), "%s/shared", temp_dir);
    mkdir(shared_dir, 0700);

    char bad_file[512];
    snprintf(bad_file, sizeof(bad_file), "%s/shared/bad_layering.c", temp_dir);
    FILE *fp = fopen(bad_file, "w");
    assert(fp != NULL);
    fprintf(fp, "/* Test file violating layering */\n");
    fprintf(fp, "#include \"run_clustering.h\"\n");
    fprintf(fp, "int bad_fn(void) { return 0; }\n");
    fclose(fp);

    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "src_dir", temp_dir);

    cJSON *res = cJSON_CreateObject();
    int rc = mcp_tool_dev_check_layering(args, res);
    assert(rc == 0);

    cJSON *status = cJSON_GetObjectItemCaseSensitive(res, "status");
    assert(status != NULL && cJSON_IsString(status));
    assert(strcmp(status->valuestring, "VIOLATIONS_FOUND") == 0);

    cJSON *v_count = cJSON_GetObjectItemCaseSensitive(res, "violations_count");
    assert(v_count != NULL && cJSON_IsNumber(v_count));
    assert(v_count->valueint == 1);

    cJSON *viols = cJSON_GetObjectItemCaseSensitive(res, "violations");
    assert(viols != NULL && cJSON_IsArray(viols));
    assert(cJSON_GetArraySize(viols) == 1);

    cJSON *first_v = cJSON_GetArrayItem(viols, 0);
    cJSON *target_l = cJSON_GetObjectItemCaseSensitive(first_v, "target_layer");
    assert(target_l != NULL && target_l->valueint == 2);

    cJSON_Delete(args);
    cJSON_Delete(res);

    unlink(bad_file);
    rmdir(shared_dir);
    rmdir(temp_dir);
} // test_synthetic_violation

int main(void)
{
    test_registry_lookup();
    test_error_handling();
    test_clean_source_scan();
    test_synthetic_violation();

    printf("[test_mcp_layering] All layering linter tests PASSED.\n");
    return 0;
}
