/**
 * @file test_mcp_stream_window.c
 * @brief Unit tests for gric_stream_window tool and FPS parameter bounds validation.
 */

#include "mcp_tools.h"
#include "mcp_registry.h"
#include "shared/gric_stream_layout.h"
#include "shared/cjson/cJSON.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_IMAGESTREAMIO
#include <ImageStreamIO/ImageStreamIO.h>
#include <ImageStreamIO/ImageStruct.h>
#endif

#define TEST_STREAM_NAME "test_mcp_window_assign"

/**
 * test_registry_lookup() - Verify gric_stream_window registration in MCP registry.
 */
static void test_registry_lookup(void)
{
    printf("[test_mcp_stream_window] Running test_registry_lookup...\n");

    struct mcp_server_config cfg = {
        .toolsets        = MCP_TS_ALL,
        .read_only       = 0,
        .has_source_tree = 1,
    };
    mcp_registry_configure(&cfg);

    const struct mcp_tool_def *tool = mcp_registry_find("gric_stream_window");
    assert(tool != NULL);
    assert(strcmp(tool->name, "gric_stream_window") == 0);
    assert(tool->toolset == MCP_TS_OPS);
    assert(tool->side_effects == 0);
    assert(tool->fn != NULL);
    assert(tool->input_schema != NULL);
} // test_registry_lookup

/**
 * test_error_cases() - Verify error handling for invalid arguments and missing streams.
 */
static void test_error_cases(void)
{
    printf("[test_mcp_stream_window] Running test_error_cases...\n");

    /* 1. NULL arguments */
    {
        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_stream_window(NULL, res);
        assert(rc != 0);
        cJSON *err = cJSON_GetObjectItemCaseSensitive(res, "error");
        assert(err != NULL && cJSON_IsString(err));
        cJSON_Delete(res);
    }

    /* 2. Missing stream_name */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddNumberToObject(args, "num_frames", 50);

        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_stream_window(args, res);
        assert(rc != 0);
        cJSON *err = cJSON_GetObjectItemCaseSensitive(res, "error");
        assert(err != NULL && cJSON_IsString(err));

        cJSON_Delete(args);
        cJSON_Delete(res);
    }

    /* 3. Invalid identifier with illegal characters */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddStringToObject(args, "stream_name", "bad stream name!@#");

        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_stream_window(args, res);
        assert(rc != 0);
        cJSON *err = cJSON_GetObjectItemCaseSensitive(res, "error");
        assert(err != NULL && cJSON_IsString(err));

        cJSON_Delete(args);
        cJSON_Delete(res);
    }

    /* 4. Nonexistent stream */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddStringToObject(args, "stream_name", "nonexistent_stream_xyz_123");

        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_stream_window(args, res);
        assert(rc != 0);
        cJSON *err = cJSON_GetObjectItemCaseSensitive(res, "error");
        assert(err != NULL && cJSON_IsString(err));

        cJSON_Delete(args);
        cJSON_Delete(res);
    }
} // test_error_cases

/**
 * test_fps_bounds_validation() - Verify numerical parameter range enforcement in gric_fps_set.
 */
static void test_fps_bounds_validation(void)
{
    printf("[test_mcp_stream_window] Running test_fps_bounds_validation...\n");

    /* 1. rlim <= 0 must fail */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddStringToObject(args, "module", "cluster");
        cJSON_AddStringToObject(args, "fps_name", "test_fps");
        cJSON_AddStringToObject(args, "param", "rlim");
        cJSON_AddNumberToObject(args, "value", -0.5);

        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_fps_set(args, res);
        assert(rc != 0);
        cJSON *err = cJSON_GetObjectItemCaseSensitive(res, "error");
        assert(err != NULL && strstr(err->valuestring, "rlim") != NULL);

        cJSON_Delete(args);
        cJSON_Delete(res);
    }

    /* 2. dprob out of [0, 1] must fail */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddStringToObject(args, "module", "cluster");
        cJSON_AddStringToObject(args, "fps_name", "test_fps");
        cJSON_AddStringToObject(args, "param", "dprob");
        cJSON_AddNumberToObject(args, "value", 1.5);

        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_fps_set(args, res);
        assert(rc != 0);
        cJSON *err = cJSON_GetObjectItemCaseSensitive(res, "error");
        assert(err != NULL &&
               (strstr(err->valuestring, "deltaprob") != NULL ||
                strstr(err->valuestring, "dprob") != NULL));

        cJSON_Delete(args);
        cJSON_Delete(res);
    }

    /* 3. maxcl < 1 must fail */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddStringToObject(args, "module", "cluster");
        cJSON_AddStringToObject(args, "fps_name", "test_fps");
        cJSON_AddStringToObject(args, "param", "maxcl");
        cJSON_AddNumberToObject(args, "value", 0);

        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_fps_set(args, res);
        assert(rc != 0);
        cJSON *err = cJSON_GetObjectItemCaseSensitive(res, "error");
        assert(err != NULL &&
               (strstr(err->valuestring, "maxnbclust") != NULL ||
                strstr(err->valuestring, "maxcl") != NULL));

        cJSON_Delete(args);
        cJSON_Delete(res);
    }

    /* 4. maxim < 0 must fail */
    {
        cJSON *args = cJSON_CreateObject();
        cJSON_AddStringToObject(args, "module", "cluster");
        cJSON_AddStringToObject(args, "fps_name", "test_fps");
        cJSON_AddStringToObject(args, "param", "maxim");
        cJSON_AddNumberToObject(args, "value", -10);

        cJSON *res = cJSON_CreateObject();
        int rc = mcp_tool_fps_set(args, res);
        assert(rc != 0);
        cJSON *err = cJSON_GetObjectItemCaseSensitive(res, "error");
        assert(err != NULL &&
               (strstr(err->valuestring, "max_frames") != NULL ||
                strstr(err->valuestring, "maxim") != NULL));

        cJSON_Delete(args);
        cJSON_Delete(res);
    }
} // test_fps_bounds_validation

#ifdef USE_IMAGESTREAMIO
/**
 * test_streaming_window_live() - Verify window calculation on live test shared memory stream.
 */
static void test_streaming_window_live(void)
{
    printf("[test_mcp_stream_window] Running test_streaming_window_live...\n");

    /* Create test telemetry ImageStreamIO stream */
    IMAGE img;
    memset(&img, 0, sizeof(IMAGE));
    uint32_t dims[2] = {GRIC_ASSIGN_NFIELDS, 1};

    int rc = ImageStreamIO_createIm_gpu(
        &img, TEST_STREAM_NAME, 2, dims, _DATATYPE_FLOAT, -1, 1, 10, 0, 0, 0);
    assert(rc == 0);

    /* Populate initial frame */
    float *ptr = (float *)img.array.raw;
    ptr[GRIC_ASSIGN_FRAME_INDEX] = 100.0f;
    ptr[GRIC_ASSIGN_CLUSTER_ID] = 5.0f;
    ptr[GRIC_ASSIGN_DIST] = 0.025f;
    ptr[GRIC_ASSIGN_IS_NEW] = 1.0f;
    ptr[GRIC_ASSIGN_TOTAL_CLUSTERS] = 6.0f;
    ptr[GRIC_ASSIGN_LATENCY_US] = 12.5f;
    ptr[GRIC_ASSIGN_WITHIN_RLIM] = 1.0f;
    ptr[GRIC_ASSIGN_QUERY_MODE] = 0.0f;
    img.md[0].cnt0 = 1;

    /* Sample stream with timeout (idle fallback path) */
    cJSON *args = cJSON_CreateObject();
    cJSON_AddStringToObject(args, "stream_name", TEST_STREAM_NAME);
    cJSON_AddNumberToObject(args, "num_frames", 10);
    cJSON_AddNumberToObject(args, "timeout_ms", 50);

    cJSON *res = cJSON_CreateObject();
    int mcp_rc = mcp_tool_stream_window(args, res);
    assert(mcp_rc == 0);

    cJSON *status = cJSON_GetObjectItemCaseSensitive(res, "status");
    assert(status != NULL && strcmp(status->valuestring, "STREAM_IDLE") == 0);

    cJSON *lat = cJSON_GetObjectItemCaseSensitive(res, "latency_us");
    assert(lat != NULL);
    assert(cJSON_GetObjectItemCaseSensitive(lat, "p50_us")->valuedouble == 12.5);

    cJSON *dist = cJSON_GetObjectItemCaseSensitive(res, "distances");
    assert(dist != NULL);
    assert(cJSON_GetObjectItemCaseSensitive(dist, "mean")->valuedouble > 0.024 &&
           cJSON_GetObjectItemCaseSensitive(dist, "mean")->valuedouble < 0.026);

    cJSON *cl = cJSON_GetObjectItemCaseSensitive(res, "clustering_dynamics");
    assert(cl != NULL);
    assert(cJSON_GetObjectItemCaseSensitive(cl, "new_anchors_in_window")->valuedouble == 1.0);

    cJSON_Delete(args);
    cJSON_Delete(res);

    /* Clean up ImageStreamIO stream */
    ImageStreamIO_destroyIm(&img);
} // test_streaming_window_live
#endif

int main(void)
{
    test_registry_lookup();
    test_error_cases();
    test_fps_bounds_validation();

#ifdef USE_IMAGESTREAMIO
    test_streaming_window_live();
#endif

    printf("[test_mcp_stream_window] All tests passed!\n");
    return 0;
} // main
