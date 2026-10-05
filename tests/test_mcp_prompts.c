/**
 * @file test_mcp_prompts.c
 * @brief Unit tests for MCP Prompts and Resource Templates protocol handlers.
 */

#include "mcp_dispatch.h"
#include "shared/cjson/cJSON.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * test_initialize_capabilities() - Verify prompts capability advertised in initialize handshake.
 */
static void test_initialize_capabilities(void)
{
    printf("[test_mcp_prompts] Running test_initialize_capabilities...\n");

    const char *req = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}";
    char *resp_str = mcp_dispatch_message(req);
    assert(resp_str != NULL);

    cJSON *resp = cJSON_Parse(resp_str);
    free(resp_str);
    assert(resp != NULL);

    cJSON *res = cJSON_GetObjectItemCaseSensitive(resp, "result");
    assert(res != NULL);

    cJSON *caps = cJSON_GetObjectItemCaseSensitive(res, "capabilities");
    assert(caps != NULL);

    cJSON *prompts = cJSON_GetObjectItemCaseSensitive(caps, "prompts");
    assert(prompts != NULL && cJSON_IsObject(prompts));

    cJSON *resources = cJSON_GetObjectItemCaseSensitive(caps, "resources");
    assert(resources != NULL && cJSON_IsObject(resources));

    cJSON *tools = cJSON_GetObjectItemCaseSensitive(caps, "tools");
    assert(tools != NULL && cJSON_IsObject(tools));

    cJSON_Delete(resp);
} // test_initialize_capabilities

/**
 * test_prompts_list() - Verify listing and argument schemas of available prompts.
 */
static void test_prompts_list(void)
{
    printf("[test_mcp_prompts] Running test_prompts_list...\n");

    const char *req = "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"prompts/list\"}";
    char *resp_str = mcp_dispatch_message(req);
    assert(resp_str != NULL);

    cJSON *resp = cJSON_Parse(resp_str);
    free(resp_str);
    assert(resp != NULL);

    cJSON *res = cJSON_GetObjectItemCaseSensitive(resp, "result");
    assert(res != NULL);

    cJSON *prompts = cJSON_GetObjectItemCaseSensitive(res, "prompts");
    assert(prompts != NULL && cJSON_IsArray(prompts));
    assert(cJSON_GetArraySize(prompts) == 3);

    int found_calib = 0;
    int found_milk = 0;
    int found_quant = 0;

    for (int ii = 0; ii < cJSON_GetArraySize(prompts); ii++)
    {
        cJSON *p = cJSON_GetArrayItem(prompts, ii);
        cJSON *name = cJSON_GetObjectItemCaseSensitive(p, "name");
        assert(name != NULL && cJSON_IsString(name));

        cJSON *args = cJSON_GetObjectItemCaseSensitive(p, "arguments");
        assert(args != NULL && cJSON_IsArray(args));

        if (strcmp(name->valuestring, "calibrate-and-cluster") == 0)
        {
            found_calib = 1;
        }
        else if (strcmp(name->valuestring, "setup-milk-stream") == 0)
        {
            found_milk = 1;
        }
        else if (strcmp(name->valuestring, "compare-quantization") == 0)
        {
            found_quant = 1;
        }
    }

    assert(found_calib && found_milk && found_quant);
    cJSON_Delete(resp);
} // test_prompts_list

/**
 * test_prompts_get() - Verify prompt rendering with default and custom arguments.
 */
static void test_prompts_get(void)
{
    printf("[test_mcp_prompts] Running test_prompts_get...\n");

    /* 1. calibrate-and-cluster with custom args */
    {
        const char *req =
            "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"prompts/get\","
            "\"params\":{\"name\":\"calibrate-and-cluster\","
            "\"arguments\":{\"input_path\":\"spiral.fits\",\"target_clusters\":\"50\"}}}";
        char *resp_str = mcp_dispatch_message(req);
        assert(resp_str != NULL);

        cJSON *resp = cJSON_Parse(resp_str);
        free(resp_str);
        assert(resp != NULL);

        cJSON *res = cJSON_GetObjectItemCaseSensitive(resp, "result");
        assert(res != NULL);

        cJSON *msgs = cJSON_GetObjectItemCaseSensitive(res, "messages");
        assert(msgs != NULL && cJSON_IsArray(msgs));
        assert(cJSON_GetArraySize(msgs) == 1);

        cJSON *msg = cJSON_GetArrayItem(msgs, 0);
        cJSON *content = cJSON_GetObjectItemCaseSensitive(msg, "content");
        assert(content != NULL);

        cJSON *text = cJSON_GetObjectItemCaseSensitive(content, "text");
        assert(text != NULL && cJSON_IsString(text));
        assert(strstr(text->valuestring, "spiral.fits") != NULL);
        assert(strstr(text->valuestring, "50") != NULL);
        assert(strstr(text->valuestring, "gric_calibrate_radius") != NULL);

        cJSON_Delete(resp);
    }

    /* 2. setup-milk-stream */
    {
        const char *req =
            "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"prompts/get\","
            "\"params\":{\"name\":\"setup-milk-stream\","
            "\"arguments\":{\"in_stream\":\"cam_stream\",\"fps_name\":\"fps_cam\"}}}";
        char *resp_str = mcp_dispatch_message(req);
        assert(resp_str != NULL);

        cJSON *resp = cJSON_Parse(resp_str);
        free(resp_str);
        assert(resp != NULL);

        cJSON *res = cJSON_GetObjectItemCaseSensitive(resp, "result");
        assert(res != NULL);

        cJSON *msgs = cJSON_GetObjectItemCaseSensitive(res, "messages");
        assert(msgs != NULL && cJSON_IsArray(msgs));

        cJSON *msg = cJSON_GetArrayItem(msgs, 0);
        cJSON *content = cJSON_GetObjectItemCaseSensitive(msg, "content");
        cJSON *text = cJSON_GetObjectItemCaseSensitive(content, "text");
        assert(strstr(text->valuestring, "cam_stream") != NULL);
        assert(strstr(text->valuestring, "fps_cam") != NULL);
        assert(strstr(text->valuestring, "gric_stream_window") != NULL);

        cJSON_Delete(resp);
    }

    /* 3. compare-quantization */
    {
        const char *req =
            "{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"prompts/get\","
            "\"params\":{\"name\":\"compare-quantization\","
            "\"arguments\":{\"input_path\":\"data.txt\",\"quant_modes\":\"fp32,eq16\"}}}";
        char *resp_str = mcp_dispatch_message(req);
        assert(resp_str != NULL);

        cJSON *resp = cJSON_Parse(resp_str);
        free(resp_str);
        assert(resp != NULL);

        cJSON *res = cJSON_GetObjectItemCaseSensitive(resp, "result");
        assert(res != NULL);

        cJSON *msgs = cJSON_GetObjectItemCaseSensitive(res, "messages");
        assert(msgs != NULL && cJSON_IsArray(msgs));

        cJSON *msg = cJSON_GetArrayItem(msgs, 0);
        cJSON *content = cJSON_GetObjectItemCaseSensitive(msg, "content");
        cJSON *text = cJSON_GetObjectItemCaseSensitive(content, "text");
        assert(strstr(text->valuestring, "data.txt") != NULL);
        assert(strstr(text->valuestring, "fp32,eq16") != NULL);
        assert(strstr(text->valuestring, "gric_knn_quality") != NULL);

        cJSON_Delete(resp);
    }

    /* 4. Error case: unknown prompt */
    {
        const char *req =
            "{\"jsonrpc\":\"2.0\",\"id\":6,\"method\":\"prompts/get\","
            "\"params\":{\"name\":\"nonexistent-prompt\"}}";
        char *resp_str = mcp_dispatch_message(req);
        assert(resp_str != NULL);

        cJSON *resp = cJSON_Parse(resp_str);
        free(resp_str);
        assert(resp != NULL);

        cJSON *err = cJSON_GetObjectItemCaseSensitive(resp, "error");
        assert(err != NULL && cJSON_IsObject(err));
        cJSON *code = cJSON_GetObjectItemCaseSensitive(err, "code");
        assert(code != NULL && code->valueint == -32602);

        cJSON_Delete(resp);
    }
} // test_prompts_get

/**
 * test_resource_templates_list() - Verify listing of dynamic resource URI templates.
 */
static void test_resource_templates_list(void)
{
    printf("[test_mcp_prompts] Running test_resource_templates_list...\n");

    const char *req = "{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"resources/templates/list\"}";
    char *resp_str = mcp_dispatch_message(req);
    assert(resp_str != NULL);

    cJSON *resp = cJSON_Parse(resp_str);
    free(resp_str);
    assert(resp != NULL);

    cJSON *res = cJSON_GetObjectItemCaseSensitive(resp, "result");
    assert(res != NULL);

    cJSON *templates = cJSON_GetObjectItemCaseSensitive(res, "resourceTemplates");
    assert(templates != NULL && cJSON_IsArray(templates));
    assert(cJSON_GetArraySize(templates) == 3);

    int found_run = 0;
    int found_stream = 0;
    int found_recipe = 0;

    for (int ii = 0; ii < cJSON_GetArraySize(templates); ii++)
    {
        cJSON *t = cJSON_GetArrayItem(templates, ii);
        cJSON *uri_t = cJSON_GetObjectItemCaseSensitive(t, "uriTemplate");
        assert(uri_t != NULL && cJSON_IsString(uri_t));

        if (strcmp(uri_t->valuestring, "gric://run/{path}/summary") == 0)
        {
            found_run = 1;
        }
        else if (strcmp(uri_t->valuestring, "gric://stream/{name}/stats") == 0)
        {
            found_stream = 1;
        }
        else if (strcmp(uri_t->valuestring, "gric://recipe/{name}") == 0)
        {
            found_recipe = 1;
        }
    }

    assert(found_run && found_stream && found_recipe);
    cJSON_Delete(resp);
} // test_resource_templates_list

/**
 * test_dynamic_resource_read() - Verify resolution and reading of dynamic resource URIs.
 */
static void test_dynamic_resource_read(void)
{
    printf("[test_mcp_prompts] Running test_dynamic_resource_read...\n");

    /* 1. Recipe URI template resolution */
    {
        const char *req =
            "{\"jsonrpc\":\"2.0\",\"id\":8,\"method\":\"resources/read\","
            "\"params\":{\"uri\":\"gric://recipe/image_cube_clustering\"}}";
        char *resp_str = mcp_dispatch_message(req);
        assert(resp_str != NULL);

        cJSON *resp = cJSON_Parse(resp_str);
        free(resp_str);
        assert(resp != NULL);

        cJSON *res = cJSON_GetObjectItemCaseSensitive(resp, "result");
        assert(res != NULL);

        cJSON *contents = cJSON_GetObjectItemCaseSensitive(res, "contents");
        assert(contents != NULL && cJSON_IsArray(contents));
        assert(cJSON_GetArraySize(contents) == 1);

        cJSON *item = cJSON_GetArrayItem(contents, 0);
        cJSON *text = cJSON_GetObjectItemCaseSensitive(item, "text");
        assert(text != NULL && cJSON_IsString(text));
        assert(strstr(text->valuestring, "Image Cube") != NULL ||
               strstr(text->valuestring, "image cube") != NULL ||
               strstr(text->valuestring, "clustering") != NULL);

        cJSON_Delete(resp);
    }

    /* 2. Stream stats template resolution (nonexistent stream handles gracefully) */
    {
        const char *req =
            "{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"resources/read\","
            "\"params\":{\"uri\":\"gric://stream/missing_shm_test/stats\"}}";
        char *resp_str = mcp_dispatch_message(req);
        assert(resp_str != NULL);

        cJSON *resp = cJSON_Parse(resp_str);
        free(resp_str);
        assert(resp != NULL);

        cJSON *res = cJSON_GetObjectItemCaseSensitive(resp, "result");
        assert(res != NULL);

        cJSON *contents = cJSON_GetObjectItemCaseSensitive(res, "contents");
        assert(contents != NULL && cJSON_IsArray(contents));
        cJSON *item = cJSON_GetArrayItem(contents, 0);
        cJSON *text = cJSON_GetObjectItemCaseSensitive(item, "text");
        assert(text != NULL && cJSON_IsString(text));

        cJSON_Delete(resp);
    }
} // test_dynamic_resource_read

int main(void)
{
    test_initialize_capabilities();
    test_prompts_list();
    test_prompts_get();
    test_resource_templates_list();
    test_dynamic_resource_read();

    printf("[test_mcp_prompts] All prompts and resource templates tests PASSED.\n");
    return 0;
}
