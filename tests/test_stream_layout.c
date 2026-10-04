/**
 * @file test_stream_layout.c
 * @brief Unit tests for canonical telemetry packet encoding, decoding, and documentation coverage.
 */

#include "mcp_tools.h"
#include "shared/gric_stream_layout.h"
#include "tests/mcp/mcp_test_util.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * test_pack_decode_roundtrip() - Verify packing and JSON decoding preserves all fields.
 */
static void test_pack_decode_roundtrip(void)
{
    struct gric_assign_sample sample = {
        .frame_index    = 1001.0f,
        .cluster_id     = 42.0f,
        .distance       = 0.285f,
        .is_new_anchor  = 1.0f,
        .total_clusters = 64.0f,
        .latency_us     = 87.5f,
        .within_rlim    = 1.0f,
        .query_mode     = 0.0f,
    };

    float buffer[GRIC_ASSIGN_NFIELDS];
    memset(buffer, 0, sizeof(buffer));

    gric_assign_pack(buffer, &sample);

    cJSON *out = cJSON_CreateObject();
    CHECK(out != NULL);

    int ret = mcp_decode_assign(buffer, GRIC_ASSIGN_NFIELDS, out);
    CHECK(ret == 0);

    /* Verify frame index */
    cJSON *item = cJSON_GetObjectItemCaseSensitive(out, "frame_index");
    CHECK(item != NULL && cJSON_IsNumber(item));
    CHECK(item->valuedouble == 1001.0);

    /* Verify cluster id */
    item = cJSON_GetObjectItemCaseSensitive(out, "cluster_id");
    CHECK(item != NULL && cJSON_IsNumber(item));
    CHECK(item->valuedouble == 42.0);

    /* Verify distance */
    item = cJSON_GetObjectItemCaseSensitive(out, "distance");
    CHECK(item != NULL && cJSON_IsNumber(item));
    CHECK(fabs(item->valuedouble - 0.285) < 1e-4);

    /* Verify is_new_anchor */
    item = cJSON_GetObjectItemCaseSensitive(out, "is_new_anchor");
    CHECK(item != NULL && cJSON_IsBool(item));
    CHECK(cJSON_IsTrue(item));

    /* Verify total_clusters */
    item = cJSON_GetObjectItemCaseSensitive(out, "total_clusters");
    CHECK(item != NULL && cJSON_IsNumber(item));
    CHECK(item->valuedouble == 64.0);

    /* Verify latency_us */
    item = cJSON_GetObjectItemCaseSensitive(out, "latency_us");
    CHECK(item != NULL && cJSON_IsNumber(item));
    CHECK(fabs(item->valuedouble - 87.5) < 1e-4);

    /* Verify within_rlim */
    item = cJSON_GetObjectItemCaseSensitive(out, "within_rlim");
    CHECK(item != NULL && cJSON_IsNumber(item));
    CHECK(item->valuedouble == 1.0);

    /* Verify query_mode */
    item = cJSON_GetObjectItemCaseSensitive(out, "query_mode");
    CHECK(item != NULL && cJSON_IsNumber(item));
    CHECK(item->valuedouble == 0.0);

    cJSON_Delete(out);
    printf("PASS: test_pack_decode_roundtrip\n");
} // test_pack_decode_roundtrip

/**
 * test_docs_field_coverage() - Verify all canonical telemetry keys appear in milk_streams.md.
 */
static void test_docs_field_coverage(void)
{
    char root[512];
    mcp_get_project_root(root, sizeof(root));

    char doc_path[1024];
    snprintf(doc_path, sizeof(doc_path), "%s/docs/help/milk/milk_streams.md", root);

    FILE *fp = fopen(doc_path, "r");
    CHECK_MSG(fp != NULL, "Failed to open docs/help/milk/milk_streams.md");

    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    CHECK(sz > 0);

    char *doc = malloc((size_t)sz + 1);
    CHECK(doc != NULL);
    size_t read_bytes = fread(doc, 1, (size_t)sz, fp);
    fclose(fp);
    doc[read_bytes] = '\0';

#define CHECK_FIELD_IN_DOC(NAME, IDX, KEY, DESCR) \
    CHECK_MSG(strstr(doc, KEY) != NULL, "Missing field key in milk_streams.md: " KEY);

    GRIC_ASSIGN_FIELDS(CHECK_FIELD_IN_DOC)
#undef CHECK_FIELD_IN_DOC

    free(doc);
    printf("PASS: test_docs_field_coverage\n");
} // test_docs_field_coverage

int main(void)
{
    printf("=== Running Stream Layout & Telemetry Drift Tests ===\n");
    test_pack_decode_roundtrip();
    test_docs_field_coverage();
    printf("=== All Stream Layout Tests Passed ===\n");
    return 0;
}
