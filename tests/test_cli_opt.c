/**
 * @file test_cli_opt.c
 * @brief Unit tests for the shared declarative CLI option parser engine.
 */

#include "cli_opt.h"
#include "cjson/cJSON.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct TestConfig
{
    double      rlim;
    int         k;
    int         use_entropy;
    int         use_sq8;
    int         use_double;
    uint32_t    max_clusters;
    char        outdir[128];
    char        custom_msg[128];
};

static int custom_handler(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)key;
    struct TestConfig *cfg = (struct TestConfig *)ctx;
    if (val != NULL)
    {
        strncpy(cfg->custom_msg, val, sizeof(cfg->custom_msg) - 1);
        cfg->custom_msg[sizeof(cfg->custom_msg) - 1] = '\0';
    }
    return 1;
}

static void test_flag_and_inline_equals(void)
{
    struct TestConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.rlim = 0.5;
    cfg.k = 10;
    cfg.use_entropy = 0;
    cfg.use_sq8 = 1;

    const struct gric_opt opts[] = {
        {"rlim", 'r', GRIC_OPT_DOUBLE, &cfg.rlim, 0, "0.5", "<val>", "Radius", "Core", NULL, 0},
        {"knn-k", 'k', GRIC_OPT_INT, &cfg.k, 0, "10", "<N>", "Neighbors", "Core", "k", 0},
        {"entropy", '\0', GRIC_OPT_FLAG, &cfg.use_entropy, 0, "OFF", NULL, "Entropy", "Algo",
         NULL, 1},
        {"sq8", '\0', GRIC_OPT_FLAG, &cfg.use_sq8, 0, "ON", NULL, "SQ8", "Quant", "sq_8", 1},
        {"out", 'o', GRIC_OPT_STRING, cfg.outdir, sizeof(cfg.outdir), "", "<dir>", "Out", "I/O",
         "outdir", 0}
    };
    size_t nopts = sizeof(opts) / sizeof(opts[0]);

    /* Test 1: Space-separated --rlim 0.25 */
    {
        char *argv[] = {"prog", "--rlim", "0.25"};
        int argc = 3;
        int idx = 1;
        int res = gric_opt_parse_arg(argc, argv, &idx, opts, nopts, NULL);
        assert(res == 1);
        assert(idx == 3);
        assert(cfg.rlim > 0.249 && cfg.rlim < 0.251);
    }

    /* Test 2: Inline equals --rlim=0.75 */
    {
        char *argv[] = {"prog", "--rlim=0.75"};
        int argc = 2;
        int idx = 1;
        int res = gric_opt_parse_arg(argc, argv, &idx, opts, nopts, NULL);
        assert(res == 1);
        assert(idx == 2);
        assert(cfg.rlim > 0.749 && cfg.rlim < 0.751);
    }

    /* Test 3: Short flag with inline equals -k=25 */
    {
        char *argv[] = {"prog", "-k=25"};
        int argc = 2;
        int idx = 1;
        int res = gric_opt_parse_arg(argc, argv, &idx, opts, nopts, NULL);
        assert(res == 1);
        assert(idx == 2);
        assert(cfg.k == 25);
    }

    /* Test 4: Negation flag --no-entropy and --no-sq8 */
    {
        cfg.use_entropy = 1;
        cfg.use_sq8 = 1;
        char *argv[] = {"prog", "--no-entropy", "--no-sq8"};
        int argc = 3;
        int idx = 1;
        int res = gric_opt_parse_arg(argc, argv, &idx, opts, nopts, NULL);
        assert(res == 1);
        assert(cfg.use_entropy == 0);
        assert(idx == 2);

        res = gric_opt_parse_arg(argc, argv, &idx, opts, nopts, NULL);
        assert(res == 1);
        assert(cfg.use_sq8 == 0);
        assert(idx == 3);
    }

    /* Test 5: Legacy alias -outdir and dash/underscore equivalence */
    {
        char *argv[] = {"prog", "-outdir", "/tmp/clusters_out"};
        int argc = 3;
        int idx = 1;
        int res = gric_opt_parse_arg(argc, argv, &idx, opts, nopts, NULL);
        assert(res == 1);
        assert(idx == 3);
        assert(strcmp(cfg.outdir, "/tmp/clusters_out") == 0);
    }

    printf("PASS: test_flag_and_inline_equals\n");
}

static void test_config_file_parser(void)
{
    struct TestConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.rlim = 0.1;
    cfg.k = 5;
    cfg.max_clusters = 100;

    const struct gric_opt opts[] = {
        {"rlim", 'r', GRIC_OPT_DOUBLE, &cfg.rlim, 0, "0.1", "<val>", "Radius", "Core", NULL, 0},
        {"k", 'k', GRIC_OPT_INT, &cfg.k, 0, "5", "<N>", "Neighbors", "Core", NULL, 0},
        {"max-clusters", 'm', GRIC_OPT_UINT32, &cfg.max_clusters, 0, "100", "<N>", "Limit", "Core",
         "maxcl,maxnbclust", 0},
        {"custom", 'c', GRIC_OPT_CUSTOM, (void *)custom_handler, 0, "", "<msg>", "Custom", "Misc",
         NULL, 0}
    };
    size_t nopts = sizeof(opts) / sizeof(opts[0]);

    const char *tmp_conf = "/tmp/test_cli_opt.conf";
    FILE *fp = fopen(tmp_conf, "w");
    assert(fp != NULL);
    fprintf(fp, "# Test configuration file\n");
    fprintf(fp, "rlim = 0.42\n");
    fprintf(fp, "// Comment line\n");
    fprintf(fp, "k 40\n");
    fprintf(fp, "maxcl = 512\n");
    fprintf(fp, "custom = hello_gric\n");
    fclose(fp);

    int res = gric_opt_parse_config_file(tmp_conf, opts, nopts, &cfg);
    assert(res == 0);
    assert(cfg.rlim > 0.419 && cfg.rlim < 0.421);
    assert(cfg.k == 40);
    assert(cfg.max_clusters == 512);
    assert(cfg.custom_msg[0] != '\0' && strcmp(cfg.custom_msg, "hello_gric") == 0);

    remove(tmp_conf);
    printf("PASS: test_config_file_parser\n");
}

static void test_json_introspection(void)
{
    double rlim = 0.05;
    int k = 10;
    int use_sq8 = 1;

    const struct gric_opt opts[] = {
        {"rlim", 'r', GRIC_OPT_DOUBLE, &rlim, 0, "0.05", "<val>", "Cluster radius cutoff limit",
         "Algorithm", "radius", 0},
        {"knn-k", 'k', GRIC_OPT_INT, &k, 0, "10", "<N>", "Number of nearest neighbors",
         "Core", "k", 0},
        {"sq8", '\0', GRIC_OPT_FLAG, &use_sq8, 0, "ON", NULL, "Enable SQ8 quantization",
         "Quantization", "sq_8", 1}
    };
    size_t nopts = sizeof(opts) / sizeof(opts[0]);

    cJSON *arr = gric_opt_to_json(opts, nopts);
    assert(arr != NULL);
    assert(cJSON_GetArraySize(arr) == 3);

    cJSON *first = cJSON_GetArrayItem(arr, 0);
    cJSON *name = cJSON_GetObjectItem(first, "name");
    assert(name != NULL && strcmp(name->valuestring, "rlim") == 0);
    cJSON *type = cJSON_GetObjectItem(first, "type");
    assert(type != NULL && strcmp(type->valuestring, "number") == 0);

    cJSON *third = cJSON_GetArrayItem(arr, 2);
    cJSON *neg = cJSON_GetObjectItem(third, "negatable");
    assert(neg != NULL && cJSON_IsTrue(neg));

    cJSON_Delete(arr);

    cJSON *schema = gric_opt_to_json_schema(opts, nopts);
    assert(schema != NULL);
    cJSON *props = cJSON_GetObjectItem(schema, "properties");
    assert(props != NULL);
    assert(cJSON_GetObjectItem(props, "rlim") != NULL);
    assert(cJSON_GetObjectItem(props, "knn-k") != NULL);
    assert(cJSON_GetObjectItem(props, "sq8") != NULL);

    cJSON_Delete(schema);
    printf("PASS: test_json_introspection\n");
}

int main(void)
{
    printf("=== Running CLI Option Parser Unit Tests ===\n");
    test_flag_and_inline_equals();
    test_config_file_parser();
    test_json_introspection();
    printf("=== All CLI Option Parser Unit Tests PASSED ===\n");
    return 0;
}
