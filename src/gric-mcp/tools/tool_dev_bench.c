/**
 * @file tool_dev_bench.c
 * @brief Deterministic micro-benchmark runner for SIMD kernels, clustering loops, and k-NN.
 */

#include "mcp_tools.h"
#include "mcp_registry.h"
#include "mcp_exec.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define MAX_BENCH_TRIALS 50
#define MAX_BENCH_OUTPUT_BYTES 131072 /* 128 KB */

/**
 * get_time_ms() - High-resolution monotonic clock in milliseconds.
 *
 * Return: Current monotonic time in milliseconds.
 */
static double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
} // get_time_ms

/**
 * detect_cpu_environment() - Query CPU frequency governor and environment jitter flags.
 * @gov_buf:  Target buffer for governor name.
 * @gov_size: Capacity of governor buffer.
 * @warn_buf: Target buffer for jitter warning message.
 * @warn_size: Capacity of warning buffer.
 */
static void detect_cpu_environment(
    char   *gov_buf,
    size_t  gov_size,
    char   *warn_buf,
    size_t  warn_size)
{
    gov_buf[0] = '\0';
    warn_buf[0] = '\0';

    FILE *fp = fopen("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "r");
    if (fp != NULL)
    {
        if (fgets(gov_buf, (int)gov_size, fp) != NULL)
        {
            size_t len = strlen(gov_buf);
            while (len > 0 && isspace((unsigned char)gov_buf[len - 1]))
            {
                gov_buf[--len] = '\0';
            }
        }
        fclose(fp);
    }

    if (gov_buf[0] == '\0')
    {
        snprintf(gov_buf, gov_size, "unknown");
    }
    else if (strcmp(gov_buf, "performance") != 0)
    {
        snprintf(
            warn_buf, warn_size,
            "CPU governor is '%s'; benchmark results may exhibit noise. "
            "Set governor to 'performance' for minimal jitter.",
            gov_buf);
    }
} // detect_cpu_environment

/**
 * compare_doubles() - Comparison callback for sorting latency samples.
 * @a: Pointer to first double.
 * @b: Pointer to second double.
 *
 * Return: -1 if a < b, 1 if a > b, 0 if equal.
 */
static int compare_doubles(
    const void *a,
    const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db)
    {
        return -1;
    }
    if (da > db)
    {
        return 1;
    }
    return 0;
} // compare_doubles

/**
 * compute_statistics() - Compute median, IQR, min, max, and mean of sample array.
 * @samples:    Array of measurement samples (will be sorted in-place).
 * @n:          Number of samples.
 * @out_median: Pointer to store median value.
 * @out_iqr:    Pointer to store interquartile range.
 * @out_min:    Pointer to store minimum value.
 * @out_max:    Pointer to store maximum value.
 * @out_mean:   Pointer to store mean value.
 */
static void compute_statistics(
    double *samples,
    int     n,
    double *out_median,
    double *out_iqr,
    double *out_min,
    double *out_max,
    double *out_mean)
{
    if (n <= 0)
    {
        *out_median = *out_iqr = *out_min = *out_max = *out_mean = 0.0;
        return;
    }

    qsort(samples, (size_t)n, sizeof(double), compare_doubles);

    *out_min = samples[0];
    *out_max = samples[n - 1];

    double sum = 0.0;
    for (int ii = 0; ii < n; ii++)
    {
        sum += samples[ii];
    }
    *out_mean = sum / (double)n;

    /* Median */
    if (n % 2 == 1)
    {
        *out_median = samples[n / 2];
    }
    else
    {
        *out_median = 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
    }

    /* Interquartile Range (IQR) */
    if (n >= 4)
    {
        int mid = n / 2;
        double q1, q3;
        if (mid % 2 == 1)
        {
            q1 = samples[mid / 2];
            q3 = samples[n - 1 - (mid / 2)];
        }
        else
        {
            q1 = 0.5 * (samples[mid / 2 - 1] + samples[mid / 2]);
            q3 = 0.5 * (samples[n - mid + mid / 2 - 1] + samples[n - mid + mid / 2]);
        }
        *out_iqr = (q3 >= q1) ? (q3 - q1) : 0.0;
    }
    else
    {
        *out_iqr = *out_max - *out_min;
    }
} // compute_statistics

/**
 * parse_simd_bench_output() - Parse kernel speedups from gric-simd-bench output.
 * @output:      Raw benchmark stdout.
 * @kernels_arr: cJSON array to append parsed kernel objects to.
 *
 * Return: Total AVX2 kernel execution time in seconds.
 */
static double parse_simd_bench_output(
    const char *output,
    cJSON      *kernels_arr)
{
    if (output == NULL)
    {
        return 0.0;
    }

    double total_simd_time = 0.0;
    const char *cur = output;
    char line[1024];

    while (*cur != '\0')
    {
        const char *next = strchr(cur, '\n');
        size_t len = (next != NULL) ? (size_t)(next - cur) : strlen(cur);

        if (len < sizeof(line))
        {
            memcpy(line, cur, len);
            line[len] = '\0';

            /* Parse rows containing '|' delimiters: Kernel | Dim | Scalar | AVX2 | ... */
            char *pipe1 = strchr(line, '|');
            if (pipe1 != NULL && strstr(line, "Kernel") == NULL && strstr(line, "---") == NULL)
            {
                char k_name[64];
                int dim = 0;
                double t_scalar = 0.0;
                double t_avx2 = 0.0;
                double speedup = 0.0;

                size_t nlen = (size_t)(pipe1 - line);
                if (nlen >= sizeof(k_name))
                {
                    nlen = sizeof(k_name) - 1;
                }
                memcpy(k_name, line, nlen);
                k_name[nlen] = '\0';

                /* Trim trailing spaces from kernel name */
                while (nlen > 0 && isspace((unsigned char)k_name[nlen - 1]))
                {
                    k_name[--nlen] = '\0';
                }

                char *pipe2 = strchr(pipe1 + 1, '|');
                char *pipe3 = (pipe2 != NULL) ? strchr(pipe2 + 1, '|') : NULL;
                char *pipe4 = (pipe3 != NULL) ? strchr(pipe3 + 1, '|') : NULL;
                char *pipe5 = (pipe4 != NULL) ? strchr(pipe4 + 1, '|') : NULL;

                if (pipe2 != NULL && pipe3 != NULL && pipe4 != NULL)
                {
                    dim = atoi(pipe1 + 1);
                    t_scalar = strtod(pipe2 + 1, NULL);
                    t_avx2 = strtod(pipe3 + 1, NULL);
                    if (pipe5 != NULL)
                    {
                        speedup = strtod(pipe5 + 1, NULL);
                    }

                    if (t_avx2 > 0.0)
                    {
                        total_simd_time += t_avx2;
                    }

                    if (kernels_arr != NULL)
                    {
                        cJSON *k_obj = cJSON_CreateObject();
                        cJSON_AddStringToObject(k_obj, "kernel", k_name);
                        cJSON_AddNumberToObject(k_obj, "dimension", dim);
                        cJSON_AddNumberToObject(k_obj, "scalar_sec", t_scalar);
                        cJSON_AddNumberToObject(k_obj, "avx2_sec", t_avx2);
                        cJSON_AddNumberToObject(k_obj, "avx2_speedup", speedup);
                        cJSON_AddItemToArray(kernels_arr, k_obj);
                    }
                }
            } // if pipe1
        }

        if (next == NULL)
        {
            break;
        }
        cur = next + 1;
    } // while cur

    return total_simd_time;
} // parse_simd_bench_output

int mcp_tool_dev_bench(
    const cJSON *args,
    cJSON       *res)
{
    char root[1024];
    mcp_get_project_root(root, sizeof(root));
    if (root[0] == '\0')
    {
        cJSON_AddStringToObject(res, "error", "Could not locate project root directory");
        return -1;
    }

    const char *target = "clustering_loop";
    int trials = 5;
    int warmup = 1;
    const char *custom_dataset = NULL;
    const char *baseline_file = NULL;
    int save_baseline = 0;
    double threshold_pct = 10.0;

    if (args != NULL)
    {
        cJSON *t_item = cJSON_GetObjectItemCaseSensitive(args, "target");
        if (t_item != NULL && cJSON_IsString(t_item) && t_item->valuestring[0] != '\0')
        {
            target = t_item->valuestring;
        }

        cJSON *tr_item = cJSON_GetObjectItemCaseSensitive(args, "trials");
        if (tr_item != NULL && cJSON_IsNumber(tr_item))
        {
            trials = tr_item->valueint;
            if (trials < 1)
            {
                trials = 1;
            }
            if (trials > MAX_BENCH_TRIALS)
            {
                trials = MAX_BENCH_TRIALS;
            }
        }

        cJSON *w_item = cJSON_GetObjectItemCaseSensitive(args, "warmup");
        if (w_item != NULL && cJSON_IsNumber(w_item))
        {
            warmup = w_item->valueint;
            if (warmup < 0)
            {
                warmup = 0;
            }
        }

        cJSON *d_item = cJSON_GetObjectItemCaseSensitive(args, "dataset");
        if (d_item != NULL && cJSON_IsString(d_item) && d_item->valuestring[0] != '\0')
        {
            custom_dataset = d_item->valuestring;
        }

        cJSON *b_item = cJSON_GetObjectItemCaseSensitive(args, "baseline_file");
        if (b_item != NULL && cJSON_IsString(b_item) && b_item->valuestring[0] != '\0')
        {
            baseline_file = b_item->valuestring;
        }

        cJSON *s_item = cJSON_GetObjectItemCaseSensitive(args, "save_baseline");
        if (s_item != NULL && cJSON_IsBool(s_item))
        {
            save_baseline = cJSON_IsTrue(s_item);
        }

        cJSON *th_item = cJSON_GetObjectItemCaseSensitive(args, "regression_threshold_pct");
        if (th_item != NULL && cJSON_IsNumber(th_item))
        {
            threshold_pct = th_item->valuedouble;
        }
    }

    char resolved_baseline[1060];
    if (baseline_file != NULL)
    {
        mcp_resolve_path(baseline_file, resolved_baseline, sizeof(resolved_baseline));
    }
    else
    {
        snprintf(
            resolved_baseline, sizeof(resolved_baseline),
            "%s/.gric_bench_baseline.json", root);
    }

    /* 1. Detect CPU environment */
    char gov_name[64];
    char warn_msg[256];
    detect_cpu_environment(gov_name, sizeof(gov_name), warn_msg, sizeof(warn_msg));

    cJSON *env_obj = cJSON_CreateObject();
    cJSON_AddStringToObject(env_obj, "cpu_governor", gov_name);
    if (warn_msg[0] != '\0')
    {
        cJSON_AddStringToObject(env_obj, "warning", warn_msg);
    }
    else
    {
        cJSON_AddNullToObject(env_obj, "warning");
    }

    /* 2. Execute benchmark trials */
    double samples[MAX_BENCH_TRIALS];
    int valid_samples = 0;
    cJSON *kernels_arr = NULL;
    double throughput_val = 0.0;

    if (strcmp(target, "simd_kernels") == 0)
    {
        char exe_bin[1060];
        snprintf(exe_bin, sizeof(exe_bin), "%s/build/gric-simd-bench", root);
        const char *const argv[] = {exe_bin, NULL};

        char *out_buf = malloc(MAX_BENCH_OUTPUT_BYTES);
        if (out_buf == NULL)
        {
            cJSON_Delete(env_obj);
            cJSON_AddStringToObject(res, "error", "Out of memory allocating bench buffer");
            return -1;
        }

        /* Warmup runs */
        for (int w = 0; w < warmup; w++)
        {
            int status = 0;
            mcp_exec_capture(argv, out_buf, MAX_BENCH_OUTPUT_BYTES, 30000, &status);
        }

        /* Timed trials */
        for (int ii = 0; ii < trials; ii++)
        {
            int status = 0;
            double t0 = get_time_ms();
            int ret = mcp_exec_capture(
                argv, out_buf, MAX_BENCH_OUTPUT_BYTES, 30000, &status);
            double t1 = get_time_ms();

            if (ret == 0 && status == 0)
            {
                samples[valid_samples++] = t1 - t0;
                if (kernels_arr == NULL)
                {
                    kernels_arr = cJSON_CreateArray();
                    parse_simd_bench_output(out_buf, kernels_arr);
                }
            }
        }
        free(out_buf);
    }
    else if (strcmp(target, "clustering_loop") == 0)
    {
        char dataset_path[1024];
        if (custom_dataset != NULL)
        {
            mcp_resolve_path(custom_dataset, dataset_path, sizeof(dataset_path));
        }
        else
        {
            snprintf(dataset_path, sizeof(dataset_path), "/tmp/ctest_spiral.txt");
            if (access(dataset_path, R_OK) != 0)
            {
                char gen_bin[1060];
                snprintf(gen_bin, sizeof(gen_bin), "%s/build/gric-mktxtseq", root);
                const char *const gen_argv[] = {
                    gen_bin, "1000", dataset_path, "2Dspiral", NULL
                };
                char dummy[256];
                int s = 0;
                mcp_exec_capture(gen_argv, dummy, sizeof(dummy), 15000, &s);
            }
        }

        char tmp_out[256];
        snprintf(tmp_out, sizeof(tmp_out), "/tmp/gric_bench_%d.clusterdat", (int)getpid());

        char cluster_bin[1060];
        snprintf(cluster_bin, sizeof(cluster_bin), "%s/build/gric-cluster", root);

        const char *const argv[] = {
            cluster_bin, "0.1", dataset_path, "-maxim", "1000",
            "-outdir", tmp_out, "-txt", NULL
        };

        char dummy_out[512];
        for (int w = 0; w < warmup; w++)
        {
            int status = 0;
            mcp_exec_capture(argv, dummy_out, sizeof(dummy_out), 30000, &status);
        }

        for (int ii = 0; ii < trials; ii++)
        {
            int status = 0;
            double t0 = get_time_ms();
            int ret = mcp_exec_capture(argv, dummy_out, sizeof(dummy_out), 30000, &status);
            double t1 = get_time_ms();

            if (ret == 0 && status == 0)
            {
                samples[valid_samples++] = t1 - t0;
            }
        }

        /* Clean up temporary directory */
        const char *const rm_argv[] = {"rm", "-rf", tmp_out, NULL};
        int s = 0;
        mcp_exec_capture(rm_argv, dummy_out, sizeof(dummy_out), 5000, &s);
    }
    else
    {
        cJSON_Delete(env_obj);
        cJSON_AddStringToObject(
            res, "error", "Unsupported benchmark target (use simd_kernels or clustering_loop)");
        return -1;
    }

    if (valid_samples == 0)
    {
        cJSON_Delete(env_obj);
        if (kernels_arr != NULL)
        {
            cJSON_Delete(kernels_arr);
        }
        cJSON_AddStringToObject(res, "error", "All benchmark trial runs failed");
        return -1;
    }

    /* 3. Compute statistics */
    double median_ms, iqr_ms, min_ms, max_ms, mean_ms;
    compute_statistics(
        samples, valid_samples, &median_ms, &iqr_ms, &min_ms, &max_ms, &mean_ms);

    if (median_ms > 0.0)
    {
        throughput_val = (1000.0 / median_ms) * 1000.0; /* items/sec for 1000-frame batch */
    }

    /* 4. Compare vs baseline or store baseline */
    cJSON *baseline_root = NULL;
    FILE *bfp = fopen(resolved_baseline, "r");
    if (bfp != NULL)
    {
        fseek(bfp, 0, SEEK_END);
        long sz = ftell(bfp);
        fseek(bfp, 0, SEEK_SET);
        if (sz > 0 && sz < 1048576)
        {
            char *buf = malloc((size_t)sz + 1);
            if (buf != NULL)
            {
                size_t rd = fread(buf, 1, (size_t)sz, bfp);
                buf[rd] = '\0';
                baseline_root = cJSON_Parse(buf);
                free(buf);
            }
        }
        fclose(bfp);
    }

    if (baseline_root == NULL)
    {
        baseline_root = cJSON_CreateObject();
    }

    double base_median = -1.0;
    cJSON *t_entry = cJSON_GetObjectItemCaseSensitive(baseline_root, target);
    if (t_entry != NULL)
    {
        cJSON *bm = cJSON_GetObjectItemCaseSensitive(t_entry, "median_ms");
        if (bm != NULL && cJSON_IsNumber(bm))
        {
            base_median = bm->valuedouble;
        }
    }

    if (save_baseline)
    {
        cJSON *new_entry = cJSON_CreateObject();
        cJSON_AddNumberToObject(new_entry, "median_ms", median_ms);
        cJSON_AddNumberToObject(new_entry, "throughput", throughput_val);
        cJSON_AddNumberToObject(new_entry, "timestamp", (double)time(NULL));

        if (t_entry != NULL)
        {
            cJSON_DeleteItemFromObject(baseline_root, target);
        }
        cJSON_AddItemToObject(baseline_root, target, new_entry);

        char *printed = cJSON_Print(baseline_root);
        if (printed != NULL)
        {
            FILE *wfp = fopen(resolved_baseline, "w");
            if (wfp != NULL)
            {
                fputs(printed, wfp);
                fclose(wfp);
            }
            free(printed);
        }
    }

    cJSON *comp_obj = cJSON_CreateObject();
    if (base_median > 0.0)
    {
        double delta_pct = ((median_ms - base_median) / base_median) * 100.0;
        cJSON_AddBoolToObject(comp_obj, "has_baseline", 1);
        cJSON_AddNumberToObject(comp_obj, "baseline_median_ms", base_median);
        cJSON_AddNumberToObject(comp_obj, "delta_pct", delta_pct);

        if (delta_pct > threshold_pct)
        {
            cJSON_AddStringToObject(comp_obj, "status", "REGRESSION");
        }
        else
        {
            cJSON_AddStringToObject(comp_obj, "status", "PASS");
        }
    }
    else
    {
        cJSON_AddBoolToObject(comp_obj, "has_baseline", 0);
        cJSON_AddStringToObject(comp_obj, "status", "PASS");
    }

    cJSON_Delete(baseline_root);

    /* 5. Assemble JSON response */
    cJSON_AddStringToObject(res, "target", target);
    cJSON_AddNumberToObject(res, "trials", valid_samples);
    cJSON_AddItemToObject(res, "environment", env_obj);

    cJSON *res_metrics = cJSON_CreateObject();
    cJSON_AddNumberToObject(res_metrics, "median_ms", median_ms);
    cJSON_AddNumberToObject(res_metrics, "iqr_ms", iqr_ms);
    cJSON_AddNumberToObject(res_metrics, "min_ms", min_ms);
    cJSON_AddNumberToObject(res_metrics, "max_ms", max_ms);
    cJSON_AddNumberToObject(res_metrics, "mean_ms", mean_ms);
    cJSON_AddNumberToObject(res_metrics, "throughput", throughput_val);
    cJSON_AddItemToObject(res, "results", res_metrics);

    cJSON_AddItemToObject(res, "comparison", comp_obj);

    if (kernels_arr != NULL)
    {
        cJSON_AddItemToObject(res, "kernels", kernels_arr);
    }

    return 0;
} // mcp_tool_dev_bench

const struct mcp_tool_def mcp_tooldef_dev_bench = {
    .name         = "gric_dev_bench",
    .toolset      = MCP_TS_DEV,
    .side_effects = 0,
    .fn           = mcp_tool_dev_bench,
    .description  = "Deterministic micro-benchmark runner for SIMD kernels and clustering loops "
                    "with statistical aggregation (median, IQR) and baseline regression checks.",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"target\": {\n"
        "      \"type\": \"string\",\n"
        "      \"enum\": [\"simd_kernels\", \"clustering_loop\"],\n"
        "      \"description\": \"Benchmark target (default: clustering_loop).\"\n"
        "    },\n"
        "    \"trials\": {\n"
        "      \"type\": \"integer\",\n"
        "      \"description\": \"Number of benchmark runs (default: 5, max: 50).\"\n"
        "    },\n"
        "    \"warmup\": {\n"
        "      \"type\": \"integer\",\n"
        "      \"description\": \"Number of untimed warmup runs (default: 1).\"\n"
        "    },\n"
        "    \"dataset\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Dataset path for clustering_loop.\"\n"
        "    },\n"
        "    \"baseline_file\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Path to stored baseline JSON.\"\n"
        "    },\n"
        "    \"save_baseline\": {\n"
        "      \"type\": \"boolean\",\n"
        "      \"description\": \"Save current results as new baseline (default: false).\"\n"
        "    },\n"
        "    \"regression_threshold_pct\": {\n"
        "      \"type\": \"number\",\n"
        "      \"description\": \"Regression threshold % increase (default: 10.0).\"\n"
        "    }\n"
        "  }\n"
        "}",
};
