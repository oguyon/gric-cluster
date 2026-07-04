/**
 * @file config_utils.c
 * @brief Configuration parsing and management for the clustering engine.
 *
 * Handles reading, writing, and applying configurations, command-line arguments, and
 * configuration files for the clustering process.
 *
 * Main Functions:
 * - apply_option: Parses and applies a single command line or configuration option.
 * - read_config_file: Reads options from a key-value configuration file.
 * - write_config_file: Dumps the active configuration to a file.
 */
#include "config_utils.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper to match options with or without leading dash
static int matches(const char *key, const char *opt)
{
    if (strcmp(key, opt) == 0)
        return 1;
    if (key[0] != '-' && opt[0] == '-' && strcmp(key, opt + 1) == 0)
        return 1;
    if (key[0] == '-' && opt[0] != '-' && strcmp(key + 1, opt) == 0)
        return 1;
    return 0;
}

/**
 * apply_option() - Parse a single CLI flag and apply it
 *                  to the configuration.
 * @config: Configuration struct to modify.
 * @key:    Option key string (with or without leading dash).
 * @value:  Option value string, or NULL for boolean flags.
 *
 * Return: 0 if a boolean flag was consumed (no value),
 *         1 if a flag + value pair was consumed,
 *        -1 if the flag is unknown.
 */
int apply_option(ClusterConfig *config, const char *key, const char *value)
{
    if (matches(key, "-dprob"))
    {
        if (!value)
            return -1;
        config->algo.deltaprob = atof(value);
        return 1;
    }
    else if (matches(key, "-maxcl"))
    {
        if (!value)
            return -1;
        config->algo.maxnbclust = atoi(value);
        return 1;
    }
    else if (matches(key, "-ncpu"))
    {
        if (!value)
            return -1;
        config->optim.ncpu = atoi(value);
        return 1;
    }
    else if (matches(key, "-maxim"))
    {
        if (!value)
            return -1;
        config->input.maxnbfr = atol(value);
        return 1;
    }
    else if (matches(key, "-avg"))
    {
        config->output.average_mode = 1;
        return 0;
    }
    else if (matches(key, "-distall"))
    {
        config->output.distall_mode = 1;
        return 0;
    }
    else if (matches(key, "-outdir"))
    {
        if (!value)
            return -1;
        config->output.user_outdir =
            strdup(value); // We strdup here to manage memory consistent with config file reading
        return 1;
    }
    else if (matches(key, "-progress"))
    {
        config->output.progress_mode = 1;
        return 0;
    }
    else if (matches(key, "-gprob"))
    {
        config->optim.gprob_mode = 1;
        return 0;
    }
    else if (matches(key, "-verbose"))
    {
        config->output.verbose_level = 1;
        return 0;
    }
    else if (matches(key, "-veryverbose"))
    {
        config->output.verbose_level = 2;
        return 0;
    }
    else if (matches(key, "-fitsout"))
    {
        config->output.fitsout_mode = 1;
        return 0;
    }
    else if (matches(key, "-pngout"))
    {
        config->output.pngout_mode = 1;
        return 0;
    }
    else if (matches(key, "-filelist"))
    {
        config->input.filelist_mode = 1;
        return 0;
    }
    else if (matches(key, "-stream"))
    {
        config->input.stream_input_mode = 1;
        return 0;
    }
    else if (matches(key, "-cnt2sync"))
    {
        config->input.cnt2sync_mode = 1;
        return 0;
    }
    else if (matches(key, "-fmatcha"))
    {
        if (!value)
            return -1;
        config->optim.fmatch_a = atof(value);
        return 1;
    }
    else if (matches(key, "-fmatchb"))
    {
        if (!value)
            return -1;
        config->optim.fmatch_b = atof(value);
        return 1;
    }
    else if (matches(key, "-maxvis"))
    {
        if (!value)
            return -1;
        config->optim.max_gprob_visitors = atoi(value);
        return 1;
    }
    else if (matches(key, "-te4"))
    {
        config->optim.te4_mode = 1;
        return 0;
    }
    else if (matches(key, "-te5"))
    {
        config->optim.te5_mode = 1;
        return 0;
    }
    else if (matches(key, "-entropy"))
    {
        config->optim.entropy_mode = 1;
        return 0;
    }
    else if (matches(key, "-entropy_max_targets"))
    {
        if (!value)
            return -1;
        config->optim.entropy_max_targets = atoi(value);
        return 1;
    }
    else if (matches(key, "-entropy_min_prob"))
    {
        if (!value)
            return -1;
        config->optim.entropy_min_prob = atof(value);
        return 1;
    }
    else if (matches(key, "-entropy_gate"))
    {
        if (!value)
            return -1;
        config->optim.entropy_gate_bits = atof(value);
        return 1;
    }
    else if (matches(key, "-entropy_first_gate"))
    {
        if (!value)
            return -1;
        config->optim.entropy_first_gate_bits = atof(value);
        return 1;
    }
    else if (matches(key, "-entropy_fast"))
    {
        config->optim.entropy_fast_mode = 1;
        return 0;
    }
    else if (matches(key, "-sparse_dcc"))
    {
        config->optim.sparse_dcc_mode = 1;
        return 0;
    }
    else if (matches(key, "-sparse_dcc_extra_evals"))
    {
        if (!value)
            return -1;
        config->optim.sparse_dcc_extra_evals = atoi(value);
        return 1;
    }
    else if (matches(key, "-soft_bayesian"))
    {
        config->optim.soft_bayesian_mode = 1;
        return 0;
    }
    else if (matches(key, "-soft_bayesian_sigma"))
    {
        if (!value)
            return -1;
        config->optim.soft_bayesian_sigma_coeff = atof(value);
        return 1;
    }
    else if (matches(key, "-tm"))
    {
        if (!value)
            return -1;
        config->algo.tm_mixing_coeff = atof(value);
        return 1;
    }
    else if (matches(key, "-maxcl_strategy"))
    {
        if (!value)
            return -1;
        if (strcmp(value, "stop") == 0)
            config->algo.maxcl_strategy = MAXCL_STOP;
        else if (strcmp(value, "discard") == 0)
            config->algo.maxcl_strategy = MAXCL_DISCARD;
        else if (strcmp(value, "merge") == 0)
            config->algo.maxcl_strategy = MAXCL_MERGE;
        else
            fprintf(stderr, "Warning: Unknown maxcl_strategy '%s'\n", value);
        return 1;
    }
    else if (matches(key, "-discard_frac"))
    {
        if (!value)
            return -1;
        config->algo.discard_fraction = atof(value);
        return 1;
    }
    else if (matches(key, "-tm_out"))
    {
        config->output.output_tm = 1;
        return 0;
    }
    else if (matches(key, "-dcc"))
    {
        config->output.output_dcc = 1;
        return 0;
    }
    else if (matches(key, "-no_dcc"))
    {
        config->output.output_dcc = 0;
        return 0;
    }
    else if (matches(key, "-anchors"))
    {
        config->output.output_anchors = 1;
        return 0;
    }
    else if (matches(key, "-counts"))
    {
        config->output.output_counts = 1;
        return 0;
    }
    else if (matches(key, "-membership"))
    {
        config->output.output_membership = 1;
        return 0;
    }
    else if (matches(key, "-no_membership"))
    {
        config->output.output_membership = 0;
        return 0;
    }
    else if (matches(key, "-discarded"))
    {
        config->output.output_discarded = 1;
        return 0;
    }
    else if (matches(key, "-clustered"))
    {
        config->output.output_clustered = 1;
        return 0;
    }
    else if (matches(key, "-clusters"))
    {
        config->output.output_clusters = 1;
        return 0;
    }
    else if (strncmp(key, "-pred", 5) == 0 || strncmp(key, "pred", 4) == 0)
    {
        config->optim.pred_mode = 1;
        const char *params = strchr(key, '[');
        if (params)
        {
            params++; // Skip [
            int l, h, n;
            if (sscanf(params, "%d,%d,%d", &l, &h, &n) == 3)
            {
                config->optim.pred_len = l;
                config->optim.pred_h = h;
                config->optim.pred_n = n;
            }
        }
        return 0;
    }
    else if (matches(key, "-scandist"))
    {
        config->input.scandist_mode = 1;
        return 0;
    }
    else if (matches(key, "-rlim"))
    { // Explicit rlim
        if (!value)
            return -1;
        if (value[0] == 'a')
        {
            config->algo.auto_rlim_factor = atof(value + 1);
            config->algo.auto_rlim_mode = 1;
        }
        else
        {
            config->algo.rlim = atof(value);
        }
        return 1;
    }
    else if (matches(key, "-input") || matches(key, "-in"))
    { // Explicit input
        if (!value)
            return -1;
        config->input.fits_filename = strdup(value);
        return 1;
    }
    else if (matches(key, "-shm") || matches(key, "-shm-file"))
    {
        if (!value)
            return -1;
        config->output.shm_filename = strdup(value);
        return 1;
    }

    return -1; // Unknown option
}

/**
 * read_config_file() - Read a config file and apply each line
 *                      as a key-value option pair.
 * @filename: Path to the configuration file.
 * @config:   Configuration struct to populate.
 *
 * Lines starting with '#' or empty lines are skipped.
 * Each non-comment line is split into a key and optional
 * value, then forwarded to apply_option().
 *
 * Return: 0 on success, 1 if the file cannot be opened.
 */
int read_config_file(const char *filename, ClusterConfig *config)
{
    FILE *f = fopen(filename, "r");
    if (!f)
        return 1;

    char line[1024];
    while (fgets(line, sizeof(line), f))
    {
        char *p = line;
        while (isspace(*p))
            p++;
        if (*p == '\0' || *p == '#')
            continue;

        // Split key value
        char *key = p;
        char *val = NULL;
        char *end = key;
        while (*end && !isspace(*end))
            end++;
        if (*end)
        {
            *end = '\0';
            val = end + 1;
            while (isspace(*val))
                val++;
            // Trim newline from val
            char *v_end = val + strlen(val) - 1;
            while (v_end >= val && isspace(*v_end))
            {
                *v_end = '\0';
                v_end--;
            }
            if (*val == '\0')
                val = NULL;
        }

        apply_option(config, key, val);
    }
    fclose(f);
    return 0;
}

/**
 * write_config_file() - Serialize the current config to a file
 *                       for reproducibility.
 * @filename: Output file path.
 * @config:   Configuration struct to serialize.
 *
 * Writes all active configuration parameters in a format
 * readable by read_config_file().
 *
 * Return: 0 on success, 1 if the file cannot be opened.
 */
int write_config_file(const char *filename, ClusterConfig *config)
{
    FILE *f = fopen(filename, "w");
    if (!f)
        return 1;

    fprintf(f, "# gric-cluster configuration file\n");
    fprintf(f, "rlim %f\n", config->algo.rlim);
    if (config->algo.auto_rlim_mode)
        fprintf(f, "# auto_rlim enabled (factor %f)\n", config->algo.auto_rlim_factor);
    if (config->input.fits_filename)
        fprintf(f, "input %s\n", config->input.fits_filename);
    if (config->output.user_outdir)
        fprintf(f, "outdir %s\n", config->output.user_outdir);
    fprintf(f, "dprob %f\n", config->algo.deltaprob);
    fprintf(f, "maxcl %d\n", config->algo.maxnbclust);
    fprintf(f, "maxim %ld\n", config->input.maxnbfr);
    fprintf(f, "ncpu %d\n", config->optim.ncpu);

    if (config->output.average_mode)
        fprintf(f, "avg\n");
    if (config->output.distall_mode)
        fprintf(f, "distall\n");
    if (config->output.progress_mode)
        fprintf(f, "progress\n");
    if (config->optim.gprob_mode)
        fprintf(f, "gprob\n");
    if (config->output.verbose_level == 1)
        fprintf(f, "verbose\n");
    if (config->output.verbose_level == 2)
        fprintf(f, "veryverbose\n");
    if (config->output.fitsout_mode)
        fprintf(f, "fitsout\n");
    if (config->output.pngout_mode)
        fprintf(f, "pngout\n");
    if (config->input.filelist_mode)
        fprintf(f, "filelist\n");
    if (config->input.stream_input_mode)
        fprintf(f, "stream\n");
    if (config->input.cnt2sync_mode)
        fprintf(f, "cnt2sync\n");

    fprintf(f, "fmatcha %f\n", config->optim.fmatch_a);
    fprintf(f, "fmatchb %f\n", config->optim.fmatch_b);
    fprintf(f, "maxvis %d\n", config->optim.max_gprob_visitors);

    if (config->optim.te4_mode)
        fprintf(f, "te4\n");
    if (config->optim.te5_mode)
        fprintf(f, "te5\n");
    if (config->optim.entropy_mode)
    {
        fprintf(f, "entropy\n");
        fprintf(f, "entropy_max_targets %d\n",
                config->optim.entropy_max_targets);
        fprintf(f, "entropy_min_prob %f\n",
                config->optim.entropy_min_prob);
        fprintf(f, "entropy_gate %f\n",
                config->optim.entropy_gate_bits);
        fprintf(f, "entropy_first_gate %f\n",
                config->optim.entropy_first_gate_bits);
        if (config->optim.entropy_fast_mode)
        {
            fprintf(f, "entropy_fast\n");
        }
    }
    if (config->optim.sparse_dcc_mode)
    {
        fprintf(f, "sparse_dcc\n");
        fprintf(f, "sparse_dcc_extra_evals %d\n", config->optim.sparse_dcc_extra_evals);
    }
    if (config->optim.soft_bayesian_mode)
    {
        fprintf(f, "soft_bayesian\n");
        fprintf(f, "soft_bayesian_sigma %f\n", config->optim.soft_bayesian_sigma_coeff);
    }

    fprintf(f, "tm %f\n", config->algo.tm_mixing_coeff);

    const char *strat = "stop";
    if (config->algo.maxcl_strategy == MAXCL_DISCARD)
        strat = "discard";
    else if (config->algo.maxcl_strategy == MAXCL_MERGE)
        strat = "merge";
    fprintf(f, "maxcl_strategy %s\n", strat);
    fprintf(f, "discard_frac %f\n", config->algo.discard_fraction);

    if (config->output.output_dcc)
        fprintf(f, "dcc\n");
    if (!config->output.output_dcc)
        fprintf(f, "no_dcc\n");
    if (config->output.output_tm)
        fprintf(f, "tm_out\n");
    if (config->output.output_anchors)
        fprintf(f, "anchors\n");
    if (config->output.output_counts)
        fprintf(f, "counts\n");
    if (config->output.output_membership)
        fprintf(f, "membership\n");
    if (!config->output.output_membership)
        fprintf(f, "no_membership\n");
    if (config->output.output_discarded)
        fprintf(f, "discarded\n");
    if (config->output.output_clustered)
        fprintf(f, "clustered\n");
    if (config->output.output_clusters)
        fprintf(f, "clusters\n");

    if (config->optim.pred_mode)
    {
        fprintf(f, "# Prediction mode enabled: pred[%d,%d,%d]\n", config->optim.pred_len, config->optim.pred_h,
                config->optim.pred_n);
        // Writing as -pred is weird, maybe write separate params?
        // But the parser handles -pred[...]. Let's write it as such.
        fprintf(f, "-pred[%d,%d,%d]\n", config->optim.pred_len, config->optim.pred_h, config->optim.pred_n);
    }

    if (config->input.scandist_mode)
        fprintf(f, "scandist\n");

    if (config->output.shm_filename)
        fprintf(f, "shm %s\n", config->output.shm_filename);

    fclose(f);
    return 0;
}
