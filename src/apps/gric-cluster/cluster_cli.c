/**
 * @file cluster_cli.c
 * @brief Command-line interface parser and configuration defaults for gric-cluster.
 */

#define _POSIX_C_SOURCE 200809L
#include "cluster_cli.h"
#include "cluster_help.h"
#include "config_utils.h"
#include "gric_profile.h"
#include "probe_engine.h"
#include "cli_colors.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

void print_args_on_error(int argc, char *argv[]);

void cluster_cli_init_config_defaults(
    ClusterConfig *config)
{
    if (config == NULL)
    {
        return;
    }
    memset(config, 0, sizeof(ClusterConfig));

    config->algo.deltaprob = 0.01;
    config->algo.maxnbclust = 1000;
    config->optim.ncpu = 1;
    config->input.maxnbfr = 1000000;
    config->optim.fmatch_a = 2.0;
    config->optim.fmatch_b = 0.5;
    config->optim.max_gprob_visitors = 1000;
    config->output.progress_mode = 1;
    config->optim.pred_len = 10;
    config->optim.pred_h = 1000;
    config->optim.pred_n = 2;
    config->optim.te4_max_anchors = 3;
    config->algo.maxcl_strategy = MAXCL_STOP;
    config->algo.discard_fraction = 0.5;
    config->optim.entropy_max_targets = 15;
    config->optim.entropy_min_prob = 0.001;
    config->optim.entropy_gate_bits = 2.0;
    config->optim.entropy_first_gate_bits = 4.0;
    config->optim.entropy_fast_mode = 0;
    config->optim.entropy_leader_shortcut = 0;
    config->optim.entropy_leader_cutoff = 0.50;
    config->optim.sparse_dcc_mode = 0;
    config->optim.sparse_dcc_extra_evals = 0;
    config->optim.soft_bayesian_mode = 0;
    config->optim.soft_bayesian_sigma_coeff = 1.0;
    config->optim.disable_pass2 = 1;
    config->optim.xtile_mode = 0;
    config->optim.xtile_decay = 1.0;
    config->optim.use_sq8 = -1;
    config->optim.use_sq16 = -1;
    config->optim.use_eq16 = -1;
    config->optim.use_eq16_adc = 1;
    config->optim.use_memo = 1;
    config->optim.sq16_ratio = 0.05;
    config->optim.use_batch_dist = 1;
    config->optim.gpu_prune_mode = -1;

    config->input.tile_grid_x = 0;
    config->input.tile_grid_y = 0;
    config->input.tile_map_file = NULL;
    config->input.tile_config_file = NULL;
    config->input.retrieval_window = 1000;

    config->output.output_dcc = 1;
    config->output.dcc_sq16_output = 0;
    config->output.output_tm = 0;
    config->output.output_anchors = 1;
    config->output.output_counts = 1;
    config->output.output_membership = 1;
    config->output.output_evals = 1;
    config->output.output_discarded = 0;
    config->output.output_clustered = 0;
    config->output.output_clusters = 0;
    config->output.no_txt = 1;
}

/**
 * record_cmdline() - Capture full CLI invocation string for provenance logging.
 * @argc: Argument count.
 * @argv: Argument vector.
 *
 * Return: Allocated command string or NULL on failure.
 */
static char *record_cmdline(
    int    argc,
    char **argv)
{
    char *cmdline = (char *)malloc(8192);
    if (cmdline == NULL)
    {
        return NULL;
    }

    size_t pos = 0;
    size_t rem = 8192;
    for (int ii = 0; ii < argc; ii++)
    {
        size_t len = strlen(argv[ii]);
        if (len + 2 < rem)
        {
            if (ii > 0)
            {
                cmdline[pos++] = ' ';
                rem--;
            }
            memcpy(cmdline + pos, argv[ii], len);
            pos += len;
            rem -= len;
            cmdline[pos] = '\0';
        }
    }
    return cmdline;
}

/**
 * check_help_flags() - Intercept help requests before configuration parsing.
 * @argc: Argument count.
 * @argv: Argument vector.
 *
 * Return: 2 if help was handled and program should exit, 0 otherwise.
 */
static int check_help_flags(
    int    argc,
    char **argv)
{
    for (int ii = 1; ii < argc; ii++)
    {
        if (strcmp(argv[ii], "-h") == 0 || strcmp(argv[ii], "-help") == 0 ||
            strcmp(argv[ii], "--help") == 0)
        {
            if (ii + 1 < argc)
            {
                print_help_keyword(argv[ii + 1]);
            }
            else
            {
                print_help(argv[0]);
            }
            return 2;
        }
    }

    if (argc < 2)
    {
        print_usage(argv[0]);
        return 1;
    }
    return 0;
}

/**
 * parse_positional_arg() - Process positional CLI token as rlim, auto-rlim, or filename.
 * @key:      CLI argument string.
 * @config:   Cluster configuration.
 * @rlim_set: Pointer to flag indicating whether rlim has been set.
 *
 * Return: 0 on success, non-zero on error.
 */
static int parse_positional_arg(
    char          *key,
    ClusterConfig *config,
    int           *rlim_set)
{
    if (!config->input.scandist_mode && !*rlim_set)
    {
        char *endptr = NULL;
        double val = strtod(key, &endptr);
        if (*endptr == '\0')
        {
            config->algo.rlim = val;
            *rlim_set = 1;
            return 0;
        }
        if (key[0] == 'a' && isdigit(key[1]))
        {
            config->algo.auto_rlim_factor = atof(key + 1);
            config->algo.auto_rlim_mode = 1;
            *rlim_set = 1;
            return 0;
        }
    }

    if (config->input.fits_filename != NULL)
    {
        fprintf(stderr, "Error: Multiple input files specified ('%s', '%s')\n",
                config->input.fits_filename, key);
        return -1;
    }
    config->input.fits_filename = key;
    return 0;
}

/**
 * parse_cluster_options() - Parse command line options into ClusterConfig.
 * @argc:     Argument count.
 * @argv:     Argument vector.
 * @config:   Configuration structure.
 * @rlim_set: Pointer to flag set when rlim is provided.
 *
 * Return: 0 on success, non-zero on error.
 */
static int parse_cluster_options(
    int            argc,
    char         **argv,
    ClusterConfig *config,
    int           *rlim_set)
{
    int arg_idx = 1;
    const char *confw_filename = NULL;
    size_t nopts = 0;
    const struct gric_opt *opts = cluster_get_options(config, &nopts);

    while (arg_idx < argc)
    {
        char *key = argv[arg_idx];
        char *val = (arg_idx + 1 < argc) ? argv[arg_idx + 1] : NULL;

        if (strcmp(key, "-conf") == 0 || strcmp(key, "--conf") == 0)
        {
            if (val == NULL || read_config_file(val, config) != 0)
            {
                fprintf(stderr, "Error: Could not read config file %s\n", val ? val : "");
                return 1;
            }
            arg_idx += 2;
            continue;
        }

        if (strcmp(key, "-confw") == 0 || strcmp(key, "--confw") == 0)
        {
            if (val == NULL)
            {
                fprintf(stderr, "Error: -confw requires a filename\n");
                return 1;
            }
            confw_filename = val;
            arg_idx += 2;
            continue;
        }

        int res = gric_opt_parse_arg(argc, argv, &arg_idx, opts, nopts, config);
        if (res == 1)
        {
            if (strncmp(key, "-rlim", 5) == 0 || strncmp(key, "--rlim", 6) == 0 ||
                strncmp(key, "rlim", 4) == 0)
            {
                *rlim_set = 1;
            }
            continue;
        }
        if (res < 0 || (key[0] == '-' && !isdigit(key[1])))
        {
            print_usage(argv[0]);
            print_args_on_error(argc, argv);
            return 1;
        }

        if (parse_positional_arg(key, config, rlim_set) != 0)
        {
            print_args_on_error(argc, argv);
            return 1;
        }
        arg_idx++;
    }

    if (confw_filename != NULL && write_config_file(confw_filename, config) == 0)
    {
        printf("Configuration written to %s\n", confw_filename);
    }
    return 0;
}

/**
 * auto_probe_profile() - Run micro-probe on input dataset if profile is absent.
 * @config:       Cluster configuration.
 * @dataset_prof: Profile structure to populate.
 *
 * Return: 1 if profile was generated, 0 otherwise.
 */
static int auto_probe_profile(
    const ClusterConfig *config,
    GricProfile         *dataset_prof)
{
    ProbeConfig pcfg;
    memset(&pcfg, 0, sizeof(pcfg));
    snprintf(pcfg.dataset_path, sizeof(pcfg.dataset_path), "%s", config->input.fits_filename);
    pcfg.sample_limit = 500;
    pcfg.verbose_level = 0;
    pcfg.use_double = config->algo.use_double;

    ProbeResults pres;
    memset(&pres, 0, sizeof(pres));
    if (probe_run(&pcfg, &pres) != 0)
    {
        return 0;
    }

    *dataset_prof = pres.profile;
    memset(&pres.profile, 0, sizeof(pres.profile));
    dataset_prof->tiles_x = 1;
    dataset_prof->tiles_y = 1;

    char auto_save_path[1024];
    snprintf(auto_save_path, sizeof(auto_save_path), "%s.gricprof",
             config->input.fits_filename);
    if (gric_profile_write_json(auto_save_path, dataset_prof) == 0)
    {
        printf("%s[INFO]%s Auto-profiled dataset (500 frames) -> generated %s%s%s\n",
               ANSI_BOLD_GREEN, ANSI_COLOR_RESET, ANSI_BOLD, auto_save_path, ANSI_COLOR_RESET);
    }
    else
    {
        printf("%s[INFO]%s Auto-profiled dataset (500 frames) in memory\n",
               ANSI_BOLD_GREEN, ANSI_COLOR_RESET);
    }
    return 1;
}

/**
 * locate_or_probe_profile() - Find existing profile or trigger auto-probing.
 * @config:       Cluster configuration.
 * @dataset_prof: Profile structure to populate.
 * @rlim_set:     Whether rlim was explicitly supplied.
 *
 * Return: 1 if profile is available, 0 otherwise.
 */
static int locate_or_probe_profile(
    const ClusterConfig *config,
    GricProfile         *dataset_prof,
    int                  rlim_set)
{
    char located_prof_path[1024] = "";
    int has_prof = 0;

    if (config->optim.prof_filename != NULL)
    {
        snprintf(located_prof_path, sizeof(located_prof_path), "%s", config->optim.prof_filename);
        has_prof = (gric_profile_read_json(located_prof_path, dataset_prof) == 0);
    }
    else if (gric_profile_find_auto(config->input.fits_filename,
                                    located_prof_path, sizeof(located_prof_path)))
    {
        has_prof = (gric_profile_read_json(located_prof_path, dataset_prof) == 0);
    }

    if (!has_prof && !config->input.stream_input_mode && !config->input.filelist_mode &&
        (config->optim.use_eq16 || config->optim.use_sq16 || !rlim_set))
    {
        has_prof = auto_probe_profile(config, dataset_prof);
    }

    if (has_prof && located_prof_path[0] != '\0')
    {
        printf("%s[INFO]%s Auto-loaded dataset profile: %s%s%s\n",
               ANSI_BOLD_GREEN, ANSI_COLOR_RESET, ANSI_BOLD, located_prof_path, ANSI_COLOR_RESET);
    }
    return has_prof;
}

/**
 * apply_profile_rlim() - Set recommended rlim from profile preset.
 * @config:       Cluster configuration to update.
 * @dataset_prof: Loaded profile.
 */
static void apply_profile_rlim(
    ClusterConfig     *config,
    const GricProfile *dataset_prof)
{
    const char *p = config->optim.preset_name;
    if (strcasecmp(p, "p01") == 0 || strcasecmp(p, "1%") == 0 || strcasecmp(p, "ultrafine") == 0)
    {
        config->algo.rlim = dataset_prof->rlim_p01;
    }
    else if (strcasecmp(p, "p03") == 0 || strcasecmp(p, "3%") == 0 || strcasecmp(p, "vfine") == 0)
    {
        config->algo.rlim = dataset_prof->rlim_p03;
    }
    else if (strcasecmp(p, "fine") == 0 || strcasecmp(p, "p05") == 0 || strcasecmp(p, "5%") == 0)
    {
        config->algo.rlim = dataset_prof->rlim_fine;
    }
    else if (strcasecmp(p, "coarse") == 0 || strcasecmp(p, "p25") == 0 ||
             strcasecmp(p, "25%") == 0)
    {
        config->algo.rlim = dataset_prof->rlim_coarse;
    }
    else
    {
        config->algo.rlim = dataset_prof->rlim_recommended;
    }
}

/**
 * apply_profile_quantization() - Configure EQ16 / SQ16 scales from profile.
 * @config:       Cluster configuration.
 * @dataset_prof: Loaded profile.
 */
static void apply_profile_quantization(
    ClusterConfig     *config,
    const GricProfile *dataset_prof)
{
    if (config->optim.use_eq16 && (dataset_prof->use_sq16 || dataset_prof->dim >= 8))
    {
        float min_v = dataset_prof->sq16_params.min_val;
        float max_v = dataset_prof->sq16_params.max_val;
        long fdim = dataset_prof->dim;
        eq16_init_params(&config->optim.eq16_params, min_v, max_v, fdim);
        config->optim.eq16_calibrated = 1;
        if (config->optim.sq16_ratio > 0.0 && config->algo.rlim > 0.0 && fdim > 0)
        {
            float target_scale = (float)((config->optim.sq16_ratio * config->algo.rlim) /
                                         sqrt((double)fdim));
            config->optim.eq16_params.scale = target_scale;
            config->optim.eq16_params.inv_scale = 1.0f / target_scale;
            config->optim.eq16_params.center = 0.5f * (min_v + max_v);
            long kk = fdim / 8;
            long rem = fdim % 8;
            float cov_sq = (float)kk * 1.0f + (float)rem * 0.25f;
            config->optim.eq16_params.err_radius = sqrtf(cov_sq) * target_scale;
        }
    }
    else if (config->optim.use_sq16 && dataset_prof->use_sq16)
    {
        config->optim.sq16_params = dataset_prof->sq16_params;
        config->optim.sq16_calibrated = 1;
        long fdim = dataset_prof->dim;
        if (config->optim.sq16_ratio > 0.0 && config->algo.rlim > 0.0 && fdim > 0)
        {
            float target_scale = (float)((config->optim.sq16_ratio * config->algo.rlim) /
                                         sqrt((double)fdim));
            float center = 0.5f * (config->optim.sq16_params.min_val +
                                   config->optim.sq16_params.max_val);
            config->optim.sq16_params.scale = target_scale;
            config->optim.sq16_params.inv_scale = 1.0f / target_scale;
            config->optim.sq16_params.min_val = center - 16384.0f * target_scale;
            config->optim.sq16_params.max_val = center + 16383.0f * target_scale;
            config->optim.sq16_params.err_radius = sqrtf((float)fdim) * target_scale * 0.5f;
        }
    }
    else if (config->optim.use_sq8 && dataset_prof->use_sq8)
    {
        config->optim.sq8_params = dataset_prof->sq8_params;
        config->optim.sq8_calibrated = 1;
    }
}

/**
 * apply_dataset_profile() - Apply all inferred dataset profile settings to config.
 * @config:       Cluster configuration.
 * @dataset_prof: Loaded profile.
 * @rlim_set:     Pointer to rlim set flag.
 */
static void apply_dataset_profile(
    ClusterConfig     *config,
    const GricProfile *dataset_prof,
    int               *rlim_set)
{
    if (!*rlim_set)
    {
        apply_profile_rlim(config, dataset_prof);
        *rlim_set = 1;
    }

    apply_profile_quantization(config, dataset_prof);

    if (config->input.tile_grid_x == 0 && config->input.tile_grid_y == 0 &&
        (dataset_prof->tiles_x > 1 || dataset_prof->tiles_y > 1))
    {
        config->input.tile_grid_x = dataset_prof->tiles_x;
        config->input.tile_grid_y = dataset_prof->tiles_y;
    }

    if (config->optim.pred_mode == 0 && dataset_prof->pred_enabled)
    {
        config->optim.pred_mode = 1;
        config->optim.pred_len = dataset_prof->pred_len;
        config->optim.pred_h = dataset_prof->pred_h;
    }

    if (config->optim.te4_mode == 0 && config->optim.te5_mode == 0)
    {
        if (dataset_prof->te5_enabled)
        {
            config->optim.te5_mode = 1;
        }
        else if (dataset_prof->te4_enabled)
        {
            config->optim.te4_mode = 1;
        }
    }

    if (config->algo.tm_mixing_coeff == 0.0 && dataset_prof->tm_mixing_coeff > 0.0)
    {
        config->algo.tm_mixing_coeff = dataset_prof->tm_mixing_coeff;
    }
    if (config->optim.sparse_dcc_mode == 0 && dataset_prof->sparse_dcc_enabled)
    {
        config->optim.sparse_dcc_mode = 1;
    }
    if (config->optim.entropy_mode == 0 && dataset_prof->entropy_enabled)
    {
        config->optim.entropy_mode = 1;
        config->optim.entropy_gate_bits = dataset_prof->entropy_gate;
    }
    if (config->optim.soft_bayesian_mode == 0 && dataset_prof->soft_bayesian_enabled)
    {
        config->optim.soft_bayesian_mode = 1;
        config->optim.soft_bayesian_sigma_coeff = dataset_prof->soft_bayesian_sigma_coeff;
    }
    if (config->algo.use_double == 0 && dataset_prof->recommend_double)
    {
        config->algo.use_double = 1;
    }
}

/**
 * sanitize_config_bounds() - Enforce non-negative bounds across configuration parameters.
 * @config: Configuration structure to sanitize.
 */
static void sanitize_config_bounds(
    ClusterConfig *config)
{
    if (config->optim.te4_mode < 0) config->optim.te4_mode = 0;
    if (config->optim.te4_max_anchors < 0) config->optim.te4_max_anchors = 0;
    if (config->optim.te5_mode < 0) config->optim.te5_mode = 0;
    if (config->optim.entropy_mode < 0) config->optim.entropy_mode = 0;
    if (config->optim.sparse_dcc_mode < 0) config->optim.sparse_dcc_mode = 0;
    if (config->optim.soft_bayesian_mode < 0) config->optim.soft_bayesian_mode = 0;
    if (config->optim.pred_mode < 0) config->optim.pred_mode = 0;
    if (config->algo.tm_mixing_coeff < 0.0) config->algo.tm_mixing_coeff = 0.0;
    if (config->input.tile_grid_x < 0) config->input.tile_grid_x = 0;
    if (config->input.tile_grid_y < 0) config->input.tile_grid_y = 0;
}

int cluster_cli_parse(
    int            argc,
    char         **argv,
    ClusterConfig *config,
    GricProfile   *profile_out,
    char         **cmdline_out)
{
    char *cmdline = record_cmdline(argc, argv);
    int help_rc = check_help_flags(argc, argv);
    if (help_rc != 0)
    {
        free(cmdline);
        return help_rc;
    }

    cluster_cli_init_config_defaults(config);

    int rlim_set = 0;
    if (parse_cluster_options(argc, argv, config, &rlim_set) != 0)
    {
        free(cmdline);
        return 1;
    }

    if (config->input.fits_filename == NULL)
    {
        fprintf(stderr, "Error: Missing input file or stream name.\n");
        if (!config->input.scandist_mode)
        {
            print_usage(argv[0]);
        }
        free(cmdline);
        print_args_on_error(argc, argv);
        return 1;
    }

    GricProfile dataset_prof;
    memset(&dataset_prof, 0, sizeof(GricProfile));
    if (!config->optim.no_prof)
    {
        if (locate_or_probe_profile(config, &dataset_prof, rlim_set))
        {
            apply_dataset_profile(config, &dataset_prof, &rlim_set);
        }
    }

    sanitize_config_bounds(config);

    if (!config->input.scandist_mode && !rlim_set)
    {
        fprintf(stderr, "Error: Missing rlim parameter (no profile and no rlim provided).\n");
        print_usage(argv[0]);
        free(cmdline);
        gric_profile_free(&dataset_prof);
        return 1;
    }

    if (profile_out != NULL)
    {
        *profile_out = dataset_prof;
    }
    else
    {
        gric_profile_free(&dataset_prof);
    }

    if (cmdline_out != NULL)
    {
        *cmdline_out = cmdline;
    }
    else
    {
        free(cmdline);
    }
    return 0;
}
