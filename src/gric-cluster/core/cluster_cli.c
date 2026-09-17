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

        // Set defaults
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
        config->optim.use_sq8 = 1; // Enabled by default for 8-bit metric pre-filtering
        config->optim.use_sq16 = 0;
        config->optim.use_memo = 1; // Enabled by default when SQ16 is used
        config->optim.sq16_ratio = 0.05; // Default ratio: sqrt(D)*scale <= 0.05*rlim
        config->optim.use_batch_dist = 1; // Enabled by default for multi-vector SIMD batching
    
        // Tiling defaults (M=1, no tiling)
        config->input.tile_grid_x = 0;
        config->input.tile_grid_y = 0;
        config->input.tile_map_file = NULL;
        config->input.tile_config_file = NULL;
        config->input.retrieval_window = 1000;
    
        // Output defaults (enabled by default: dcc, anchors, counts, membership)
        config->output.output_dcc = 1;
        config->output.output_tm = 0;
        config->output.output_anchors = 1;
        config->output.output_counts = 1;
        config->output.output_membership = 1;
        config->output.output_evals = 1;
        config->output.output_discarded = 0;
        config->output.output_clustered = 0;
        config->output.output_clusters = 0;
        config->output.no_txt = 1; // Binary (.bin) format is default; use -txt to write ASCII text
    }

int cluster_cli_parse(
    int            argc,
    char         **argv,
    ClusterConfig *config,
    GricProfile   *profile_out,
    char         **cmdline_out)
{
    char *cmdline = (char *)malloc(8192);
    if (cmdline)
    {
        size_t cmdline_pos = 0;
        size_t cmdline_remaining = 8192;
        for (int arg_idx = 0; arg_idx < argc; arg_idx++)
        {
            int written = snprintf(cmdline + cmdline_pos, cmdline_remaining,
                                   "%s%s", argv[arg_idx],
                                   (arg_idx < argc - 1) ? " " : "");
            if (written < 0 || (size_t)written >= cmdline_remaining)
            {
                fprintf(stderr, "Warning: command line string truncated\n");
                break;
            }
            cmdline_pos += (size_t)written;
            cmdline_remaining -= (size_t)written;
        }
    }

    // Check for help option early
    for (int arg_idx = 1; arg_idx < argc; arg_idx++)
    {
        if (strcmp(argv[arg_idx], "-h") == 0 || strcmp(argv[arg_idx], "--help") == 0)
        {
            if (arg_idx + 1 < argc)
            {
                print_help_keyword(argv[arg_idx + 1]);
            }
            else
            {
                print_help(argv[0]);
            }
            if (cmdline)
                free(cmdline);
            return 2;
        }
    }

    if (argc < 2)
    {
        print_usage(argv[0]);
        if (cmdline)
            free(cmdline);
        return 1;
    }

    cluster_cli_init_config_defaults(config);

        int arg_idx = 1;
        int rlim_set = 0;
        const char *confw_filename = NULL;
    
        while (arg_idx < argc)
        {
            char *key = argv[arg_idx];
            char *val = (arg_idx + 1 < argc) ? argv[arg_idx + 1] : NULL;
    
            // Check for -conf and -confw explicitly or handle via apply_option?
            // apply_option handles config-specific logic mostly, but -conf is meta.
    
            if (strcmp(key, "-conf") == 0)
            {
                if (!val)
                {
                    fprintf(stderr, "Error: -conf requires a filename\n");
                    if (cmdline)
                        free(cmdline);
                    return 1;
                }
                if (read_config_file(val, config) != 0)
                {
                    fprintf(stderr, "Error: Could not read config file %s\n", val);
                    if (cmdline)
                        free(cmdline);
                    return 1;
                }
                // If config loaded rlim, we can consider it set?
                // But we don't know for sure if it was set in config->
                // However, our smart positional logic below handles this.
                arg_idx += 2;
                continue;
            }
    
            if (strcmp(key, "-confw") == 0)
            {
                if (!val)
                {
                    fprintf(stderr, "Error: -confw requires a filename\n");
                    if (cmdline)
                        free(cmdline);
                    return 1;
                }
                confw_filename = val;
                arg_idx += 2;
                continue;
            }
    
            int res = apply_option(config, key, val);
            if (res >= 0)
            {
                arg_idx += (1 + res);
                // If explicit -rlim was used, mark it
                if (strcmp(key, "-rlim") == 0 || strcmp(key, "rlim") == 0)
                    rlim_set = 1;
                continue;
            }
    
            // Positional or Unknown
            if (key[0] == '-' && !isdigit(key[1]))
            {
                fprintf(stderr, "Error: Unknown option: %s\n", key);
                print_usage(argv[0]);
                if (cmdline)
                    free(cmdline);
                print_args_on_error(argc, argv);
                return 1;
            }
    
            // Positional logic
            if (!config->input.scandist_mode && !rlim_set)
            {
                // Try parse as rlim
                char *endptr;
                double v = strtod(key, &endptr);
                if (*endptr == '\0')
                {
                    config->algo.rlim = v;
                    rlim_set = 1;
                }
                else if (key[0] == 'a' && isdigit(key[1]))
                {
                    config->algo.auto_rlim_factor = atof(key + 1);
                    config->algo.auto_rlim_mode = 1;
                    rlim_set = 1;
                }
                else
                {
                    // Not a number, assume filename
                    if (config->input.fits_filename != NULL)
                    {
                        fprintf(stderr,
                                "Error: Multiple input files specified "
                                "(already have '%s', found '%s')\n",
                                config->input.fits_filename, key);
                        if (cmdline)
                            free(cmdline);
                        print_args_on_error(argc, argv);
                        return 1;
                    }
                    config->input.fits_filename = key;
                }
            }
            else
            {
                if (config->input.fits_filename != NULL)
                {
                    fprintf(stderr,
                            "Error: Multiple input files specified "
                            "(already have '%s', found '%s')\n",
                            config->input.fits_filename, key);
                    if (cmdline)
                        free(cmdline);
                    print_args_on_error(argc, argv);
                    return 1;
                }
                config->input.fits_filename = key;
            }
            arg_idx++;
        }
    
        if (confw_filename)
        {
            if (write_config_file(confw_filename, config) != 0)
            {
                fprintf(stderr, "Error: Could not write config file %s\n", confw_filename);
                if (cmdline)
                    free(cmdline);
                return 1;
            }
            printf("Configuration written to %s\n", confw_filename);
        }
    
        if (!config->input.fits_filename)
        {
            fprintf(stderr, "Error: Missing input file or stream name.\n");
            if (!config->input.scandist_mode)
                print_usage(argv[0]);
            if (cmdline)
                free(cmdline);
            print_args_on_error(argc, argv);
            return 1;
        }
    
        GricProfile dataset_prof;
        memset(&dataset_prof, 0, sizeof(GricProfile));
        int has_prof = 0;
        char located_prof_path[1024] = "";
    
        if (!config->optim.no_prof && config->input.fits_filename != NULL)
        {
            if (config->optim.prof_filename != NULL)
            {
                snprintf(located_prof_path, sizeof(located_prof_path), "%s",
                         config->optim.prof_filename);
                has_prof = (gric_profile_read_json(located_prof_path, &dataset_prof) == 0);
            }
            else if (gric_profile_find_auto(config->input.fits_filename,
                                            located_prof_path, sizeof(located_prof_path)))
            {
                has_prof = (gric_profile_read_json(located_prof_path, &dataset_prof) == 0);
            }

            /* Automatic micro-probe fallback for file inputs when SQ16 is requested
             * or rlim is omitted. */
            if (!has_prof && !config->optim.no_prof && !config->input.stream_input_mode &&
                !config->input.filelist_mode && (config->optim.use_sq16 || !rlim_set))
            {
                ProbeConfig pcfg;
                memset(&pcfg, 0, sizeof(pcfg));
                snprintf(pcfg.dataset_path, sizeof(pcfg.dataset_path), "%s",
                         config->input.fits_filename);
                pcfg.sample_limit = 500;
                pcfg.verbose_level = 0;
                pcfg.use_double = config->algo.use_double;
                ProbeResults pres;
                memset(&pres, 0, sizeof(pres));
                if (probe_run(&pcfg, &pres) == 0)
                {
                    dataset_prof = pres.profile;
                    memset(&pres.profile, 0, sizeof(pres.profile));
                    dataset_prof.tiles_x = 1;
                    dataset_prof.tiles_y = 1;
                    has_prof = 1;

                    char auto_save_path[1024];
                    snprintf(auto_save_path, sizeof(auto_save_path), "%s.gricprof",
                             config->input.fits_filename);
                    if (gric_profile_write_json(auto_save_path, &dataset_prof) == 0)
                    {
                        printf("%s[INFO]%s Auto-profiled dataset (500 frames) -> "
                               "generated %s%s%s\n",
                               ANSI_BOLD_GREEN, ANSI_COLOR_RESET,
                               ANSI_BOLD, auto_save_path, ANSI_COLOR_RESET);
                    }
                    else
                    {
                        printf("%s[INFO]%s Auto-profiled dataset (500 frames) in memory\n",
                               ANSI_BOLD_GREEN, ANSI_COLOR_RESET);
                    }
                }
            }

            if (has_prof)
            {
                if (located_prof_path[0] != '\0')
                {
                    printf("%s[INFO]%s Auto-loaded dataset profile: %s%s%s\n",
                           ANSI_BOLD_GREEN, ANSI_COLOR_RESET,
                           ANSI_BOLD, located_prof_path, ANSI_COLOR_RESET);
                }

                if (!rlim_set)
                {
                    if (strcasecmp(config->optim.preset_name, "p01") == 0 ||
                        strcasecmp(config->optim.preset_name, "1%") == 0 ||
                        strcasecmp(config->optim.preset_name, "ultrafine") == 0)
                    {
                        config->algo.rlim = dataset_prof.rlim_p01;
                    }
                    else if (strcasecmp(config->optim.preset_name, "p03") == 0 ||
                             strcasecmp(config->optim.preset_name, "3%") == 0 ||
                             strcasecmp(config->optim.preset_name, "vfine") == 0)
                    {
                        config->algo.rlim = dataset_prof.rlim_p03;
                    }
                    else if (strcasecmp(config->optim.preset_name, "fine") == 0 ||
                             strcasecmp(config->optim.preset_name, "p05") == 0 ||
                             strcasecmp(config->optim.preset_name, "5%") == 0)
                    {
                        config->algo.rlim = dataset_prof.rlim_fine;
                    }
                    else if (strcasecmp(config->optim.preset_name, "coarse") == 0 ||
                             strcasecmp(config->optim.preset_name, "p25") == 0 ||
                             strcasecmp(config->optim.preset_name, "25%") == 0)
                    {
                        config->algo.rlim = dataset_prof.rlim_coarse;
                    }
                    else
                    {
                        config->algo.rlim = dataset_prof.rlim_recommended;
                    }
                    rlim_set = 1;
                }

                if (config->optim.use_sq16 && dataset_prof.use_sq16)
                {
                    config->optim.sq16_params = dataset_prof.sq16_params;
                    config->optim.sq16_calibrated = 1;
                    long fdim = dataset_prof.dim;
                    if (config->optim.sq16_ratio > 0.0 && config->algo.rlim > 0.0 && fdim > 0)
                    {
                        float target_scale = (float)((config->optim.sq16_ratio *
                                                      config->algo.rlim) /
                                                     sqrt((double)fdim));
                        float center = 0.5f * (config->optim.sq16_params.min_val +
                                               config->optim.sq16_params.max_val);
                        config->optim.sq16_params.scale = target_scale;
                        config->optim.sq16_params.inv_scale = 1.0f / target_scale;
                        config->optim.sq16_params.min_val = center - 16384.0f * target_scale;
                        config->optim.sq16_params.max_val = center + 16383.0f * target_scale;
                        config->optim.sq16_params.err_radius =
                            sqrtf((float)fdim) * target_scale * 0.5f;
                    }
                }
                else if (config->optim.use_sq8 && dataset_prof.use_sq8)
                {
                    config->optim.sq8_params = dataset_prof.sq8_params;
                    config->optim.sq8_calibrated = 1;
                }
    
                if (config->input.tile_grid_x == 0 && config->input.tile_grid_y == 0 &&
                    (dataset_prof.tiles_x > 1 || dataset_prof.tiles_y > 1))
                {
                    config->input.tile_grid_x = dataset_prof.tiles_x;
                    config->input.tile_grid_y = dataset_prof.tiles_y;
                }
    
                if (config->optim.pred_mode == 0 && dataset_prof.pred_enabled)
                {
                    config->optim.pred_mode = 1;
                    config->optim.pred_len = dataset_prof.pred_len;
                    config->optim.pred_h = dataset_prof.pred_h;
                }
    
                if (config->optim.te4_mode == 0 && config->optim.te5_mode == 0)
                {
                    if (dataset_prof.te5_enabled)
                    {
                        config->optim.te5_mode = 1;
                    }
                    else if (dataset_prof.te4_enabled)
                    {
                        config->optim.te4_mode = 1;
                    }
                }
    
                if (config->algo.tm_mixing_coeff == 0.0 && dataset_prof.tm_mixing_coeff > 0.0)
                {
                    config->algo.tm_mixing_coeff = dataset_prof.tm_mixing_coeff;
                }
    
                if (config->optim.sparse_dcc_mode == 0 && dataset_prof.sparse_dcc_enabled)
                {
                    config->optim.sparse_dcc_mode = 1;
                }
    
                if (config->optim.entropy_mode == 0 && dataset_prof.entropy_enabled)
                {
                    config->optim.entropy_mode = 1;
                    config->optim.entropy_gate_bits = dataset_prof.entropy_gate;
                }
    
                if (config->optim.soft_bayesian_mode == 0 && dataset_prof.soft_bayesian_enabled)
                {
                    config->optim.soft_bayesian_mode = 1;
                    config->optim.soft_bayesian_sigma_coeff =
                        dataset_prof.soft_bayesian_sigma_coeff;
                }
    
                if (config->algo.use_double == 0 && dataset_prof.recommend_double)
                {
                    config->algo.use_double = 1;
                }
            } // if (has_prof)
        } // if (!config->optim.no_prof)
    
        if (config->optim.te4_mode < 0) config->optim.te4_mode = 0;
        if (config->optim.te5_mode < 0) config->optim.te5_mode = 0;
        if (config->optim.entropy_mode < 0) config->optim.entropy_mode = 0;
        if (config->optim.sparse_dcc_mode < 0) config->optim.sparse_dcc_mode = 0;
        if (config->optim.soft_bayesian_mode < 0) config->optim.soft_bayesian_mode = 0;
    
        if (!config->input.scandist_mode && !rlim_set)
        {
            fprintf(stderr,
                    "Error: Missing rlim parameter (no profile found and no rlim provided).\n");
            print_usage(argv[0]);
            if (cmdline)
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

    if (cmdline_out)
    {
        *cmdline_out = cmdline;
    }
    else if (cmdline)
    {
        free(cmdline);
    }

    return 0;
}
