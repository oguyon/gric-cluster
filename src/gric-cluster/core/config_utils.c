/**
 * @file config_utils.c
 * @brief Declarative configuration parsing and serialization for gric-cluster.
 */

#include "config_utils.h"
#include "cli_opt.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int is_negation_key(
    const char *key)
{
    if (key == NULL)
    {
        return 0;
    }
    while (*key == '-')
    {
        key++;
    }
    return (strncmp(key, "no-", 3) == 0 || strncmp(key, "no_", 3) == 0 ||
            strncmp(key, "nosq", 4) == 0 || strncmp(key, "noeq", 4) == 0 ||
            strncmp(key, "nomemo", 6) == 0 || strncmp(key, "nobatch", 7) == 0 ||
            strncmp(key, "noprof", 6) == 0 || strncmp(key, "notm", 4) == 0);
}

static int cb_rlim(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)key;
    if (val == NULL)
    {
        return -1;
    }
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        if (val[0] == 'a')
        {
            cfg->algo.auto_rlim_factor = atof(val + 1);
            cfg->algo.auto_rlim_mode = 1;
        }
        else
        {
            cfg->algo.rlim = atof(val);
        }
    }
    return 1;
}

static int cb_sq8(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)val;
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        if (is_negation_key(key))
        {
            cfg->optim.use_sq8 = 0;
        }
        else
        {
            cfg->optim.use_sq8 = 1;
            cfg->optim.use_sq16 = 0;
            cfg->optim.use_eq16 = 0;
        }
    }
    return 0;
}

static int cb_sq16(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)val;
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        if (is_negation_key(key))
        {
            cfg->optim.use_sq16 = 0;
        }
        else
        {
            cfg->optim.use_sq16 = 1;
            cfg->optim.use_sq8 = 0;
            cfg->optim.use_eq16 = 0;
        }
    }
    return 0;
}

static int cb_eq16(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)val;
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        if (is_negation_key(key))
        {
            cfg->optim.use_eq16 = 0;
        }
        else
        {
            cfg->optim.use_eq16 = 1;
            cfg->optim.use_sq8 = 0;
            cfg->optim.use_sq16 = 0;
        }
    }
    return 0;
}

static int cb_gpu(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)val;
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        if (is_negation_key(key))
        {
            cfg->optim.use_gpu = 0;
            cfg->optim.use_gpu_pass1 = 0;
        }
        else
        {
            cfg->optim.use_gpu = 1;
            cfg->optim.use_gpu_pass1 = 1;
        }
    }
    return 0;
}

static int cb_cpu(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)key;
    (void)val;
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        cfg->optim.use_gpu = 0;
        cfg->optim.use_gpu_pass1 = 0;
    }
    return 0;
}

static int cb_gpu_pass1(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)key;
    (void)val;
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        cfg->optim.use_gpu = 1;
        cfg->optim.use_gpu_pass1 = 1;
    }
    return 0;
}

static int cb_gpu_pass2(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)key;
    (void)val;
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        cfg->optim.use_gpu = 1;
        cfg->optim.use_gpu_pass1 = 0;
    }
    return 0;
}

static int cb_cpu_pass1(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)key;
    (void)val;
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        cfg->optim.use_gpu_pass1 = 0;
    }
    return 0;
}

static int cb_gpu_batch_size(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)key;
    if (val == NULL)
    {
        return -1;
    }
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        cfg->optim.use_gpu = 1;
        cfg->optim.use_gpu_pass1 = 1;
        cfg->optim.gpu_micro_batch_size = atoi(val);
    }
    return 1;
}

static int cb_gpu_device(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)key;
    if (val == NULL)
    {
        return -1;
    }
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        cfg->optim.use_gpu = 1;
        cfg->optim.gpu_device_id = atoi(val);
    }
    return 1;
}

static int cb_gpu_prune(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)key;
    (void)val;
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        cfg->optim.use_gpu = 1;
        cfg->optim.gpu_prune_mode = 1;
    }
    return 0;
}

static int cb_gpu_gemm(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)key;
    (void)val;
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        cfg->optim.use_gpu = 1;
        cfg->optim.gpu_prune_mode = 0;
    }
    return 0;
}

static int cb_gpu_auto_prune(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)key;
    (void)val;
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        cfg->optim.use_gpu = 1;
        cfg->optim.gpu_prune_mode = -1;
    }
    return 0;
}

static int cb_tiles(
    const char *key,
    const char *val,
    void       *ctx)
{
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (is_negation_key(key))
    {
        if (cfg != NULL)
        {
            cfg->input.tile_grid_x = -1;
            cfg->input.tile_grid_y = -1;
        }
        return 0;
    }
    if (val == NULL)
    {
        return -1;
    }
    if (cfg != NULL)
    {
        if (sscanf(val, "%dx%d", &cfg->input.tile_grid_x, &cfg->input.tile_grid_y) != 2)
        {
            int n = atoi(val);
            if (n > 0)
            {
                int side = 1;
                while (side * side < n)
                {
                    side++;
                }
                cfg->input.tile_grid_x = side;
                cfg->input.tile_grid_y = (n + side - 1) / side;
            }
        }
    }
    return 1;
}

static int cb_xtile(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)key;
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (val != NULL && (strcmp(val, "1") == 0 || strcmp(val, "2") == 0))
    {
        if (cfg != NULL)
        {
            cfg->optim.xtile_mode = atoi(val);
        }
        return 1;
    }
    if (cfg != NULL)
    {
        cfg->optim.xtile_mode = 2;
    }
    return 0;
}

static int cb_jtf(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)val;
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        cfg->optim.disable_pass2 = is_negation_key(key) ? 1 : 0;
    }
    return 0;
}

static int cb_pred(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)val;
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (is_negation_key(key))
    {
        if (cfg != NULL)
        {
            cfg->optim.pred_mode = -1;
        }
        return 0;
    }
    int mode = 1;
    const char *p = key;
    while (*p == '-')
    {
        p++;
    }
    if (strncmp(p, "predf", 5) == 0)
    {
        mode = 2;
    }
    if (cfg != NULL)
    {
        cfg->optim.pred_mode = mode;
        const char *params = strchr(key, '[');
        if (params != NULL)
        {
            int l, h, n;
            if (sscanf(params + 1, "%d,%d,%d", &l, &h, &n) == 3)
            {
                cfg->optim.pred_len = l;
                cfg->optim.pred_h = h;
                cfg->optim.pred_n = n;
            }
        }
    }
    return 0;
}

static int cb_maxcl_strategy(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)key;
    if (val == NULL)
    {
        return -1;
    }
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        if (strcmp(val, "stop") == 0)
        {
            cfg->algo.maxcl_strategy = MAXCL_STOP;
        }
        else if (strcmp(val, "discard") == 0)
        {
            cfg->algo.maxcl_strategy = MAXCL_DISCARD;
        }
        else if (strcmp(val, "merge") == 0)
        {
            cfg->algo.maxcl_strategy = MAXCL_MERGE;
        }
        else
        {
            fprintf(stderr, "Warning: Unknown maxcl_strategy '%s'\n", val);
        }
    }
    return 1;
}

static int cb_txt(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)val;
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        cfg->output.no_txt = is_negation_key(key) ? 1 : 0;
    }
    return 0;
}

static int cb_veryverbose(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)key;
    (void)val;
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        cfg->output.verbose_level = 2;
    }
    return 0;
}

static int cb_entropy(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)val;
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        cfg->optim.entropy_mode = is_negation_key(key) ? -1 : 1;
    }
    return 0;
}

static int cb_te4(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)val;
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        cfg->optim.te4_mode = is_negation_key(key) ? -1 : 1;
    }
    return 0;
}

static int cb_te5(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)val;
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        cfg->optim.te5_mode = is_negation_key(key) ? -1 : 1;
    }
    return 0;
}

static int cb_sparse_dcc(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)val;
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        cfg->optim.sparse_dcc_mode = is_negation_key(key) ? -1 : 1;
    }
    return 0;
}

static int cb_soft_bayesian(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)val;
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        cfg->optim.soft_bayesian_mode = is_negation_key(key) ? -1 : 1;
    }
    return 0;
}

static int cb_tm(
    const char *key,
    const char *val,
    void       *ctx)
{
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (is_negation_key(key))
    {
        if (cfg != NULL)
        {
            cfg->algo.tm_mixing_coeff = -1.0;
        }
        return 0;
    }
    if (val == NULL)
    {
        return -1;
    }
    if (cfg != NULL)
    {
        cfg->algo.tm_mixing_coeff = atof(val);
    }
    return 1;
}

static int cb_outdir(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)key;
    if (val == NULL)
    {
        return -1;
    }
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        cfg->output.user_outdir = strdup(val);
    }
    return 1;
}

static int cb_input(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)key;
    if (val == NULL)
    {
        return -1;
    }
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        cfg->input.fits_filename = strdup(val);
    }
    return 1;
}

static int cb_prof(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)key;
    if (val == NULL)
    {
        return -1;
    }
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        cfg->optim.prof_filename = strdup(val);
    }
    return 1;
}

static int cb_shm(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)key;
    if (val == NULL)
    {
        return -1;
    }
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        cfg->output.shm_filename = strdup(val);
    }
    return 1;
}

static int cb_tilemap(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)key;
    if (val == NULL)
    {
        return -1;
    }
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        cfg->input.tile_map_file = strdup(val);
    }
    return 1;
}

static int cb_tileconf(
    const char *key,
    const char *val,
    void       *ctx)
{
    (void)key;
    if (val == NULL)
    {
        return -1;
    }
    ClusterConfig *cfg = (ClusterConfig *)ctx;
    if (cfg != NULL)
    {
        cfg->input.tile_config_file = strdup(val);
    }
    return 1;
}

const struct gric_opt *cluster_get_options(
    ClusterConfig *config,
    size_t        *nopts)
{
    static struct gric_opt opts[] = {
        /* Core Algorithm */
        {"rlim", 'r', GRIC_OPT_CUSTOM, (void *)cb_rlim, 0, "", "<val>",
         "Cluster radius cutoff limit", "Algorithm", "radius", 0},
        {"dprob", '\0', GRIC_OPT_DOUBLE, NULL, 0, "0.01", "<val>",
         "Delta probability convergence threshold", "Algorithm", "deltaprob", 0},
        {"maxcl", '\0', GRIC_OPT_INT, NULL, 0, "1000", "<N>",
         "Maximum number of clusters", "Algorithm", "maxnbclust,maxclusters", 0},
        {"maxcl-strategy", '\0', GRIC_OPT_CUSTOM, (void *)cb_maxcl_strategy, 0, "stop", "<strat>",
         "Action when maxcl is reached (stop, discard, merge)", "Algorithm", "maxcl_strategy", 0},
        {"discard-frac", '\0', GRIC_OPT_DOUBLE, NULL, 0, "0.5", "<val>",
         "Fraction of smallest clusters to discard", "Algorithm", "discard_frac", 0},
        {"swap-remove", '\0', GRIC_OPT_FLAG, NULL, 0, "0", NULL,
         "Enable O(1) swap-with-last cluster eviction", "Algorithm", "swap_remove", 0},
        {"double", '\0', GRIC_OPT_FLAG, NULL, 0, "0", NULL,
         "Force double-precision distance math", "Algorithm", NULL, 0},
        {"tm", '\0', GRIC_OPT_CUSTOM, (void *)cb_tm, 0, "0.0", "<val>",
         "Temporal mixing coefficient", "Algorithm", "tm_mix,no-tm,no_tm", 1},
        {"pass2-nearest", '\0', GRIC_OPT_FLAG, NULL, 0, "0", NULL,
         "Reassign frames to nearest centroid in pass 2", "Algorithm",
         "pass2nearest,pass2_nearest,reassign,second_pass,second-pass,reallocate,no_reassign", 1},

        /* Input / Output */
        {"input", 'i', GRIC_OPT_CUSTOM, (void *)cb_input, 0, "", "<path>",
         "Input FITS/stream/text file path", "Input/Output", "in", 0},
        {"outdir", 'o', GRIC_OPT_CUSTOM, (void *)cb_outdir, 0, "", "<dir>",
         "Directory to store output clusters and sidecars", "Input/Output", "out,output", 0},
        {"maxim", '\0', GRIC_OPT_INT64, NULL, 0, "1000000", "<N>",
         "Maximum number of input frames to process", "Input/Output", "maxnbfr,maxframes", 0},
        {"progress", 'p', GRIC_OPT_FLAG, NULL, 0, "1", NULL,
         "Display live console progress bar", "Input/Output", NULL, 0},
        {"verbose", 'v', GRIC_OPT_FLAG, NULL, 0, "0", NULL,
         "Enable verbose diagnostic logging", "Input/Output", NULL, 0},
        {"veryverbose", '\0', GRIC_OPT_CUSTOM, (void *)cb_veryverbose, 0, "0", NULL,
         "Enable very verbose diagnostic logging", "Input/Output", "vv", 0},
        {"fitsout", '\0', GRIC_OPT_FLAG, NULL, 0, "0", NULL,
         "Write cluster centroids to FITS format", "Input/Output", NULL, 0},
        {"pngout", '\0', GRIC_OPT_FLAG, NULL, 0, "0", NULL,
         "Render cluster images as PNG files", "Input/Output", NULL, 0},
        {"filelist", '\0', GRIC_OPT_FLAG, NULL, 0, "0", NULL,
         "Read input file as a list of image paths", "Input/Output", NULL, 0},
        {"stream", '\0', GRIC_OPT_FLAG, NULL, 0, "0", NULL,
         "Read input frames from ImageStreamIO shared memory", "Input/Output", NULL, 0},
        {"cnt2sync", '\0', GRIC_OPT_FLAG, NULL, 0, "0", NULL,
         "Synchronize stream reading via cnt2 counter", "Input/Output", NULL, 0},
        {"tm-out", '\0', GRIC_OPT_FLAG, NULL, 0, "0", NULL,
         "Write transition matrix sidecar file", "Input/Output", "tm_out", 0},
        {"dcc", '\0', GRIC_OPT_FLAG, NULL, 0, "1", NULL,
         "Write inter-cluster distance matrix", "Input/Output", NULL, 1},
        {"dcc-sq16", '\0', GRIC_OPT_FLAG, NULL, 0, "0", NULL,
         "Write quantized 16-bit DCC matrix", "Input/Output", NULL, 1},
        {"txt", '\0', GRIC_OPT_CUSTOM, (void *)cb_txt, 0, "0", NULL,
         "Write output metrics in ASCII text format", "Input/Output", "no-txt,no_txt", 1},
        {"anchors", '\0', GRIC_OPT_FLAG, NULL, 0, "1", NULL,
         "Save cluster anchor frame coordinates", "Input/Output", NULL, 1},
        {"counts", '\0', GRIC_OPT_FLAG, NULL, 0, "1", NULL,
         "Save per-cluster frame count statistics", "Input/Output", NULL, 1},
        {"membership", '\0', GRIC_OPT_FLAG, NULL, 0, "1", NULL,
         "Write frame-to-cluster membership mapping", "Input/Output", NULL, 1},
        {"evals", '\0', GRIC_OPT_FLAG, NULL, 0, "1", NULL,
         "Record per-frame distance evaluation counts", "Input/Output", "trace", 1},
        {"discarded", '\0', GRIC_OPT_FLAG, NULL, 0, "0", NULL,
         "Write list of discarded outlier frame indices", "Input/Output", NULL, 0},
        {"clustered", '\0', GRIC_OPT_FLAG, NULL, 0, "0", NULL,
         "Write list of clustered frame indices", "Input/Output", NULL, 0},
        {"clusters", '\0', GRIC_OPT_FLAG, NULL, 0, "0", NULL,
         "Write cluster center centroids", "Input/Output", NULL, 0},
        {"scandist", '\0', GRIC_OPT_FLAG, NULL, 0, "0", NULL,
         "Run in distance scanning mode without clustering", "Input/Output", NULL, 0},
        {"shm", '\0', GRIC_OPT_CUSTOM, (void *)cb_shm, 0, "", "<name>",
         "Write cluster streams to ImageStreamIO shared memory", "Input/Output", "shm-file", 0},
        {"avg", '\0', GRIC_OPT_FLAG, NULL, 0, "0", NULL,
         "Compute centroid average across assigned frames", "Input/Output", NULL, 0},
        {"distall", '\0', GRIC_OPT_FLAG, NULL, 0, "0", NULL,
         "Calculate all pairwise frame distances", "Input/Output", NULL, 0},

        /* Quantization & FastScan */
        {"sq8", '\0', GRIC_OPT_CUSTOM, (void *)cb_sq8, 0, "auto", NULL,
         "Scalar 8-bit quantization", "Quantization", "sq_8,nosq8", 1},
        {"sq16", '\0', GRIC_OPT_CUSTOM, (void *)cb_sq16, 0, "auto", NULL,
         "Scalar 16-bit quantization", "Quantization", "sq_16,nosq16", 1},
        {"eq16", '\0', GRIC_OPT_CUSTOM, (void *)cb_eq16, 0, "auto", NULL,
         "Exact 16-bit block lattice quantization", "Quantization", "eq_16,noeq16", 1},
        {"eq16-adc", '\0', GRIC_OPT_FLAG, NULL, 0, "1", NULL,
         "Asymmetric distance computation for EQ16", "Quantization", "eq16_adc", 1},
        {"sq16-ratio", '\0', GRIC_OPT_DOUBLE, NULL, 0, "0.05", "<ratio>",
         "Scale ratio bound sqrt(D)*scale <= ratio*rlim", "Quantization",
         "eq16-ratio,sq16_ratio,eq16_ratio", 0},
        {"memo", '\0', GRIC_OPT_FLAG, NULL, 0, "1", NULL,
         "Quantized candidate memoization table", "Quantization", "use_memo,nomemo", 1},
        {"batch-dist", '\0', GRIC_OPT_FLAG, NULL, 0, "1", NULL,
         "Multi-vector SIMD batch distance evaluation", "Quantization",
         "batchdist,nobatchdist", 1},

        /* Optimization & Pruning */
        {"ncpu", '\0', GRIC_OPT_INT, NULL, 0, "1", "<N>",
         "Number of CPU worker threads", "Optimization", NULL, 0},
        {"gprob", '\0', GRIC_OPT_FLAG, NULL, 0, "0", NULL,
         "Enable geometric probability candidate ordering", "Optimization", NULL, 0},
        {"fmatcha", '\0', GRIC_OPT_DOUBLE, NULL, 0, "2.0", "<val>",
         "Geometric probability slope parameter A", "Optimization", NULL, 0},
        {"fmatchb", '\0', GRIC_OPT_DOUBLE, NULL, 0, "0.5", "<val>",
         "Geometric probability offset parameter B", "Optimization", NULL, 0},
        {"maxvis", '\0', GRIC_OPT_INT, NULL, 0, "1000", "<N>",
         "Maximum candidate visitors per frame in gprob", "Optimization", NULL, 0},
        {"te4", '\0', GRIC_OPT_CUSTOM, (void *)cb_te4, 0, "0", NULL,
         "Enable TE4 triangle inequality bounding", "Optimization", "no_te4", 1},
        {"te4-max-anchors", '\0', GRIC_OPT_INT, NULL, 0, "3", "<N>",
         "Maximum anchor pivots evaluated in TE4", "Optimization", "te4_max_anchors", 0},
        {"te5", '\0', GRIC_OPT_CUSTOM, (void *)cb_te5, 0, "0", NULL,
         "Enable TE5 triangle inequality bounding", "Optimization", "no_te5", 1},
        {"entropy", '\0', GRIC_OPT_CUSTOM, (void *)cb_entropy, 0, "0", NULL,
         "Information-theoretic entropy-guided search", "Optimization", "no_entropy", 1},
        {"entropy-max-targets", '\0', GRIC_OPT_INT, NULL, 0, "15", "<N>",
         "Maximum clusters in entropy priority queue", "Optimization", "entropy_max_targets", 0},
        {"entropy-min-prob", '\0', GRIC_OPT_FLOAT, NULL, 0, "0.001", "<val>",
         "Probability threshold for entropy candidate selection", "Optimization",
         "entropy_min_prob", 0},
        {"entropy-gate", '\0', GRIC_OPT_FLOAT, NULL, 0, "2.0", "<bits>",
         "Entropy reduction gate threshold in bits", "Optimization", "entropy_gate", 0},
        {"entropy-first-gate", '\0', GRIC_OPT_FLOAT, NULL, 0, "4.0", "<bits>",
         "Entropy reduction gate threshold for initial frame", "Optimization",
         "entropy_first_gate", 0},
        {"entropy-fast", '\0', GRIC_OPT_FLAG, NULL, 0, "0", NULL,
         "Accelerated entropy reduction mode", "Optimization", "entropy_fast", 0},
        {"entropy-leader", '\0', GRIC_OPT_FLAG, NULL, 0, "0", NULL,
         "Early termination when cluster probability exceeds cutoff", "Optimization",
         "entropy_leader", 0},
        {"entropy-leader-cutoff", '\0', GRIC_OPT_DOUBLE, NULL, 0, "0.50", "<val>",
         "Cutoff probability for leader shortcut", "Optimization", "entropy_leader_cutoff", 0},
        {"sparse-dcc", '\0', GRIC_OPT_CUSTOM, (void *)cb_sparse_dcc, 0, "0", NULL,
         "Lazy on-demand sparse cluster distance matrix", "Optimization", "sparse_dcc", 1},
        {"sparse-dcc-extra-evals", '\0', GRIC_OPT_INT, NULL, 0, "0", "<N>",
         "Additional neighbor evaluations for sparse DCC", "Optimization",
         "sparse_dcc_extra_evals", 0},
        {"soft-bayesian", '\0', GRIC_OPT_CUSTOM, (void *)cb_soft_bayesian, 0, "0", NULL,
         "Soft Bayesian probability updates", "Optimization", "soft_bayesian", 1},
        {"soft-bayesian-sigma", '\0', GRIC_OPT_DOUBLE, NULL, 0, "1.0", "<val>",
         "Soft Bayesian Gaussian standard deviation scale", "Optimization",
         "soft_bayesian_sigma,soft_sigma", 0},
        {"jtf", '\0', GRIC_OPT_CUSTOM, (void *)cb_jtf, 0, "0", NULL,
         "Pass 2 joint temporal filtering", "Optimization",
         "pass2,no_pass2,no-pass2,no_xtile", 1},
        {"xtile", '\0', GRIC_OPT_CUSTOM, (void *)cb_xtile, 0, "0", "[1|2]",
         "Cross-tile candidate sharing mode", "Optimization", NULL, 0},
        {"xtile-decay", '\0', GRIC_OPT_DOUBLE, NULL, 0, "1.0", "<val>",
         "Cross-tile distance score decay factor", "Optimization", "xtile_decay", 0},
        {"tiles", '\0', GRIC_OPT_CUSTOM, (void *)cb_tiles, 0, "1x1", "<WxH|N>",
         "Spatial image domain tiling grid", "Optimization", "no-tiles,no_tiles", 1},
        {"tilemap", '\0', GRIC_OPT_CUSTOM, (void *)cb_tilemap, 0, "", "<file>",
         "Custom tile assignment map image", "Optimization", NULL, 0},
        {"tileconf", '\0', GRIC_OPT_CUSTOM, (void *)cb_tileconf, 0, "", "<file>",
         "Per-tile configuration file", "Optimization", NULL, 0},
        {"retrieval-window", '\0', GRIC_OPT_INT, NULL, 0, "1000", "<N>",
         "Temporal sliding window for cross-tile frame lookup", "Optimization",
         "retrieval_window", 0},
        {"pred", '\0', GRIC_OPT_CUSTOM, (void *)cb_pred, 0, "0", "[len,h,n]",
         "Predictive pattern search", "Optimization", "predf,no-pred,no_pred", 1},
        {"prof", '\0', GRIC_OPT_CUSTOM, (void *)cb_prof, 0, "", "<file>",
         "Dataset distribution profile (.gricprof)", "Optimization", "profile", 0},
        {"no-prof", '\0', GRIC_OPT_FLAG, NULL, 0, "0", NULL,
         "Disable automatic loading of dataset profile", "Optimization", "noprof", 0},
        {"preset", '\0', GRIC_OPT_STRING, NULL, 0, "", "<preset>",
         "Quantization/search parameter preset name", "Optimization", NULL, 0},

        /* GPU Acceleration */
        {"gpu", '\0', GRIC_OPT_CUSTOM, (void *)cb_gpu, 0, "0", NULL,
         "Enable CUDA GPU acceleration", "GPU Acceleration", NULL, 1},
        {"cpu", '\0', GRIC_OPT_CUSTOM, (void *)cb_cpu, 0, "1", NULL,
         "Force CPU execution (disable GPU)", "GPU Acceleration", NULL, 0},
        {"gpu-pass1", '\0', GRIC_OPT_CUSTOM, (void *)cb_gpu_pass1, 0, "0", NULL,
         "Enable GPU acceleration for Pass 1", "GPU Acceleration",
         "gpu-bf,gpu-brute-force", 0},
        {"gpu-pass2", '\0', GRIC_OPT_CUSTOM, (void *)cb_gpu_pass2, 0, "0", NULL,
         "Enable GPU acceleration only for Pass 2", "GPU Acceleration", NULL, 0},
        {"cpu-pass1", '\0', GRIC_OPT_CUSTOM, (void *)cb_cpu_pass1, 0, "1", NULL,
         "Execute Pass 1 on CPU", "GPU Acceleration", "no-gpu-pass1", 0},
        {"gpu-batch-size", '\0', GRIC_OPT_CUSTOM, (void *)cb_gpu_batch_size, 0, "auto", "<size>",
         "Micro-batch size for GPU execution", "GPU Acceleration", "gpu-micro-batch", 0},
        {"gpu-device", '\0', GRIC_OPT_CUSTOM, (void *)cb_gpu_device, 0, "0", "<id>",
         "CUDA GPU device ordinal", "GPU Acceleration", NULL, 0},
        {"gpu-prune", '\0', GRIC_OPT_CUSTOM, (void *)cb_gpu_prune, 0, "auto", NULL,
         "Force metric-pruned GPU kernel", "GPU Acceleration", NULL, 0},
        {"gpu-gemm", '\0', GRIC_OPT_CUSTOM, (void *)cb_gpu_gemm, 0, "auto", NULL,
         "Force dense GEMM matrix multiply GPU kernel", "GPU Acceleration", "gpu-no-prune", 0},
        {"gpu-auto-prune", '\0', GRIC_OPT_CUSTOM, (void *)cb_gpu_auto_prune, 0, "auto", NULL,
         "Select GPU kernel dynamically based on dimensionality", "GPU Acceleration", NULL, 0}
    };

    size_t count = sizeof(opts) / sizeof(opts[0]);

    if (config != NULL)
    {
        /* Bind target pointers to active configuration struct */
        for (size_t ii = 0; ii < count; ii++)
        {
            const char *name = opts[ii].long_name;
            if (strcmp(name, "dprob") == 0)
                opts[ii].target = &config->algo.deltaprob;
            else if (strcmp(name, "maxcl") == 0)
                opts[ii].target = &config->algo.maxnbclust;
            else if (strcmp(name, "discard-frac") == 0)
                opts[ii].target = &config->algo.discard_fraction;
            else if (strcmp(name, "swap-remove") == 0)
                opts[ii].target = &config->algo.swap_remove;
            else if (strcmp(name, "double") == 0)
                opts[ii].target = &config->algo.use_double;
            else if (strcmp(name, "pass2-nearest") == 0)
                opts[ii].target = &config->algo.pass2_nearest_mode;
            else if (strcmp(name, "maxim") == 0)
                opts[ii].target = &config->input.maxnbfr;
            else if (strcmp(name, "progress") == 0)
                opts[ii].target = &config->output.progress_mode;
            else if (strcmp(name, "verbose") == 0)
                opts[ii].target = &config->output.verbose_level;
            else if (strcmp(name, "fitsout") == 0)
                opts[ii].target = &config->output.fitsout_mode;
            else if (strcmp(name, "pngout") == 0)
                opts[ii].target = &config->output.pngout_mode;
            else if (strcmp(name, "filelist") == 0)
                opts[ii].target = &config->input.filelist_mode;
            else if (strcmp(name, "stream") == 0)
                opts[ii].target = &config->input.stream_input_mode;
            else if (strcmp(name, "cnt2sync") == 0)
                opts[ii].target = &config->input.cnt2sync_mode;
            else if (strcmp(name, "tm-out") == 0)
                opts[ii].target = &config->output.output_tm;
            else if (strcmp(name, "dcc") == 0)
                opts[ii].target = &config->output.output_dcc;
            else if (strcmp(name, "dcc-sq16") == 0)
                opts[ii].target = &config->output.dcc_sq16_output;
            else if (strcmp(name, "anchors") == 0)
                opts[ii].target = &config->output.output_anchors;
            else if (strcmp(name, "counts") == 0)
                opts[ii].target = &config->output.output_counts;
            else if (strcmp(name, "membership") == 0)
                opts[ii].target = &config->output.output_membership;
            else if (strcmp(name, "evals") == 0)
                opts[ii].target = &config->output.output_evals;
            else if (strcmp(name, "discarded") == 0)
                opts[ii].target = &config->output.output_discarded;
            else if (strcmp(name, "clustered") == 0)
                opts[ii].target = &config->output.output_clustered;
            else if (strcmp(name, "clusters") == 0)
                opts[ii].target = &config->output.output_clusters;
            else if (strcmp(name, "scandist") == 0)
                opts[ii].target = &config->input.scandist_mode;
            else if (strcmp(name, "avg") == 0)
                opts[ii].target = &config->output.average_mode;
            else if (strcmp(name, "distall") == 0)
                opts[ii].target = &config->output.distall_mode;
            else if (strcmp(name, "eq16-adc") == 0)
                opts[ii].target = &config->optim.use_eq16_adc;
            else if (strcmp(name, "sq16-ratio") == 0)
                opts[ii].target = &config->optim.sq16_ratio;
            else if (strcmp(name, "memo") == 0)
                opts[ii].target = &config->optim.use_memo;
            else if (strcmp(name, "batch-dist") == 0)
                opts[ii].target = &config->optim.use_batch_dist;
            else if (strcmp(name, "ncpu") == 0)
                opts[ii].target = &config->optim.ncpu;
            else if (strcmp(name, "gprob") == 0)
                opts[ii].target = &config->optim.gprob_mode;
            else if (strcmp(name, "fmatcha") == 0)
                opts[ii].target = &config->optim.fmatch_a;
            else if (strcmp(name, "fmatchb") == 0)
                opts[ii].target = &config->optim.fmatch_b;
            else if (strcmp(name, "maxvis") == 0)
                opts[ii].target = &config->optim.max_gprob_visitors;
            else if (strcmp(name, "te4-max-anchors") == 0)
                opts[ii].target = &config->optim.te4_max_anchors;
            else if (strcmp(name, "entropy-max-targets") == 0)
                opts[ii].target = &config->optim.entropy_max_targets;
            else if (strcmp(name, "entropy-min-prob") == 0)
                opts[ii].target = &config->optim.entropy_min_prob;
            else if (strcmp(name, "entropy-gate") == 0)
                opts[ii].target = &config->optim.entropy_gate_bits;
            else if (strcmp(name, "entropy-first-gate") == 0)
                opts[ii].target = &config->optim.entropy_first_gate_bits;
            else if (strcmp(name, "entropy-fast") == 0)
                opts[ii].target = &config->optim.entropy_fast_mode;
            else if (strcmp(name, "entropy-leader") == 0)
                opts[ii].target = &config->optim.entropy_leader_shortcut;
            else if (strcmp(name, "entropy-leader-cutoff") == 0)
                opts[ii].target = &config->optim.entropy_leader_cutoff;
            else if (strcmp(name, "sparse-dcc-extra-evals") == 0)
                opts[ii].target = &config->optim.sparse_dcc_extra_evals;
            else if (strcmp(name, "soft-bayesian-sigma") == 0)
                opts[ii].target = &config->optim.soft_bayesian_sigma_coeff;
            else if (strcmp(name, "xtile-decay") == 0)
                opts[ii].target = &config->optim.xtile_decay;
            else if (strcmp(name, "retrieval-window") == 0)
                opts[ii].target = &config->input.retrieval_window;
            else if (strcmp(name, "no-prof") == 0)
                opts[ii].target = &config->optim.no_prof;
            else if (strcmp(name, "preset") == 0)
            {
                opts[ii].target = config->optim.preset_name;
                opts[ii].target_size = sizeof(config->optim.preset_name);
            }
        }
    }

    if (nopts != NULL)
    {
        *nopts = count;
    }
    return opts;
}

int apply_option(
    ClusterConfig *config,
    const char    *key,
    const char    *value)
{
    if (config == NULL || key == NULL)
    {
        return -1;
    }

    size_t nopts = 0;
    const struct gric_opt *opts = cluster_get_options(config, &nopts);

    char *fake_argv[2] = {(char *)key, (char *)value};
    int fake_argc = (value != NULL) ? 2 : 1;
    int fake_idx = 0;

    int res = gric_opt_parse_arg(fake_argc, fake_argv, &fake_idx, opts, nopts, config);
    if (res <= 0)
    {
        return -1;
    }

    return (fake_idx == 2) ? 1 : 0;
}

int read_config_file(
    const char    *filename,
    ClusterConfig *config)
{
    if (filename == NULL || config == NULL)
    {
        return 1;
    }

    size_t nopts = 0;
    const struct gric_opt *opts = cluster_get_options(config, &nopts);

    int res = gric_opt_parse_config_file(filename, opts, nopts, config);
    return (res == 0) ? 0 : 1;
}

int write_config_file(
    const char    *filename,
    ClusterConfig *config)
{
    FILE *f = fopen(filename, "w");
    if (!f)
    {
        return 1;
    }

    fprintf(f, "# gric-cluster configuration file\n");
    fprintf(f, "rlim %f\n", config->algo.rlim);
    if (config->algo.auto_rlim_mode)
    {
        fprintf(f, "# auto_rlim enabled (factor %f)\n", config->algo.auto_rlim_factor);
    }
    if (config->input.fits_filename)
    {
        fprintf(f, "input %s\n", config->input.fits_filename);
    }
    if (config->output.user_outdir)
    {
        fprintf(f, "outdir %s\n", config->output.user_outdir);
    }
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
    if (config->algo.use_double)
        fprintf(f, "double\n");

    fprintf(f, "fmatcha %f\n", config->optim.fmatch_a);
    fprintf(f, "fmatchb %f\n", config->optim.fmatch_b);
    fprintf(f, "maxvis %d\n", config->optim.max_gprob_visitors);

    if (config->optim.te4_mode)
    {
        fprintf(f, "te4\n");
        if (config->optim.te4_max_anchors > 0)
        {
            fprintf(f, "te4_max_anchors %d\n", config->optim.te4_max_anchors);
        }
    }
    if (config->optim.te5_mode)
    {
        fprintf(f, "te5\n");
    }
    if (config->optim.entropy_mode)
    {
        fprintf(f, "entropy\n");
        fprintf(f, "entropy_max_targets %d\n", config->optim.entropy_max_targets);
        fprintf(f, "entropy_min_prob %f\n", config->optim.entropy_min_prob);
        fprintf(f, "entropy_gate %f\n", config->optim.entropy_gate_bits);
        fprintf(f, "entropy_first_gate %f\n", config->optim.entropy_first_gate_bits);
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
    if (config->algo.pass2_nearest_mode)
    {
        fprintf(f, "pass2_nearest\n");
    }
    if (config->optim.use_sq16)
    {
        fprintf(f, "sq16\n");
        if (config->optim.sq16_ratio > 0.0)
        {
            fprintf(f, "sq16_ratio %f\n", config->optim.sq16_ratio);
        }
        if (config->optim.use_memo)
        {
            fprintf(f, "use_memo 1\n");
        }
        else
        {
            fprintf(f, "use_memo 0\n");
        }
    }
    else if (config->optim.use_sq8)
    {
        fprintf(f, "sq8\n");
    }
    else
    {
        fprintf(f, "no_sq8\n");
    }

    fprintf(f, "tm %f\n", config->algo.tm_mixing_coeff);

    const char *strat = "stop";
    if (config->algo.maxcl_strategy == MAXCL_DISCARD)
    {
        strat = "discard";
    }
    else if (config->algo.maxcl_strategy == MAXCL_MERGE)
    {
        strat = "merge";
    }
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
        const char *tag = (config->optim.pred_mode == 2) ? "predf" : "pred";
        fprintf(f, "-%s[%d,%d,%d]\n",
                tag, config->optim.pred_len,
                config->optim.pred_h,
                config->optim.pred_n);
    }

    if (config->input.scandist_mode)
        fprintf(f, "scandist\n");

    if (config->output.shm_filename)
        fprintf(f, "shm %s\n", config->output.shm_filename);

    if (config->input.tile_grid_x > 0 && config->input.tile_grid_y > 0)
    {
        fprintf(f, "tiles %dx%d\n",
                config->input.tile_grid_x,
                config->input.tile_grid_y);
    }
    if (config->input.tile_map_file)
    {
        fprintf(f, "tilemap %s\n", config->input.tile_map_file);
    }
    if (config->input.tile_config_file)
    {
        fprintf(f, "tileconf %s\n", config->input.tile_config_file);
    }
    if (config->input.retrieval_window != 1000)
    {
        fprintf(f, "retrieval_window %d\n", config->input.retrieval_window);
    }
    if (config->optim.xtile_mode)
    {
        fprintf(f, "xtile %d\n", config->optim.xtile_mode);
    }
    if (config->optim.xtile_decay != 1.0)
    {
        fprintf(f, "xtile_decay %f\n", config->optim.xtile_decay);
    }

    fclose(f);
    return 0;
}
