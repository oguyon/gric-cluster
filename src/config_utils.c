#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "config_utils.h"

// Helper to match options with or without leading dash
static int matches(const char *key, const char *opt) {
    if (strcmp(key, opt) == 0) return 1;
    if (key[0] != '-' && opt[0] == '-' && strcmp(key, opt + 1) == 0) return 1;
    if (key[0] == '-' && opt[0] != '-' && strcmp(key + 1, opt) == 0) return 1;
    return 0;
}

int apply_option(ClusterConfig *config, const char *key, const char *value) {
    if (matches(key, "-dprob")) {
        if (!value) return -1;
        config->deltaprob = atof(value);
        return 1;
    } else if (matches(key, "-maxcl")) {
        if (!value) return -1;
        config->maxnbclust = atoi(value);
        return 1;
    } else if (matches(key, "-ncpu")) {
        if (!value) return -1;
        config->ncpu = atoi(value);
        return 1;
    } else if (matches(key, "-maxim")) {
        if (!value) return -1;
        config->maxnbfr = atol(value);
        return 1;
    } else if (matches(key, "-avg")) {
        config->average_mode = 1;
        return 0;
    } else if (matches(key, "-distall")) {
        config->distall_mode = 1;
        return 0;
    } else if (matches(key, "-outdir")) {
        if (!value) return -1;
        config->user_outdir = strdup(value); // We strdup here to manage memory consistent with config file reading
        return 1;
    } else if (matches(key, "-progress")) {
        config->progress_mode = 1;
        return 0;
    } else if (matches(key, "-gprob")) {
        config->gprob_mode = 1;
        return 0;
    } else if (matches(key, "-verbose")) {
        config->verbose_level = 1;
        return 0;
    } else if (matches(key, "-veryverbose")) {
        config->verbose_level = 2;
        return 0;
    } else if (matches(key, "-fitsout")) {
        config->fitsout_mode = 1;
        return 0;
    } else if (matches(key, "-pngout")) {
        config->pngout_mode = 1;
        return 0;
    } else if (matches(key, "-filelist")) {
        config->filelist_mode = 1;
        return 0;
    } else if (matches(key, "-stream")) {
        config->stream_input_mode = 1;
        return 0;
    } else if (matches(key, "-cnt2sync")) {
        config->cnt2sync_mode = 1;
        return 0;
    } else if (matches(key, "-fmatcha")) {
        if (!value) return -1;
        config->fmatch_a = atof(value);
        return 1;
    } else if (matches(key, "-fmatchb")) {
        if (!value) return -1;
        config->fmatch_b = atof(value);
        return 1;
    } else if (matches(key, "-maxvis")) {
        if (!value) return -1;
        config->max_gprob_visitors = atoi(value);
        return 1;
    } else if (matches(key, "-te4")) {
        config->te4_mode = 1;
        return 0;
    } else if (matches(key, "-te5")) {
        config->te5_mode = 1;
        return 0;
    } else if (matches(key, "-tm")) {
        if (!value) return -1;
        config->tm_mixing_coeff = atof(value);
        return 1;
    } else if (matches(key, "-maxcl_strategy")) {
        if (!value) return -1;
        if (strcmp(value, "stop") == 0) config->maxcl_strategy = MAXCL_STOP;
        else if (strcmp(value, "discard") == 0) config->maxcl_strategy = MAXCL_DISCARD;
        else if (strcmp(value, "merge") == 0) config->maxcl_strategy = MAXCL_MERGE;
        else fprintf(stderr, "Warning: Unknown maxcl_strategy '%s'\n", value);
        return 1;
    } else if (matches(key, "-discard_frac")) {
        if (!value) return -1;
        config->discard_fraction = atof(value);
        return 1;
    } else if (matches(key, "-tm_out")) {
        config->output_tm = 1;
        return 0;
    } else if (matches(key, "-anchors")) {
        config->output_anchors = 1;
        return 0;
    } else if (matches(key, "-counts")) {
        config->output_counts = 1;
        return 0;
    } else if (matches(key, "-membership")) {
        config->output_membership = 1;
        return 0;
    } else if (matches(key, "-no_membership")) {
        config->output_membership = 0;
        return 0;
    } else if (matches(key, "-discarded")) {
        config->output_discarded = 1;
        return 0;
    } else if (matches(key, "-clustered")) {
        config->output_clustered = 1;
        return 0;
    } else if (matches(key, "-clusters")) {
        config->output_clusters = 1;
        return 0;
    } else if (strncmp(key, "-pred", 5) == 0 || strncmp(key, "pred", 4) == 0) {
        config->pred_mode = 1;
        const char *params = strchr(key, '[');
        if (params) {
            params++; // Skip [
            int l, h, n;
            if (sscanf(params, "%d,%d,%d", &l, &h, &n) == 3) {
                 config->pred_len = l;
                 config->pred_h = h;
                 config->pred_n = n;
            }
        }
        return 0;
    } else if (matches(key, "-scandist")) {
        config->scandist_mode = 1;
        return 0;
    } else if (matches(key, "-rlim")) { // Explicit rlim
        if (!value) return -1;
        if (value[0] == 'a') {
             config->auto_rlim_factor = atof(value+1);
             config->auto_rlim_mode = 1;
        } else {
             config->rlim = atof(value);
        }
        return 1;
    } else if (matches(key, "-input") || matches(key, "-in")) { // Explicit input
        if (!value) return -1;
        config->fits_filename = strdup(value);
        return 1;
    }
    
    return -1; // Unknown option
}

int read_config_file(const char *filename, ClusterConfig *config) {
    FILE *f = fopen(filename, "r");
    if (!f) return 1;

    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        while (isspace(*p)) p++;
        if (*p == '\0' || *p == '#') continue;

        // Split key value
        char *key = p;
        char *val = NULL;
        char *end = key;
        while (*end && !isspace(*end)) end++;
        if (*end) {
            *end = '\0';
            val = end + 1;
            while (isspace(*val)) val++;
            // Trim newline from val
            char *v_end = val + strlen(val) - 1;
            while (v_end >= val && isspace(*v_end)) { *v_end = '\0'; v_end--; }
            if (*val == '\0') val = NULL;
        }

        apply_option(config, key, val);
    }
    fclose(f);
    return 0;
}

int write_config_file(const char *filename, ClusterConfig *config) {
    FILE *f = fopen(filename, "w");
    if (!f) return 1;

    fprintf(f, "# gric-cluster configuration file\n");
    fprintf(f, "rlim %f\n", config->rlim);
    if (config->auto_rlim_mode) fprintf(f, "# auto_rlim enabled (factor %f)\n", config->auto_rlim_factor);
    if (config->fits_filename) fprintf(f, "input %s\n", config->fits_filename);
    if (config->user_outdir) fprintf(f, "outdir %s\n", config->user_outdir);
    fprintf(f, "dprob %f\n", config->deltaprob);
    fprintf(f, "maxcl %d\n", config->maxnbclust);
    fprintf(f, "maxim %ld\n", config->maxnbfr);
    fprintf(f, "ncpu %d\n", config->ncpu);
    
    if (config->average_mode) fprintf(f, "avg\n");
    if (config->distall_mode) fprintf(f, "distall\n");
    if (config->progress_mode) fprintf(f, "progress\n");
    if (config->gprob_mode) fprintf(f, "gprob\n");
    if (config->verbose_level == 1) fprintf(f, "verbose\n");
    if (config->verbose_level == 2) fprintf(f, "veryverbose\n");
    if (config->fitsout_mode) fprintf(f, "fitsout\n");
    if (config->pngout_mode) fprintf(f, "pngout\n");
    if (config->filelist_mode) fprintf(f, "filelist\n");
    if (config->stream_input_mode) fprintf(f, "stream\n");
    if (config->cnt2sync_mode) fprintf(f, "cnt2sync\n");
    
    fprintf(f, "fmatcha %f\n", config->fmatch_a);
    fprintf(f, "fmatchb %f\n", config->fmatch_b);
    fprintf(f, "maxvis %d\n", config->max_gprob_visitors);
    
    if (config->te4_mode) fprintf(f, "te4\n");
    if (config->te5_mode) fprintf(f, "te5\n");
    
    fprintf(f, "tm %f\n", config->tm_mixing_coeff);
    
    const char *strat = "stop";
    if (config->maxcl_strategy == MAXCL_DISCARD) strat = "discard";
    else if (config->maxcl_strategy == MAXCL_MERGE) strat = "merge";
    fprintf(f, "maxcl_strategy %s\n", strat);
    fprintf(f, "discard_frac %f\n", config->discard_fraction);
    
    if (config->output_tm) fprintf(f, "tm_out\n");
    if (config->output_anchors) fprintf(f, "anchors\n");
    if (config->output_counts) fprintf(f, "counts\n");
    if (config->output_membership) fprintf(f, "membership\n");
    if (!config->output_membership) fprintf(f, "no_membership\n");
    if (config->output_discarded) fprintf(f, "discarded\n");
    if (config->output_clustered) fprintf(f, "clustered\n");
    if (config->output_clusters) fprintf(f, "clusters\n");
    
    if (config->pred_mode) {
        fprintf(f, "# Prediction mode enabled: pred[%d,%d,%d]\n", config->pred_len, config->pred_h, config->pred_n);
        // Writing as -pred is weird, maybe write separate params? 
        // But the parser handles -pred[...]. Let's write it as such.
        fprintf(f, "-pred[%d,%d,%d]\n", config->pred_len, config->pred_h, config->pred_n);
    }
    
    if (config->scandist_mode) fprintf(f, "scandist\n");

    fclose(f);
    return 0;
}
