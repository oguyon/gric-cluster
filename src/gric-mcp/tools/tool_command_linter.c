/**
 * @file tool_command_linter.c
 * @brief Pre-flight command validation and linting tool for MCP.
 */

#include "mcp_tools.h"
#include "mcp_registry.h"
#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/**
 * add_diagnostic() - Append an error or warning descriptor to a cJSON array.
 */
static void add_diagnostic(
    cJSON      *arr,
    const char *code,
    const char *fmt,
    ...)
{
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    cJSON *item = cJSON_CreateObject();
    cJSON_AddStringToObject(item, "code", code);
    cJSON_AddStringToObject(item, "message", buf);
    cJSON_AddItemToArray(arr, item);
}

/**
 * tokenize_command() - Split a command string into tokens respecting quotes.
 */
static int tokenize_command(
    const char  *cmd,
    char      ***out_argv,
    int         *out_argc)
{
    size_t cap = 16;
    size_t count = 0;
    char **argv = (char **)malloc(cap * sizeof(char *));
    if (argv == NULL)
    {
        return -1;
    }

    const char *p = cmd;
    while (*p != '\0')
    {
        while (*p != '\0' && isspace((unsigned char)*p))
        {
            p++;
        }
        if (*p == '\0')
        {
            break;
        }

        char token[1024];
        size_t tlen = 0;
        char quote = '\0';

        while (*p != '\0')
        {
            if (quote != '\0')
            {
                if (*p == quote)
                {
                    quote = '\0';
                    p++;
                }
                else
                {
                    if (tlen + 1 < sizeof(token))
                    {
                        token[tlen++] = *p;
                    }
                    p++;
                }
            }
            else
            {
                if (*p == '\'' || *p == '"')
                {
                    quote = *p++;
                }
                else if (isspace((unsigned char)*p))
                {
                    break;
                }
                else
                {
                    if (tlen + 1 < sizeof(token))
                    {
                        token[tlen++] = *p;
                    }
                    p++;
                }
            }
        } // while (*p != '\0')

        token[tlen] = '\0';
        if (count >= cap)
        {
            cap *= 2;
            char **new_argv = (char **)realloc(argv, cap * sizeof(char *));
            if (new_argv == NULL)
            {
                for (size_t ii = 0; ii < count; ii++)
                {
                    free(argv[ii]);
                }
                free(argv);
                return -1;
            }
            argv = new_argv;
        }
        argv[count] = strdup(token);
        count++;
    } // while (*p != '\0')

    *out_argv = argv;
    *out_argc = (int)count;
    return 0;
}

/**
 * free_tokens() - Free token array allocated by tokenize_command.
 */
static void free_tokens(
    char **argv,
    int    argc)
{
    if (argv != NULL)
    {
        for (int ii = 0; ii < argc; ii++)
        {
            free(argv[ii]);
        }
        free(argv);
    }
}

/**
 * is_known_cluster_opt() - Check if flag is recognized by gric-cluster.
 */
static int is_known_cluster_opt(
    const char *opt,
    int        *needs_val)
{
    *needs_val = 0;

    static const char *const val_opts[] = {
        "-maxcl", "-maxim", "-ncpu", "-threads", "-dprob", "-tiles",
        "-outdir", "-input", "-shm", "-maxcl_strategy", "-maxcl-strategy",
        "-discard-frac", "-discard_frac", "-tm", "-tilemap", "-tileconf",
        "-retrieval_window", "-xtile", "-xtile_decay", "-pred",
        "-entropy-gate", "-entropy_gate", "-sparse_dcc_extra_evals",
        "-sparse-dcc-extra-evals", "-rlim", "-sq16-ratio", "-out"
    };

    static const char *const flag_opts[] = {
        "-sq8", "-sq16", "-eq16", "-rq8", "-rabitq", "-pq",
        "-te3", "-te4", "-te5", "-double", "-stream", "-cnt2sync",
        "-fitsout", "-pngout", "-txt", "-dcc", "-dcc-sq16",
        "-anchors", "-counts", "-membership", "-evals", "-discarded",
        "-clustered", "-clusters", "-scandist", "-avg", "-distall",
        "-progress", "-verbose", "-veryverbose", "-vv",
        "-pass2-nearest", "-reassign", "-swap-remove",
        "-sparse-dcc", "-sparse_dcc", "-entropy", "-no_xtile",
        "-help", "--help", "-h"
    };

    for (size_t ii = 0; ii < sizeof(val_opts) / sizeof(val_opts[0]); ii++)
    {
        if (strcmp(opt, val_opts[ii]) == 0)
        {
            *needs_val = 1;
            return 1;
        }
    }

    for (size_t ii = 0; ii < sizeof(flag_opts) / sizeof(flag_opts[0]); ii++)
    {
        if (strcmp(opt, flag_opts[ii]) == 0)
        {
            *needs_val = 0;
            return 1;
        }
    }

    return 0;
}

/**
 * validate_cluster_command() - Validate argument vector for gric-cluster.
 */
static void validate_cluster_command(
    int    argc,
    char **argv,
    cJSON *errors,
    cJSON *warnings,
    cJSON *parsed_opts)
{
    const char *quant_scheme = NULL;
    const char *prune_scheme = NULL;
    int is_double = 0;
    int is_stream = 0;
    int pos_count = 0;
    const char *rlim_str = NULL;
    const char *input_str = NULL;

    for (int ii = 0; ii < argc; ii++)
    {
        const char *arg = argv[ii];
        if (arg[0] == '-')
        {
            int needs_val = 0;
            if (!is_known_cluster_opt(arg, &needs_val))
            {
                add_diagnostic(
                    warnings, "UNKNOWN_OPTION",
                    "Unrecognized option '%s' for gric-cluster.", arg);
            }

            /* Quantization exclusivity check */
            if (strcmp(arg, "-sq8") == 0 || strcmp(arg, "-sq16") == 0 ||
                strcmp(arg, "-eq16") == 0 || strcmp(arg, "-rq8") == 0 ||
                strcmp(arg, "-rabitq") == 0 || strcmp(arg, "-pq") == 0)
            {
                if (quant_scheme != NULL)
                {
                    add_diagnostic(
                        errors, "CONFLICTING_QUANTIZATION",
                        "Multiple quantization flags specified (%s and %s); select at most one.",
                        quant_scheme, arg);
                }
                else
                {
                    quant_scheme = arg;
                    cJSON_AddStringToObject(parsed_opts, "quantization", arg + 1);
                }
            }

            /* Metric pruning exclusivity check */
            if (strcmp(arg, "-te3") == 0 || strcmp(arg, "-te4") == 0 ||
                strcmp(arg, "-te5") == 0)
            {
                if (prune_scheme != NULL)
                {
                    add_diagnostic(
                        errors, "CONFLICTING_PRUNING",
                        "Conflicting triangle inequality flags (%s and %s); select at most one.",
                        prune_scheme, arg);
                }
                else
                {
                    prune_scheme = arg;
                    cJSON_AddStringToObject(parsed_opts, "pruning", arg + 1);
                }
            }

            if (strcmp(arg, "-double") == 0)
            {
                is_double = 1;
                cJSON_AddBoolToObject(parsed_opts, "double", 1);
            }
            else if (strcmp(arg, "-stream") == 0)
            {
                is_stream = 1;
                cJSON_AddBoolToObject(parsed_opts, "stream", 1);
            }

            /* Check options requiring values */
            if (needs_val)
            {
                if (ii + 1 >= argc ||
                    (argv[ii + 1][0] == '-' && !isdigit((unsigned char)argv[ii + 1][1])))
                {
                    add_diagnostic(
                        errors, "MISSING_OPTION_VALUE",
                        "Option '%s' requires a value.", arg);
                }
                else
                {
                    const char *val = argv[++ii];
                    if (strcmp(arg, "-maxcl") == 0)
                    {
                        int v = atoi(val);
                        if (v <= 0)
                        {
                            add_diagnostic(
                                errors, "INVALID_VALUE",
                                "Option -maxcl must be a positive integer, got %s.", val);
                        }
                        else
                        {
                            cJSON_AddNumberToObject(parsed_opts, "maxcl", v);
                        }
                    }
                    else if (strcmp(arg, "-maxim") == 0)
                    {
                        long v = atol(val);
                        if (v <= 0)
                        {
                            add_diagnostic(
                                errors, "INVALID_VALUE",
                                "Option -maxim must be a positive integer, got %s.", val);
                        }
                        else
                        {
                            cJSON_AddNumberToObject(parsed_opts, "maxim", (double)v);
                        }
                    }
                    else if (strcmp(arg, "-ncpu") == 0 || strcmp(arg, "-threads") == 0)
                    {
                        int v = atoi(val);
                        if (v <= 0)
                        {
                            add_diagnostic(
                                errors, "INVALID_VALUE",
                                "Option %s must be a positive integer, got %s.", arg, val);
                        }
                        else
                        {
                            cJSON_AddNumberToObject(parsed_opts, "ncpu", v);
                        }
                    }
                    else if (strcmp(arg, "-dprob") == 0)
                    {
                        double v = atof(val);
                        if (v <= 0.0 || v >= 1.0)
                        {
                            add_diagnostic(
                                errors, "INVALID_VALUE",
                                "Option -dprob must be strictly between 0.0 and 1.0, got %s.", val);
                        }
                        else
                        {
                            cJSON_AddNumberToObject(parsed_opts, "dprob", v);
                        }
                    }
                    else if (strcmp(arg, "-tiles") == 0)
                    {
                        int tx = 0;
                        int ty = 0;
                        if (sscanf(val, "%dx%d", &tx, &ty) != 2 || tx <= 0 || ty <= 0)
                        {
                            add_diagnostic(
                                errors, "INVALID_FORMAT",
                                "Option -tiles requires format <NxM> (e.g. 2x2), got '%s'.", val);
                        }
                        else
                        {
                            cJSON_AddStringToObject(parsed_opts, "tiles", val);
                        }
                    }
                    else if (strcmp(arg, "-outdir") == 0 || strcmp(arg, "-out") == 0)
                    {
                        cJSON_AddStringToObject(parsed_opts, "outdir", val);
                    }
                } // if (ii + 1 >= argc ...)
            } // if (needs_val)
        }
        else
        {
            /* Positional argument */
            pos_count++;
            if (pos_count == 1)
            {
                rlim_str = arg;
            }
            else if (pos_count == 2)
            {
                input_str = arg;
            }
            else
            {
                add_diagnostic(
                    warnings, "EXTRA_ARGUMENT",
                    "Unexpected extra positional argument '%s'.", arg);
            }
        }
    } // for (int ii = 0; ii < argc; ii++)

    /* Check double precision compatibility with quantization */
    if (is_double && quant_scheme != NULL)
    {
        add_diagnostic(
            errors, "CONFLICTING_PRECISION",
            "Option -double cannot be combined with quantization flag %s.", quant_scheme);
    }

    /* Validate required positional arguments */
    if (rlim_str == NULL)
    {
        add_diagnostic(
            errors, "MISSING_ARG",
            "Missing required positional argument <rlim> (cluster radius cutoff).");
    }
    else
    {
        double rlim_val = atof(rlim_str);
        if (rlim_val <= 0.0)
        {
            add_diagnostic(
                errors, "INVALID_RADIUS",
                "Radius (rlim) must be a positive float (> 0.0), got '%s'.", rlim_str);
        }
        else
        {
            cJSON_AddNumberToObject(parsed_opts, "rlim", rlim_val);
        }
    }

    if (input_str == NULL)
    {
        add_diagnostic(
            errors, "MISSING_ARG",
            "Missing required positional argument <input_file|stream_name>.");
    }
    else
    {
        cJSON_AddStringToObject(parsed_opts, "input", input_str);
        if (!is_stream)
        {
            char resolved[512];
            mcp_resolve_path(input_str, resolved, sizeof(resolved));
            if (access(resolved, R_OK) != 0)
            {
                add_diagnostic(
                    warnings, "FILE_NOT_FOUND",
                    "Input dataset '%s' was not found on disk.", input_str);
            }
        }
    }
} // validate_cluster_command

/**
 * validate_knn_command() - Validate argument vector for gric-knn.
 */
static void validate_knn_command(
    int    argc,
    char **argv,
    cJSON *errors,
    cJSON *warnings,
    cJSON *parsed_opts)
{
    int pos_count = 0;
    const char *coords_str = NULL;
    const char *clust_dir_str = NULL;

    for (int ii = 0; ii < argc; ii++)
    {
        const char *arg = argv[ii];
        if (arg[0] == '-')
        {
            if (strcmp(arg, "-k") == 0 || strcmp(arg, "-dtmin") == 0 ||
                strcmp(arg, "-o") == 0 || strcmp(arg, "-out") == 0 ||
                strcmp(arg, "-output") == 0 || strcmp(arg, "-ncpu") == 0 ||
                strcmp(arg, "-threads") == 0)
            {
                if (ii + 1 >= argc ||
                    (argv[ii + 1][0] == '-' && !isdigit((unsigned char)argv[ii + 1][1])))
                {
                    add_diagnostic(
                        errors, "MISSING_OPTION_VALUE",
                        "Option '%s' requires a value.", arg);
                }
                else
                {
                    const char *val = argv[++ii];
                    if (strcmp(arg, "-k") == 0)
                    {
                        int v = atoi(val);
                        if (v <= 0)
                        {
                            add_diagnostic(
                                errors, "INVALID_VALUE",
                                "Option -k must be a positive integer, got %s.", val);
                        }
                        else
                        {
                            cJSON_AddNumberToObject(parsed_opts, "k", v);
                        }
                    }
                    else if (strcmp(arg, "-dtmin") == 0)
                    {
                        int v = atoi(val);
                        if (v < 0)
                        {
                            add_diagnostic(
                                errors, "INVALID_VALUE",
                                "Option -dtmin must be non-negative (>= 0), got %s.", val);
                        }
                        else
                        {
                            cJSON_AddNumberToObject(parsed_opts, "dtmin", v);
                        }
                    }
                    else if (strcmp(arg, "-o") == 0 || strcmp(arg, "-out") == 0 ||
                             strcmp(arg, "-output") == 0)
                    {
                        cJSON_AddStringToObject(parsed_opts, "output", val);
                    }
                    else if (strcmp(arg, "-ncpu") == 0 || strcmp(arg, "-threads") == 0)
                    {
                        int v = atoi(val);
                        if (v <= 0)
                        {
                            add_diagnostic(
                                errors, "INVALID_VALUE",
                                "Option %s must be a positive integer, got %s.", arg, val);
                        }
                        else
                        {
                            cJSON_AddNumberToObject(parsed_opts, "ncpu", v);
                        }
                    }
                } // if (ii + 1 >= argc ...)
            }
            else if (strcmp(arg, "-double") == 0 || strcmp(arg, "-gpu") == 0 ||
                     strcmp(arg, "-no-cache") == 0 || strcmp(arg, "-sq8") == 0 ||
                     strcmp(arg, "-sq16") == 0 || strcmp(arg, "-eq16") == 0 ||
                     strcmp(arg, "-rq8") == 0 || strcmp(arg, "-rabitq") == 0)
            {
                cJSON_AddBoolToObject(parsed_opts, arg + 1, 1);
            }
            else
            {
                add_diagnostic(
                    warnings, "UNKNOWN_OPTION",
                    "Unrecognized option '%s' for gric-knn.", arg);
            }
        }
        else
        {
            pos_count++;
            if (pos_count == 1)
            {
                coords_str = arg;
            }
            else if (pos_count == 2)
            {
                clust_dir_str = arg;
            }
            else
            {
                add_diagnostic(
                    warnings, "EXTRA_ARGUMENT",
                    "Unexpected extra positional argument '%s'.", arg);
            }
        }
    } // for (int ii = 0; ii < argc; ii++)

    if (coords_str == NULL || clust_dir_str == NULL)
    {
        add_diagnostic(
            errors, "MISSING_ARG",
            "gric-knn requires <input_coords> and <cluster_dir> positional arguments.");
    }
    else
    {
        cJSON_AddStringToObject(parsed_opts, "coords", coords_str);
        cJSON_AddStringToObject(parsed_opts, "cluster_dir", clust_dir_str);

        char resolved[512];
        mcp_resolve_path(coords_str, resolved, sizeof(resolved));
        if (access(resolved, R_OK) != 0)
        {
            add_diagnostic(
                warnings, "FILE_NOT_FOUND",
                "Coordinates dataset '%s' was not found on disk.", coords_str);
        }
    }
} // validate_knn_command

/**
 * mcp_tool_validate_command() - Validate CLI commands and flag compatibility.
 */
int mcp_tool_validate_command(
    const cJSON *args,
    cJSON       *res)
{
    if (args == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Missing arguments");
        return -1;
    }

    char **argv = NULL;
    int argc = 0;
    int must_free_tokens = 0;

    cJSON *cmd_item = cJSON_GetObjectItemCaseSensitive(args, "command");
    cJSON *bin_item = cJSON_GetObjectItemCaseSensitive(args, "binary");
    cJSON *args_item = cJSON_GetObjectItemCaseSensitive(args, "args");

    if (cmd_item != NULL && cJSON_IsString(cmd_item))
    {
        if (tokenize_command(cmd_item->valuestring, &argv, &argc) != 0 || argc == 0)
        {
            cJSON_AddStringToObject(res, "error", "Failed to tokenize command string");
            return -1;
        }
        must_free_tokens = 1;
    }
    else if (bin_item != NULL && cJSON_IsString(bin_item))
    {
        int extra = (args_item != NULL && cJSON_IsArray(args_item)) ?
                    cJSON_GetArraySize(args_item) : 0;
        argc = 1 + extra;
        argv = (char **)malloc((size_t)argc * sizeof(char *));
        argv[0] = strdup(bin_item->valuestring);
        for (int ii = 0; ii < extra; ii++)
        {
            cJSON *tok = cJSON_GetArrayItem(args_item, ii);
            argv[ii + 1] = strdup(cJSON_IsString(tok) ? tok->valuestring : "");
        }
        must_free_tokens = 1;
    }
    else
    {
        cJSON_AddStringToObject(res, "error", "Either 'command' or 'binary' must be provided");
        return -1;
    }

    /* Extract base binary name */
    const char *raw_bin = argv[0];
    const char *slash = strrchr(raw_bin, '/');
    const char *base_bin = (slash != NULL) ? (slash + 1) : raw_bin;

    cJSON *errors = cJSON_CreateArray();
    cJSON *warnings = cJSON_CreateArray();
    cJSON *parsed_opts = cJSON_CreateObject();

    cJSON_AddStringToObject(res, "binary", base_bin);

    int sub_argc = argc - 1;
    char **sub_argv = argv + 1;

    if (strcmp(base_bin, "gric-cluster") == 0)
    {
        validate_cluster_command(sub_argc, sub_argv, errors, warnings, parsed_opts);
    }
    else if (strcmp(base_bin, "gric-knn") == 0 || strcmp(base_bin, "gric-knn-search") == 0)
    {
        validate_knn_command(sub_argc, sub_argv, errors, warnings, parsed_opts);
    }
    else
    {
        add_diagnostic(
            errors, "UNKNOWN_BINARY",
            "Unsupported binary '%s'; expected 'gric-cluster' or 'gric-knn'.", base_bin);
    }

    int num_errors = cJSON_GetArraySize(errors);
    cJSON_AddBoolToObject(res, "valid", num_errors == 0);
    cJSON_AddItemToObject(res, "errors", errors);
    cJSON_AddItemToObject(res, "warnings", warnings);
    cJSON_AddItemToObject(res, "parsed_options", parsed_opts);

    /* Construct normalized command string */
    char norm_cmd[2048] = "";
    size_t cur_len = 0;
    for (int ii = 0; ii < argc; ii++)
    {
        cur_len += snprintf(
            norm_cmd + cur_len, sizeof(norm_cmd) - cur_len,
            "%s%s", (ii > 0) ? " " : "", argv[ii]);
        if (cur_len >= sizeof(norm_cmd) - 1)
        {
            break;
        }
    }
    cJSON_AddStringToObject(res, "normalized_command", norm_cmd);

    if (must_free_tokens)
    {
        free_tokens(argv, argc);
    }

    return 0;
} // mcp_tool_validate_command

const struct mcp_tool_def mcp_tooldef_validate_command = {
    .name         = "gric_validate_command",
    .toolset      = MCP_TS_ANALYSIS,
    .side_effects = 0,
    .fn           = mcp_tool_validate_command,
    .description  = "Pre-flight validation and linting for GRIC CLI commands (detects conflicting "
                    "flags, invalid ranges, and missing arguments).",
    .input_schema =
        "{\n"
        "  \"type\": \"object\",\n"
        "  \"properties\": {\n"
        "    \"command\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Full CLI command string to validate.\"\n"
        "    },\n"
        "    \"binary\": {\n"
        "      \"type\": \"string\",\n"
        "      \"description\": \"Optional binary name (e.g. gric-cluster, gric-knn).\"\n"
        "    },\n"
        "    \"args\": {\n"
        "      \"type\": \"array\",\n"
        "      \"items\": { \"type\": \"string\" },\n"
        "      \"description\": \"Optional array of argument tokens.\"\n"
        "    }\n"
        "  }\n"
        "}",
};
