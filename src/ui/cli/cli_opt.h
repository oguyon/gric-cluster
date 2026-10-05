/**
 * @file cli_opt.h
 * @brief Declarative table-driven command line option parser and introspection engine.
 */

#ifndef CLI_OPT_H
#define CLI_OPT_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * enum gric_opt_type - Data types supported by the declarative option parser.
 */
enum gric_opt_type
{
    GRIC_OPT_FLAG = 0, /* Boolean flag: sets *target = 1 (or 0 if negation) */
    GRIC_OPT_BOOL,     /* Boolean: accepts optional =on|off|1|0|true|false */
    GRIC_OPT_INT,      /* Signed integer: int *target */
    GRIC_OPT_INT64,    /* 64-bit integer: int64_t *target */
    GRIC_OPT_UINT32,   /* 32-bit unsigned: uint32_t *target */
    GRIC_OPT_UINT64,   /* 64-bit unsigned: uint64_t *target */
    GRIC_OPT_FLOAT,    /* Single-precision float: float *target */
    GRIC_OPT_DOUBLE,   /* Double-precision float: double *target */
    GRIC_OPT_STRING,   /* String: copies to buffer if size > 0, else sets const char ** */
    GRIC_OPT_CUSTOM    /* Custom callback handler */
};

/**
 * gric_opt_custom_fn - Function pointer signature for custom option parsers.
 * @key: Parsed option key string.
 * @val: Option value string (may be NULL if no argument provided).
 * @ctx: Opaque user context pointer passed to parser.
 *
 * Return: Number of consumed arguments (0 for flag, 1 for consumed value), or -1 on error.
 */
typedef int (*gric_opt_custom_fn)(
    const char *key,
    const char *val,
    void       *ctx);

/**
 * struct gric_opt - Declarative definition of a command line option.
 */
struct gric_opt
{
    const char         *long_name;   /* Canonical name without dashes, e.g. "rlim" */
    char                short_name;  /* Single-character flag (e.g. 'k', 'o', '\0') */
    enum gric_opt_type  type;        /* Value data type */
    void               *target;      /* Pointer to variable to populate (or callback) */
    size_t              target_size; /* Buffer capacity for string buffers */
    const char         *default_str; /* Default value string for help & introspection */
    const char         *arg_name;    /* Value placeholder for help (e.g. "<N>", "<val>") */
    const char         *description; /* Human-readable explanation of option */
    const char         *category;    /* Grouping category (e.g. "Core", "Quantization") */
    const char         *aliases;     /* Comma-separated legacy aliases ("maxcl,maxnbclust") */
    int                 negatable;   /* 1 = automatically supports --no-<long_name> */
};

/**
 * gric_opt_match_key() - Test if an argument string matches an option definition.
 * @arg_key:      Input argument token (with or without dashes).
 * @opt:          Option definition to test against.
 * @is_negation:  Output set to 1 if matched via --no-<name> negation, else 0.
 *
 * Return: 1 if matched, 0 otherwise.
 */
int gric_opt_match_key(
    const char            *arg_key,
    const struct gric_opt *opt,
    int                   *is_negation);

/**
 * gric_opt_parse_arg() - Match and apply a single argument token from argv.
 * @argc:        Total argument count.
 * @argv:        Argument vector.
 * @arg_idx:     Pointer to current argument index (incremented if consumed).
 * @opts:        Array of option definitions.
 * @nopts:       Number of entries in opts array.
 * @custom_ctx:  Opaque context pointer passed to custom callbacks.
 *
 * Return: 1 if matched and successfully applied, 0 if not matched, -1 on parse error.
 */
int gric_opt_parse_arg(
    int                    argc,
    char                 **argv,
    int                   *arg_idx,
    const struct gric_opt *opts,
    size_t                 nopts,
    void                  *custom_ctx);

/**
 * gric_opt_parse_config_file() - Parse key-value configuration file using option table.
 * @filename:    Path to configuration file.
 * @opts:        Array of option definitions.
 * @nopts:       Number of entries in opts array.
 * @custom_ctx:  Opaque context pointer passed to custom callbacks.
 *
 * Return: 0 on success, -1 on file open or parse error.
 */
int gric_opt_parse_config_file(
    const char            *filename,
    const struct gric_opt *opts,
    size_t                 nopts,
    void                  *custom_ctx);

/**
 * gric_opt_print_help() - Print formatted, categorized help to stream.
 * @fp:            Output file stream (e.g. stdout or stderr).
 * @prog_name:     Program binary name.
 * @usage_summary: Usage headline (e.g. "<rlim> <input> [options]").
 * @opts:          Array of option definitions.
 * @nopts:         Number of entries in opts array.
 */
void gric_opt_print_help(
    FILE                  *fp,
    const char            *prog_name,
    const char            *usage_summary,
    const struct gric_opt *opts,
    size_t                 nopts);

struct cJSON;

/**
 * gric_opt_to_json() - Serialize options into a cJSON array of option objects.
 * @opts:   Array of option definitions.
 * @nopts:  Number of entries in opts array.
 *
 * Return: Pointer to new cJSON array (caller frees with cJSON_Delete), or NULL on error.
 */
struct cJSON *gric_opt_to_json(
    const struct gric_opt *opts,
    size_t                 nopts);

/**
 * gric_opt_to_json_schema() - Generate JSON Schema object for options (e.g. MCP inputSchema).
 * @opts:   Array of option definitions.
 * @nopts:  Number of entries in opts array.
 *
 * Return: Pointer to new cJSON object schema (caller frees with cJSON_Delete), or NULL on error.
 */
struct cJSON *gric_opt_to_json_schema(
    const struct gric_opt *opts,
    size_t                 nopts);

#ifdef __cplusplus
}
#endif

#endif /* CLI_OPT_H */

