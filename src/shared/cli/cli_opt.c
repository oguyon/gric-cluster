/**
 * @file cli_opt.c
 * @brief Table-driven command line and configuration parser implementation.
 */

#include "cli_opt.h"
#include "cli_colors.h"
#include "cjson/cJSON.h"
#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * normalize_char() - Convert character to lowercase and treat underscore as hyphen.
 * @c: Input character.
 *
 * Return: Normalized character.
 */
static inline char normalize_char(
    char c)
{
    if (c == '_')
    {
        return '-';
    }
    return (char)tolower((unsigned char)c);
} // normalize_char

/**
 * name_matches() - Compare option name ignoring leading dashes and dash/underscore variance.
 * @key:  Command line argument token.
 * @cand: Candidate option name.
 *
 * Return: 1 if match, 0 if different.
 */
static int name_matches(
    const char *key,
    const char *cand)
{
    if (key == NULL || cand == NULL)
    {
        return 0;
    }

    while (*key == '-')
    {
        key++;
    }
    while (*cand == '-')
    {
        cand++;
    }

    while (*key != '\0' && *cand != '\0')
    {
        if (normalize_char(*key) != normalize_char(*cand))
        {
            return 0;
        }
        key++;
        cand++;
    }

    return (*key == '\0' && *cand == '\0');
} // name_matches

int gric_opt_match_key(
    const char            *arg_key,
    const struct gric_opt *opt,
    int                   *is_negation)
{
    if (arg_key == NULL || opt == NULL)
    {
        return 0;
    }

    if (is_negation != NULL)
    {
        *is_negation = 0;
    }

    /* 1. Check single-character short flag */
    if (opt->short_name != '\0')
    {
        if (arg_key[0] == '-' && arg_key[1] == opt->short_name && arg_key[2] == '\0')
        {
            return 1;
        }
    }

    /* 2. Check canonical long option name */
    if (opt->long_name != NULL)
    {
        if (name_matches(arg_key, opt->long_name))
        {
            return 1;
        }

        /* Check automatic negation for negatable options (--no-<name>) */
        if (opt->negatable)
        {
            const char *stripped = arg_key;
            while (*stripped == '-')
            {
                stripped++;
            }
            if ((strncmp(stripped, "no-", 3) == 0 || strncmp(stripped, "no_", 3) == 0) &&
                name_matches(stripped + 3, opt->long_name))
            {
                if (is_negation != NULL)
                {
                    *is_negation = 1;
                }
                return 1;
            }
        }
    }

    /* 3. Check legacy aliases list (comma-separated tokens) */
    if (opt->aliases != NULL)
    {
        const char *cur = opt->aliases;
        while (*cur != '\0')
        {
            char token[64];
            size_t len = 0;
            while (*cur != '\0' && *cur != ',' && len < sizeof(token) - 1)
            {
                token[len++] = *cur++;
            }
            token[len] = '\0';

            if (name_matches(arg_key, token))
            {
                return 1;
            }

            /* Support explicit negation aliases (e.g. "no-sq8" or "nosq8") */
            if (opt->negatable)
            {
                const char *stripped = arg_key;
                while (*stripped == '-')
                {
                    stripped++;
                }
                if (name_matches(stripped, token))
                {
                    if (strncmp(token, "no", 2) == 0 && is_negation != NULL)
                    {
                        *is_negation = 1;
                    }
                    return 1;
                }
            }

            if (*cur == ',')
            {
                cur++;
            }
        } // while cur
    }

    return 0;
} // gric_opt_match_key

/**
 * apply_parsed_value() - Convert string value and assign to target variable.
 * @opt:         Option definition.
 * @key:         Matched key name.
 * @val:         Value string (may be NULL for flags).
 * @is_negation: 1 if matched via negation flag.
 * @custom_ctx:  Opaque context pointer.
 *
 * Return: 0 on success, -1 on conversion error.
 */
static int apply_parsed_value(
    const struct gric_opt *opt,
    const char            *key,
    const char            *val,
    int                    is_negation,
    void                  *custom_ctx)
{
    if (opt->type == GRIC_OPT_CUSTOM)
    {
        if (opt->target == NULL)
        {
            return -1;
        }
        gric_opt_custom_fn cb = (gric_opt_custom_fn)opt->target;
        return (cb(key, val, custom_ctx) >= 0) ? 0 : -1;
    }

    if (opt->target == NULL)
    {
        return 0;
    }

    switch (opt->type)
    {
        case GRIC_OPT_FLAG:
            if (val != NULL)
            {
                if (name_matches(val, "0") || name_matches(val, "off") ||
                    name_matches(val, "no") || name_matches(val, "false"))
                {
                    *(int *)opt->target = 0;
                    return 0;
                }
                if (name_matches(val, "1") || name_matches(val, "on") ||
                    name_matches(val, "yes") || name_matches(val, "true"))
                {
                    *(int *)opt->target = 1;
                    return 0;
                }
            }
            *(int *)opt->target = (is_negation ? 0 : 1);
            return 0;

        case GRIC_OPT_BOOL:
            if (val == NULL)
            {
                *(int *)opt->target = (is_negation ? 0 : 1);
                return 0;
            }
            if (name_matches(val, "1") || name_matches(val, "true") ||
                name_matches(val, "on") || name_matches(val, "yes"))
            {
                *(int *)opt->target = 1;
                return 0;
            }
            if (name_matches(val, "0") || name_matches(val, "false") ||
                name_matches(val, "off") || name_matches(val, "no"))
            {
                *(int *)opt->target = 0;
                return 0;
            }
            fprintf(stderr, "Error: Invalid boolean value '%s' for option '%s'\n", val, key);
            return -1;

        case GRIC_OPT_INT:
        {
            char *endptr = NULL;
            errno = 0;
            long num = strtol(val, &endptr, 10);
            if (errno != 0 || endptr == val || *endptr != '\0')
            {
                fprintf(stderr, "Error: Invalid integer '%s' for option '%s'\n", val, key);
                return -1;
            }
            *(int *)opt->target = (int)num;
            return 0;
        }

        case GRIC_OPT_INT64:
        {
            char *endptr = NULL;
            errno = 0;
            int64_t num = (int64_t)strtoll(val, &endptr, 10);
            if (errno != 0 || endptr == val || *endptr != '\0')
            {
                fprintf(stderr, "Error: Invalid int64 '%s' for option '%s'\n", val, key);
                return -1;
            }
            *(int64_t *)opt->target = num;
            return 0;
        }

        case GRIC_OPT_UINT32:
        {
            char *endptr = NULL;
            errno = 0;
            unsigned long num = strtoul(val, &endptr, 10);
            if (errno != 0 || endptr == val || *endptr != '\0')
            {
                fprintf(stderr, "Error: Invalid uint32 '%s' for option '%s'\n", val, key);
                return -1;
            }
            *(uint32_t *)opt->target = (uint32_t)num;
            return 0;
        }

        case GRIC_OPT_UINT64:
        {
            char *endptr = NULL;
            errno = 0;
            uint64_t num = (uint64_t)strtoull(val, &endptr, 10);
            if (errno != 0 || endptr == val || *endptr != '\0')
            {
                fprintf(stderr, "Error: Invalid uint64 '%s' for option '%s'\n", val, key);
                return -1;
            }
            *(uint64_t *)opt->target = num;
            return 0;
        }

        case GRIC_OPT_FLOAT:
        {
            char *endptr = NULL;
            errno = 0;
            float num = strtof(val, &endptr);
            if (errno != 0 || endptr == val || *endptr != '\0')
            {
                fprintf(stderr, "Error: Invalid float '%s' for option '%s'\n", val, key);
                return -1;
            }
            *(float *)opt->target = num;
            return 0;
        }

        case GRIC_OPT_DOUBLE:
        {
            char *endptr = NULL;
            errno = 0;
            double num = strtod(val, &endptr);
            if (errno != 0 || endptr == val || *endptr != '\0')
            {
                fprintf(stderr, "Error: Invalid double '%s' for option '%s'\n", val, key);
                return -1;
            }
            *(double *)opt->target = num;
            return 0;
        }

        case GRIC_OPT_STRING:
            if (opt->target_size > 0)
            {
                strncpy((char *)opt->target, val, opt->target_size - 1);
                ((char *)opt->target)[opt->target_size - 1] = '\0';
            }
            else
            {
                *(const char **)opt->target = val;
            }
            return 0;

        default:
            return -1;
    } // switch
} // apply_parsed_value

int gric_opt_parse_arg(
    int                    argc,
    char                 **argv,
    int                   *arg_idx,
    const struct gric_opt *opts,
    size_t                 nopts,
    void                  *custom_ctx)
{
    if (argv == NULL || arg_idx == NULL || opts == NULL || *arg_idx >= argc)
    {
        return 0;
    }

    const char *token = argv[*arg_idx];
    if (token == NULL || token[0] == '\0')
    {
        return 0;
    }

    /* Extract inline value if option uses --opt=value or -o=val syntax */
    char key_buf[128];
    const char *key_str = token;
    const char *inline_val = NULL;
    const char *eq = strchr(token, '=');

    if (eq != NULL)
    {
        size_t klen = (size_t)(eq - token);
        if (klen >= sizeof(key_buf))
        {
            klen = sizeof(key_buf) - 1;
        }
        memcpy(key_buf, token, klen);
        key_buf[klen] = '\0';
        key_str = key_buf;
        inline_val = eq + 1;
    }

    for (size_t ii = 0; ii < nopts; ii++)
    {
        int is_negation = 0;
        if (!gric_opt_match_key(key_str, &opts[ii], &is_negation))
        {
            continue;
        }

        /* Option matched! Determine value string */
        const char *val_str = NULL;
        int advance = 1;

        if (opts[ii].type == GRIC_OPT_FLAG)
        {
            val_str = NULL;
            advance = 1;
        }
        else if (opts[ii].type == GRIC_OPT_BOOL)
        {
            if (inline_val != NULL)
            {
                val_str = inline_val;
                advance = 1;
            }
            else if (*arg_idx + 1 < argc && argv[*arg_idx + 1][0] != '-')
            {
                /* Check if next token is a recognized boolean literal */
                const char *next_tok = argv[*arg_idx + 1];
                if (name_matches(next_tok, "on") || name_matches(next_tok, "off") ||
                    name_matches(next_tok, "true") || name_matches(next_tok, "false") ||
                    name_matches(next_tok, "1") || name_matches(next_tok, "0"))
                {
                    val_str = next_tok;
                    advance = 2;
                }
            }
        }
        else if (opts[ii].type == GRIC_OPT_CUSTOM)
        {
            if (opts[ii].target == NULL)
            {
                return -1;
            }
            gric_opt_custom_fn cb = (gric_opt_custom_fn)opts[ii].target;
            const char *cand_val = inline_val;
            if (cand_val == NULL && *arg_idx + 1 < argc)
            {
                cand_val = argv[*arg_idx + 1];
            }
            int consumed = cb(key_str, cand_val, custom_ctx);
            if (consumed < 0)
            {
                return -1;
            }
            if (consumed == 0)
            {
                *arg_idx += 1;
            }
            else
            {
                *arg_idx += (inline_val != NULL ? 1 : 2);
            }
            return 1;
        }
        else
        {
            /* Requires value */
            if (inline_val != NULL)
            {
                val_str = inline_val;
                advance = 1;
            }
            else if (*arg_idx + 1 < argc)
            {
                val_str = argv[*arg_idx + 1];
                advance = 2;
            }
            else
            {
                fprintf(stderr, "Error: Option '%s' requires an argument (%s)\n",
                        token, opts[ii].arg_name ? opts[ii].arg_name : "value");
                return -1;
            }
        }

        if (apply_parsed_value(&opts[ii], key_str, val_str, is_negation, custom_ctx) != 0)
        {
            return -1;
        }

        *arg_idx += advance;
        return 1;
    } // for ii

    return 0;
} // gric_opt_parse_arg

int gric_opt_parse_config_file(
    const char            *filename,
    const struct gric_opt *opts,
    size_t                 nopts,
    void                  *custom_ctx)
{
    if (filename == NULL || opts == NULL)
    {
        return -1;
    }

    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
        return -1;
    }

    char line[1024];
    int line_num = 0;

    while (fgets(line, sizeof(line), fp) != NULL)
    {
        line_num++;

        /* Strip trailing whitespace and newline */
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r' ||
                           line[len - 1] == ' ' || line[len - 1] == '\t'))
        {
            line[--len] = '\0';
        }

        /* Skip leading whitespace */
        const char *p = line;
        while (*p == ' ' || *p == '\t')
        {
            p++;
        }

        /* Skip blank lines and comments */
        if (*p == '\0' || *p == '#' || *p == ';' || (p[0] == '/' && p[1] == '/'))
        {
            continue;
        }

        /* Parse key and optional value (separated by '=' or whitespace) */
        char key[128] = "";
        char val[512] = "";
        const char *delim = strchr(p, '=');

        if (delim != NULL)
        {
            size_t klen = (size_t)(delim - p);
            while (klen > 0 && (p[klen - 1] == ' ' || p[klen - 1] == '\t'))
            {
                klen--;
            }
            if (klen >= sizeof(key))
            {
                klen = sizeof(key) - 1;
            }
            memcpy(key, p, klen);
            key[klen] = '\0';

            const char *v = delim + 1;
            while (*v == ' ' || *v == '\t')
            {
                v++;
            }
            strncpy(val, v, sizeof(val) - 1);
            val[sizeof(val) - 1] = '\0';
        }
        else
        {
            /* Space-separated key value or standalone flag */
            if (sscanf(p, "%127s %511s", key, val) < 1)
            {
                continue;
            }
        }

        int matched = 0;
        for (size_t ii = 0; ii < nopts; ii++)
        {
            int is_negation = 0;
            if (gric_opt_match_key(key, &opts[ii], &is_negation))
            {
                const char *vptr = (val[0] != '\0') ? val : NULL;
                if (apply_parsed_value(&opts[ii], key, vptr, is_negation, custom_ctx) != 0)
                {
                    fprintf(stderr, "Error: Config file %s:%d failed to parse key '%s'\n",
                            filename, line_num, key);
                    fclose(fp);
                    return -1;
                }
                matched = 1;
                break;
            }
        } // for ii

        if (!matched)
        {
            fprintf(stderr, "Warning: Unknown option '%s' in config file %s:%d\n",
                    key, filename, line_num);
        }
    } // while fgets

    fclose(fp);
    return 0;
} // gric_opt_parse_config_file

void gric_opt_print_help(
    FILE                  *fp,
    const char            *prog_name,
    const char            *usage_summary,
    const struct gric_opt *opts,
    size_t                 nopts)
{
    if (fp == NULL || opts == NULL)
    {
        return;
    }

    fprintf(fp, "Usage: %s %s\n\n", prog_name ? prog_name : "program",
            usage_summary ? usage_summary : "[options]");

    /* Identify unique categories and display options grouped by category */
    const char *categories[32];
    size_t ncat = 0;

    for (size_t ii = 0; ii < nopts; ii++)
    {
        const char *cat = opts[ii].category ? opts[ii].category : "General Options";
        int exists = 0;
        for (size_t c = 0; c < ncat; c++)
        {
            if (strcmp(categories[c], cat) == 0)
            {
                exists = 1;
                break;
            }
        }
        if (!exists && ncat < sizeof(categories) / sizeof(categories[0]))
        {
            categories[ncat++] = cat;
        }
    }

    for (size_t c = 0; c < ncat; c++)
    {
        fprintf(fp, "%s%s:%s\n", ANSI_BOLD, categories[c], ANSI_COLOR_RESET);

        for (size_t ii = 0; ii < nopts; ii++)
        {
            const char *cat = opts[ii].category ? opts[ii].category : "General Options";
            if (strcmp(cat, categories[c]) != 0)
            {
                continue;
            }

            char flag_buf[64];
            if (opts[ii].short_name != '\0' && opts[ii].long_name != NULL)
            {
                if (opts[ii].arg_name != NULL)
                {
                    snprintf(flag_buf, sizeof(flag_buf), "-%c, --%s %s",
                             opts[ii].short_name, opts[ii].long_name, opts[ii].arg_name);
                }
                else
                {
                    snprintf(flag_buf, sizeof(flag_buf), "-%c, --%s",
                             opts[ii].short_name, opts[ii].long_name);
                }
            }
            else if (opts[ii].long_name != NULL)
            {
                if (opts[ii].arg_name != NULL)
                {
                    snprintf(flag_buf, sizeof(flag_buf), "    --%s %s",
                             opts[ii].long_name, opts[ii].arg_name);
                }
                else
                {
                    snprintf(flag_buf, sizeof(flag_buf), "    --%s", opts[ii].long_name);
                }
            }
            else if (opts[ii].short_name != '\0')
            {
                if (opts[ii].arg_name != NULL)
                {
                    snprintf(flag_buf, sizeof(flag_buf), "-%c %s",
                             opts[ii].short_name, opts[ii].arg_name);
                }
                else
                {
                    snprintf(flag_buf, sizeof(flag_buf), "-%c", opts[ii].short_name);
                }
            }
            else
            {
                continue;
            }

            fprintf(fp, "  %-30s %s", flag_buf,
                    opts[ii].description ? opts[ii].description : "");

            if (opts[ii].default_str != NULL && opts[ii].default_str[0] != '\0')
            {
                fprintf(fp, " %s(default: %s)%s", ANSI_COLOR_GREY,
                        opts[ii].default_str, ANSI_COLOR_RESET);
            }
            fprintf(fp, "\n");
        } // for ii
        fprintf(fp, "\n");
    } // for c
} // gric_opt_print_help

struct cJSON *gric_opt_to_json(
    const struct gric_opt *opts,
    size_t                 nopts)
{
    if (opts == NULL)
    {
        return NULL;
    }

    cJSON *arr = cJSON_CreateArray();
    if (arr == NULL)
    {
        return NULL;
    }

    for (size_t ii = 0; ii < nopts; ii++)
    {
        cJSON *obj = cJSON_CreateObject();
        if (obj == NULL)
        {
            continue;
        }

        if (opts[ii].long_name != NULL)
        {
            cJSON_AddStringToObject(obj, "name", opts[ii].long_name);
        }

        if (opts[ii].short_name != '\0')
        {
            char s[2] = {opts[ii].short_name, '\0'};
            cJSON_AddStringToObject(obj, "short", s);
        }

        const char *type_str = "string";
        switch (opts[ii].type)
        {
            case GRIC_OPT_FLAG:
                type_str = "flag";
                break;
            case GRIC_OPT_BOOL:
                type_str = "boolean";
                break;
            case GRIC_OPT_INT:
            case GRIC_OPT_INT64:
            case GRIC_OPT_UINT32:
            case GRIC_OPT_UINT64:
                type_str = "integer";
                break;
            case GRIC_OPT_FLOAT:
            case GRIC_OPT_DOUBLE:
                type_str = "number";
                break;
            case GRIC_OPT_CUSTOM:
                type_str = "custom";
                break;
            case GRIC_OPT_STRING:
            default:
                type_str = "string";
                break;
        } // switch
        cJSON_AddStringToObject(obj, "type", type_str);

        if (opts[ii].description != NULL)
        {
            cJSON_AddStringToObject(obj, "description", opts[ii].description);
        }
        if (opts[ii].default_str != NULL)
        {
            cJSON_AddStringToObject(obj, "default", opts[ii].default_str);
        }
        if (opts[ii].arg_name != NULL)
        {
            cJSON_AddStringToObject(obj, "arg_name", opts[ii].arg_name);
        }
        if (opts[ii].category != NULL)
        {
            cJSON_AddStringToObject(obj, "category", opts[ii].category);
        }
        if (opts[ii].negatable)
        {
            cJSON_AddBoolToObject(obj, "negatable", cJSON_True);
        }

        if (opts[ii].aliases != NULL)
        {
            cJSON *aliases_arr = cJSON_CreateArray();
            const char *cur = opts[ii].aliases;
            while (*cur != '\0')
            {
                char token[64];
                size_t len = 0;
                while (*cur != '\0' && *cur != ',' && len < sizeof(token) - 1)
                {
                    token[len++] = *cur++;
                }
                token[len] = '\0';
                cJSON_AddItemToArray(aliases_arr, cJSON_CreateString(token));
                if (*cur == ',')
                {
                    cur++;
                }
            } // while cur
            cJSON_AddItemToObject(obj, "aliases", aliases_arr);
        }

        cJSON_AddItemToArray(arr, obj);
    } // for ii

    return arr;
} // gric_opt_to_json

struct cJSON *gric_opt_to_json_schema(
    const struct gric_opt *opts,
    size_t                 nopts)
{
    if (opts == NULL)
    {
        return NULL;
    }

    cJSON *root = cJSON_CreateObject();
    if (root == NULL)
    {
        return NULL;
    }

    cJSON_AddStringToObject(root, "type", "object");
    cJSON *props = cJSON_CreateObject();
    if (props == NULL)
    {
        cJSON_Delete(root);
        return NULL;
    }
    cJSON_AddItemToObject(root, "properties", props);

    for (size_t ii = 0; ii < nopts; ii++)
    {
        if (opts[ii].long_name == NULL)
        {
            continue;
        }

        cJSON *prop = cJSON_CreateObject();
        if (prop == NULL)
        {
            continue;
        }

        const char *schema_type = "string";
        switch (opts[ii].type)
        {
            case GRIC_OPT_FLAG:
            case GRIC_OPT_BOOL:
                schema_type = "boolean";
                break;
            case GRIC_OPT_INT:
            case GRIC_OPT_INT64:
            case GRIC_OPT_UINT32:
            case GRIC_OPT_UINT64:
                schema_type = "integer";
                break;
            case GRIC_OPT_FLOAT:
            case GRIC_OPT_DOUBLE:
                schema_type = "number";
                break;
            case GRIC_OPT_STRING:
            case GRIC_OPT_CUSTOM:
            default:
                schema_type = "string";
                break;
        } // switch
        cJSON_AddStringToObject(prop, "type", schema_type);

        if (opts[ii].description != NULL)
        {
            char desc_buf[512];
            if (opts[ii].default_str != NULL && opts[ii].default_str[0] != '\0')
            {
                snprintf(desc_buf, sizeof(desc_buf), "%s (default: %s)",
                         opts[ii].description, opts[ii].default_str);
            }
            else
            {
                snprintf(desc_buf, sizeof(desc_buf), "%s", opts[ii].description);
            }
            cJSON_AddStringToObject(prop, "description", desc_buf);
        }

        cJSON_AddItemToObject(props, opts[ii].long_name, prop);
    } // for ii

    return root;
} // gric_opt_to_json_schema

