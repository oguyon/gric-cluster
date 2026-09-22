/**
 * @file tool_align_parameters.c
 * @brief Column-alignment validator and formatter for multi-line C function prototypes.
 */

#include "mcp_tools.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PARAMS 32

typedef struct
{
    char type_prefix[256];
    char param_name[128];
} ParamEntry;

/**
 * trim() - Remove leading and trailing whitespace from string in-place.
 * @str: String buffer.
 *
 * Return: Pointer to first non-whitespace character.
 */
static char *trim(
    char *str)
{
    while (*str && isspace((unsigned char)*str))
    {
        str++;
    }
    if (*str == '\0')
    {
        return str;
    }
    char *end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end))
    {
        *end = '\0';
        end--;
    }
    return str;
} // trim

/**
 * parse_param() - Decompose a parameter string into type prefix and parameter name.
 * @raw:   Raw parameter string (e.g. "const float *restrict in" or "char *tagname").
 * @entry: Target entry to store parsed components.
 *
 * Return: 0 on success, -1 on parse failure.
 */
static int parse_param(
    const char *raw,
    ParamEntry *entry)
{
    char buf[512];
    strncpy(buf, raw, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    char *clean = trim(buf);
    size_t len = strlen(clean);
    if (len == 0)
    {
        return -1;
    }

    /* Strip trailing comma if present */
    if (clean[len - 1] == ',')
    {
        clean[len - 1] = '\0';
        clean = trim(clean);
        len = strlen(clean);
    }

    if (strcmp(clean, "void") == 0)
    {
        strncpy(entry->type_prefix, "void", sizeof(entry->type_prefix) - 1);
        entry->param_name[0] = '\0';
        return 0;
    }

    /* Find the last token (the parameter name) */
    char *last_space = NULL;
    char *last_star = NULL;

    for (size_t ii = 0; ii < len; ii++)
    {
        if (isspace((unsigned char)clean[ii]))
        {
            last_space = &clean[ii];
        }
        else if (clean[ii] == '*')
        {
            last_star = &clean[ii];
        }
    } // for ii

    char *name_start = NULL;
    if (last_space != NULL && (last_star == NULL || last_space > last_star))
    {
        *last_space = '\0';
        name_start = trim(last_space + 1);
    }
    else if (last_star != NULL)
    {
        *last_star = '\0';
        name_start = trim(last_star + 1);
        /* Append star back to type prefix */
        strcat(clean, " *");
    }
    else
    {
        return -1;
    }

    strncpy(entry->type_prefix, trim(clean), sizeof(entry->type_prefix) - 1);
    strncpy(entry->param_name, name_start, sizeof(entry->param_name) - 1);

    return 0;
} // parse_param

int mcp_tool_align_parameters(
    const cJSON *args,
    cJSON       *res)
{
    if (args == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Missing arguments");
        return -1;
    }

    cJSON *proto_item = cJSON_GetObjectItemCaseSensitive(args, "prototype");
    if (proto_item == NULL || !cJSON_IsString(proto_item))
    {
        cJSON_AddStringToObject(res, "error", "prototype string is required");
        return -1;
    }

    const char *proto_str = proto_item->valuestring;
    const char *open_paren = strchr(proto_str, '(');
    const char *close_paren = strrchr(proto_str, ')');

    if (open_paren == NULL || close_paren == NULL || close_paren <= open_paren)
    {
        cJSON_AddStringToObject(res, "error", "Invalid prototype format (missing parentheses)");
        return -1;
    }

    /* Extract return type and function name */
    char header[512];
    size_t header_len = (size_t)(open_paren - proto_str + 1);
    if (header_len >= sizeof(header))
    {
        header_len = sizeof(header) - 1;
    }
    strncpy(header, proto_str, header_len);
    header[header_len] = '\0';

    /* Parse parameter strings between parentheses */
    char params_buf[2048];
    size_t params_len = (size_t)(close_paren - open_paren - 1);
    if (params_len >= sizeof(params_buf))
    {
        params_len = sizeof(params_buf) - 1;
    }
    strncpy(params_buf, open_paren + 1, params_len);
    params_buf[params_len] = '\0';

    ParamEntry entries[MAX_PARAMS];
    int param_count = 0;
    size_t max_type_len = 0;

    /* Tokenize by line or comma */
    char *saveptr = NULL;
    char *line = strtok_r(params_buf, "\n\r", &saveptr);
    while (line != NULL && param_count < MAX_PARAMS)
    {
        char *t = trim(line);
        if (strlen(t) > 0)
        {
            if (parse_param(t, &entries[param_count]) == 0)
            {
                size_t t_len = strlen(entries[param_count].type_prefix);
                if (t_len > max_type_len)
                {
                    max_type_len = t_len;
                }
                param_count++;
            }
        }
        line = strtok_r(NULL, "\n\r", &saveptr);
    } // while line

    if (param_count <= 1)
    {
        cJSON_AddStringToObject(res, "status", "ALIGNED");
        cJSON_AddStringToObject(res, "note", "Single parameter or void (no alignment required)");
        cJSON_AddStringToObject(res, "formatted", proto_str);
        return 0;
    }

    /* Build aligned output string */
    char output[4096];
    int offset = snprintf(output, sizeof(output), "%s\n", header);

    for (int ii = 0; ii < param_count; ii++)
    {
        int pad = (int)(max_type_len - strlen(entries[ii].type_prefix) + 1);
        offset += snprintf(
            output + offset, sizeof(output) - (size_t)offset,
            "    %s%*s%s%s\n",
            entries[ii].type_prefix,
            pad, "",
            entries[ii].param_name,
            (ii == param_count - 1) ? ")" : ",");
    } // for ii

    cJSON_AddStringToObject(res, "status", "FORMATTED");
    cJSON_AddNumberToObject(res, "parameter_count", param_count);
    cJSON_AddNumberToObject(res, "alignment_column", (double)(max_type_len + 5));
    cJSON_AddStringToObject(res, "formatted", output);

    return 0;
} // mcp_tool_align_parameters
