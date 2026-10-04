/**
 * @file mcp_validate.c
 * @brief Input validation and strict parsing primitives for gric-mcp.
 */

#include "mcp_validate.h"
#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

int mcp_valid_identifier(
    const char *s,
    size_t      max_len)
{
    if (s == NULL || *s == '\0')
    {
        return 0;
    }

    size_t len = strlen(s);
    if (len > max_len)
    {
        return 0;
    }

    if (strstr(s, "..") != NULL)
    {
        return 0;
    }

    for (size_t ii = 0; ii < len; ii++)
    {
        unsigned char c = (unsigned char)s[ii];
        if (!isalnum(c) && c != '_' && c != '-' && c != '.')
        {
            return 0;
        }
    }

    return 1;
} // mcp_valid_identifier

int mcp_valid_path(
    const char *s,
    size_t      max_len)
{
    if (s == NULL || *s == '\0')
    {
        return 0;
    }

    size_t len = strlen(s);
    if (len > max_len)
    {
        return 0;
    }

    for (size_t ii = 0; ii < len; ii++)
    {
        unsigned char c = (unsigned char)s[ii];
        if (!isalnum(c) && c != '_' && c != '-' && c != '.' && c != '/')
        {
            return 0;
        }
    }

    return 1;
} // mcp_valid_path

int mcp_parse_double_strict(
    const char *s,
    double     *out)
{
    if (s == NULL || *s == '\0')
    {
        return -1;
    }

    while (isspace((unsigned char)*s))
    {
        s++;
    }
    if (*s == '\0')
    {
        return -1;
    }

    char *endptr = NULL;
    errno = 0;
    double val = strtod(s, &endptr);
    if (errno != 0 || endptr == s)
    {
        return -1;
    }

    while (isspace((unsigned char)*endptr))
    {
        endptr++;
    }

    if (*endptr != '\0')
    {
        return -1;
    }

    if (out != NULL)
    {
        *out = val;
    }

    return 0;
} // mcp_parse_double_strict

int mcp_parse_int64_strict(
    const char *s,
    int64_t    *out)
{
    if (s == NULL || *s == '\0')
    {
        return -1;
    }

    while (isspace((unsigned char)*s))
    {
        s++;
    }
    if (*s == '\0')
    {
        return -1;
    }

    char *endptr = NULL;
    errno = 0;
    long long val = strtoll(s, &endptr, 10);
    if (errno != 0 || endptr == s)
    {
        return -1;
    }

    while (isspace((unsigned char)*endptr))
    {
        endptr++;
    }

    if (*endptr != '\0')
    {
        return -1;
    }

    if (out != NULL)
    {
        *out = (int64_t)val;
    }

    return 0;
} // mcp_parse_int64_strict

int mcp_parse_onoff(
    const cJSON *v,
    int         *out)
{
    if (v == NULL || out == NULL)
    {
        return -1;
    }

    if (cJSON_IsBool(v))
    {
        *out = cJSON_IsTrue(v) ? 1 : 0;
        return 0;
    }

    if (cJSON_IsNumber(v))
    {
        *out = (v->valueint != 0 || v->valuedouble != 0.0) ? 1 : 0;
        return 0;
    }

    if (cJSON_IsString(v) && v->valuestring != NULL)
    {
        const char *s = v->valuestring;
        if (strcasecmp(s, "1") == 0 || strcasecmp(s, "true") == 0 ||
            strcasecmp(s, "on") == 0 || strcasecmp(s, "yes") == 0 ||
            strcasecmp(s, "enable") == 0 || strcasecmp(s, "enabled") == 0)
        {
            *out = 1;
            return 0;
        }
        if (strcasecmp(s, "0") == 0 || strcasecmp(s, "false") == 0 ||
            strcasecmp(s, "off") == 0 || strcasecmp(s, "no") == 0 ||
            strcasecmp(s, "disable") == 0 || strcasecmp(s, "disabled") == 0)
        {
            *out = 0;
            return 0;
        }
    }

    return -1;
} // mcp_parse_onoff
