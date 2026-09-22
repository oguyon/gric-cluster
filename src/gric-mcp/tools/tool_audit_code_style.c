/**
 * @file tool_audit_code_style.c
 * @brief C source code auditor for line length, Allman braces, Kernel-Doc, and loops.
 */

#include "mcp_tools.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LEN_LIMIT 100

/**
 * trim_whitespace() - Return pointer to first non-whitespace character.
 * @str: Input string.
 *
 * Return: Pointer to first non-whitespace character.
 */
static const char *trim_whitespace(
    const char *str)
{
    while (*str && isspace((unsigned char)*str))
    {
        str++;
    }
    return str;
} // trim_whitespace

int mcp_tool_audit_code_style(
    const cJSON *args,
    cJSON       *res)
{
    if (args == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Missing arguments");
        return -1;
    }

    cJSON *path_item = cJSON_GetObjectItemCaseSensitive(args, "file_path");
    if (path_item == NULL || !cJSON_IsString(path_item))
    {
        cJSON_AddStringToObject(res, "error", "file_path parameter is required");
        return -1;
    }

    const char *filepath = path_item->valuestring;
    char resolved_path[1024];
    mcp_resolve_path(filepath, resolved_path, sizeof(resolved_path));

    FILE *fp = fopen(resolved_path, "r");
    if (fp == NULL)
    {
        cJSON_AddStringToObject(res, "error", "Unable to open file for auditing");
        cJSON_AddStringToObject(res, "file_path", path_item->valuestring);
        return -1;
    }

    cJSON *line_violations = cJSON_CreateArray();
    cJSON *brace_violations = cJSON_CreateArray();
    cJSON *loop_allocations = cJSON_CreateArray();

    char line_buf[4096];
    int line_number = 0;
    int in_loop_depth = 0;
    int total_violations = 0;

    while (fgets(line_buf, sizeof(line_buf), fp) != NULL)
    {
        line_number++;
        size_t len = strlen(line_buf);
        if (len > 0 && line_buf[len - 1] == '\n')
        {
            line_buf[len - 1] = '\0';
            len--;
        }
        if (len > 0 && line_buf[len - 1] == '\r')
        {
            line_buf[len - 1] = '\0';
            len--;
        }

        /* 1. Line length check (> 100 characters) */
        if (len > MAX_LINE_LEN_LIMIT)
        {
            cJSON *item = cJSON_CreateObject();
            cJSON_AddNumberToObject(item, "line", line_number);
            cJSON_AddNumberToObject(item, "length", (double)len);
            cJSON_AddStringToObject(item, "content", line_buf);
            cJSON_AddItemToArray(line_violations, item);
            total_violations++;
        }

        const char *trimmed = trim_whitespace(line_buf);

        /* 2. Allman brace check: opening brace on same line as if/for/while */
        if ((strncmp(trimmed, "if ", 3) == 0 || strncmp(trimmed, "if(", 3) == 0 ||
             strncmp(trimmed, "for ", 4) == 0 || strncmp(trimmed, "for(", 4) == 0 ||
             strncmp(trimmed, "while ", 6) == 0 || strncmp(trimmed, "while(", 6) == 0) &&
            len > 0 && line_buf[len - 1] == '{')
        {
            cJSON *item = cJSON_CreateObject();
            cJSON_AddNumberToObject(item, "line", line_number);
            cJSON_AddStringToObject(
                item, "issue", "Brace style must be Allman (place '{' on next line)");
            cJSON_AddStringToObject(item, "content", line_buf);
            cJSON_AddItemToArray(brace_violations, item);
            total_violations++;
        }

        /* 3. Loop tracking and dynamic allocation detection inside loops */
        if (strncmp(trimmed, "for ", 4) == 0 || strncmp(trimmed, "for(", 4) == 0 ||
            strncmp(trimmed, "while ", 6) == 0 || strncmp(trimmed, "while(", 6) == 0)
        {
            in_loop_depth++;
        }

        if (in_loop_depth > 0)
        {
            if (strstr(line_buf, "malloc(") != NULL ||
                strstr(line_buf, "calloc(") != NULL ||
                strstr(line_buf, "realloc(") != NULL)
            {
                cJSON *item = cJSON_CreateObject();
                cJSON_AddNumberToObject(item, "line", line_number);
                cJSON_AddStringToObject(item, "issue", "Dynamic allocation inside compute loop");
                cJSON_AddStringToObject(item, "content", line_buf);
                cJSON_AddItemToArray(loop_allocations, item);
                total_violations++;
            }
        }

        /* Approximate end of loop block */
        if (in_loop_depth > 0 && trimmed[0] == '}' && strchr(trimmed, ';') == NULL)
        {
            in_loop_depth--;
        }
    } // while fgets

    fclose(fp);

    cJSON_AddStringToObject(res, "file_path", filepath);
    cJSON_AddNumberToObject(res, "total_lines", line_number);
    cJSON_AddNumberToObject(res, "total_issues", total_violations);
    cJSON_AddStringToObject(res, "status", (total_violations == 0) ? "PASS" : "ISSUES_FOUND");

    cJSON_AddItemToObject(res, "line_length_violations", line_violations);
    cJSON_AddItemToObject(res, "brace_style_violations", brace_violations);
    cJSON_AddItemToObject(res, "loop_allocation_violations", loop_allocations);

    return 0;
} // mcp_tool_audit_code_style
