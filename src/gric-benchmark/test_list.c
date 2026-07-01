#include "benchmark.h"
#include <ctype.h>
#include <stdio.h>
#include <string.h>

/**
 * @brief Load the list of test patterns from an external file.
 *
 * Each non-empty line of the file that does not start with '#' is treated as a pattern.
 * Leading and trailing whitespace is stripped.
 *
 * @param filepath      Path to the test list file.
 * @param patterns      Array of pattern strings to populate.
 * @param pattern_count Pointer to the pattern count to be updated.
 * @param max_patterns  Maximum capacity of the patterns array.
 * @return 0 on success, non-zero on failure.
 */
int load_test_file(
    const char  *filepath,
    char        *patterns[],
    int         *pattern_count,
    int          max_patterns)
{
    FILE *fp = fopen(filepath, "r");
    if (fp == NULL)
    {
        return -1;
    }

    char line[1024];
    while (fgets(line, sizeof(line), fp) != NULL)
    {
        /* Trim leading whitespace */
        char *start = line;
        while (*start != '\0' && isspace((unsigned char)*start))
        {
            start++;
        }

        /* Trim trailing whitespace */
        char *end = start + strlen(start) - 1;
        while (end >= start && isspace((unsigned char)*end))
        {
            *end = '\0';
            end--;
        }

        /* Skip empty or comment lines */
        if (*start == '\0' || *start == '#')
        {
            continue;
        }

        if (*pattern_count < max_patterns)
        {
            patterns[*pattern_count] = strdup(start);
            if (patterns[*pattern_count] != NULL)
            {
                (*pattern_count)++;
            }
        }
        else
        {
            fprintf(stderr,
                    "Warning: Maximum patterns (%d) reached. Ignoring remaining.\n",
                    max_patterns);
            break;
        }
    }

    fclose(fp);
    return 0;
}
