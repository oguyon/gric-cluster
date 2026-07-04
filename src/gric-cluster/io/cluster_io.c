/**
 * @file cluster_io.c
 * @brief Common filesystem and helper functions for cluster I/O.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "cluster_io.h"
#include "shared/cli_colors.h"

/**
 * create_output_dir_name() - Build output directory path from input filename.
 * @input_file: Path to the input file (FITS, MP4, or text).
 *
 * Strips the directory prefix and recognized extensions
 * (.fits.fz, .fits, .mp4, .txt) from @input_file, then
 * appends ".clusterdat" to form the output directory name.
 *
 * Return: Heap-allocated directory name string, or NULL on
 *         allocation failure.  Caller must free().
 */
char *create_output_dir_name(
    const char *input_file)
{
    const char *base = strrchr(input_file, '/');

    if (base)
    {
        base++;
    }
    else
    {
        base = input_file;
    }

    char *name = strdup(base);

    if (!name)
    {
        return NULL;
    }

    size_t len = strlen(name);

    if (len > 8 && strcmp(name + len - 8, ".fits.fz") == 0)
    {
        name[len - 8] = '\0';
    }
    else if (len > 5 && strcmp(name + len - 5, ".fits") == 0)
    {
        name[len - 5] = '\0';
    }
    else if (len > 4 && strcmp(name + len - 4, ".mp4") == 0)
    {
        name[len - 4] = '\0';
    }
    else if (len > 4 && strcmp(name + len - 4, ".txt") == 0)
    {
        name[len - 4] = '\0';
    }

    size_t new_len = strlen(name) + strlen(".clusterdat") + 1;
    char *out_dir = (char *)malloc(new_len);

    if (out_dir)
    {
        sprintf(out_dir, "%s.clusterdat", name);
    }

    free(name);

    return out_dir;
}

/**
 * safe_mkdir() - Create directory if it doesn't already exist.
 * @path: Path to the directory to create.
 *
 * Return: 0 on success, or -1 on failure.
 */
int safe_mkdir(
    const char *path)
{
    struct stat st = {0};

    if (stat(path, &st) == -1)
    {
        return mkdir(path, 0777);
    }

    return 0;
}

/**
 * init_colors_io() - Initialize color support by calling shared init.
 */
void init_colors_io(void)
{
    cli_colors_init();
}