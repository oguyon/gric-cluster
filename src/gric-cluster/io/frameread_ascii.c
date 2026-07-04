/**
 * @file frameread_ascii.c
 * @brief ASCII format reader implementation.
 */

#include "frameread_internal.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * init_ascii() - Initialize the ASCII format frame reader.
 * @filename: Path to the ASCII (.txt) file containing coordinates.
 *
 * Opens the ASCII file, reads all lines to count the number of frames, records the byte
 * offset of each line for fast random-access seeking, and determines the width (number of
 * columns) from the first line.
 *
 * Return: 0 on success, or -1 on failure.
 */
int init_ascii(
    char *filename)
{
    ascii_ptr = fopen(filename, "r");
    if (ascii_ptr == NULL)
    {
        perror("Failed to open ASCII file");
        return -1;
    }

    is_ascii_mode = 1;
    num_frames = 0;

    size_t capacity = 1024;
    ascii_line_offsets = (long *)malloc(capacity * sizeof(long));
    if (ascii_line_offsets == NULL)
    {
        perror("Memory allocation failed");
        fclose(ascii_ptr);
        ascii_ptr = NULL;
        return -1;
    }

    {
        /* Read lines and index offset locations */
        char *line = NULL;
        size_t len = 0;
        long offset = ftell(ascii_ptr);
        int first_line = 1;

        while (getline(&line, &len, ascii_ptr) != -1)
        {
            if (num_frames >= capacity)
            {
                capacity *= 2;
                long *new_offsets = (long *)realloc(ascii_line_offsets, capacity * sizeof(long));
                if (new_offsets == NULL)
                {
                    perror("Memory reallocation failed");
                    free(line);
                    free(ascii_line_offsets);
                    ascii_line_offsets = NULL;
                    fclose(ascii_ptr);
                    ascii_ptr = NULL;
                    return -1;
                }
                ascii_line_offsets = new_offsets;
            }
            ascii_line_offsets[num_frames] = offset;
            num_frames++;

            if (first_line)
            {
                int cols = 0;
                char *p = line;
                int in_num = 0;
                while (*p)
                {
                    if (!isspace((unsigned char)*p))
                    {
                        if (!in_num)
                        {
                            cols++;
                            in_num = 1;
                        }
                    }
                    else
                    {
                        in_num = 0;
                    }
                    p++;
                }
                frame_width = cols;
                frame_height = 1;
                first_line = 0;
            }

            offset = ftell(ascii_ptr);
        }

        free(line);
    }

    if (num_frames == 0)
    {
        fprintf(stderr, "Error: Empty ASCII file.\n");
        free(ascii_line_offsets);
        ascii_line_offsets = NULL;
        fclose(ascii_ptr);
        ascii_ptr = NULL;
        return -1;
    }

    rewind(ascii_ptr);
    return 0;
}

/**
 * getframe_ascii() - Read a frame from the ASCII file.
 * @frame_struct: Pointer to the Frame struct to populate.
 * @index:        Zero-based index of the frame to retrieve.
 *
 * Seeks to the recorded byte offset of the line in the ASCII file and parses the
 * floating-point values into the frame's data buffer.
 *
 * Return: 0 on success, or -1 on failure.
 */
int getframe_ascii(
    Frame *frame_struct,
    long   index)
{
    long nelements = frame_width * frame_height;

    if (fseek(ascii_ptr, ascii_line_offsets[index], SEEK_SET) != 0)
    {
        perror("fseek failed");
        return -1;
    }

    for (long ii = 0; ii < nelements; ii++)
    {
        if (fscanf(ascii_ptr, "%lf", &frame_struct->data[ii]) != 1)
        {
            return -1;
        }
    }

    return 0;
}

/**
 * close_ascii() - Close the ASCII reader and free line offsets.
 */
void close_ascii(void)
{
    if (ascii_ptr != NULL)
    {
        fclose(ascii_ptr);
        ascii_ptr = NULL;
    }
    if (ascii_line_offsets != NULL)
    {
        free(ascii_line_offsets);
        ascii_line_offsets = NULL;
    }
    is_ascii_mode = 0;
}

/**
 * reset_ascii() - Reset the ASCII file pointer to the beginning.
 */
void reset_ascii(void)
{
    if (ascii_ptr != NULL)
    {
        rewind(ascii_ptr);
    }
}
