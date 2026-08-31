/**
 * @file gric_bin_io.c
 * @brief Implementation of I/O utilities for self-describing GRIC binary files.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gric_bin_io.h"

/**
 * gric_bin_write_header() - Write binary header to file.
 * @fp:      Destination file pointer.
 * @hdr:     Pointer to header struct with metadata populated.
 * @comment: Optional description text (or NULL).
 *
 * Return: 0 on success, -1 on error.
 */
int gric_bin_write_header(
    FILE                    *fp,
    const gric_bin_header_t *hdr,
    const char              *comment)
{
    if (fp == NULL || hdr == NULL)
    {
        return -1;
    }

    gric_bin_header_t local_hdr = *hdr;
    memcpy(local_hdr.magic, GRIC_BIN_MAGIC, 4);
    local_hdr.version = GRIC_BIN_VERSION;
    local_hdr.endian = GRIC_BIN_ENDIAN_LITTLE;

    size_t comment_len = (comment != NULL) ? strlen(comment) : 0;
    size_t padded_comment_len = (comment_len > 0) ? ((comment_len + 7) & ~((size_t)7)) : 0;
    local_hdr.header_bytes = (uint16_t)(GRIC_BIN_HEADER_DEFAULT_SIZE + padded_comment_len);

    if (fwrite(&local_hdr, sizeof(gric_bin_header_t), 1, fp) != 1)
    {
        return -1;
    }

    if (padded_comment_len > 0)
    {
        char pad_buf[256];
        memset(pad_buf, 0, sizeof(pad_buf));
        if (comment_len > 0)
        {
            size_t copy_len = (comment_len < sizeof(pad_buf)) ? comment_len : sizeof(pad_buf);
            memcpy(pad_buf, comment, copy_len);
        }
        if (fwrite(pad_buf, 1, padded_comment_len, fp) != padded_comment_len)
        {
            return -1;
        }
    }

    return 0;
}

/**
 * gric_bin_read_header() - Read and validate binary header from file.
 * @fp:          Source file pointer.
 * @hdr:         Pointer to header struct to populate.
 * @comment_out: Optional pointer to store allocated comment string (or NULL).
 *
 * Return: 0 on success, -1 on invalid format or read failure.
 */
int gric_bin_read_header(
    FILE              *fp,
    gric_bin_header_t *hdr,
    char             **comment_out)
{
    if (fp == NULL || hdr == NULL)
    {
        return -1;
    }

    if (comment_out != NULL)
    {
        *comment_out = NULL;
    }

    if (fread(hdr, sizeof(gric_bin_header_t), 1, fp) != 1)
    {
        return -1;
    }

    if (memcmp(hdr->magic, GRIC_BIN_MAGIC, 4) != 0)
    {
        return -1;
    }

    if (hdr->header_bytes > GRIC_BIN_HEADER_DEFAULT_SIZE)
    {
        size_t extra_bytes = hdr->header_bytes - GRIC_BIN_HEADER_DEFAULT_SIZE;
        if (comment_out != NULL)
        {
            char *buf = (char *)malloc(extra_bytes + 1);
            if (buf == NULL)
            {
                return -1;
            }
            if (fread(buf, 1, extra_bytes, fp) != extra_bytes)
            {
                free(buf);
                return -1;
            }
            buf[extra_bytes] = '\0';
            *comment_out = buf;
        }
        else
        {
            if (fseek(fp, (long)extra_bytes, SEEK_CUR) != 0)
            {
                return -1;
            }
        }
    }

    return 0;
}

/**
 * gric_bin_file_type_str() - Convert file type enum to human-readable string.
 * @type: Semantic file type.
 *
 * Return: Constant string describing file type.
 */
const char *gric_bin_file_type_str(
    gric_bin_file_type_t type)
{
    switch (type)
    {
        case GRIC_BIN_TYPE_ANCHORS:     return "ANCHORS";
        case GRIC_BIN_TYPE_DCC:         return "DCC_MATRIX";
        case GRIC_BIN_TYPE_MEMBERSHIP:  return "FRAME_MEMBERSHIP";
        case GRIC_BIN_TYPE_COUNTS:      return "CLUSTER_COUNTS";
        case GRIC_BIN_TYPE_EVALS:       return "FRAME_EVALS";
        case GRIC_BIN_TYPE_COORDINATES: return "COORDINATES";
        case GRIC_BIN_TYPE_GENERIC:
        default:                        return "GENERIC_TENSOR";
    }
}

/**
 * gric_bin_file_type_from_str() - Parse file type enum from name string.
 * @str: String name of type.
 *
 * Return: Corresponding enum value.
 */
gric_bin_file_type_t gric_bin_file_type_from_str(
    const char *str)
{
    if (str == NULL)
    {
        return GRIC_BIN_TYPE_GENERIC;
    }

    if (strcasecmp(str, "anchors") == 0 || strcasecmp(str, "centroids") == 0)
    {
        return GRIC_BIN_TYPE_ANCHORS;
    }
    if (strcasecmp(str, "dcc") == 0 || strcasecmp(str, "dcc_matrix") == 0)
    {
        return GRIC_BIN_TYPE_DCC;
    }
    if (strcasecmp(str, "membership") == 0 || strcasecmp(str, "assign") == 0)
    {
        return GRIC_BIN_TYPE_MEMBERSHIP;
    }
    if (strcasecmp(str, "counts") == 0)
    {
        return GRIC_BIN_TYPE_COUNTS;
    }
    if (strcasecmp(str, "evals") == 0)
    {
        return GRIC_BIN_TYPE_EVALS;
    }
    if (strcasecmp(str, "coords") == 0 || strcasecmp(str, "coordinates") == 0)
    {
        return GRIC_BIN_TYPE_COORDINATES;
    }

    return GRIC_BIN_TYPE_GENERIC;
}

/**
 * gric_bin_data_type_str() - Convert data type enum to string.
 * @dtype: Primitive data type code.
 *
 * Return: Constant string description.
 */
const char *gric_bin_data_type_str(
    gric_bin_data_type_t dtype)
{
    switch (dtype)
    {
        case GRIC_BIN_DTYPE_FLOAT32: return "FLOAT32";
        case GRIC_BIN_DTYPE_FLOAT64: return "FLOAT64";
        case GRIC_BIN_DTYPE_UINT32:  return "UINT32";
        case GRIC_BIN_DTYPE_INT32:   return "INT32";
        case GRIC_BIN_DTYPE_UINT16:  return "UINT16";
        case GRIC_BIN_DTYPE_UINT8:   return "UINT8";
        case GRIC_BIN_DTYPE_UNKNOWN:
        default:                     return "UNKNOWN";
    }
}

/**
 * gric_bin_data_type_size() - Return element byte size for a data type.
 * @dtype: Primitive data type code.
 *
 * Return: Size in bytes (1, 2, 4, or 8).
 */
size_t gric_bin_data_type_size(
    gric_bin_data_type_t dtype)
{
    switch (dtype)
    {
        case GRIC_BIN_DTYPE_FLOAT32: return sizeof(float);
        case GRIC_BIN_DTYPE_FLOAT64: return sizeof(double);
        case GRIC_BIN_DTYPE_UINT32:  return sizeof(uint32_t);
        case GRIC_BIN_DTYPE_INT32:   return sizeof(int32_t);
        case GRIC_BIN_DTYPE_UINT16:  return sizeof(uint16_t);
        case GRIC_BIN_DTYPE_UINT8:   return sizeof(uint8_t);
        case GRIC_BIN_DTYPE_UNKNOWN:
        default:                     return 0;
    }
}

/**
 * gric_bin_print_header_info() - Print formatted metadata summary to stream.
 * @out:     Output stream (e.g. stdout or stderr).
 * @hdr:     Pointer to parsed header.
 * @comment: Optional comment string (or NULL).
 */
void gric_bin_print_header_info(
    FILE                    *out,
    const gric_bin_header_t *hdr,
    const char              *comment)
{
    if (out == NULL || hdr == NULL)
    {
        return;
    }

    fprintf(out, "========================================\n");
    fprintf(out, "       GRIC Binary File Header\n");
    fprintf(out, "========================================\n");
    fprintf(out, "  Magic Signature : %c%c%c%c\n",
            hdr->magic[0], hdr->magic[1], hdr->magic[2], hdr->magic[3]);
    fprintf(out, "  Format Version  : %u\n", hdr->version);
    fprintf(out, "  File Type       : %s (%u)\n",
            gric_bin_file_type_str((gric_bin_file_type_t)hdr->file_type), hdr->file_type);
    fprintf(out, "  Data Type       : %s (%u, %zu bytes/elem)\n",
            gric_bin_data_type_str((gric_bin_data_type_t)hdr->data_type),
            hdr->data_type,
            gric_bin_data_type_size((gric_bin_data_type_t)hdr->data_type));
    fprintf(out, "  Endianness      : %s\n",
            (hdr->endian == GRIC_BIN_ENDIAN_LITTLE) ? "Little-Endian" : "Big-Endian");
    fprintf(out, "  Header Size     : %u bytes\n", hdr->header_bytes);
    fprintf(out, "  Dimensions (ndim): %u\n", hdr->ndim);

    fprintf(out, "  Shape [dims]    : [");
    for (uint16_t i = 0; i < hdr->ndim; i++)
    {
        fprintf(out, "%lu%s", (unsigned long)hdr->dims[i], (i + 1 < hdr->ndim) ? " x " : "");
    }
    fprintf(out, "]\n");

    fprintf(out, "  Layout Flags    : 0x%04X (%s)\n",
            hdr->flags,
            (hdr->flags & GRIC_BIN_FLAG_ROW_MAJOR) ? "Row-Major/Interleaved" : "Standard");
    fprintf(out, "  Total Elements  : %lu\n", (unsigned long)hdr->num_elements);
    fprintf(out, "  Payload Size    : %lu bytes (%.2f MB)\n",
            (unsigned long)hdr->data_bytes,
            (double)hdr->data_bytes / (1024.0 * 1024.0));

    if (comment != NULL && strlen(comment) > 0)
    {
        fprintf(out, "  Comment/Desc    : \"%s\"\n", comment);
    }
    fprintf(out, "========================================\n");
}
