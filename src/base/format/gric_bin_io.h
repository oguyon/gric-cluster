/**
 * @file gric_bin_io.h
 * @brief I/O and utility functions for self-describing GRIC binary data files.
 */

#ifndef GRIC_BIN_IO_H
#define GRIC_BIN_IO_H

#include <stdio.h>
#include <stddef.h>
#include "gric_bin_header.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * gric_bin_write_header() - Write binary header to file.
 */
int gric_bin_write_header(
    FILE                    *fp,
    const gric_bin_header_t *hdr,
    const char              *comment);

/**
 * gric_bin_read_header() - Read and validate binary header from file.
 */
int gric_bin_read_header(
    FILE              *fp,
    gric_bin_header_t *hdr,
    char             **comment_out);

/**
 * gric_bin_file_type_str() - Convert file type enum to human-readable string.
 */
const char *gric_bin_file_type_str(
    gric_bin_file_type_t type);

/**
 * gric_bin_file_type_from_str() - Parse file type enum from name string.
 */
gric_bin_file_type_t gric_bin_file_type_from_str(
    const char *str);

/**
 * gric_bin_data_type_str() - Convert data type enum to string.
 */
const char *gric_bin_data_type_str(
    gric_bin_data_type_t dtype);

/**
 * gric_bin_data_type_size() - Return element byte size for a data type.
 */
size_t gric_bin_data_type_size(
    gric_bin_data_type_t dtype);

/**
 * gric_bin_print_header_info() - Print formatted metadata summary to stream.
 */
void gric_bin_print_header_info(
    FILE                    *out,
    const gric_bin_header_t *hdr,
    const char              *comment);

#ifdef __cplusplus
}
#endif

#endif /* GRIC_BIN_IO_H */
