/**
 * @file gric_bin_header.h
 * @brief Self-describing 64-byte binary header definition for GRIC data files.
 */

#ifndef GRIC_BIN_HEADER_H
#define GRIC_BIN_HEADER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GRIC_BIN_MAGIC "GRIC"
#define GRIC_BIN_VERSION 1
#define GRIC_BIN_HEADER_DEFAULT_SIZE 64

/**
 * enum gric_bin_file_type - Semantic category of data in binary file.
 */
typedef enum
{
    GRIC_BIN_TYPE_GENERIC     = 0,
    GRIC_BIN_TYPE_ANCHORS     = 1,
    GRIC_BIN_TYPE_DCC         = 2,
    GRIC_BIN_TYPE_MEMBERSHIP  = 3,
    GRIC_BIN_TYPE_COUNTS      = 4,
    GRIC_BIN_TYPE_EVALS       = 5,
    GRIC_BIN_TYPE_COORDINATES = 6
} gric_bin_file_type_t;

/**
 * enum gric_bin_data_type - Primitive element data type.
 */
typedef enum
{
    GRIC_BIN_DTYPE_UNKNOWN = 0,
    GRIC_BIN_DTYPE_FLOAT32 = 1,
    GRIC_BIN_DTYPE_FLOAT64 = 2,
    GRIC_BIN_DTYPE_UINT32  = 3,
    GRIC_BIN_DTYPE_INT32   = 4,
    GRIC_BIN_DTYPE_UINT16  = 5,
    GRIC_BIN_DTYPE_UINT8   = 6
} gric_bin_data_type_t;

/**
 * enum gric_bin_endian - Byte endianness of numeric payload.
 */
typedef enum
{
    GRIC_BIN_ENDIAN_LITTLE = 1,
    GRIC_BIN_ENDIAN_BIG    = 2
} gric_bin_endian_t;

/**
 * enum gric_bin_flags - Bit flags for array layout and properties.
 */
typedef enum
{
    GRIC_BIN_FLAG_ROW_MAJOR = 0x0001,
    GRIC_BIN_FLAG_COL_MAJOR = 0x0002,
    GRIC_BIN_FLAG_SYMMETRIC = 0x0004,
    GRIC_BIN_FLAG_SPARSE    = 0x0008
} gric_bin_flags_t;

/**
 * struct gric_bin_header - 64-byte self-describing binary header.
 * @magic:        Magic string "GRIC" (4 bytes).
 * @version:      Header format version (1 byte).
 * @file_type:    Semantic file category code (1 byte).
 * @data_type:    Primitive element type code (1 byte).
 * @endian:       Byte endianness (1 = Little-Endian, 2 = Big-Endian).
 * @header_bytes: Total header size in bytes including optional comments (uint16_t).
 * @ndim:         Number of dimensional axes (uint16_t, 1 to 4).
 * @flags:        Layout and property flags (uint32_t).
 * @num_elements: Total count of elements in payload (uint64_t).
 * @data_bytes:   Payload length in bytes (uint64_t).
 * @dims:         Extent along each axis [dim0, dim1, dim2, dim3] (4 x uint64_t).
 */
typedef struct __attribute__((packed))
{
    char     magic[4];
    uint8_t  version;
    uint8_t  file_type;
    uint8_t  data_type;
    uint8_t  endian;
    uint16_t header_bytes;
    uint16_t ndim;
    uint32_t flags;
    uint64_t num_elements;
    uint64_t data_bytes;
    uint64_t dims[4];
} gric_bin_header_t;

#ifdef __cplusplus
}
#endif

#endif /* GRIC_BIN_HEADER_H */
