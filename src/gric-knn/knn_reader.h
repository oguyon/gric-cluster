#ifndef KNN_READER_H
#define KNN_READER_H

/**
 * @file knn_reader.h
 * @brief Out-of-core random-access frame reader for FITS and ASCII datasets.
 */

#include "knn_defs.h"

#ifdef USE_CFITSIO
#include <fitsio.h>
#endif

/** Context for random-access dataset reading */
typedef struct
{
    char         *input_path;
    int           is_fits;
    int           is_bin;
    uint32_t      bin_data_type;
    uint64_t      bin_header_bytes;
    const double *memory_data; /**< Optional in-memory dataset buffer */
    long          total_frames;
    long          frame_width;
    long          frame_height;
    long          frame_elements;
    uint64_t     *line_offsets; /**< 64-bit file byte offsets for ASCII lines */
    FILE         *ascii_file;
    FILE         *bin_file;
#ifdef USE_CFITSIO
    fitsfile     *fits_ptr;
#endif
} KnnFrameReader;

int knn_reader_inspect(
    const char *path,
    long       *total_frames,
    long       *frame_width,
    long       *frame_height);

int knn_reader_open(
    KnnFrameReader *reader,
    const char     *input_path,
    long            total_frames,
    long            frame_width,
    long            frame_height);

int knn_reader_open_memory(
    KnnFrameReader *reader,
    const double   *memory_data,
    long            total_frames,
    long            frame_elements);

int knn_reader_read_frame(
    KnnFrameReader *reader,
    long            frame_id,
    double         *out_data);

int knn_reader_clone_thread(
    const KnnFrameReader *src,
    KnnFrameReader       *dst);

void knn_reader_close_thread(
    KnnFrameReader *reader);

void knn_reader_close(
    KnnFrameReader *reader);

#endif // KNN_READER_H
