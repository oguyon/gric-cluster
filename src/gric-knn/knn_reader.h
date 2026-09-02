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

/**
 * @brief Inspect dataset file header to determine total frame count and dimensions.
 * @param path         File path to dataset.
 * @param total_frames Output for total frame count.
 * @param frame_width  Output for frame width in pixels.
 * @param frame_height Output for frame height in pixels.
 * @return 0 on success, -1 on error.
 */
int knn_reader_inspect(
    const char *path,
    long       *total_frames,
    long       *frame_width,
    long       *frame_height);

/**
 * @brief Open dataset file for random-access frame reading.
 * @param reader       Pointer to KnnFrameReader context.
 * @param input_path   Path to input file.
 * @param total_frames Expected total frames.
 * @param frame_width  Frame width.
 * @param frame_height Frame height.
 * @return 0 on success, -1 on error.
 */
int knn_reader_open(
    KnnFrameReader *reader,
    const char     *input_path,
    long            total_frames,
    long            frame_width,
    long            frame_height);

/**
 * @brief Open an in-memory dataset buffer for zero-copy random access.
 * @param reader         Pointer to KnnFrameReader context.
 * @param memory_data    Pointer to contiguous frame pixel data.
 * @param total_frames   Number of frames.
 * @param frame_elements Number of elements per frame.
 * @return 0 on success, -1 on error.
 */
int knn_reader_open_memory(
    KnnFrameReader *reader,
    const double   *memory_data,
    long            total_frames,
    long            frame_elements);

/**
 * @brief Read a single frame by index into the destination buffer.
 * @param reader   Pointer to KnnFrameReader context.
 * @param frame_id 0-based frame index.
 * @param out_data Output pixel buffer of size frame_elements.
 * @return 0 on success, -1 on error.
 */
int knn_reader_read_frame(
    KnnFrameReader *reader,
    long            frame_id,
    double         *out_data);

/**
 * @brief Clone a reader handle for thread-local usage in parallel OpenMP workers.
 * @param src Source KnnFrameReader.
 * @param dst Destination KnnFrameReader.
 * @return 0 on success, -1 on error.
 */
int knn_reader_clone_thread(
    const KnnFrameReader *src,
    KnnFrameReader       *dst);

/**
 * @brief Close a thread-local cloned reader.
 * @param reader Pointer to KnnFrameReader context.
 */
void knn_reader_close_thread(
    KnnFrameReader *reader);

/**
 * @brief Close master reader and release memory and file handles.
 * @param reader Pointer to KnnFrameReader context.
 */
void knn_reader_close(
    KnnFrameReader *reader);

#endif // KNN_READER_H
