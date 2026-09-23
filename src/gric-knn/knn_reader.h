#ifndef KNN_READER_H
#define KNN_READER_H

/**
 * @file knn_reader.h
 * @brief Out-of-core random-access frame reader for FITS and ASCII datasets.
 *
 * Declares the KnnFrameReader structure and file I/O operations for reading frames on-demand.
 * Supports binary formats (GRIC .bin via mmap), multi-extension FITS files, and line-indexed
 * ASCII datasets. Includes thread-local cloning and thread-safe random-access frame retrieval.
 */

#include "knn_defs.h"

#ifdef USE_CFITSIO
#include <fitsio.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** Context for random-access dataset reading */
typedef struct
{
    char         *input_path;
    int           is_fits;
    int           is_bin;
    uint32_t      bin_data_type;
    uint64_t      bin_header_bytes;
    int           use_double;
    const void   *memory_data; /**< Optional in-memory dataset buffer */
    long          total_frames;
    long          frame_width;
    long          frame_height;
    long          frame_elements;
    uint64_t     *line_offsets; /**< 64-bit file byte offsets for ASCII lines */
    FILE         *ascii_file;
    FILE         *bin_file;
    void         *bin_mmap_addr;   /**< Mmap base address for .bin files */
    size_t        bin_mmap_size;   /**< Mmap length in bytes */
#ifdef USE_CFITSIO
    fitsfile     *fits_ptr;
#endif
} KnnFrameReader;

/**
 * knn_reader_inspect() - Discover dataset sample count and coordinate dimensions
 * @path:         Path to dataset file.
 * @total_frames: Output pointer for sample/frame count.
 * @frame_width:  Output pointer for frame width in pixels.
 * @frame_height: Output pointer for frame height in pixels.
 *
 * Inspects header of binary (.bin), FITS, or ASCII dataset without reading full file.
 *
 * Return: 0 on success, or -1 on error.
 */
int knn_reader_inspect(
    const char *path,
    long       *total_frames,
    long       *frame_width,
    long       *frame_height);

/**
 * knn_reader_open() - Open dataset file for random-access frame reading
 * @reader:       Pointer to KnnFrameReader context to initialize.
 * @input_path:   Path to input file (.bin, FITS, or ASCII).
 * @total_frames: Expected total frames in dataset.
 * @frame_width:  Frame width in elements.
 * @frame_height: Frame height in elements.
 * @use_double:   1 for double precision, 0 for single-precision float.
 *
 * Initializes file handles, memory mapping, or ASCII byte offsets.
 *
 * Return: 0 on success, or -1 on error.
 */
int knn_reader_open(
    KnnFrameReader *reader,
    const char     *input_path,
    long            total_frames,
    long            frame_width,
    long            frame_height,
    int             use_double);

/**
 * knn_reader_open_memory() - Open an in-memory dataset buffer for zero-copy random access
 * @reader:         Pointer to KnnFrameReader context to initialize.
 * @memory_data:    Pointer to contiguous frame pixel data.
 * @total_frames:   Number of frames in memory.
 * @frame_elements: Number of elements per frame vector.
 * @use_double:     1 for double precision, 0 for single-precision float.
 *
 * Configures reader for direct memory-backed reading without disk I/O.
 *
 * Return: 0 on success, or -1 on error.
 */
int knn_reader_open_memory(
    KnnFrameReader *reader,
    const void     *memory_data,
    long            total_frames,
    long            frame_elements,
    int             use_double);

/**
 * knn_reader_read_frame() - Read a single frame by index into destination buffer
 * @reader:   Pointer to KnnFrameReader context.
 * @frame_id: 0-based frame index.
 * @out_data: Output pixel buffer of size frame_elements (float* or double*).
 *
 * Reads vector data using mmap, thread-safe FITS reading, or pre-indexed ASCII seek.
 *
 * Return: 0 on success, or -1 on error.
 */
int knn_reader_read_frame(
    KnnFrameReader *reader,
    long            frame_id,
    void           *out_data);

/**
 * knn_reader_clone_thread() - Clone a reader handle for thread-local usage in OpenMP
 * @src: Source KnnFrameReader context.
 * @dst: Destination KnnFrameReader context to initialize.
 *
 * Creates independent thread-local file descriptors sharing base index offsets.
 *
 * Return: 0 on success, or -1 on error.
 */
int knn_reader_clone_thread(
    const KnnFrameReader *src,
    KnnFrameReader       *dst);

/**
 * knn_reader_close_thread() - Close a thread-local cloned reader
 * @reader: Pointer to thread-local KnnFrameReader context.
 *
 * Releases thread-local file descriptors without freeing shared memory maps.
 */
void knn_reader_close_thread(
    KnnFrameReader *reader);

/**
 * knn_reader_close() - Close master reader and release memory and file handles
 * @reader: Pointer to KnnFrameReader context.
 *
 * Unmaps binary files, frees index tables, and closes all open file handles.
 */
void knn_reader_close(
    KnnFrameReader *reader);

#ifdef __cplusplus
}
#endif

#endif // KNN_READER_H
