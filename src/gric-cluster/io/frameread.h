#ifndef FRAMEREAD_H
#define FRAMEREAD_H

#include "common.h"

/**
 * @brief Initialize the frame reader with the specified source and options.
 */
int init_frameread(
    char *filename,
    int   stream_mode,
    int   cnt2sync_mode,
    int   filelist_mode);

/**
 * @brief Retrieve the next sequential frame.
 */
Frame *getframe(void);

/**
 * @brief Retrieve a frame at a specific index.
 */
Frame *getframe_at(
    long index);

/**
 * @brief Free resources associated with a Frame structure.
 */
void free_frame(
    Frame *frame);

/**
 * @brief Close resources associated with the active frame reader.
 */
void close_frameread(void);

/**
 * @brief Reset the frame reader to the first frame.
 */
void reset_frameread(void);

/**
 * @brief Get the total number of frames in the source.
 */
long get_num_frames(void);

/**
 * @brief Get the number of frames missed during streaming.
 */
long get_missed_frames(void);

/**
 * @brief Get the index of the stream slice currently being read.
 */
long get_stream_read_slice(void);

/**
 * @brief Get the index of the stream slice currently being written.
 */
long get_stream_write_slice(void);

/**
 * @brief Get the stream processing lag.
 */
long get_stream_lag(void);

/**
 * @brief Check if the active stream is in 3D mode.
 */
int is_3d_stream_mode(void);

/**
 * @brief Get the cumulative time spent waiting for stream frames.
 */
double get_stream_wait_time(void);

/**
 * @brief Get the width of the ingested frames.
 */
long get_frame_width(void);

/**
 * @brief Get the height of the ingested frames.
 */
long get_frame_height(void);

/**
 * @brief Check if the reader is in ASCII input mode.
 */
int is_ascii_input_mode(void);

#endif // FRAMEREAD_H
