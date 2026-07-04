#ifndef FRAMEREAD_INTERNAL_H
#define FRAMEREAD_INTERNAL_H

#include "common.h"
#include <stdio.h>

#ifdef USE_CFITSIO
#include <fitsio.h>
#endif

#ifdef USE_FFMPEG
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#endif

#ifdef USE_IMAGESTREAMIO
#include <ImageStreamIO/ImageStreamIO.h>
#include <ImageStreamIO/ImageStruct.h>
#endif

/* Reader global state variables */

#ifdef USE_CFITSIO
extern fitsfile *fptr;
#endif

extern FILE *ascii_ptr;
extern long *ascii_line_offsets;
extern int is_ascii_mode;

extern char **file_list;
extern int is_filelist_mode;

#ifdef USE_FFMPEG
extern AVFormatContext *fmt_ctx;
extern AVCodecContext *dec_ctx;
extern int video_stream_idx;
extern AVFrame *frame;
extern AVPacket *pkt;
extern struct SwsContext *sws_ctx;
extern int is_mp4_mode;
extern int internal_mp4_index;
#endif

#ifdef USE_IMAGESTREAMIO
extern IMAGE stream_image;
extern int is_stream_mode;
extern uint64_t last_cnt0;
extern int first_stream_frame;
extern long cumulative_missed_frames;
extern long stream_depth;
extern long current_read_slice;
extern long current_write_slice;
extern long stream_read_counter;
extern int is_3d;
extern double cumulative_wait_time_sec;
extern int cnt2sync_enabled;
#endif

extern long num_frames;
extern long frame_width;
extern long frame_height;
extern int current_frame_idx;

#define FRAME_DATA_POOL_SIZE 64
extern double *frame_data_pool[FRAME_DATA_POOL_SIZE];
extern int frame_data_pool_count;

/* Format-specific helper prototypes */

/**
 * @brief Initialize ASCII format frame reader.
 */
int init_ascii(
    char *filename);

/**
 * @brief Retrieve a frame from the ASCII file at the specified index.
 */
int getframe_ascii(
    Frame *frame_struct,
    long   index);

/**
 * @brief Close resources associated with the ASCII frame reader.
 */
void close_ascii(void);

/**
 * @brief Reset the ASCII frame reader position to the beginning.
 */
void reset_ascii(void);

#ifdef USE_CFITSIO
/**
 * @brief Initialize FITS format frame reader.
 */
int init_fits(
    char *filename);

/**
 * @brief Retrieve a frame from the FITS file at the specified index.
 */
int getframe_fits(
    Frame *frame_struct,
    long   index);

/**
 * @brief Close resources associated with the FITS frame reader.
 */
void close_fits(void);

/**
 * @brief Reset the FITS frame reader position to the beginning.
 */
void reset_fits(void);
#endif // USE_CFITSIO

#ifdef USE_FFMPEG
/**
 * @brief Initialize FFmpeg MP4 format frame reader.
 */
int init_mp4(
    char *filename);

/**
 * @brief Retrieve a frame from the MP4 video stream at the specified index.
 */
int getframe_mp4(
    Frame *frame_struct,
    long   index);

/**
 * @brief Close resources associated with the MP4 frame reader.
 */
void close_mp4(void);

/**
 * @brief Reset the MP4 frame reader position to the beginning.
 */
void reset_mp4(void);
#endif // USE_FFMPEG

#ifdef USE_IMAGESTREAMIO
/**
 * @brief Initialize ImageStreamIO streaming frame reader.
 */
int init_stream(
    char *stream_name);

/**
 * @brief Retrieve a frame from the shared memory stream at the specified index.
 */
int getframe_stream(
    Frame *frame_struct,
    long   index);

/**
 * @brief Close resources associated with the ImageStreamIO frame reader.
 */
void close_stream(void);

/**
 * @brief Reset the ImageStreamIO frame reader position/counter.
 */
void reset_stream(void);
#endif // USE_IMAGESTREAMIO

#endif // FRAMEREAD_INTERNAL_H
