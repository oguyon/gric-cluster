/**
 * @file frameread.c
 * @brief Dispatcher for coordinate and image frame reader formats.
 */

#include "common.h"
#include "frameread.h"
#include "png_io.h"
#include "frameread_internal.h"
#include "gric_bin_io.h"
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef USE_CFITSIO
fitsfile *fptr = NULL;
#endif

FILE *ascii_ptr = NULL;
long *ascii_line_offsets = NULL;
int is_ascii_mode = 0;

static FILE *bin_input_ptr = NULL;
static uint64_t bin_input_data_offset = 64;
static gric_bin_data_type_t bin_input_dtype = GRIC_BIN_DTYPE_FLOAT32;
static int is_bin_mode = 0;

char **file_list = NULL;
int is_filelist_mode = 0;

#ifdef USE_FFMPEG
AVFormatContext *fmt_ctx = NULL;
AVCodecContext *dec_ctx = NULL;
int video_stream_idx = -1;
AVFrame *frame = NULL;
AVPacket *pkt = NULL;
struct SwsContext *sws_ctx = NULL;
int is_mp4_mode = 0;
int internal_mp4_index = 0;
#endif

#ifdef USE_IMAGESTREAMIO
IMAGE stream_image;
int is_stream_mode = 0;
uint64_t last_cnt0 = 0;
int first_stream_frame = 1;
long cumulative_missed_frames = 0;
long stream_depth = 1;
long current_read_slice = 0;
long current_write_slice = 0;
long stream_read_counter = 0;
int is_3d = 0;
double cumulative_wait_time_sec = 0.0;
int cnt2sync_enabled = 0;
#endif

long num_frames = 0;
long frame_width = 0;
long frame_height = 0;
int current_frame_idx = 0;

double *frame_data_pool[FRAME_DATA_POOL_SIZE];
int frame_data_pool_count = 0;

/**
 * init_filelist() - Initialize the frame reader in file list (PNG sequence) mode.
 * @filename: Path to the text file listing PNG paths (one per line).
 *
 * Reads filenames into memory, queries the first frame to discover width and height,
 * and sets up list-based frame ingestion.
 *
 * Return: 0 on success, or -1 on failure.
 */
static int init_filelist(
    char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
        perror("Failed to open file list");
        return -1;
    }

    is_filelist_mode = 1;
    num_frames = 0;
    size_t capacity = 1024;
    file_list = (char **)malloc(capacity * sizeof(char *));
    if (file_list == NULL)
    {
        perror("Memory allocation failed");
        fclose(fp);
        return -1;
    }

    {
        /* Read line entries */
        char *line = NULL;
        size_t len = 0;
        ssize_t read_bytes;

        while ((read_bytes = getline(&line, &len, fp)) != -1)
        {
            if (read_bytes > 0 && line[read_bytes - 1] == '\n')
            {
                line[read_bytes - 1] = '\0';
            }
            if (strlen(line) == 0)
            {
                continue;
            }

            if ((size_t)num_frames >= capacity)
            {
                capacity *= 2;
                char **new_list = (char **)realloc(file_list, capacity * sizeof(char *));
                if (new_list == NULL)
                {
                    perror("Memory reallocation failed");
                    free(line);
                    fclose(fp);
                    return -1;
                }
                file_list = new_list;
            }
            file_list[num_frames] = strdup(line);
            num_frames++;
        }
        free(line);
    }
    fclose(fp);

    if (num_frames == 0)
    {
        fprintf(stderr, "Error: Empty file list.\n");
        return -1;
    }

    {
        /* Read first PNG frame to get dimensions */
        int w = 0;
        int h = 0;
        double *tmp = read_png_frame(file_list[0], &w, &h);
        if (tmp == NULL)
        {
            fprintf(stderr, "Failed to read first frame from list: %s\n", file_list[0]);
            return -1;
        }
        frame_width = w;
        frame_height = h;
        free(tmp);
    }

    printf("File list mode initialized. %ld frames, %ldx%ld\n", num_frames, frame_width,
           frame_height);
    return 0;
}

/**
 * init_bin() - Initialize binary (.bin) coordinate/image dataset reader.
 * @filename: Path to .bin file.
 *
 * Return: 0 on success, -1 on error.
 */
static int init_bin(
    char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL)
    {
        perror("Failed to open binary dataset");
        return -1;
    }

    gric_bin_header_t hdr;
    char *comment = NULL;
    if (gric_bin_read_header(fp, &hdr, &comment) != 0)
    {
        fprintf(stderr, "Error: '%s' is not a valid GRIC binary dataset\n", filename);
        fclose(fp);
        return -1;
    }
    if (comment != NULL)
    {
        free(comment);
    }

    num_frames = (long)hdr.dims[0];
    frame_width = (hdr.ndim > 1) ? (long)hdr.dims[1] : 1;
    frame_height = 1;
    bin_input_data_offset = hdr.header_bytes;
    bin_input_dtype = (gric_bin_data_type_t)hdr.data_type;
    bin_input_ptr = fp;
    is_bin_mode = 1;

    printf("Binary dataset mode initialized. %ld frames, %ld dimensions (%s)\n",
           num_frames, frame_width, gric_bin_data_type_str(bin_input_dtype));
    return 0;
}

/**
 * getframe_bin() - Read frame from binary dataset.
 * @frame_struct: Frame struct to populate.
 * @index:        Frame index.
 *
 * Return: 0 on success, -1 on error.
 */
static int getframe_bin(
    Frame *frame_struct,
    long   index)
{
    if (bin_input_ptr == NULL || index < 0 || index >= num_frames)
    {
        return -1;
    }

    size_t elem_size = gric_bin_data_type_size(bin_input_dtype);
    off_t offset = (off_t)bin_input_data_offset +
                   (off_t)index * (off_t)frame_width * (off_t)elem_size;
    if (fseeko(bin_input_ptr, offset, SEEK_SET) != 0)
    {
        return -1;
    }

    if (bin_input_dtype == GRIC_BIN_DTYPE_FLOAT32)
    {
        float *fbuf = (float *)malloc(frame_width * sizeof(float));
        if (fbuf == NULL)
        {
            return -1;
        }
        if (fread(fbuf, sizeof(float), frame_width, bin_input_ptr) != (size_t)frame_width)
        {
            free(fbuf);
            return -1;
        }
        for (long k = 0; k < frame_width; k++)
        {
            frame_struct->data[k] = (double)fbuf[k];
        }
        free(fbuf);
    }
    else if (bin_input_dtype == GRIC_BIN_DTYPE_FLOAT64)
    {
        if (fread(frame_struct->data, sizeof(double), frame_width, bin_input_ptr) !=
            (size_t)frame_width)
        {
            return -1;
        }
    }
    else
    {
        uint32_t *ubuf = (uint32_t *)malloc(frame_width * sizeof(uint32_t));
        if (ubuf == NULL)
        {
            return -1;
        }
        if (fread(ubuf, sizeof(uint32_t), frame_width, bin_input_ptr) != (size_t)frame_width)
        {
            free(ubuf);
            return -1;
        }
        for (long k = 0; k < frame_width; k++)
        {
            frame_struct->data[k] = (double)ubuf[k];
        }
        free(ubuf);
    }

    return 0;
}

/**
 * close_bin() - Close binary dataset file handle.
 */
static void close_bin(void)
{
    if (bin_input_ptr != NULL)
    {
        fclose(bin_input_ptr);
        bin_input_ptr = NULL;
    }
    is_bin_mode = 0;
}

/**
 * reset_bin() - Reset reading position for binary dataset.
 */
static void reset_bin(void)
{
    if (bin_input_ptr != NULL)
    {
        fseeko(bin_input_ptr, (off_t)bin_input_data_offset, SEEK_SET);
    }
}

/**
 * init_frameread() - Public entrypoint to initialize the frame reader.
 * @filename:      Path to the file or stream name.
 * @stream_mode:   Whether to use shared memory streaming mode.
 * @cnt2sync_mode: Synchronization mode flag.
 * @filelist_mode: Whether the filename is a list of image filepaths.
 *
 * Directs execution to the specific initialization logic based on input mode flags
 * and file extension.
 *
 * Return: 0 on success, or non-zero error code on failure.
 */
int init_frameread(
    char *filename,
    int   stream_mode,
    int   cnt2sync_mode,
    int   filelist_mode)
{
    if (filelist_mode)
    {
        return init_filelist(filename);
    }

#ifdef USE_IMAGESTREAMIO
    if (stream_mode)
    {
        cnt2sync_enabled = cnt2sync_mode;
        return init_stream(filename);
    }
#else
    if (stream_mode)
    {
        fprintf(stderr, "Error: ImageStreamIO support is not compiled in.\n");
        return -1;
    }
#endif

    {
        /* Check file extension for BIN, TXT, or video formats */
        char *ext = strrchr(filename, '.');
        if (ext != NULL)
        {
            if (strcmp(ext, ".bin") == 0)
            {
                return init_bin(filename);
            }
            if (strcmp(ext, ".txt") == 0)
            {
                return init_ascii(filename);
            }
            if (strcmp(ext, ".mp4") == 0 || strcmp(ext, ".avi") == 0 ||
                strcmp(ext, ".mov") == 0 || strcmp(ext, ".mkv") == 0)
            {
#ifdef USE_FFMPEG
                return init_mp4(filename);
#else
                fprintf(stderr,
                        "Error: FFmpeg support is not compiled in. Cannot read video file.\n");
                return -1;
#endif
            }
        }
    }

#ifdef USE_CFITSIO
    return init_fits(filename);
#else
    fprintf(stderr,
            "Error: FITS support is not compiled in. Cannot read file %s. ASCII (.txt) "
            "supported.\n", filename);
    return -1;
#endif
}

/**
 * getframe() - Retrieve the next sequential frame.
 *
 * Calls getframe_at with the current frame index.
 *
 * Return: Pointer to the populated Frame struct, or NULL on error or EOF.
 */
Frame *getframe(void)
{
#ifndef USE_IMAGESTREAMIO
    if (current_frame_idx >= num_frames)
    {
        return NULL;
    }
#endif

    return getframe_at(current_frame_idx++);
}

/**
 * getframe_at() - Retrieve the frame at the specified index.
 * @index: Zero-based index of the frame.
 *
 * Allocates or reuses a Frame struct, dispatches file/stream-specific reading logic,
 * and handles error cleanups.
 *
 * Return: Pointer to the populated Frame struct, or NULL on error.
 */
Frame *getframe_at(
    long index)
{
    if (index >= num_frames || index < 0)
    {
        return NULL;
    }

    long nelements = frame_width * frame_height;
    Frame *frame_struct = (Frame *)malloc(sizeof(Frame));
    if (frame_struct == NULL)
    {
        return NULL;
    }

    frame_struct->width = frame_width;
    frame_struct->height = frame_height;
    frame_struct->id = index;

    if (!is_filelist_mode)
    {
        int got_from_pool = 0;
#ifdef _OPENMP
#pragma omp critical(frame_pool)
#endif
        {
            if (frame_data_pool_count > 0)
            {
                frame_struct->data = frame_data_pool[--frame_data_pool_count];
                got_from_pool = 1;
            }
        }
        if (!got_from_pool)
        {
            frame_struct->data = (double *)malloc(nelements * sizeof(double));
        }
        if (frame_struct->data == NULL)
        {
            free(frame_struct);
            return NULL;
        }
    }
    else
    {
        frame_struct->data = NULL;
    }

    frame_struct->cnt0 = 0;
    frame_struct->atime.tv_sec = 0;
    frame_struct->atime.tv_nsec = 0;

    if (is_filelist_mode)
    {
        int w = 0;
        int h = 0;
        frame_struct->data = read_png_frame(file_list[index], &w, &h);
        if (frame_struct->data == NULL)
        {
            fprintf(stderr, "Error reading frame %ld: %s\n", index, file_list[index]);
            free(frame_struct);
            return NULL;
        }
        if (w != frame_width || h != frame_height)
        {
            fprintf(stderr,
                    "Error: Frame dimension mismatch in file list. Expected %ldx%ld, got %dx%d\n",
                    frame_width, frame_height, w, h);
            free(frame_struct->data);
            free(frame_struct);
            return NULL;
        }
    }
    else if (is_ascii_mode)
    {
        if (getframe_ascii(frame_struct, index) != 0)
        {
            free(frame_struct->data);
            free(frame_struct);
            return NULL;
        }
    }
    else if (is_bin_mode)
    {
        if (getframe_bin(frame_struct, index) != 0)
        {
            free(frame_struct->data);
            free(frame_struct);
            return NULL;
        }
    }
#ifdef USE_IMAGESTREAMIO
    else if (is_stream_mode)
    {
        if (getframe_stream(frame_struct, index) != 0)
        {
            free(frame_struct->data);
            free(frame_struct);
            return NULL;
        }
    }
#endif
#ifdef USE_FFMPEG
    else if (is_mp4_mode)
    {
        if (getframe_mp4(frame_struct, index) != 0)
        {
            free(frame_struct->data);
            free(frame_struct);
            return NULL;
        }
    }
#endif
#ifdef USE_CFITSIO
    else if (fptr != NULL)
    {
        if (getframe_fits(frame_struct, index) != 0)
        {
            free(frame_struct->data);
            free(frame_struct);
            return NULL;
        }
    }
#endif
    else
    {
        free(frame_struct->data);
        free(frame_struct);
        return NULL;
    }

    return frame_struct;
}

/**
 * free_frame() - Return a frame structure and buffer to the reuse pool or release memory.
 * @frame_ptr: Pointer to the Frame struct to free.
 */
void free_frame(
    Frame *frame_ptr)
{
    if (frame_ptr != NULL)
    {
        if (frame_ptr->data != NULL)
        {
            int returned_to_pool = 0;
#ifdef _OPENMP
#pragma omp critical(frame_pool)
#endif
            {
                if (frame_data_pool_count < FRAME_DATA_POOL_SIZE)
                {
                    frame_data_pool[frame_data_pool_count++] = frame_ptr->data;
                    returned_to_pool = 1;
                }
            }
            if (!returned_to_pool)
            {
                free(frame_ptr->data);
            }
        }
        free(frame_ptr);
    }
}

/**
 * close_frameread() - Finalize the frame reader, freeing all global state and format readers.
 */
void close_frameread(void)
{
#ifdef _OPENMP
#pragma omp critical(frame_pool)
#endif
    {
        while (frame_data_pool_count > 0)
        {
            free(frame_data_pool[--frame_data_pool_count]);
        }
    }

    if (is_filelist_mode)
    {
        if (file_list != NULL)
        {
            for (long ii = 0; ii < num_frames; ii++)
            {
                if (file_list[ii] != NULL)
                {
                    free(file_list[ii]);
                }
            }
            free(file_list);
            file_list = NULL;
        }
        is_filelist_mode = 0;
    }
    else if (is_ascii_mode)
    {
        close_ascii();
    }
    else if (is_bin_mode)
    {
        close_bin();
    }
#ifdef USE_IMAGESTREAMIO
    else if (is_stream_mode)
    {
        close_stream();
    }
#endif
#ifdef USE_FFMPEG
    else if (is_mp4_mode)
    {
        close_mp4();
    }
#endif
#ifdef USE_CFITSIO
    else if (fptr != NULL)
    {
        close_fits();
    }
#endif
}

/**
 * reset_frameread() - Reset frame reading to the start index/frame.
 */
void reset_frameread(void)
{
    current_frame_idx = 0;

    if (is_ascii_mode)
    {
        reset_ascii();
    }
    else if (is_bin_mode)
    {
        reset_bin();
    }
#ifdef USE_IMAGESTREAMIO
    else if (is_stream_mode)
    {
        reset_stream();
    }
#endif
#ifdef USE_FFMPEG
    else if (is_mp4_mode)
    {
        reset_mp4();
    }
#endif
#ifdef USE_CFITSIO
    else if (fptr != NULL)
    {
        reset_fits();
    }
#endif
}

/**
 * get_num_frames() - Get the total number of frames in the source.
 *
 * Return: Total number of frames.
 */
long get_num_frames(void)
{
    return num_frames;
}

/**
 * get_missed_frames() - Get the number of missed frames in ImageStreamIO stream mode.
 *
 * Return: Cumulative count of missed frames.
 */
long get_missed_frames(void)
{
#ifdef USE_IMAGESTREAMIO
    return cumulative_missed_frames;
#else
    return 0;
#endif
}

/**
 * get_frame_width() - Get width of the frame.
 *
 * Return: Width in pixels.
 */
long get_frame_width(void)
{
    return frame_width;
}

/**
 * get_frame_height() - Get height of the frame.
 *
 * Return: Height in pixels.
 */
long get_frame_height(void)
{
    return frame_height;
}

/**
 * is_ascii_input_mode() - Query if the reader is reading an ASCII coordinate file.
 *
 * Return: 1 if ASCII mode is active, 0 otherwise.
 */
int is_ascii_input_mode(void)
{
    return is_ascii_mode;
}

#ifdef USE_IMAGESTREAMIO
/**
 * is_stream_input_mode() - Query if the reader is reading from an ImageStreamIO stream.
 *
 * Return: 1 if stream mode is active, 0 otherwise.
 */
int is_stream_input_mode(void)
{
    return is_stream_mode;
}

/**
 * get_stream_read_slice() - Query the active read slice index for 3D streams.
 *
 * Return: The slice index.
 */
long get_stream_read_slice(void)
{
    return current_read_slice;
}

/**
 * get_stream_write_slice() - Query the active write slice index for 3D streams.
 *
 * Return: The slice index.
 */
long get_stream_write_slice(void)
{
    return current_write_slice;
}

/**
 * get_stream_lag() - Query the lagging frame count between write and read heads in a stream.
 *
 * Return: Lag in frame count.
 */
long get_stream_lag(void)
{
    if (is_stream_mode)
    {
        return (long)(stream_image.md[0].cnt0 - last_cnt0);
    }
    return 0;
}

/**
 * is_3d_stream_mode() - Query if the active stream is in 3D mode.
 *
 * Return: 1 if 3D mode is active, 0 otherwise.
 */
int is_3d_stream_mode(void)
{
    return is_3d;
}

/**
 * get_stream_wait_time() - Get cumulative time spent waiting for stream frames.
 *
 * Return: Wait duration in seconds.
 */
double get_stream_wait_time(void)
{
    return cumulative_wait_time_sec;
}
#else
int is_stream_input_mode(void)
{
    return 0;
}

long get_stream_read_slice(void)
{
    return 0;
}

long get_stream_write_slice(void)
{
    return 0;
}

long get_stream_lag(void)
{
    return 0;
}

int is_3d_stream_mode(void)
{
    return 0;
}

double get_stream_wait_time(void)
{
    return 0.0;
}
#endif

/**
 * is_bin_input_mode() - Query if active input dataset is in binary (.bin) format.
 *
 * Return: 1 if binary format, 0 otherwise.
 */
int is_bin_input_mode(void)
{
    return is_bin_mode;
}
