/**
 * @file gric_fps_common.c
 * @brief Common adapter implementation for GRIC Milk streaming integration.
 */

#define _POSIX_C_SOURCE 200809L
#include "gric_fps_common.h"
#include "gric_fps_params.h"
#include "cluster_shm.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

char     fps_in_name[FUNCTION_PARAMETER_STRMAXLEN]          = "";
char     fps_out_assign_name[FUNCTION_PARAMETER_STRMAXLEN]  = "";
char     fps_out_anchors_name[FUNCTION_PARAMETER_STRMAXLEN] = "";
char     fps_out_counts_name[FUNCTION_PARAMETER_STRMAXLEN]  = "";
int32_t  fps_stream_anchors   = 0;
int32_t  fps_stream_counts    = 0;
int32_t  fps_allow_frame_drop = 0;
double   fps_rlim             = 0.5;
double   fps_deltaprob        = 0.01;
uint32_t fps_maxnbclust       = 256;
int64_t  fps_maxcl_strategy   = 0;
uint64_t fps_max_frames        = 0;
uint32_t fps_ncpu             = 0;
int32_t  fps_use_double       = 0;
int32_t  fps_use_sq16         = 0;
int32_t  fps_entropy_mode     = 0;
int32_t  fps_reset_state      = 0;

uint64_t fps_status_frames_processed                   = 0;
uint32_t fps_status_num_clusters                       = 0;
uint64_t fps_status_new_clusters                       = 0;
uint64_t fps_status_distance_evals                     = 0;
double   fps_status_pruning_ratio                      = 0.0;
double   fps_status_latency_us                         = 0.0;
double   fps_status_fps                                = 0.0;
int64_t  fps_status_stream_lag                         = 0;
double   fps_status_memory_rss_mb                      = 0.0;
char     fps_shm_status_file[FUNCTION_PARAMETER_STRMAXLEN] = "";
char     fps_save_dir[FUNCTION_PARAMETER_STRMAXLEN]        = "";

static GricClusterShmStatus *status_shm_ptr        = NULL;
static char                  active_shm_path[PATH_MAX] = "";
static struct timespec       session_start_time;
static struct timespec       prev_fps_calc_time;
static uint64_t              prev_fps_calc_frames  = 0;

static gric_cluster_t *cluster_ctx        = NULL;
static double         *coord_conv_buf     = NULL;
static size_t          current_ndim       = 0;
static uint32_t        prev_cluster_count = 0;

/**
 * gric_fps_custom_conf_check() - Validate runtime configuration parameters.
 *
 * Checks input stream existence, valid radius limits, thread limits, and cluster limits.
 *
 * Return: 0 on valid configuration, non-zero if invalid.
 */
errno_t gric_fps_custom_conf_check(void)
{
    if (fps_in_name[0] == '\0')
    {
        return 1;
    }

    if (fps_rlim <= 0.0)
    {
        fps_rlim = 0.5;
    }

    if (fps_deltaprob <= 0.0 || fps_deltaprob > 1.0)
    {
        fps_deltaprob = 0.01;
    }

    if (fps_maxnbclust < 2)
    {
        fps_maxnbclust = 256;
    }

    if (fps_ncpu == 0)
    {
        fps_ncpu = 1;
    }

    /* Probe input stream in shared memory */
    IMAGE test_img;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(fps_in_name, &test_img) == 0)
    {
        ImageStreamIO_closeIm(&test_img);
    }

    return 0;
}

/**
 * gric_fps_init_engine() - Allocate and initialize the libgric session.
 * @ndim: Total elements per frame.
 *
 * Return: 0 on success, or non-zero on failure.
 */
errno_t gric_fps_init_engine(
    uint32_t ndim)
{
    gric_fps_cleanup_engine();

    current_ndim = (size_t)ndim;
    coord_conv_buf = (double *)calloc(current_ndim, sizeof(double));
    if (!coord_conv_buf)
    {
        return 1;
    }

    gric_cluster_config_t cfg;
    gric_cluster_config_default(&cfg);

    cfg.rlim = fps_rlim;
    cfg.maxnbclust = (int)fps_maxnbclust;
    cfg.tm_mixing_coeff = fps_deltaprob;
    cfg.ncpu = (int)fps_ncpu;
    cfg.use_double = (int)fps_use_double;
    cfg.use_sq16 = (int)fps_use_sq16;
    cfg.entropy_mode = (int)fps_entropy_mode;
    cfg.maxcl_strategy = (int)fps_maxcl_strategy;

    cluster_ctx = gric_cluster_create(&cfg, current_ndim);
    if (!cluster_ctx)
    {
        free(coord_conv_buf);
        coord_conv_buf = NULL;
        return 1;
    }

    prev_cluster_count = 0;
    return 0;
}

/**
 * gric_fps_cleanup_engine() - Free the libgric clustering session and buffers.
 */
void gric_fps_cleanup_engine(void)
{
    if (cluster_ctx)
    {
        gric_cluster_destroy(cluster_ctx);
        cluster_ctx = NULL;
    }
    if (coord_conv_buf)
    {
        free(coord_conv_buf);
        coord_conv_buf = NULL;
    }
    current_ndim = 0;
    prev_cluster_count = 0;
}

/**
 * gric_fps_save_results() - Export clustering results to disk in GRIC binary format.
 * @out_dir: Destination output directory path.
 *
 * Return: 0 on success, non-zero on error.
 */
errno_t gric_fps_save_results(
    const char *out_dir)
{
    if (cluster_ctx == NULL || out_dir == NULL || out_dir[0] == '\0')
    {
        return 1;
    }
    return (gric_cluster_save_results(cluster_ctx, out_dir) == GRIC_SUCCESS) ? 0 : 1;
}

/**
 * gric_fps_init_output_streams() - Create all output ImageStreamIO streams.
 * @xsize:        Frame width.
 * @ysize:        Frame height.
 * @max_clusters: Cluster capacity.
 * @out_assign:   Assignment output stream.
 * @out_anchors:  Centroid vectors output stream.
 * @out_counts:   Cluster count output stream.
 *
 * Return: 0 on success, non-zero on failure.
 */
errno_t gric_fps_init_output_streams(
    uint32_t xsize,
    uint32_t ysize,
    uint32_t max_clusters,
    IMAGE   *out_assign,
    IMAGE   *out_anchors,
    IMAGE   *out_counts)
{
    const char *assign_name = (fps_out_assign_name[0] != '\0')
                                  ? fps_out_assign_name : "gric_assign";

    /* Create 2D assignment stream [GRIC_ASSIGN_PACKET_ELEMS x 1] */
    uint32_t assign_dims[2] = { GRIC_ASSIGN_PACKET_ELEMS, 1 };
    if (ImageStreamIO_createIm_gpu(out_assign, assign_name, 2, assign_dims,
                                   _DATATYPE_FLOAT, -1, 1, 10, 0, 0, 0) != 0)
    {
        return 1;
    }

    /* Optionally create 3D anchors stream [xsize x ysize x max_clusters] */
    if (fps_stream_anchors != 0 && out_anchors)
    {
        const char *anchors_name = (fps_out_anchors_name[0] != '\0')
                                       ? fps_out_anchors_name : "gric_anchors";
        uint32_t anchor_dims[3] = { xsize, ysize, max_clusters };
        ImageStreamIO_createIm_gpu(out_anchors, anchors_name, 3, anchor_dims,
                                   _DATATYPE_FLOAT, -1, 1, 10, 0, 0, 0);
    }

    /* Optionally create 1D counts stream [max_clusters] */
    if (fps_stream_counts != 0 && out_counts)
    {
        const char *counts_name = (fps_out_counts_name[0] != '\0')
                                      ? fps_out_counts_name : "gric_counts";
        uint32_t count_dims[1] = { max_clusters };
        ImageStreamIO_createIm_gpu(out_counts, counts_name, 1, count_dims,
                                   _DATATYPE_UINT32, -1, 1, 10, 0, 0, 0);
    }

    return 0;
}

/**
 * gric_fps_close_output_streams() - Safely disconnect output ImageStreamIO streams.
 * @out_assign:  Assignment stream handle.
 * @out_anchors: Centroids stream handle.
 * @out_counts:  Counts stream handle.
 */
void gric_fps_close_output_streams(
    IMAGE *out_assign,
    IMAGE *out_anchors,
    IMAGE *out_counts)
{
    if (out_assign && out_assign->used == 1)
    {
        ImageStreamIO_closeIm(out_assign);
    }
    if (out_anchors && out_anchors->used == 1)
    {
        ImageStreamIO_closeIm(out_anchors);
    }
    if (out_counts && out_counts->used == 1)
    {
        ImageStreamIO_closeIm(out_counts);
    }
}

/**
 * gric_fps_process_frame() - Cluster one incoming frame and broadcast output.
 * @raw_pixels:  Source pixel buffer pointer.
 * @datatype:    Input stream pixel data type code.
 * @ndim:        Dimension count.
 * @frame_index: Frame count (cnt0).
 * @frame_time:  Arrival timestamp.
 * @latency_us:  Algorithm execution latency in microseconds.
 * @out_assign:  Assignment stream handle.
 * @out_anchors: Centroids stream handle.
 * @out_counts:  Counts stream handle.
 *
 * Return: 0 on success, non-zero on failure.
 */
errno_t gric_fps_process_frame(
    const void      *raw_pixels,
    int              datatype,
    uint32_t         ndim,
    uint64_t         frame_index,
    struct timespec  frame_time,
    double           latency_us,
    IMAGE           *out_assign,
    IMAGE           *out_anchors,
    IMAGE           *out_counts)
{
    (void)frame_time;
    if (!cluster_ctx || !coord_conv_buf || ndim != current_ndim)
    {
        return 1;
    }

    /* Check dynamic state reset trigger */
    if (fps_reset_state != 0)
    {
        fps_reset_state = 0;
        gric_cluster_reset(cluster_ctx);
        prev_cluster_count = 0;
    }

    /* Convert input pixels to coordinate buffer */
    {
        switch (datatype)
        {
        case _DATATYPE_FLOAT:
        {
            const float *f_in = (const float *)raw_pixels;
            for (uint32_t ii = 0; ii < ndim; ii++)
            {
                coord_conv_buf[ii] = (double)f_in[ii];
            }
            break;
        }
        case _DATATYPE_DOUBLE:
        {
            memcpy(coord_conv_buf, raw_pixels, ndim * sizeof(double));
            break;
        }
        case _DATATYPE_UINT16:
        {
            const uint16_t *u16_in = (const uint16_t *)raw_pixels;
            for (uint32_t ii = 0; ii < ndim; ii++)
            {
                coord_conv_buf[ii] = (double)u16_in[ii];
            }
            break;
        }
        case _DATATYPE_UINT8:
        {
            const uint8_t *u8_in = (const uint8_t *)raw_pixels;
            for (uint32_t ii = 0; ii < ndim; ii++)
            {
                coord_conv_buf[ii] = (double)u8_in[ii];
            }
            break;
        }
        default:
            return 1;
        }
    }

    /* Ingest frame through libgric clustering pipeline */
    int64_t assigned_cluster_id = -1;
    gric_status_t status = gric_cluster_feed_frame(cluster_ctx, coord_conv_buf,
                                                   &assigned_cluster_id);
    if (status != GRIC_SUCCESS)
    {
        return 1;
    }

    int64_t total_clusters = gric_cluster_get_num_clusters(cluster_ctx);
    int is_new = (total_clusters > (int64_t)prev_cluster_count);

    /* Write 1D telemetry packet to out_assign */
    if (out_assign && out_assign->used == 1)
    {
        out_assign->md[0].write = 1;
        float *assign_data = (float *)out_assign->array.raw;
        assign_data[0] = (float)frame_index;
        assign_data[1] = (float)assigned_cluster_id;
        assign_data[2] = 0.0f;
        assign_data[3] = is_new ? 1.0f : 0.0f;
        assign_data[4] = (float)total_clusters;
        assign_data[5] = (float)latency_us;
        assign_data[6] = 1.0f;
        assign_data[7] = 0.0f;

        out_assign->md[0].cnt0++;
        out_assign->md[0].write = 0;
        ImageStreamIO_sempost(out_assign, -1);
    }

    /* Update anchors stream if a new cluster was spawned */
    if (is_new && out_anchors && out_anchors->used == 1 &&
        assigned_cluster_id >= 0 && assigned_cluster_id < (int64_t)out_anchors->md[0].size[2])
    {
        out_anchors->md[0].write = 1;
        float *anchor_slice = (float *)out_anchors->array.raw +
                              ((size_t)assigned_cluster_id * ndim);
        for (uint32_t ii = 0; ii < ndim; ii++)
        {
            anchor_slice[ii] = (float)coord_conv_buf[ii];
        }
        out_anchors->md[0].cnt0++;
        out_anchors->md[0].write = 0;
        ImageStreamIO_sempost(out_anchors, -1);
    }

    /* Update counts stream */
    if (out_counts && out_counts->used == 1 && assigned_cluster_id >= 0 &&
        assigned_cluster_id < (int64_t)out_counts->md[0].size[0])
    {
        out_counts->md[0].write = 1;
        uint32_t *counts_data = (uint32_t *)out_counts->array.raw;
        counts_data[assigned_cluster_id]++;
        out_counts->md[0].cnt0++;
        out_counts->md[0].write = 0;
        ImageStreamIO_sempost(out_counts, -1);
    }

    prev_cluster_count = (uint32_t)total_clusters;
    return 0;
}

/**
 * gric_fps_get_cluster_count() - Query discovered cluster count.
 *
 * Return: Cluster count, or 0 if uninitialized.
 */
int64_t gric_fps_get_cluster_count(void)
{
    return cluster_ctx ? gric_cluster_get_num_clusters(cluster_ctx) : 0;
}

/**
 * get_current_rss_mb() - Query current process resident set size in megabytes.
 *
 * Return: Memory RSS in MB, or 0.0 on error.
 */
static double get_current_rss_mb(void)
{
    FILE *f = fopen("/proc/self/statm", "r");
    if (!f)
    {
        return 0.0;
    }
    long pages = 0;
    if (fscanf(f, "%*d %ld", &pages) != 1)
    {
        fclose(f);
        return 0.0;
    }
    fclose(f);
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size < 0)
    {
        page_size = 4096;
    }
    return (double)((uint64_t)pages * (uint64_t)page_size) / (1024.0 * 1024.0);
}

/**
 * gric_fps_status_init() - Initialize file-mapped shared memory status bridge.
 * @fps_name:    Name of the active FPS daemon instance.
 * @custom_path: Optional custom path to SHM status file.
 *
 * Return: 0 on success, non-zero on error.
 */
errno_t gric_fps_status_init(
    const char *fps_name,
    const char *custom_path)
{
    if (status_shm_ptr != NULL)
    {
        gric_fps_status_close(0);
    }

    if (custom_path != NULL && custom_path[0] != '\0')
    {
        strncpy(active_shm_path, custom_path, sizeof(active_shm_path) - 1);
        active_shm_path[sizeof(active_shm_path) - 1] = '\0';
    }
    else
    {
        const char *name = (fps_name && fps_name[0] != '\0') ? fps_name : "gric_cluster";
        const char *shmdir = getenv("MILK_SHM_DIR");
        if (shmdir == NULL && access("/milk/shm", W_OK) == 0)
        {
            shmdir = "/milk/shm";
        }
        if (shmdir == NULL && access("/dev/shm", W_OK) == 0)
        {
            shmdir = "/dev/shm";
        }
        if (shmdir == NULL)
        {
            shmdir = "/tmp";
        }
        snprintf(active_shm_path, sizeof(active_shm_path), "%s/fps.%s.status.shm", shmdir, name);
    }

    int fd = open(active_shm_path, O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (fd < 0)
    {
        const char *name = (fps_name && fps_name[0] != '\0') ? fps_name : "gric_cluster";
        snprintf(active_shm_path, sizeof(active_shm_path), "/tmp/fps.%s.status.shm", name);
        fd = open(active_shm_path, O_RDWR | O_CREAT | O_TRUNC, 0666);
    }

    if (fd < 0)
    {
        active_shm_path[0] = '\0';
        return 1;
    }

    if (ftruncate(fd, sizeof(GricClusterShmStatus)) < 0)
    {
        close(fd);
        active_shm_path[0] = '\0';
        return 1;
    }

    void *ptr = mmap(NULL, sizeof(GricClusterShmStatus), PROT_READ | PROT_WRITE,
                     MAP_SHARED, fd, 0);
    close(fd);

    if (ptr == MAP_FAILED)
    {
        active_shm_path[0] = '\0';
        return 1;
    }

    status_shm_ptr = (GricClusterShmStatus *)ptr;
    memset(status_shm_ptr, 0, sizeof(GricClusterShmStatus));

    status_shm_ptr->magic = GRIC_SHM_MAGIC;
    status_shm_ptr->version = GRIC_SHM_VERSION;
    status_shm_ptr->pid = (uint32_t)getpid();
    status_shm_ptr->status_state = GRIC_STATUS_RUNNING;
    status_shm_ptr->total_frames = fps_max_frames;
    strncpy(status_shm_ptr->input_source, fps_in_name, sizeof(status_shm_ptr->input_source) - 1);

    if (getcwd(status_shm_ptr->config_cwd, sizeof(status_shm_ptr->config_cwd) - 1) == NULL)
    {
        status_shm_ptr->config_cwd[0] = '\0';
    }

    status_shm_ptr->config_rlim = fps_rlim;
    status_shm_ptr->config_maxnbclust = fps_maxnbclust;
    status_shm_ptr->config_dprob = fps_deltaprob;
    status_shm_ptr->config_maxcl_strategy = (uint32_t)fps_maxcl_strategy;
    status_shm_ptr->config_entropy_mode = (uint32_t)fps_entropy_mode;
    status_shm_ptr->active_threads = fps_ncpu;

    clock_gettime(CLOCK_MONOTONIC, &session_start_time);
    prev_fps_calc_time = session_start_time;
    prev_fps_calc_frames = 0;

    return 0;
}

/**
 * gric_fps_status_update() - Update real-time status parameters and bridge shared memory.
 * @frame_index: Frame counter (cnt0).
 * @latency_us:  Algorithm execution latency in microseconds.
 * @stream_lag:  Input stream lag between write and read heads.
 * @write_slice: Current circular buffer write index.
 * @read_slice:  Current circular buffer read index.
 */
void gric_fps_status_update(
    uint64_t frame_index,
    double   latency_us,
    long     stream_lag,
    long     write_slice,
    long     read_slice)
{
    (void)frame_index;

    gric_cluster_stats_t stats;
    memset(&stats, 0, sizeof(stats));
    if (cluster_ctx)
    {
        gric_cluster_get_stats(cluster_ctx, &stats);
    }

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);

    double elapsed_ms = (now.tv_sec - session_start_time.tv_sec) * 1000.0 +
                        (now.tv_nsec - session_start_time.tv_nsec) / 1000000.0;

    /* Compute processing throughput (FPS) */
    double dt_fps = (now.tv_sec - prev_fps_calc_time.tv_sec) +
                    (now.tv_nsec - prev_fps_calc_time.tv_nsec) / 1000000000.0;
    if (dt_fps >= 0.1)
    {
        uint64_t dframes = stats.total_frames_processed - prev_fps_calc_frames;
        fps_status_fps = (double)dframes / dt_fps;
        prev_fps_calc_time = now;
        prev_fps_calc_frames = stats.total_frames_processed;
    }

    /* Update FPS parameter variables */
    fps_status_frames_processed = stats.total_frames_processed;
    fps_status_num_clusters = stats.num_clusters;
    fps_status_new_clusters = stats.num_new_clusters;
    fps_status_distance_evals = stats.framedist_calls;

    uint64_t total_possible = stats.total_frames_processed * (uint64_t)stats.num_clusters;
    if (total_possible > 0)
    {
        double ratio = 1.0 - ((double)stats.framedist_calls / (double)total_possible);
        fps_status_pruning_ratio = (ratio < 0.0) ? 0.0 : ratio;
    }
    else
    {
        fps_status_pruning_ratio = 0.0;
    }

    fps_status_latency_us = latency_us;
    fps_status_stream_lag = (int64_t)stream_lag;
    fps_status_memory_rss_mb = get_current_rss_mb();

    /* Update bridge shared memory file if active */
    if (status_shm_ptr != NULL)
    {
        status_shm_ptr->status_state = GRIC_STATUS_RUNNING;
        status_shm_ptr->total_frames_processed = stats.total_frames_processed;
        status_shm_ptr->num_clusters = stats.num_clusters;
        status_shm_ptr->framedist_calls = stats.framedist_calls;
        status_shm_ptr->framedist_calls_sample = stats.framedist_calls_sample;
        status_shm_ptr->framedist_calls_intercluster = stats.framedist_calls_intercluster;
        status_shm_ptr->clusters_pruned = stats.clusters_pruned;
        status_shm_ptr->elapsed_ms = elapsed_ms;

        status_shm_ptr->stream_read_slice = read_slice;
        status_shm_ptr->stream_write_slice = write_slice;
        status_shm_ptr->stream_lag = stream_lag;
        status_shm_ptr->last_assignment_dist = stats.last_assignment_dist;
        status_shm_ptr->num_new_clusters = stats.num_new_clusters;

        status_shm_ptr->memory_rss_kb = (uint64_t)(fps_status_memory_rss_mb * 1024.0);
        status_shm_ptr->active_threads = fps_ncpu;

        status_shm_ptr->last_frame_dists = stats.last_frame_dists;
        status_shm_ptr->last_frame_dfc = stats.last_frame_dfc;
        status_shm_ptr->last_frame_dcc = stats.last_frame_dcc;
        status_shm_ptr->time_io_ms = stats.time_io_ms;
        status_shm_ptr->time_step_1 = stats.time_step_1;
        status_shm_ptr->time_step_2 = stats.time_step_2;
        status_shm_ptr->time_step_3a = stats.time_step_3a;
        status_shm_ptr->time_step_3b = stats.time_step_3b;
        status_shm_ptr->time_step_3c = stats.time_step_3c;
        status_shm_ptr->time_step_4 = stats.time_step_4;
        status_shm_ptr->time_step_5 = stats.time_step_5;
        status_shm_ptr->time_step_refine = stats.time_step_refine;

        status_shm_ptr->entropy_last_initial = stats.entropy_last_initial;
        status_shm_ptr->entropy_avg_initial = stats.entropy_avg_initial;
        status_shm_ptr->entropy_gate_ratio = stats.entropy_gate_ratio;
        status_shm_ptr->dcc_entries_populated = stats.dcc_entries_populated;
        status_shm_ptr->dcc_pairs_total = stats.dcc_pairs_total;
        status_shm_ptr->memo_hits = stats.memo_hits;
        status_shm_ptr->memo_lookups = stats.memo_lookups;
        status_shm_ptr->memo_cache_entries = stats.memo_cache_entries;
        status_shm_ptr->memo_cache_capacity = stats.memo_cache_capacity;

        status_shm_ptr->config_rlim = fps_rlim;
        status_shm_ptr->config_maxnbclust = fps_maxnbclust;
        status_shm_ptr->config_dprob = fps_deltaprob;
        status_shm_ptr->total_frames = fps_max_frames;

        struct timespec real_now;
        clock_gettime(CLOCK_REALTIME, &real_now);
        status_shm_ptr->last_update_time = (uint64_t)real_now.tv_sec * 1000000000ULL +
                                           (uint64_t)real_now.tv_nsec;
    }
}

/**
 * gric_fps_status_close() - Finalize and unmap the status shared memory bridge.
 * @exit_state: Exit code (0 for success, non-zero for error).
 */
void gric_fps_status_close(
    int exit_state)
{
    if (status_shm_ptr != NULL)
    {
        status_shm_ptr->status_state = (exit_state == 0) ? GRIC_STATUS_SUCCESS
                                                         : GRIC_STATUS_ERROR;
        munmap(status_shm_ptr, sizeof(GricClusterShmStatus));
        status_shm_ptr = NULL;
    }
}
