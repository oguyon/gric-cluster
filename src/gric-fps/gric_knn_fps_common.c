/**
 * @file gric_knn_fps_common.c
 * @brief Common adapter implementation for GRIC k-NN Milk streaming integration.
 */

#define _POSIX_C_SOURCE 200809L
#include "gric_knn_fps_common.h"
#include "gric_knn_fps_params.h"
#include "knn_defs.h"
#include "knn_loader.h"
#include "knn_reader.h"
#include "knn_engine.h"
#include "knn_cross_dataset.h"
#include "knn_heap.h"
#include "cluster_shm.h"
#include "shared/sys/gric_rss.h"
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

char     fps_knn_in_name[FUNCTION_PARAMETER_STRMAXLEN]      = "";
char     fps_knn_out_name[FUNCTION_PARAMETER_STRMAXLEN]     = "gric_knn";
uint64_t fps_knn_cnt2sync                                   = 0;
uint64_t fps_knn_allow_frame_drop                           = 0;
char     fps_knn_cluster_dir[FUNCTION_PARAMETER_STRMAXLEN]  = "";
char     fps_knn_ref_data[FUNCTION_PARAMETER_STRMAXLEN]     = "";
uint32_t fps_knn_k                                          = 10;
double   fps_knn_rlim                                       = 0.0;
double   fps_knn_eps                                        = 0.0;
uint64_t fps_knn_max_frames                                 = 0;
uint32_t fps_knn_ncpu                                       = 1;
uint64_t fps_knn_use_double                                 = 0;
uint64_t fps_knn_use_sq8                                    = 0;
uint64_t fps_knn_use_sq16                                   = 0;
uint64_t fps_knn_use_eq16                                   = 0;
uint64_t fps_knn_use_rq8                                    = 0;
uint64_t fps_knn_approx_mode                                = 0;
uint64_t fps_knn_use_cluster_graph                          = 1;

uint64_t fps_knn_status_queries                             = 0;
uint32_t fps_knn_status_k                                   = 0;
uint64_t fps_knn_status_dist_evals                          = 0;
double   fps_knn_status_latency_us                          = 0.0;
double   fps_knn_status_fps                                 = 0.0;
double   fps_knn_status_min_dist                            = 0.0;
double   fps_knn_status_max_dist                            = 0.0;
int64_t  fps_knn_status_stream_lag                          = 0;
double   fps_knn_status_memory_rss_mb                       = 0.0;
char     fps_knn_shm_status_file[FUNCTION_PARAMETER_STRMAXLEN] = "";

static GricClusterShmStatus *status_shm_ptr        = NULL;
static char                  active_shm_path[PATH_MAX] = "";
static struct timespec       session_start_time;
static struct timespec       prev_fps_calc_time;
static uint64_t              prev_fps_calc_frames  = 0;

static KnnModel              knn_model;
static KnnConfig             knn_cfg;
static int                   model_initialized     = 0;
static KnnFrameReader        cand_reader;
static int                   reader_initialized    = 0;

/* Pre-allocated per-frame scratch buffers */
static KnnMaxHeap            heap;
static int                   heap_initialized      = 0;
static void                 *query_buf             = NULL;
static void                 *cand_buf              = NULL;
static uint8_t              *query_sq8             = NULL;
static int16_t              *query_sq16            = NULL;
static int16_t              *query_eq16            = NULL;
static double               *anchor_dists          = NULL;
static ClusterScore         *scores_buf            = NULL;
static KnnVisitedTracker     visited;
static KnnTrajectoryTracker  tracker;
static int                  *out_indices           = NULL;
static double               *out_dists             = NULL;
static size_t                current_ndim          = 0;

/**
 * gric_knn_fps_custom_conf_check() - Validate runtime configuration parameters.
 *
 * Return: 0 on valid configuration, non-zero if invalid.
 */
errno_t gric_knn_fps_custom_conf_check(void)
{
    if (fps_knn_in_name[0] == '\0')
    {
        return 1;
    }

    if (fps_knn_cluster_dir[0] == '\0')
    {
        return 1;
    }

    if (fps_knn_ref_data[0] == '\0')
    {
        return 1;
    }

    if (fps_knn_k == 0)
    {
        fps_knn_k = 10;
    }

    /* Probe input stream in shared memory */
    IMAGE test_img;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(fps_knn_in_name, &test_img) == 0)
    {
        ImageStreamIO_closeIm(&test_img);
    }

    return 0;
}

/**
 * gric_knn_fps_init_engine() - Allocate and initialize resident model and buffers.
 * @ndim: Total elements per query frame.
 *
 * Return: 0 on success, or non-zero on failure.
 */
errno_t gric_knn_fps_init_engine(
    uint32_t ndim)
{
    gric_knn_fps_cleanup_engine();

    current_ndim = (size_t)ndim;

    memset(&knn_cfg, 0, sizeof(KnnConfig));
    knn_cfg.cluster_dir = fps_knn_cluster_dir;
    knn_cfg.input_data_path = fps_knn_ref_data;
    knn_cfg.k = (fps_knn_k > 0) ? (int)fps_knn_k : 10;
    knn_cfg.rlim_cutoff = fps_knn_rlim;
    knn_cfg.epsilon = fps_knn_eps;
    knn_cfg.nthreads = (fps_knn_ncpu > 0) ? (int)fps_knn_ncpu : 1;
    knn_cfg.use_double = fps_knn_use_double;
    knn_cfg.use_sq8 = fps_knn_use_sq8;
    knn_cfg.use_sq16 = fps_knn_use_sq16;
    knn_cfg.use_eq16 = fps_knn_use_eq16;
    knn_cfg.use_rq8 = fps_knn_use_rq8;
    knn_cfg.approx_mode = fps_knn_approx_mode;
    knn_cfg.use_cluster_graph = fps_knn_use_cluster_graph;
    knn_cfg.no_cache_dataset = 0;
    knn_cfg.use_batch_dist = 1;

    if (knn_model_load(fps_knn_cluster_dir, fps_knn_ref_data,
                       &knn_model, fps_knn_use_double) != 0)
    {
        fprintf(stderr, "Error: Failed to load k-NN model from '%s'\n",
                fps_knn_cluster_dir);
        return 1;
    }
    model_initialized = 1;

    if ((size_t)knn_model.frame_elements != current_ndim)
    {
        fprintf(stderr, "Error: Stream dim (%zu) does not match model dim (%ld)\n",
                current_ndim, knn_model.frame_elements);
        gric_knn_fps_cleanup_engine();
        return 1;
    }

    knn_model_cache_dataset(&knn_model, &knn_cfg);

    if (knn_cfg.use_sq8)
    {
        knn_model_build_or_load_sq8(&knn_model, &knn_cfg);
    }
    if (knn_cfg.use_sq16)
    {
        knn_model_build_or_load_sq16(&knn_model, &knn_cfg);
    }
    if (knn_cfg.use_eq16)
    {
        knn_model_build_or_load_eq16(&knn_model, &knn_cfg);
    }
    if (knn_cfg.use_rq8)
    {
        knn_model_build_or_load_rq8(&knn_model, &knn_cfg);
    }

    int open_res = -1;
    if (knn_model.dataset_buffer != NULL)
    {
        open_res = knn_reader_open_memory(
            &cand_reader,
            knn_model.dataset_buffer,
            knn_model.total_dataset_frames,
            knn_model.frame_elements,
            knn_model.is_double);
    }
    else
    {
        open_res = knn_reader_open(
            &cand_reader,
            fps_knn_ref_data,
            knn_model.total_dataset_frames,
            knn_model.frame_width,
            knn_model.frame_height,
            knn_model.is_double);
    }

    if (open_res != 0)
    {
        fprintf(stderr, "Error: Failed to open candidate reader for '%s'\n",
                fps_knn_ref_data);
        gric_knn_fps_cleanup_engine();
        return 1;
    }
    reader_initialized = 1;

    int k = knn_cfg.k;
    if (knn_heap_init(&heap, k) != 0)
    {
        gric_knn_fps_cleanup_engine();
        return 1;
    }
    heap_initialized = 1;

    size_t elem_size = knn_model.is_double ? sizeof(double) : sizeof(float);
    query_buf = malloc(current_ndim * elem_size);
    cand_buf  = malloc(4 * current_ndim * elem_size);
    if (knn_cfg.use_sq8)
    {
        query_sq8 = (uint8_t *)malloc(current_ndim);
    }
    if (knn_cfg.use_sq16)
    {
        query_sq16 = (int16_t *)malloc(current_ndim * sizeof(int16_t));
    }
    if (knn_cfg.use_eq16)
    {
        query_eq16 = (int16_t *)malloc(current_ndim * sizeof(int16_t));
    }

    anchor_dists = (double *)malloc((size_t)knn_model.num_clusters * sizeof(double));
    scores_buf   = (ClusterScore *)malloc((size_t)knn_model.num_clusters * sizeof(ClusterScore));

    memset(&visited, 0, sizeof(KnnVisitedTracker));
    visited.tags = (uint32_t *)calloc((size_t)knn_model.total_dataset_frames, sizeof(uint32_t));
    visited.epoch = 1;
    visited.query_sq8 = query_sq8;
    visited.query_sq16 = query_sq16;
    visited.query_eq16 = query_eq16;

    memset(&tracker, 0, sizeof(KnnTrajectoryTracker));
    tracker.prev_query_id = -999;
    tracker.prev_cluster_id = -1;
    tracker.prev_best_seed = -1;
    tracker.prev_best_dist = 1e20;

    out_indices = (int *)malloc((size_t)k * sizeof(int));
    out_dists   = (double *)malloc((size_t)k * sizeof(double));

    if (!query_buf || !cand_buf || !anchor_dists || !scores_buf ||
        !visited.tags || !out_indices || !out_dists)
    {
        gric_knn_fps_cleanup_engine();
        return 1;
    }

    fps_knn_status_k = (uint32_t)k;
    return 0;
}

/**
 * gric_knn_fps_cleanup_engine() - Free the resident model and scratch buffers.
 */
void gric_knn_fps_cleanup_engine(void)
{
    if (heap_initialized)
    {
        knn_heap_free(&heap);
        heap_initialized = 0;
    }
    if (reader_initialized)
    {
        knn_reader_close(&cand_reader);
        reader_initialized = 0;
    }
    if (model_initialized)
    {
        knn_model_free(&knn_model);
        model_initialized = 0;
    }

    if (query_buf)
    {
        free(query_buf);
        query_buf = NULL;
    }
    if (cand_buf)
    {
        free(cand_buf);
        cand_buf = NULL;
    }
    if (query_sq8)
    {
        free(query_sq8);
        query_sq8 = NULL;
    }
    if (query_sq16)
    {
        free(query_sq16);
        query_sq16 = NULL;
    }
    if (query_eq16)
    {
        free(query_eq16);
        query_eq16 = NULL;
    }
    if (anchor_dists)
    {
        free(anchor_dists);
        anchor_dists = NULL;
    }
    if (scores_buf)
    {
        free(scores_buf);
        scores_buf = NULL;
    }
    if (visited.tags)
    {
        free(visited.tags);
        visited.tags = NULL;
    }
    if (out_indices)
    {
        free(out_indices);
        out_indices = NULL;
    }
    if (out_dists)
    {
        free(out_dists);
        out_dists = NULL;
    }

    current_ndim = 0;
}

/**
 * gric_knn_fps_init_output_streams() - Create output ImageStreamIO stream.
 * @k:       Number of nearest neighbors.
 * @out_knn: Output IMAGE struct handle.
 *
 * Return: 0 on success, non-zero on failure.
 */
errno_t gric_knn_fps_init_output_streams(
    uint32_t  k,
    IMAGE    *out_knn)
{
    const char *stream_name = (fps_knn_out_name[0] != '\0')
                                  ? fps_knn_out_name : "gric_knn";

    /* Create 2D output stream [k x 2] (Row 0 = indices, Row 1 = distances) */
    uint32_t dims[2] = { k, 2 };
    if (ImageStreamIO_createIm_gpu(out_knn, stream_name, 2, dims,
                                   _DATATYPE_FLOAT, -1, 1, 10,
                                   IMAGE_NB_SEMAPHORE, 0, 0) != 0)
    {
        return 1;
    }

    return 0;
}

/**
 * gric_knn_fps_close_output_streams() - Safely disconnect output stream.
 * @out_knn: Output stream handle.
 */
void gric_knn_fps_close_output_streams(
    IMAGE *out_knn)
{
    if (out_knn && out_knn->used == 1)
    {
        ImageStreamIO_closeIm(out_knn);
    }
}

/**
 * gric_knn_fps_process_frame() - Search top-k neighbors for one incoming frame.
 * @raw_pixels:  Source pixel buffer pointer.
 * @datatype:    Input stream pixel data type code.
 * @ndim:        Dimension count.
 * @frame_index: Frame count (cnt0).
 * @frame_time:  Arrival timestamp.
 * @latency_us:  Algorithm execution latency in microseconds.
 * @out_knn:     Output k-NN matches stream handle.
 *
 * Return: 0 on success, non-zero on failure.
 */
errno_t gric_knn_fps_process_frame(
    const void      *raw_pixels,
    int              datatype,
    uint32_t         ndim,
    uint64_t         frame_index,
    struct timespec  frame_time,
    double           latency_us,
    IMAGE           *out_knn)
{
    (void)frame_time;
    (void)latency_us;
    if (!model_initialized || !query_buf || ndim != current_ndim)
    {
        return 1;
    }

    /* Convert input pixels to query buffer */
    {
        switch (datatype)
        {
        case _DATATYPE_FLOAT:
        {
            if (knn_model.is_double)
            {
                const float *f_in = (const float *)raw_pixels;
                double *d_out = (double *)query_buf;
                for (uint32_t ii = 0; ii < ndim; ii++)
                {
                    d_out[ii] = (double)f_in[ii];
                }
            }
            else
            {
                memcpy(query_buf, raw_pixels, ndim * sizeof(float));
            }
            break;
        }
        case _DATATYPE_DOUBLE:
        {
            if (knn_model.is_double)
            {
                memcpy(query_buf, raw_pixels, ndim * sizeof(double));
            }
            else
            {
                const double *d_in = (const double *)raw_pixels;
                float *f_out = (float *)query_buf;
                for (uint32_t ii = 0; ii < ndim; ii++)
                {
                    f_out[ii] = (float)d_in[ii];
                }
            }
            break;
        }
        case _DATATYPE_UINT16:
        {
            const uint16_t *u16_in = (const uint16_t *)raw_pixels;
            if (knn_model.is_double)
            {
                double *d_out = (double *)query_buf;
                for (uint32_t ii = 0; ii < ndim; ii++)
                {
                    d_out[ii] = (double)u16_in[ii];
                }
            }
            else
            {
                float *f_out = (float *)query_buf;
                for (uint32_t ii = 0; ii < ndim; ii++)
                {
                    f_out[ii] = (float)u16_in[ii];
                }
            }
            break;
        }
        case _DATATYPE_UINT8:
        {
            const uint8_t *u8_in = (const uint8_t *)raw_pixels;
            if (knn_model.is_double)
            {
                double *d_out = (double *)query_buf;
                for (uint32_t ii = 0; ii < ndim; ii++)
                {
                    d_out[ii] = (double)u8_in[ii];
                }
            }
            else
            {
                float *f_out = (float *)query_buf;
                for (uint32_t ii = 0; ii < ndim; ii++)
                {
                    f_out[ii] = (float)u8_in[ii];
                }
            }
            break;
        }
        default:
            return 1;
        }
    }

    /* Quantize query buffer if filters enabled */
    if (knn_cfg.use_sq8 && query_sq8 != NULL)
    {
        if (knn_model.is_double)
        {
            sq8_quantize_double((const double *)query_buf, query_sq8, &knn_model.sq8_params);
        }
        else
        {
            sq8_quantize_float((const float *)query_buf, query_sq8, &knn_model.sq8_params);
        }
    }

    if (knn_cfg.use_sq16 && query_sq16 != NULL)
    {
        if (knn_model.is_double)
        {
            sq16_quantize_double((const double *)query_buf, query_sq16, &knn_model.sq16_params);
        }
        else
        {
            sq16_quantize_float((const float *)query_buf, query_sq16, &knn_model.sq16_params);
        }
    }

    if (knn_cfg.use_eq16 && query_eq16 != NULL)
    {
        if (knn_model.is_double)
        {
            eq16_quantize_double((const double *)query_buf, query_eq16, &knn_model.eq16_params);
        }
        else
        {
            eq16_quantize_float((const float *)query_buf, query_eq16, &knn_model.eq16_params);
        }
    }

    /* Reset search state */
    knn_heap_reset(&heap);
    visited.epoch++;
    if (visited.epoch == 0)
    {
        memset(visited.tags, 0, (size_t)knn_model.total_dataset_frames * sizeof(uint32_t));
        visited.epoch = 1;
    }

    KnnTelemetry thread_telem;
    memset(&thread_telem, 0, sizeof(KnnTelemetry));

    /* Execute k-NN search */
    knn_search_cross_dataset_frame(
        (long)frame_index, query_buf, &knn_model, &knn_cfg,
        &cand_reader, cand_buf, anchor_dists, scores_buf,
        &heap, &tracker, &visited, &thread_telem);

    int k = knn_cfg.k;
    knn_heap_extract_sorted(&heap, out_indices, out_dists, k);

    /* Publish to output stream [k x 2] */
    if (out_knn && out_knn->used == 1)
    {
        out_knn->md[0].write = 1;
        float *raw_out = (float *)out_knn->array.raw;
        for (int j = 0; j < k; j++)
        {
            raw_out[j] = (float)out_indices[j];
            raw_out[k + j] = (float)out_dists[j];
        }
    }

    fps_knn_status_queries++;
    fps_knn_status_dist_evals += thread_telem.framedist_calls;
    fps_knn_status_min_dist = (heap.count > 0) ? out_dists[0] : 0.0;
    fps_knn_status_max_dist = (heap.count > 0) ? out_dists[heap.count - 1] : 0.0;

    return 0;
}

/**
 * gric_knn_fps_status_init() - Initialize file-mapped shared memory status bridge.
 * @fps_name:    Name of the active FPS daemon instance.
 * @custom_path: Optional custom path to SHM status file.
 *
 * Return: 0 on success, non-zero on error.
 */
errno_t gric_knn_fps_status_init(
    const char *fps_name,
    const char *custom_path)
{
    clock_gettime(CLOCK_MONOTONIC, &session_start_time);
    prev_fps_calc_time = session_start_time;
    prev_fps_calc_frames = 0;

    if (custom_path && custom_path[0] != '\0')
    {
        strncpy(active_shm_path, custom_path, sizeof(active_shm_path) - 1);
        active_shm_path[sizeof(active_shm_path) - 1] = '\0';
    }
    else
    {
        snprintf(active_shm_path, sizeof(active_shm_path), "/dev/shm/gric_knn_status_%s.shm",
                 (fps_name && fps_name[0] != '\0') ? fps_name : "default");
    }

    int fd = open(active_shm_path, O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (fd < 0)
    {
        return 1;
    }

    if (ftruncate(fd, (off_t)sizeof(GricClusterShmStatus)) != 0)
    {
        close(fd);
        return 1;
    }

    void *mapped = mmap(NULL, sizeof(GricClusterShmStatus),
                        PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    if (mapped == MAP_FAILED)
    {
        status_shm_ptr = NULL;
        return 1;
    }

    status_shm_ptr = (GricClusterShmStatus *)mapped;
    memset(status_shm_ptr, 0, sizeof(GricClusterShmStatus));
    status_shm_ptr->magic = GRIC_SHM_MAGIC;
    status_shm_ptr->version = GRIC_SHM_VERSION;
    status_shm_ptr->pid = (uint32_t)getpid();
    status_shm_ptr->status_state = GRIC_STATUS_RUNNING;
    status_shm_ptr->total_frames = fps_knn_max_frames;
    status_shm_ptr->num_clusters = fps_knn_k;
    strncpy(status_shm_ptr->input_source, fps_knn_in_name,
            sizeof(status_shm_ptr->input_source) - 1);

    if (getcwd(status_shm_ptr->config_cwd, sizeof(status_shm_ptr->config_cwd) - 1) == NULL)
    {
        status_shm_ptr->config_cwd[0] = '\0';
    }

    status_shm_ptr->config_rlim = fps_knn_rlim;
    status_shm_ptr->active_threads = fps_knn_ncpu;

    return 0;
}

/**
 * gric_knn_fps_status_update() - Refresh real-time status parameters and bridge SHM.
 * @frame_index: Frame counter (cnt0).
 * @latency_us:  Algorithm execution latency in microseconds.
 * @stream_lag:  Input stream frame lag.
 * @write_slice: Current input ring buffer write slice.
 * @read_slice:  Current input ring buffer read slice.
 */
void gric_knn_fps_status_update(
    uint64_t frame_index,
    double   latency_us,
    long     stream_lag,
    long     write_slice,
    long     read_slice)
{
    (void)frame_index;
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);

    fps_knn_status_latency_us = latency_us;
    fps_knn_status_stream_lag = (int64_t)stream_lag;
    fps_knn_status_memory_rss_mb = gric_rss_mb_sampled();

    double elapsed_fps = (now.tv_sec - prev_fps_calc_time.tv_sec) +
                         (now.tv_nsec - prev_fps_calc_time.tv_nsec) * 1e-9;
    if (elapsed_fps >= 0.5)
    {
        uint64_t frames_delta = fps_knn_status_queries - prev_fps_calc_frames;
        fps_knn_status_fps = (double)frames_delta / elapsed_fps;
        prev_fps_calc_time = now;
        prev_fps_calc_frames = fps_knn_status_queries;
    }

    if (status_shm_ptr != NULL)
    {
        status_shm_ptr->status_state = GRIC_STATUS_RUNNING;
        status_shm_ptr->total_frames_processed = fps_knn_status_queries;
        status_shm_ptr->num_clusters = fps_knn_status_k;
        status_shm_ptr->framedist_calls = fps_knn_status_dist_evals;
        status_shm_ptr->elapsed_ms = (now.tv_sec - session_start_time.tv_sec) * 1000.0 +
                                     (now.tv_nsec - session_start_time.tv_nsec) / 1000000.0;
        status_shm_ptr->stream_read_slice = read_slice;
        status_shm_ptr->stream_write_slice = write_slice;
        status_shm_ptr->stream_lag = stream_lag;
        status_shm_ptr->last_assignment_dist = fps_knn_status_min_dist;
        status_shm_ptr->memory_rss_kb = (uint64_t)(fps_knn_status_memory_rss_mb * 1024.0);
        status_shm_ptr->active_threads = fps_knn_ncpu;
        status_shm_ptr->config_rlim = fps_knn_rlim;
        status_shm_ptr->total_frames = fps_knn_max_frames;

        struct timespec real_now;
        clock_gettime(CLOCK_REALTIME, &real_now);
        status_shm_ptr->last_update_time = (uint64_t)real_now.tv_sec * 1000000000ULL +
                                           (uint64_t)real_now.tv_nsec;
    }
}

/**
 * gric_knn_fps_status_close() - Finalize and unmap the status shared memory bridge.
 * @exit_state: Exit code (0 for success, non-zero for failure).
 */
void gric_knn_fps_status_close(
    int exit_state)
{
    if (status_shm_ptr != NULL)
    {
        status_shm_ptr->status_state = (exit_state == 0) ? GRIC_STATUS_SUCCESS
                                                         : GRIC_STATUS_ERROR;
        munmap(status_shm_ptr, sizeof(GricClusterShmStatus));
        status_shm_ptr = NULL;
    }
    if (active_shm_path[0] != '\0')
    {
        unlink(active_shm_path);
        active_shm_path[0] = '\0';
    }
}
