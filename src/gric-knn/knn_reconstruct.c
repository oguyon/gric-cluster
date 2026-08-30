/**
 * @file knn_reconstruct.c
 * @brief Computes a reconstructed dataset using k-NN indices and distances.
 */

#include "knn_defs.h"
#include "knn_reader.h"
#include "shared/cli_colors.h"
#include "../../shared/gric_bin_io.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void print_usage(
    const char *prog_name)
{
    printf(
        "Usage: %s <knn_result_dir> <dataset_B>"
        " [-o <output_D>] [-w uniform|idw]"
        " [-alpha 1.0] [-v] [-h]\n",
        prog_name);
}

int main(
    int    argc,
    char **argv)
{
    if (argc < 3)
    {
        print_usage(argv[0]);
        return 1;
    }

    cli_colors_init();

    const char *knn_dir = argv[1];
    const char *data_b_path = argv[2];
    
    char out_path[1024];
    char qual_path[1024];
    snprintf(out_path, sizeof(out_path), "%s/knn_recon.bin", knn_dir);
    snprintf(qual_path, sizeof(qual_path), "%s/knn_quality.bin", knn_dir);
    
    int use_idw = 0;
    float alpha = 1.0f;
    int verbose = 0;

    for (int i = 3; i < argc; i++)
    {
        if (strcmp(argv[i], "-h") == 0)
        {
            print_usage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc)
        {
            strncpy(out_path, argv[++i], sizeof(out_path) - 1);
        }
        else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc)
        {
            const char *mode = argv[++i];
            if (strcmp(mode, "idw") == 0)
            {
                use_idw = 1;
            }
            else if (strcmp(mode, "uniform") != 0)
            {
                fprintf(stderr, "Unknown weighting mode: %s\n", mode);
                return 1;
            }
        }
        else if (strcmp(argv[i], "-alpha") == 0 && i + 1 < argc)
        {
            alpha = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-v") == 0)
        {
            verbose = 1;
        }
    }

    struct timespec t_start;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    char idx_path[1024], dist_path[1024];
    snprintf(idx_path, sizeof(idx_path), "%s/knn_indices.bin", knn_dir);
    snprintf(dist_path, sizeof(dist_path), "%s/knn_distances.bin", knn_dir);

    FILE* fp_idx = fopen(idx_path, "rb");
    if (!fp_idx)
    {
        fprintf(stderr, "Error: cannot open %s\n", idx_path);
        return 1;
    }

    gric_bin_header_t hdr_idx;
    if (gric_bin_read_header(fp_idx, &hdr_idx, NULL) != 0)
    {
        fprintf(stderr, "Error: invalid header in %s\n", idx_path);
        fclose(fp_idx);
        return 1;
    }
    
    FILE* fp_dist = fopen(dist_path, "rb");
    if (!fp_dist)
    {
        fprintf(stderr, "Error: cannot open %s\n", dist_path);
        fclose(fp_idx);
        return 1;
    }

    gric_bin_header_t hdr_dist;
    if (gric_bin_read_header(fp_dist, &hdr_dist, NULL) != 0)
    {
        fprintf(stderr, "Error: invalid header in %s\n", dist_path);
        fclose(fp_idx);
        fclose(fp_dist);
        return 1;
    }

    uint64_t N = hdr_idx.dims[0];
    uint64_t k = hdr_idx.dims[1];

    fseek(fp_idx, hdr_idx.header_bytes, SEEK_SET);
    fseek(fp_dist, hdr_dist.header_bytes, SEEK_SET);

    uint32_t* indices = malloc((size_t)(N * k * sizeof(uint32_t)));
    float* distances = malloc((size_t)(N * k * sizeof(float)));

    if (!indices || !distances)
    {
        fprintf(stderr, "Error: out of memory\n");
        if (indices) free(indices);
        if (distances) free(distances);
        fclose(fp_idx);
        fclose(fp_dist);
        return 1;
    }

    if (fread(indices, sizeof(uint32_t), N * k, fp_idx) != N * k)
    {
        fprintf(stderr, "Error: failed to read indices\n");
        free(indices);
        free(distances);
        fclose(fp_idx);
        fclose(fp_dist);
        return 1;
    }
    if (fread(distances, sizeof(float), N * k, fp_dist) != N * k)
    {
        fprintf(stderr, "Error: failed to read distances\n");
        free(indices);
        free(distances);
        fclose(fp_idx);
        fclose(fp_dist);
        return 1;
    }

    fclose(fp_idx); fp_idx = NULL;
    fclose(fp_dist); fp_dist = NULL;

    long b_n, b_w, b_h;
    if (knn_reader_inspect(data_b_path, &b_n, &b_w, &b_h) != 0)
    {
        fprintf(stderr, "Error: failed to inspect %s\n", data_b_path);
        free(indices);
        free(distances);
        return 1;
    }

    uint64_t b_dim = (uint64_t)(b_w * b_h);

    KnnFrameReader b_ctx;
    if (knn_reader_open(&b_ctx, data_b_path, b_n, b_w, b_h) != 0)
    {
        fprintf(stderr, "Error: failed to open %s\n", data_b_path);
        free(indices);
        free(distances);
        return 1;
    }

    float* data_b = malloc((size_t)(b_n * b_dim * sizeof(float)));
    double* temp_frame = malloc((size_t)(b_dim * sizeof(double)));
    if (!data_b || !temp_frame)
    {
        fprintf(stderr, "Error: out of memory for dataset B\n");
        if (data_b) free(data_b);
        if (temp_frame) free(temp_frame);
        knn_reader_close(&b_ctx);
        free(indices);
        free(distances);
        return 1;
    }

    for (long i = 0; i < b_n; i++)
    {
        if (knn_reader_read_frame(&b_ctx, i, temp_frame) != 0)
        {
            fprintf(stderr, "Error: failed to read frame %ld from %s\n", i, data_b_path);
            knn_reader_close(&b_ctx);
            free(data_b);
            free(temp_frame);
            free(indices);
            free(distances);
            return 1;
        }
        for (long d = 0; d < (long)b_dim; d++)
        {
            data_b[i * b_dim + d] = (float)temp_frame[d];
        }
    }
    free(temp_frame);
    knn_reader_close(&b_ctx);

    float* D = calloc((size_t)(N * b_dim), sizeof(float));
    float* qual = calloc((size_t)(N * 2), sizeof(float));
    float* weights = malloc((size_t)(k * sizeof(float)));
    
    if (!D || !qual || !weights)
    {
        fprintf(stderr, "Error: out of memory for D/qual/weights\n");
        if (D) free(D);
        if (qual) free(qual);
        if (weights) free(weights);
        free(indices);
        free(distances);
        free(data_b);
        return 1;
    }

    double total_kth_dist = 0.0;
    double total_variance = 0.0;

    for (uint64_t i = 0; i < N; i++)
    {
        uint32_t* restrict idx = indices + i * k;
        float* restrict dist = distances + i * k;
        
        int exact_match = -1;
        for (uint64_t j = 0; j < k; j++)
        {
            if (dist[j] < 1e-9f)
            {
                exact_match = (int)j;
                break;
            }
        }

        if (exact_match >= 0)
        {
            for (uint64_t j = 0; j < k; j++)
            {
                weights[j] = 0.0f;
            }
            weights[exact_match] = 1.0f;
        }
        else if (!use_idw)
        {
            for (uint64_t j = 0; j < k; j++)
            {
                weights[j] = 1.0f / (float)k;
            }
        }
        else
        {
            float sum_w = 0.0f;
            for (uint64_t j = 0; j < k; j++)
            {
                float d = fmaxf(dist[j], 1e-7f);
                weights[j] = powf(1.0f / d, alpha);
                sum_w += weights[j];
            }
            for (uint64_t j = 0; j < k; j++)
            {
                weights[j] /= sum_w;
            }
        }

        float* restrict out_D = D + i * b_dim;
        for (uint64_t j = 0; j < k; j++)
        {
            uint32_t neighbor_idx = idx[j];
            float w = weights[j];
            float* restrict neighbor_b = data_b + neighbor_idx * b_dim;
            
            for (uint64_t d = 0; d < b_dim; d++)
            {
                out_D[d] += w * neighbor_b[d];
            }
        }

        float variance = 0.0f;
        for (uint64_t j = 0; j < k; j++)
        {
            uint32_t neighbor_idx = idx[j];
            float w = weights[j];
            float* restrict neighbor_b = data_b + neighbor_idx * b_dim;
            
            float dist_sq = 0.0f;
            for (uint64_t d = 0; d < b_dim; d++)
            {
                float diff = neighbor_b[d] - out_D[d];
                dist_sq += diff * diff;
            }
            variance += w * dist_sq;
        }

        qual[i * 2 + 0] = dist[k - 1];
        qual[i * 2 + 1] = variance;

        total_kth_dist += dist[k - 1];
        total_variance += variance;
    }

    FILE* fp_out = fopen(out_path, "wb");
    if (!fp_out)
    {
        fprintf(stderr, "Error: cannot write to %s\n", out_path);
        free(indices); free(distances); free(data_b); free(D); free(qual); free(weights);
        return 1;
    }

    gric_bin_header_t hdr_out;
    memset(&hdr_out, 0, sizeof(hdr_out));
    hdr_out.file_type = GRIC_BIN_TYPE_GENERIC;
    hdr_out.data_type = GRIC_BIN_DTYPE_FLOAT32;
    hdr_out.flags = GRIC_BIN_FLAG_ROW_MAJOR;
    hdr_out.ndim = 2;
    hdr_out.dims[0] = N;
    hdr_out.dims[1] = b_dim;
    hdr_out.num_elements = N * b_dim;
    hdr_out.data_bytes = hdr_out.num_elements * sizeof(float);
    if (gric_bin_write_header(fp_out, &hdr_out, "knn_reconstruct output D") != 0)
    {
        fprintf(stderr, "Error writing header to %s\n", out_path);
    }
    fwrite(D, sizeof(float), N * b_dim, fp_out);
    fclose(fp_out);

    FILE* fp_qual = fopen(qual_path, "wb");
    if (!fp_qual)
    {
        fprintf(stderr, "Error: cannot write to %s\n", qual_path);
        free(indices); free(distances); free(data_b); free(D); free(qual); free(weights);
        return 1;
    }

    gric_bin_header_t hdr_qual;
    memset(&hdr_qual, 0, sizeof(hdr_qual));
    hdr_qual.file_type = GRIC_BIN_TYPE_GENERIC;
    hdr_qual.data_type = GRIC_BIN_DTYPE_FLOAT32;
    hdr_qual.flags = GRIC_BIN_FLAG_ROW_MAJOR;
    hdr_qual.ndim = 2;
    hdr_qual.dims[0] = N;
    hdr_qual.dims[1] = 2;
    hdr_qual.num_elements = N * 2;
    hdr_qual.data_bytes = hdr_qual.num_elements * sizeof(float);
    if (gric_bin_write_header(fp_qual, &hdr_qual, "knn_reconstruct output quality") != 0)
    {
        fprintf(stderr, "Error writing header to %s\n", qual_path);
    }
    fwrite(qual, sizeof(float), N * 2, fp_qual);
    fclose(fp_qual);

    struct timespec t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double wall_time = (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    if (verbose)
    {
        printf("\n%s--- KNN Reconstruction Complete ---%s\n", ansi_bold_green, ansi_reset);
        printf("Queries       : %llu\n", (unsigned long long)N);
        printf("Neighbors (k) : %llu\n", (unsigned long long)k);
        printf("Output Dim    : %llu\n", (unsigned long long)b_dim);
        printf("Weighting     : %s%s%s\n", ansi_bold_cyan, use_idw ? "idw" : "uniform", ansi_reset);
        if (use_idw) printf("IDW Alpha     : %.2f\n", alpha);
        printf("Avg kth dist  : %f\n", total_kth_dist / N);
        printf("Avg variance  : %f\n", total_variance / N);
        printf("Wall time     : %.3f s\n", wall_time);
    }
    else
    {
        printf("\n%s--- KNN Reconstruction Complete ---%s\n", ansi_bold_green, ansi_reset);
        printf("Queries       : %llu\n", (unsigned long long)N);
        printf("Output Dim    : %llu\n", (unsigned long long)b_dim);
        printf("Weighting     : %s%s%s\n", ansi_bold_cyan, use_idw ? "idw" : "uniform", ansi_reset);
        printf("Avg kth dist  : %f\n", total_kth_dist / N);
        printf("Avg variance  : %f\n", total_variance / N);
        printf("Wall time     : %.3f s\n", wall_time);
    }

    free(indices);
    free(distances);
    free(data_b);
    free(D);
    free(qual);
    free(weights);

    return 0;
}
