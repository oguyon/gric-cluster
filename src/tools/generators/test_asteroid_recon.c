/**
 * @file test_asteroid_recon.c
 * @brief Benchmark cross-view reconstruction for rotating asteroid image sequences.
 *
 * Evaluates whether an unobserved viewpoint (e.g. View Y) can be reconstructed
 * from an observed viewpoint (e.g. View X) using non-parametric k-NN manifold
 * reconstruction learned from paired training sequences.
 *
 * Usage:
 *   gric-asteroid-recon-test -train-in <fits> -train-out <fits>
 *                            -test-in <fits>  -test-true <fits> [options]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fitsio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * struct Neighbor - Track neighbor frame index and distance.
 * @index: Frame index in training dataset.
 * @dist:  Euclidean distance to query frame.
 */
typedef struct
{
    int   index;
    float dist;
} Neighbor;

/**
 * read_fits_cube() - Read a 3D FITS image cube into newly allocated float buffer.
 * @path:    Path to FITS file.
 * @out_W:   Pointer to store width (NAXIS1).
 * @out_H:   Pointer to store height (NAXIS2).
 * @out_N:   Pointer to store frame count (NAXIS3).
 *
 * Return: Pointer to allocated float buffer, or NULL on error.
 */
static float *read_fits_cube(
    const char *path,
    int        *out_W,
    int        *out_H,
    int        *out_N)
{
    fitsfile *fptr = NULL;
    int status = 0;

    fits_open_file(&fptr, path, READONLY, &status);
    if (status)
    {
        fprintf(stderr, "Error: cannot open FITS file %s (status %d)\n", path, status);
        return NULL;
    }

    int naxis = 0;
    long naxes[3] = { 0, 0, 0 };
    fits_get_img_dim(fptr, &naxis, &status);
    if (status || naxis != 3)
    {
        fprintf(stderr, "Error: %s is not a 3D image cube (naxis=%d)\n", path, naxis);
        fits_close_file(fptr, &status);
        return NULL;
    }

    fits_get_img_size(fptr, 3, naxes, &status);
    if (status)
    {
        fprintf(stderr, "Error reading dimensions of %s\n", path);
        fits_close_file(fptr, &status);
        return NULL;
    }

    int W = (int)naxes[0];
    int H = (int)naxes[1];
    int N = (int)naxes[2];
    size_t total_pix = (size_t)W * (size_t)H * (size_t)N;

    float *data = malloc(total_pix * sizeof(float));
    if (!data)
    {
        fprintf(stderr, "Error: malloc failed for %zu elements\n", total_pix);
        fits_close_file(fptr, &status);
        return NULL;
    }

    long fpixel[3] = { 1, 1, 1 };
    fits_read_pix(fptr, TFLOAT, fpixel, (long)total_pix, NULL, data, NULL, &status);
    fits_close_file(fptr, &status);

    if (status)
    {
        fprintf(stderr, "Error reading pixel data from %s (status %d)\n", path, status);
        free(data);
        return NULL;
    }

    *out_W = W;
    *out_H = H;
    *out_N = N;
    return data;
}

/**
 * write_fits_cube() - Write a 3D float image cube to FITS file.
 * @path: Path to output file (overwriting with '!' prefix).
 * @data: Float image cube buffer.
 * @W:    Width (NAXIS1).
 * @H:    Height (NAXIS2).
 * @N:    Depth / frame count (NAXIS3).
 *
 * Return: 0 on success, or non-zero CFITSIO status.
 */
static int write_fits_cube(
    const char  *path,
    const float *data,
    int          W,
    int          H,
    int          N)
{
    fitsfile *fptr = NULL;
    int status = 0;
    char bangpath[1024];
    snprintf(bangpath, sizeof(bangpath), "!%s", path);

    fits_create_file(&fptr, bangpath, &status);
    if (status)
    {
        fprintf(stderr, "Error: cannot create %s (status %d)\n", path, status);
        return status;
    }

    long naxes[3] = { (long)W, (long)H, (long)N };
    fits_create_img(fptr, FLOAT_IMG, 3, naxes, &status);
    if (status)
    {
        fprintf(stderr, "Error: cannot create img in %s (status %d)\n", path, status);
        fits_close_file(fptr, &status);
        return status;
    }

    long fpixel[3] = { 1, 1, 1 };
    long total_pix = (long)W * (long)H * (long)N;
    fits_write_pix(fptr, TFLOAT, fpixel, total_pix, (void *)data, &status);
    fits_close_file(fptr, &status);

    return status;
}

/**
 * find_top_k_neighbors() - Find k nearest training frames for a query frame.
 * @query_frame: Pointer to query frame pixels (length npix).
 * @train_in:    All training input frames (size n_train * npix).
 * @n_train:     Number of training frames.
 * @npix:        Pixels per frame (W * H).
 * @k:           Number of nearest neighbors.
 * @neighbors:   Output array of size k (sorted in ascending distance).
 */
static void find_top_k_neighbors(
    const float *restrict query_frame,
    const float *restrict train_in,
    int                   n_train,
    size_t                npix,
    int                   k,
    Neighbor    *restrict neighbors)
{
    /* Initialize neighbor array with sentinel large distances */
    for (int j = 0; j < k; j++)
    {
        neighbors[j].index = -1;
        neighbors[j].dist  = 1e30f;
    } // for j

    for (int i = 0; i < n_train; i++)
    {
        const float *tr_ptr = train_in + (size_t)i * npix;
        float d2 = 0.0f;

        for (size_t p = 0; p < npix; p++)
        {
            float diff = query_frame[p] - tr_ptr[p];
            d2 += diff * diff;
        } // for p

        float dist = sqrtf(d2);

        /* Insertion into sorted top-k array */
        if (dist < neighbors[k - 1].dist)
        {
            int pos = k - 1;
            while (pos > 0 && dist < neighbors[pos - 1].dist)
            {
                neighbors[pos] = neighbors[pos - 1];
                pos--;
            }
            neighbors[pos].index = i;
            neighbors[pos].dist  = dist;
        }
    } // for i
}

/**
 * render_ascii_comparison() - Print ASCII side-by-side terminal rendering of a frame.
 * @in_pix:    Input query frame.
 * @true_pix:  Ground-truth target frame.
 * @recon_pix: Reconstructed target frame.
 * @W:         Frame width.
 * @H:         Frame height.
 * @frame_idx: Test frame index.
 */
static void render_ascii_comparison(
    const float *in_pix,
    const float *true_pix,
    const float *recon_pix,
    int          W,
    int          H,
    int          frame_idx)
{
    static const char ramp[] = " .:-=+*#%@";
    int num_shades = (int)strlen(ramp);

    printf("\n=== Visual Comparison for Test Frame %d ===\n", frame_idx);
    printf("   [Input View]                 [True Target View]           "
           "[Recon Target View]          [|Diff| x 3]\n");

    for (int y = 0; y < H; y++)
    {
        /* 1. Input View */
        for (int x = 0; x < W; x++)
        {
            float v = in_pix[y * W + x];
            int idx = (int)(v * (float)(num_shades - 1));
            idx = (idx < 0) ? 0 : ((idx >= num_shades) ? num_shades - 1 : idx);
            putchar(ramp[idx]);
        }
        printf("    ");

        /* 2. True Target View */
        for (int x = 0; x < W; x++)
        {
            float v = true_pix[y * W + x];
            int idx = (int)(v * (float)(num_shades - 1));
            idx = (idx < 0) ? 0 : ((idx >= num_shades) ? num_shades - 1 : idx);
            putchar(ramp[idx]);
        }
        printf("    ");

        /* 3. Reconstructed Target View */
        for (int x = 0; x < W; x++)
        {
            float v = recon_pix[y * W + x];
            int idx = (int)(v * (float)(num_shades - 1));
            idx = (idx < 0) ? 0 : ((idx >= num_shades) ? num_shades - 1 : idx);
            putchar(ramp[idx]);
        }
        printf("    ");

        /* 4. Absolute Difference (boosted 3x for contrast) */
        for (int x = 0; x < W; x++)
        {
            float diff = fabsf(recon_pix[y * W + x] - true_pix[y * W + x]) * 3.0f;
            int idx = (int)(diff * (float)(num_shades - 1));
            idx = (idx < 0) ? 0 : ((idx >= num_shades) ? num_shades - 1 : idx);
            putchar(ramp[idx]);
        }
        putchar('\n');
    } // for y
    printf("\n");
}

static void print_usage(
    const char *prog)
{
    printf("Usage: %s [options]\n", prog);
    printf("Required:\n");
    printf("  -train-in <fits>    Training input FITS cube (e.g. View X)\n");
    printf("  -train-out <fits>   Training target FITS cube (e.g. View Y)\n");
    printf("  -test-in <fits>     Test input FITS cube (e.g. View X test)\n");
    printf("  -test-true <fits>   Test ground-truth target FITS cube (e.g. View Y test)\n");
    printf("Options:\n");
    printf("  -k <int>            Number of nearest neighbors   (default: 10)\n");
    printf("  -w <uniform|idw>    Weighting mode                (default: idw)\n");
    printf("  -alpha <float>      IDW distance power exponent   (default: 1.0)\n");
    printf("  -o <fits>           Save reconstructed FITS cube  (optional)\n");
    printf("  -diff <fits>        Save absolute difference cube (optional)\n");
    printf("  -show <int>         Show ASCII render of frame    (default: 0, -1 to disable)\n");
    printf("  -h, --help          Show this help message\n");
}

int main(
    int   argc,
    char *argv[])
{
    const char *path_tr_in  = NULL;
    const char *path_tr_out = NULL;
    const char *path_te_in  = NULL;
    const char *path_te_true = NULL;
    const char *path_out    = NULL;
    const char *path_diff   = NULL;

    int   k          = 10;
    int   use_idw    = 1;
    float alpha      = 1.0f;
    int   show_frame = 0;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-train-in") == 0 && i + 1 < argc)
        {
            path_tr_in = argv[++i];
        }
        else if (strcmp(argv[i], "-train-out") == 0 && i + 1 < argc)
        {
            path_tr_out = argv[++i];
        }
        else if (strcmp(argv[i], "-test-in") == 0 && i + 1 < argc)
        {
            path_te_in = argv[++i];
        }
        else if (strcmp(argv[i], "-test-true") == 0 && i + 1 < argc)
        {
            path_te_true = argv[++i];
        }
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc)
        {
            path_out = argv[++i];
        }
        else if (strcmp(argv[i], "-diff") == 0 && i + 1 < argc)
        {
            path_diff = argv[++i];
        }
        else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc)
        {
            k = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc)
        {
            const char *mode = argv[++i];
            if (strcmp(mode, "idw") == 0)
            {
                use_idw = 1;
            }
            else if (strcmp(mode, "uniform") == 0)
            {
                use_idw = 0;
            }
            else
            {
                fprintf(stderr, "Unknown weighting mode: %s\n", mode);
                return 1;
            }
        }
        else if (strcmp(argv[i], "-alpha") == 0 && i + 1 < argc)
        {
            alpha = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-show") == 0 && i + 1 < argc)
        {
            show_frame = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            print_usage(argv[0]);
            return 0;
        }
        else
        {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    } // for i

    if (!path_tr_in || !path_tr_out || !path_te_in || !path_te_true)
    {
        fprintf(stderr, "Error: missing required input datasets\n");
        print_usage(argv[0]);
        return 1;
    }

    if (k <= 0)
    {
        fprintf(stderr, "Error: k must be > 0\n");
        return 1;
    }

    printf("=== Loading Datasets ===\n");
    int w_tr_in, h_tr_in, n_tr_in;
    int w_tr_out, h_tr_out, n_tr_out;
    int w_te_in, h_te_in, n_te_in;
    int w_te_true, h_te_true, n_te_true;

    float *tr_in = read_fits_cube(path_tr_in, &w_tr_in, &h_tr_in, &n_tr_in);
    if (!tr_in) return 1;

    float *tr_out = read_fits_cube(path_tr_out, &w_tr_out, &h_tr_out, &n_tr_out);
    if (!tr_out) { free(tr_in); return 1; }

    float *te_in = read_fits_cube(path_te_in, &w_te_in, &h_te_in, &n_te_in);
    if (!te_in) { free(tr_in); free(tr_out); return 1; }

    float *te_true = read_fits_cube(path_te_true, &w_te_true, &h_te_true, &n_te_true);
    if (!te_true) { free(tr_in); free(tr_out); free(te_in); return 1; }

    /* Validate dimensions */
    if (w_tr_in != w_tr_out || h_tr_in != h_tr_out || n_tr_in != n_tr_out ||
        w_tr_in != w_te_in  || h_tr_in != h_te_in  ||
        w_te_in != w_te_true|| h_te_in != h_te_true|| n_te_in != n_te_true)
    {
        fprintf(stderr, "Error: dataset dimension mismatch!\n");
        fprintf(stderr, "  Train In:   %dx%dx%d\n", w_tr_in, h_tr_in, n_tr_in);
        fprintf(stderr, "  Train Out:  %dx%dx%d\n", w_tr_out, h_tr_out, n_tr_out);
        fprintf(stderr, "  Test In:    %dx%dx%d\n", w_te_in, h_te_in, n_te_in);
        fprintf(stderr, "  Test True:  %dx%dx%d\n", w_te_true, h_te_true, n_te_true);
        free(tr_in); free(tr_out); free(te_in); free(te_true);
        return 1;
    }

    int W = w_tr_in;
    int H = h_tr_in;
    int n_train = n_tr_in;
    int n_test  = n_te_in;
    size_t npix = (size_t)W * (size_t)H;

    printf("Image dimensions: %dx%d (%zu pixels/frame)\n", W, H, npix);
    printf("Training pairs:   %d frames\n", n_train);
    printf("Test queries:     %d frames\n", n_test);
    printf("k-NN setting:     k=%d, weighting=%s (alpha=%.2f)\n",
           k, use_idw ? "idw" : "uniform", alpha);

    size_t test_total_pix = (size_t)n_test * npix;
    float *recon_out = malloc(test_total_pix * sizeof(float));
    float *diff_out  = malloc(test_total_pix * sizeof(float));

    if (!recon_out || !diff_out)
    {
        fprintf(stderr, "Error: malloc failed for reconstruction buffers\n");
        free(tr_in); free(tr_out); free(te_in); free(te_true);
        free(recon_out); free(diff_out);
        return 1;
    }

    double sum_rmse = 0.0;
    double sum_psnr = 0.0;
    double sum_corr = 0.0;
    double min_psnr = 1e9;
    double max_psnr = -1e9;

    printf("\n=== Running Reconstruction ===\n");

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        Neighbor *local_neighbors = malloc((size_t)k * sizeof(Neighbor));
        float    *weights         = malloc((size_t)k * sizeof(float));

#ifdef _OPENMP
        #pragma omp for reduction(+:sum_rmse, sum_psnr, sum_corr) \
                        reduction(min:min_psnr) reduction(max:max_psnr)
#endif
        for (int q = 0; q < n_test; q++)
        {
            const float *q_in   = te_in + (size_t)q * npix;
            const float *q_true = te_true + (size_t)q * npix;
            float       *q_rec  = recon_out + (size_t)q * npix;
            float       *q_diff = diff_out + (size_t)q * npix;

            /* 1. Find k nearest neighbors */
            find_top_k_neighbors(q_in, tr_in, n_train, npix, k, local_neighbors);

            /* 2. Calculate neighbor weights */
            float w_sum = 0.0f;
            for (int j = 0; j < k; j++)
            {
                if (use_idw)
                {
                    float d = local_neighbors[j].dist;
                    weights[j] = 1.0f / powf(d + 1e-5f, alpha);
                }
                else
                {
                    weights[j] = 1.0f / (float)k;
                }
                w_sum += weights[j];
            } // for j

            float inv_w_sum = 1.0f / w_sum;
            for (int j = 0; j < k; j++)
            {
                weights[j] *= inv_w_sum;
            } // for j

            /* 3. Reconstruct output frame */
            for (size_t p = 0; p < npix; p++)
            {
                float val = 0.0f;
                for (int j = 0; j < k; j++)
                {
                    int n_idx = local_neighbors[j].index;
                    val += weights[j] * tr_out[(size_t)n_idx * npix + p];
                } // for j

                q_rec[p]  = val;
                q_diff[p] = fabsf(val - q_true[p]);
            } // for p

            /* 4. Evaluate metrics */
            double mse = 0.0;
            double mean_true = 0.0, mean_rec = 0.0;
            for (size_t p = 0; p < npix; p++)
            {
                double d = (double)(q_rec[p] - q_true[p]);
                mse += d * d;
                mean_true += (double)q_true[p];
                mean_rec  += (double)q_rec[p];
            }
            mse /= (double)npix;
            mean_true /= (double)npix;
            mean_rec  /= (double)npix;

            double rmse = sqrt(mse);

            /* PSNR relative to peak value 1.0 */
            double psnr = (mse > 1e-12) ? (10.0 * log10(1.0 / mse)) : 100.0;

            /* Pearson correlation */
            double var_t = 0.0, var_r = 0.0, cov_tr = 0.0;
            for (size_t p = 0; p < npix; p++)
            {
                double dt = (double)q_true[p] - mean_true;
                double dr = (double)q_rec[p]  - mean_rec;
                var_t += dt * dt;
                var_r += dr * dr;
                cov_tr += dt * dr;
            }
            double corr = (var_t > 1e-12 && var_r > 1e-12)
                        ? (cov_tr / (sqrt(var_t) * sqrt(var_r)))
                        : 1.0;

            sum_rmse += rmse;
            sum_psnr += psnr;
            sum_corr += corr;

            if (psnr < min_psnr) min_psnr = psnr;
            if (psnr > max_psnr) max_psnr = psnr;
        } // for q

        free(local_neighbors);
        free(weights);
    } // OpenMP parallel

    double mean_rmse = sum_rmse / (double)n_test;
    double mean_psnr = sum_psnr / (double)n_test;
    double mean_corr = sum_corr / (double)n_test;

    printf("Reconstruction Results across %d test frames:\n", n_test);
    printf("  Mean RMSE:        %.5f\n", mean_rmse);
    printf("  Mean PSNR:        %.2f dB (min: %.2f dB, max: %.2f dB)\n",
           mean_psnr, min_psnr, max_psnr);
    printf("  Mean Correlation: %.4f\n", mean_corr);

    /* Render ASCII comparison if requested */
    if (show_frame >= 0 && show_frame < n_test)
    {
        const float *in_p   = te_in + (size_t)show_frame * npix;
        const float *true_p = te_true + (size_t)show_frame * npix;
        const float *rec_p  = recon_out + (size_t)show_frame * npix;
        render_ascii_comparison(in_p, true_p, rec_p, W, H, show_frame);
    }

    /* Save outputs */
    if (path_out)
    {
        if (write_fits_cube(path_out, recon_out, W, H, n_test) == 0)
        {
            printf("Saved reconstructed cube to: %s\n", path_out);
        }
    }

    if (path_diff)
    {
        if (write_fits_cube(path_diff, diff_out, W, H, n_test) == 0)
        {
            printf("Saved difference cube to:     %s\n", path_diff);
        }
    }

    free(tr_in);
    free(tr_out);
    free(te_in);
    free(te_true);
    free(recon_out);
    free(diff_out);

    return 0;
}
