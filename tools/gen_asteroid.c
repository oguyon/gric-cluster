/**
 * @file gen_asteroid.c
 * @brief Generate synchronized 3D FITS cubes of a rotating non-spherical asteroid.
 *
 * Simulates continuous 3D rotation in phi and theta of a non-spherical asteroid
 * (triaxial ellipsoid with surface elevation perturbations and modulated albedo
 * features) viewed simultaneously from three fixed orthogonal viewpoints:
 *   - View X: Looking along +X axis towards origin (Y-Z projection)
 *   - View Y: Looking along +Y axis towards origin (-X-Z projection)
 *   - View Z: Looking along +Z axis towards origin (X-Y projection)
 *
 * Outputs 3D FITS cubes (W x H x N) and ground-truth orientation trajectory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fitsio.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NUM_CRATERS 8

/**
 * struct Crater - Surface crater / elevation and albedo perturbation parameterization.
 * @cx:         Body X coordinate on unit sphere.
 * @cy:         Body Y coordinate on unit sphere.
 * @cz:         Body Z coordinate on unit sphere.
 * @sigma_sq:   Angular variance parameter squared.
 * @weight_dep: Central bowl depression weight (negative).
 * @weight_rim: Raised circular rim weight (positive).
 */
typedef struct
{
    float cx;
    float cy;
    float cz;
    float sigma_sq;
    float weight_dep;
    float weight_rim;
} Crater;

static const Crater CRATERS[NUM_CRATERS] = {
    {  0.707f,  0.707f,  0.000f, 0.040f, -0.28f, 0.18f },
    { -0.500f,  0.300f,  0.812f, 0.060f, -0.32f, 0.15f },
    {  0.200f, -0.800f,  0.566f, 0.035f, -0.25f, 0.18f },
    { -0.700f, -0.600f,  0.387f, 0.070f, -0.22f, 0.12f },
    {  0.000f,  0.500f, -0.866f, 0.050f, -0.28f, 0.14f },
    {  0.600f, -0.400f, -0.693f, 0.030f, -0.30f, 0.16f },
    { -0.800f,  0.200f, -0.566f, 0.038f, -0.25f, 0.16f },
    {  0.100f,  0.200f,  0.975f, 0.045f, -0.28f, 0.14f }
};

/**
 * make_directory() - Ensure directory path exists.
 * @path: Directory path string.
 *
 * Return: 0 on success, -1 on failure.
 */
static int make_directory(
    const char *path)
{
    struct stat st;

    if (stat(path, &st) == 0)
    {
        return S_ISDIR(st.st_mode) ? 0 : -1;
    }

    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s", path);
    size_t len = strlen(tmp);
    if (len == 0)
    {
        return 0;
    }

    for (size_t i = 1; i < len; i++)
    {
        if (tmp[i] == '/')
        {
            tmp[i] = '\0';
            if (stat(tmp, &st) != 0)
            {
                mkdir(tmp, 0755);
            }
            tmp[i] = '/';
        }
    } // for i

    return mkdir(tmp, 0755);
}

/**
 * build_rotation_matrix() - Construct 3x3 rotation matrix R = Rz(phi) * Ry(theta).
 * @R:     Output 9-element row-major matrix.
 * @phi:   Azimuthal spin angle in radians.
 * @theta: Obliquity / tilt angle in radians.
 */
static void build_rotation_matrix(
    float *restrict R,
    float           phi,
    float           theta)
{
    float cp = cosf(phi);
    float sp = sinf(phi);
    float ct = cosf(theta);
    float st = sinf(theta);

    /* R = Rz(phi) * Ry(theta) */
    R[0] =  cp * ct;  R[1] = -sp;  R[2] =  cp * st;
    R[3] =  sp * ct;  R[4] =  cp;  R[5] =  sp * st;
    R[6] = -st;       R[7] = 0.0f; R[8] =  ct;
}

/**
 * compute_bump_height() - Evaluate radial bumpy surface perturbation h(u).
 * @ux: Unit direction X in body frame.
 * @uy: Unit direction Y in body frame.
 * @uz: Unit direction Z in body frame.
 *
 * Combines multi-scale harmonic lobes, ridges, and impact craters with central
 * depressions and elevated rim walls.
 *
 * Return: Surface height perturbation h(u).
 */
static inline float compute_bump_height(
    float ux,
    float uy,
    float uz)
{
    float uz2 = uz * uz;
    float rxy2 = fmaxf(0.0f, 1.0f - uz2);
    if (rxy2 < 1e-8f)
    {
        return 0.0f;
    }
    float rxy = sqrtf(rxy2);
    float inv_rxy = 1.0f / rxy;

    float c1 = ux * inv_rxy;
    float s1 = uy * inv_rxy;

    float c2 = c1 * c1 - s1 * s1;
    float s2 = 2.0f * c1 * s1;

    float c3 = c2 * c1 - s2 * s1;
    float s3 = s2 * c1 + c2 * s1;

    float c5 = c3 * c2 - s3 * s2;
    float s5 = s3 * c2 + c3 * s2;

    float c7 = c5 * c2 - s5 * s2;
    float s7 = s5 * c2 + c5 * s2;

    float sin2th = 2.0f * rxy * uz;
    float cos3th = uz * (4.0f * uz2 - 3.0f);
    float sin5th = rxy * (16.0f * uz2 * uz2 - 12.0f * uz2 + 1.0f);

    /* Harmonic 1: 0.16 * sin(2*phi) * cos(theta) */
    float h1 = 0.16f * s2 * uz;

    /* Harmonic 2: 0.12 * cos(3*phi + 0.5) * sin(2*theta) */
    static const float cos_05 = 0.87758256f;
    static const float sin_05 = 0.47942554f;
    float cos3phi05 = c3 * cos_05 - s3 * sin_05;
    float h2 = 0.12f * cos3phi05 * sin2th;

    /* Harmonic 3: 0.08 * sin(5*phi) * cos(3*theta) */
    float h3 = 0.08f * s5 * cos3th;

    /* Harmonic 4: 0.05 * cos(7*phi - 1.0) * sin(5*theta) */
    static const float cos_10 = 0.54030230f;
    static const float sin_10 = 0.84147098f;
    float cos7phiM1 = c7 * cos_10 + s7 * sin_10;
    float h4 = 0.05f * cos7phiM1 * sin5th;

    float h = h1 + h2 + h3 + h4;

    /* Discrete impact craters with depressed bowl and raised rim */
    for (int k = 0; k < NUM_CRATERS; k++)
    {
        float dot = ux * CRATERS[k].cx + uy * CRATERS[k].cy + uz * CRATERS[k].cz;
        if (dot < 0.65f)
        {
            continue;
        }
        float ang_sq = 1.0f - dot;
        float r_sq = ang_sq / CRATERS[k].sigma_sq;
        h += CRATERS[k].weight_dep * expf(-r_sq * 2.5f)
           + CRATERS[k].weight_rim * r_sq * expf(-r_sq);
    } // for k

    return h;
}

/**
 * eval_surface_f() - Evaluate implicit bumpy surface equation F(p).
 * @p:       3D position vector in body coordinates.
 * @inv_a2:  1.0 / (a * a).
 * @inv_b2:  1.0 / (b * b).
 * @inv_c2:  1.0 / (c * c).
 * @out_h:   Optional output pointer to store local bump height.
 *
 * Defined by F(p) = (x/a)^2 + (y/b)^2 + (z/c)^2 - (1 + h(u))^2.
 *
 * Return: Evaluated scalar F(p).
 */
static inline float eval_surface_f(
    const float *p,
    float        inv_a2,
    float        inv_b2,
    float        inv_c2,
    float       *out_h)
{
    float plen2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    if (plen2 < 1e-6f)
    {
        if (out_h != NULL)
        {
            *out_h = 0.0f;
        }
        return -1.0f;
    }

    float inv_len = 1.0f / sqrtf(plen2);
    float h = compute_bump_height(p[0] * inv_len, p[1] * inv_len, p[2] * inv_len);
    if (out_h != NULL)
    {
        *out_h = h;
    }

    float one_plus_h = 1.0f + h;
    return (p[0] * p[0] * inv_a2 + p[1] * p[1] * inv_b2 + p[2] * p[2] * inv_c2)
         - (one_plus_h * one_plus_h);
}

/**
 * trace_ray() - Intersect a ray with the rotating bumpy asteroid.
 * @ow:           Ray origin in world coordinates (3D vector).
 * @db:           Ray direction in body coordinates (3D vector).
 * @A:            Quadratic coefficient for ray-ellipsoid test.
 * @inv_2A:       0.5 / A.
 * @four_A:       4.0 * A.
 * @R:            3x3 row-major rotation matrix body-to-world.
 * @inv_a2:       1.0 / (a * a).
 * @inv_b2:       1.0 / (b * b).
 * @inv_c2:       1.0 / (c * c).
 * @inv_a2_out:   1.0 / (a_out * a_out) for outer bounding ellipsoid.
 * @inv_b2_out:   1.0 / (b_out * b_out) for outer bounding ellipsoid.
 * @inv_c2_out:   1.0 / (c_out * c_out) for outer bounding ellipsoid.
 * @sun_dir:      Normalized sun direction vector in world coordinates.
 * @ambient:      Ambient illumination fraction.
 * @albedo_noise: Albedo variation strength.
 *
 * Return: Surface radiance at hit point, or 0.0f on miss.
 */
static float trace_ray(
    const float *restrict ow,
    const float *restrict db,
    float                 inv_2A,
    float                 four_A,
    const float *restrict R,
    float                 inv_a2,
    float                 inv_b2,
    float                 inv_c2,
    float                 inv_a2_out,
    float                 inv_b2_out,
    float                 inv_c2_out,
    const float *restrict sun_dir,
    float                 ambient,
    float                 albedo_noise)
{
    /* Transform ray origin to body coordinates via R^T: ob = R^T * ow */
    float obx = R[0] * ow[0] + R[3] * ow[1] + R[6] * ow[2];
    float oby = R[1] * ow[0] + R[4] * ow[1] + R[7] * ow[2];
    float obz = R[2] * ow[0] + R[5] * ow[1] + R[8] * ow[2];

    float dbx = db[0];
    float dby = db[1];
    float dbz = db[2];

    /* Fast analytical intersection with outer bounding ellipsoid */
    float B = 2.0f * (obx * dbx * inv_a2_out + oby * dby * inv_b2_out + obz * dbz * inv_c2_out);
    float C = obx * obx * inv_a2_out + oby * oby * inv_b2_out + obz * obz * inv_c2_out - 1.0f;

    float discr = B * B - four_A * C;
    if (discr < 0.0f)
    {
        return 0.0f;
    }

    float sqrt_d = sqrtf(discr);
    float s_near = (-B - sqrt_d) * inv_2A;
    float s_far  = (-B + sqrt_d) * inv_2A;
    if (s_far <= 0.0f)
    {
        return 0.0f;
    }
    if (s_near < 0.0f)
    {
        s_near = 0.0f;
    }

    /* Multi-step bracketed linear probe + bisection for exact bumpy surface hit */
    const int N_PROBE = 4;
    float ds = (s_far - s_near) * 0.25f;
    float s_prev = s_near;
    float p_test[3] = { obx + s_prev * dbx, oby + s_prev * dby, obz + s_prev * dbz };
    float f_prev = eval_surface_f(p_test, inv_a2, inv_b2, inv_c2, NULL);
    float s_hit = -1.0f;

    if (f_prev <= 0.0f)
    {
        s_hit = s_near;
    }
    else
    {
        for (int step_i = 1; step_i <= N_PROBE; step_i++)
        {
            float s_curr = s_near + (float)step_i * ds;
            float p_c[3] = { obx + s_curr * dbx, oby + s_curr * dby, obz + s_curr * dbz };
            float f_curr = eval_surface_f(p_c, inv_a2, inv_b2, inv_c2, NULL);
            if (f_curr <= 0.0f)
            {
                float s_lo = s_prev;
                float s_hi = s_curr;
                for (int bi = 0; bi < 3; bi++)
                {
                    float s_mid = 0.5f * (s_lo + s_hi);
                    float p_m[3] = { obx + s_mid * dbx, oby + s_mid * dby, obz + s_mid * dbz };
                    float f_mid = eval_surface_f(p_m, inv_a2, inv_b2, inv_c2, NULL);
                    if (f_mid <= 0.0f)
                    {
                        s_hi = s_mid;
                    }
                    else
                    {
                        s_lo = s_mid;
                    }
                } // for bi
                s_hit = 0.5f * (s_lo + s_hi);
                break;
            }
            s_prev = s_curr;
        } // for step_i
    }

    if (s_hit <= 0.0f)
    {
        return 0.0f;
    }

    /* Hit point in body coordinates */
    float p_hit[3] = { obx + s_hit * dbx, oby + s_hit * dby, obz + s_hit * dbz };
    float h = 0.0f;
    float f0 = eval_surface_f(p_hit, inv_a2, inv_b2, inv_c2, &h);

    /* Surface normal in body coordinates via numerical gradient of F(p) */
    const float eps = 0.02f;
    float p_x[3] = { p_hit[0] + eps, p_hit[1], p_hit[2] };
    float p_y[3] = { p_hit[0], p_hit[1] + eps, p_hit[2] };
    float p_z[3] = { p_hit[0], p_hit[1], p_hit[2] + eps };
    float fx = eval_surface_f(p_x, inv_a2, inv_b2, inv_c2, NULL);
    float fy = eval_surface_f(p_y, inv_a2, inv_b2, inv_c2, NULL);
    float fz = eval_surface_f(p_z, inv_a2, inv_b2, inv_c2, NULL);

    float nbx = fx - f0;
    float nby = fy - f0;
    float nbz = fz - f0;
    float n_len = 1.0f / sqrtf(nbx * nbx + nby * nby + nbz * nbz + 1e-12f);
    nbx *= n_len;
    nby *= n_len;
    nbz *= n_len;

    /* Transform normal to world coordinates: nw = R * nb */
    float nwx = R[0] * nbx + R[1] * nby + R[2] * nbz;
    float nwy = R[3] * nbx + R[4] * nby + R[5] * nbz;
    float nwz = R[6] * nbx + R[7] * nby + R[8] * nbz;

    /* Lambertian diffuse shading */
    float n_dot_l = nwx * sun_dir[0] + nwy * sun_dir[1] + nwz * sun_dir[2];
    float diffuse = fmaxf(0.0f, n_dot_l);

    /* Combined albedo modulated with crater rims and weathered floors */
    float albedo = fmaxf(0.12f, fminf(1.0f, 0.60f + albedo_noise * h));

    return albedo * (ambient + (1.0f - ambient) * diffuse);
}

/**
 * render_view() - Render one viewpoint frame with 2x2 subpixel anti-aliasing.
 * @buf:          Output pixel buffer of size W * H.
 * @view_id:      0 for View X, 1 for View Y, 2 for View Z.
 * @W:            Image width.
 * @H:            Image height.
 * @R:            3x3 row-major body-to-world rotation matrix.
 * @inv_a2:       1.0 / (a * a).
 * @inv_b2:       1.0 / (b * b).
 * @inv_c2:       1.0 / (c * c).
 * @inv_a2_out:   1.0 / (a_out * a_out) for outer bounding ellipsoid.
 * @inv_b2_out:   1.0 / (b_out * b_out) for outer bounding ellipsoid.
 * @inv_c2_out:   1.0 / (c_out * c_out) for outer bounding ellipsoid.
 * @sun_dir:      Normalized sun direction vector.
 * @ambient:      Ambient illumination factor.
 * @albedo_noise: Surface albedo modulation amplitude.
 */
static void render_view(
    float       *restrict buf,
    int                   view_id,
    int                   W,
    int                   H,
    const float *restrict R,
    float                 inv_a2,
    float                 inv_b2,
    float                 inv_c2,
    float                 inv_a2_out,
    float                 inv_b2_out,
    float                 inv_c2_out,
    const float *restrict sun_dir,
    float                 ambient,
    float                 albedo_noise)
{
    float cx = (float)(W - 1) * 0.5f;
    float cy = (float)(H - 1) * 0.5f;
    const float sub_offsets[2] = { -0.25f, 0.25f };

    /* Direction vector in body frame is constant across all pixels for this frame */
    float db[3];
    if (view_id == 0)
    {
        db[0] = -R[0]; db[1] = -R[1]; db[2] = -R[2];
    }
    else if (view_id == 1)
    {
        db[0] = -R[3]; db[1] = -R[4]; db[2] = -R[5];
    }
    else
    {
        db[0] = -R[6]; db[1] = -R[7]; db[2] = -R[8];
    }

    float A = db[0] * db[0] * inv_a2_out +
              db[1] * db[1] * inv_b2_out +
              db[2] * db[2] * inv_c2_out;
    float inv_2A = 0.5f / A;
    float four_A = 4.0f * A;

    /* Screen-space bounding circle rejection */
    float min_inv_out = fminf(inv_a2_out, fminf(inv_b2_out, inv_c2_out));
    float max_r = (1.0f / sqrtf(min_inv_out)) + 0.6f;
    float max_r2 = max_r * max_r;

    for (int y = 0; y < H; y++)
    {
        float dy_center = (float)y - cy;
        float dy2 = dy_center * dy_center;

        for (int x = 0; x < W; x++)
        {
            float dx_center = (float)x - cx;
            if (dx_center * dx_center + dy2 > max_r2)
            {
                buf[y * W + x] = 0.0f;
                continue;
            }

            float accum = 0.0f;

            /* 2x2 subpixel supersampling */
            for (int sy = 0; sy < 2; sy++)
            {
                float py = dy_center + sub_offsets[sy];

                for (int sx = 0; sx < 2; sx++)
                {
                    float px = dx_center + sub_offsets[sx];
                    float ow[3];

                    if (view_id == 0)
                    {
                        ow[0] =  100.0f; ow[1] =  px;    ow[2] =  py;
                    }
                    else if (view_id == 1)
                    {
                        ow[0] = -px;     ow[1] =  100.0f; ow[2] =  py;
                    }
                    else
                    {
                        ow[0] =  px;     ow[1] =  py;     ow[2] =  100.0f;
                    }

                    accum += trace_ray(ow, db, inv_2A, four_A, R,
                                       inv_a2, inv_b2, inv_c2,
                                       inv_a2_out, inv_b2_out, inv_c2_out,
                                       sun_dir, ambient, albedo_noise);
                } // for sx
            } // for sy

            buf[y * W + x] = accum * 0.25f;
        } // for x
    } // for y
}

/**
 * create_fits_cube() - Create and initialize a 3D FITS image cube.
 * @fptr:    Output FITS file pointer.
 * @path:    File path to create (overwriting if exists with '!').
 * @W:       Image width.
 * @H:       Image height.
 * @nframes: Number of frames along depth axis.
 *
 * Return: 0 on success, or non-zero CFITSIO error code.
 */
static int create_fits_cube(
    fitsfile   **fptr,
    const char  *path,
    int          W,
    int          H,
    int          nframes)
{
    int status = 0;
    char bangpath[1024];
    snprintf(bangpath, sizeof(bangpath), "!%s", path);

    fits_create_file(fptr, bangpath, &status);
    if (status)
    {
        fprintf(stderr, "Error: cannot create FITS file %s (status %d)\n", path, status);
        return status;
    }

    long naxes[3] = { (long)W, (long)H, (long)nframes };
    fits_create_img(*fptr, FLOAT_IMG, 3, naxes, &status);
    if (status)
    {
        fprintf(stderr, "Error: cannot create image cube in %s (status %d)\n", path, status);
        fits_close_file(*fptr, &status);
        *fptr = NULL;
        return status;
    }

    return 0;
}

static int write_txt_cube(
    const char  *path,
    const float *data,
    int          nframes,
    int          npix)
{
    FILE *fp = fopen(path, "w");
    if (!fp)
    {
        fprintf(stderr, "Error: cannot open %s for writing\n", path);
        return -1;
    }

    char *buf = malloc(1024 * 1024);
    if (buf)
    {
        setvbuf(fp, buf, _IOFBF, 1024 * 1024);
    }

    for (int fi = 0; fi < nframes; fi++)
    {
        const float *frame = data + (size_t)fi * npix;
        for (int p = 0; p < npix; p++)
        {
            fprintf(fp, (p == npix - 1) ? "%.4f\n" : "%.4f ", frame[p]);
        }
    } // for fi

    fclose(fp);
    free(buf);
    return 0;
}

static void print_usage(
    const char *prog)
{
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -W <int>            Frame width in pixels      (default: 32)\n");
    printf("  -H <int>            Frame height in pixels     (default: 32)\n");
    printf("  -f <int>            Total number of frames     (default: 10000)\n");
    printf("  -split <float>      Train split fraction [0..1](default: 0.80)\n");
    printf("  -omega-phi <float>  Spin speed in deg/frame    (default: 1.0)\n");
    printf("  -omega-theta <float>Tilt speed in deg/frame    (default: 0.236)\n");
    printf("  -a <float>          Ellipsoid semi-axis X      (default: 14.5)\n");
    printf("  -b <float>          Ellipsoid semi-axis Y      (default: 10.0)\n");
    printf("  -c <float>          Ellipsoid semi-axis Z      (default: 6.5)\n");
    printf("  -ambient <float>    Ambient light fraction     (default: 0.25)\n");
    printf("  -albedo <float>     Albedo feature strength    (default: 0.40)\n");
    printf("  -txt                Write ASCII .txt datasets  (default: FITS only)\n");
    printf("  -o <dir>            Output directory           (default: .)\n");
    printf("  -prefix <str>       Dataset filename prefix    (default: asteroid)\n");
    printf("  -h, --help          Show this help message\n");
}

int main(
    int   argc,
    char *argv[])
{
    int   W           = 32;
    int   H           = 32;
    int   nframes     = 10000;
    int   write_txt   = 0;
    float split       = 0.80f;
    float omega_phi   = 1.0f;
    float omega_theta = 0.236f;
    float axis_a      = 14.0f;
    float axis_b      = 9.8f;
    float axis_c      = 6.4f;
    float ambient     = 0.25f;
    float albedo_str  = 0.40f;
    const char *outdir = ".";
    const char *prefix = "asteroid";

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-W") == 0 && i + 1 < argc)
        {
            W = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-H") == 0 && i + 1 < argc)
        {
            H = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc)
        {
            nframes = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-split") == 0 && i + 1 < argc)
        {
            split = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-omega-phi") == 0 && i + 1 < argc)
        {
            omega_phi = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-omega-theta") == 0 && i + 1 < argc)
        {
            omega_theta = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-a") == 0 && i + 1 < argc)
        {
            axis_a = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc)
        {
            axis_b = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc)
        {
            axis_c = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-ambient") == 0 && i + 1 < argc)
        {
            ambient = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-albedo") == 0 && i + 1 < argc)
        {
            albedo_str = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-txt") == 0 || strcmp(argv[i], "--txt") == 0)
        {
            write_txt = 1;
        }
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc)
        {
            outdir = argv[++i];
        }
        else if (strcmp(argv[i], "-prefix") == 0 && i + 1 < argc)
        {
            prefix = argv[++i];
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

    if (W <= 0 || H <= 0 || nframes <= 0)
    {
        fprintf(stderr, "Error: invalid dimensions or frame count\n");
        return 1;
    }

    if (make_directory(outdir) != 0)
    {
        fprintf(stderr, "Error: cannot create output directory %s\n", outdir);
        return 1;
    }

    int n_train = (int)floorf((float)nframes * split);
    int n_test  = nframes - n_train;
    int do_split = (split > 0.0f && split < 1.0f);

    printf("=== Generating Asteroid Datasets (OpenMP Accelerated) ===\n");
    printf("Resolution: %dx%d, Total frames: %d\n", W, H, nframes);
    printf("Rotation rates: omega_phi=%.3f deg/f, omega_theta=%.3f deg/f\n",
           omega_phi, omega_theta);
    printf("Ellipsoid semi-axes: a=%.1f, b=%.1f, c=%.1f\n", axis_a, axis_b, axis_c);
    if (do_split)
    {
        printf("Train frames: %d (%.1f%%), Test frames: %d (%.1f%%)\n",
               n_train, split * 100.0f, n_test, (1.0f - split) * 100.0f);
    }
    printf("Output directory: %s (prefix: %s)\n", outdir, prefix);

    float inv_a2 = 1.0f / (axis_a * axis_a);
    float inv_b2 = 1.0f / (axis_b * axis_b);
    float inv_c2 = 1.0f / (axis_c * axis_c);

    float scale_out  = 1.35f;
    float inv_a2_out = 1.0f / ((axis_a * scale_out) * (axis_a * scale_out));
    float inv_b2_out = 1.0f / ((axis_b * scale_out) * (axis_b * scale_out));
    float inv_c2_out = 1.0f / ((axis_c * scale_out) * (axis_c * scale_out));

    /* Normalized solar vector pointing from (+1, +1, +1) towards origin */
    float sun_dir[3] = { 0.57735f, 0.57735f, 0.57735f };

    /* Allocate full sequence buffers for multi-threaded parallel rendering */
    size_t npix = (size_t)(W * H);
    size_t total_pix = (size_t)nframes * npix;
    float *all_x = malloc(total_pix * sizeof(float));
    float *all_y = malloc(total_pix * sizeof(float));
    float *all_z = malloc(total_pix * sizeof(float));

    if (!all_x || !all_y || !all_z)
    {
        fprintf(stderr, "Error: memory allocation failed for full sequence buffers\n");
        free(all_x);
        free(all_y);
        free(all_z);
        return 1;
    }

    float rad_conv = (float)(M_PI / 180.0);

    #pragma omp parallel for schedule(dynamic, 64)
    for (int fi = 0; fi < nframes; fi++)
    {
        float phi_deg   = fmodf(omega_phi * (float)fi, 360.0f);
        float theta_deg = fmodf(omega_theta * (float)fi, 360.0f);
        float phi_rad   = phi_deg * rad_conv;
        float theta_rad = theta_deg * rad_conv;

        float R[9];
        build_rotation_matrix(R, phi_rad, theta_rad);

        float *dest_x = all_x + (size_t)fi * npix;
        float *dest_y = all_y + (size_t)fi * npix;
        float *dest_z = all_z + (size_t)fi * npix;

        render_view(dest_x, 0, W, H, R, inv_a2, inv_b2, inv_c2,
                    inv_a2_out, inv_b2_out, inv_c2_out,
                    sun_dir, ambient, albedo_str);
        render_view(dest_y, 1, W, H, R, inv_a2, inv_b2, inv_c2,
                    inv_a2_out, inv_b2_out, inv_c2_out,
                    sun_dir, ambient, albedo_str);
        render_view(dest_z, 2, W, H, R, inv_a2, inv_b2, inv_c2,
                    inv_a2_out, inv_b2_out, inv_c2_out,
                    sun_dir, ambient, albedo_str);
    } // for fi

    /* Open trajectory metadata log */
    char traj_path[1024];
    snprintf(traj_path, sizeof(traj_path), "%s/%s_trajectory.txt", outdir, prefix);
    FILE *fp_traj = fopen(traj_path, "w");
    if (fp_traj)
    {
        fprintf(fp_traj, "# frame phi_deg theta_deg phi_rad theta_rad split\n");
        for (int fi = 0; fi < nframes; fi++)
        {
            float phi_deg   = fmodf(omega_phi * (float)fi, 360.0f);
            float theta_deg = fmodf(omega_theta * (float)fi, 360.0f);
            float phi_rad   = phi_deg * rad_conv;
            float theta_rad = theta_deg * rad_conv;
            const char *split_tag = (fi < n_train) ? "train" : "test";
            fprintf(fp_traj, "%d %.4f %.4f %.6f %.6f %s\n",
                    fi, phi_deg, theta_deg, phi_rad, theta_rad, split_tag);
        }
        fclose(fp_traj);
    }

    /* Create and write FITS files in fast single-block operations */
    fitsfile *fp_full_x = NULL, *fp_full_y = NULL, *fp_full_z = NULL;
    fitsfile *fp_trn_x  = NULL, *fp_trn_y  = NULL, *fp_trn_z  = NULL;
    fitsfile *fp_tst_x  = NULL, *fp_tst_y  = NULL, *fp_tst_z  = NULL;

    char path_buf[1024];
    int status = 0;

    /* Full sequence FITS */
    snprintf(path_buf, sizeof(path_buf), "%s/%s_viewX.fits", outdir, prefix);
    create_fits_cube(&fp_full_x, path_buf, W, H, nframes);
    snprintf(path_buf, sizeof(path_buf), "%s/%s_viewY.fits", outdir, prefix);
    create_fits_cube(&fp_full_y, path_buf, W, H, nframes);
    snprintf(path_buf, sizeof(path_buf), "%s/%s_viewZ.fits", outdir, prefix);
    create_fits_cube(&fp_full_z, path_buf, W, H, nframes);

    long fpixel[3] = { 1, 1, 1 };
    if (fp_full_x) fits_write_pix(fp_full_x, TFLOAT, fpixel, (long)total_pix, all_x, &status);
    if (fp_full_y) fits_write_pix(fp_full_y, TFLOAT, fpixel, (long)total_pix, all_y, &status);
    if (fp_full_z) fits_write_pix(fp_full_z, TFLOAT, fpixel, (long)total_pix, all_z, &status);

    if (fp_full_x) fits_close_file(fp_full_x, &status);
    if (fp_full_y) fits_close_file(fp_full_y, &status);
    if (fp_full_z) fits_close_file(fp_full_z, &status);

    if (do_split)
    {
        long n_trn_pix = (long)n_train * (long)npix;
        long n_tst_pix = (long)n_test * (long)npix;
        const float *tst_x = all_x + (size_t)n_train * npix;
        const float *tst_y = all_y + (size_t)n_train * npix;
        const float *tst_z = all_z + (size_t)n_train * npix;

        /* Train split FITS */
        snprintf(path_buf, sizeof(path_buf), "%s/%s_train_X.fits", outdir, prefix);
        create_fits_cube(&fp_trn_x, path_buf, W, H, n_train);
        snprintf(path_buf, sizeof(path_buf), "%s/%s_train_Y.fits", outdir, prefix);
        create_fits_cube(&fp_trn_y, path_buf, W, H, n_train);
        snprintf(path_buf, sizeof(path_buf), "%s/%s_train_Z.fits", outdir, prefix);
        create_fits_cube(&fp_trn_z, path_buf, W, H, n_train);

        if (fp_trn_x) fits_write_pix(fp_trn_x, TFLOAT, fpixel, n_trn_pix, all_x, &status);
        if (fp_trn_y) fits_write_pix(fp_trn_y, TFLOAT, fpixel, n_trn_pix, all_y, &status);
        if (fp_trn_z) fits_write_pix(fp_trn_z, TFLOAT, fpixel, n_trn_pix, all_z, &status);

        if (fp_trn_x) fits_close_file(fp_trn_x, &status);
        if (fp_trn_y) fits_close_file(fp_trn_y, &status);
        if (fp_trn_z) fits_close_file(fp_trn_z, &status);

        /* Test split FITS */
        snprintf(path_buf, sizeof(path_buf), "%s/%s_test_X.fits", outdir, prefix);
        create_fits_cube(&fp_tst_x, path_buf, W, H, n_test);
        snprintf(path_buf, sizeof(path_buf), "%s/%s_test_Y.fits", outdir, prefix);
        create_fits_cube(&fp_tst_y, path_buf, W, H, n_test);
        snprintf(path_buf, sizeof(path_buf), "%s/%s_test_Z.fits", outdir, prefix);
        create_fits_cube(&fp_tst_z, path_buf, W, H, n_test);

        if (fp_tst_x)
        {
            fits_write_pix(fp_tst_x, TFLOAT, fpixel, n_tst_pix, (float *)tst_x, &status);
        }
        if (fp_tst_y)
        {
            fits_write_pix(fp_tst_y, TFLOAT, fpixel, n_tst_pix, (float *)tst_y, &status);
        }
        if (fp_tst_z)
        {
            fits_write_pix(fp_tst_z, TFLOAT, fpixel, n_tst_pix, (float *)tst_z, &status);
        }

        if (fp_tst_x) fits_close_file(fp_tst_x, &status);
        if (fp_tst_y) fits_close_file(fp_tst_y, &status);
        if (fp_tst_z) fits_close_file(fp_tst_z, &status);
    }

    if (write_txt)
    {
        printf("Writing ASCII .txt datasets...\n");
        char txt_path[1024];
        snprintf(txt_path, sizeof(txt_path), "%s/%s_viewX.txt", outdir, prefix);
        write_txt_cube(txt_path, all_x, nframes, (int)npix);
        snprintf(txt_path, sizeof(txt_path), "%s/%s_viewY.txt", outdir, prefix);
        write_txt_cube(txt_path, all_y, nframes, (int)npix);
        snprintf(txt_path, sizeof(txt_path), "%s/%s_viewZ.txt", outdir, prefix);
        write_txt_cube(txt_path, all_z, nframes, (int)npix);

        if (do_split)
        {
            snprintf(txt_path, sizeof(txt_path), "%s/%s_train_X.txt", outdir, prefix);
            write_txt_cube(txt_path, all_x, n_train, (int)npix);
            snprintf(txt_path, sizeof(txt_path), "%s/%s_train_Y.txt", outdir, prefix);
            write_txt_cube(txt_path, all_y, n_train, (int)npix);
            snprintf(txt_path, sizeof(txt_path), "%s/%s_train_Z.txt", outdir, prefix);
            write_txt_cube(txt_path, all_z, n_train, (int)npix);

            const float *tst_x = all_x + (size_t)n_train * npix;
            const float *tst_y = all_y + (size_t)n_train * npix;
            const float *tst_z = all_z + (size_t)n_train * npix;

            snprintf(txt_path, sizeof(txt_path), "%s/%s_test_X.txt", outdir, prefix);
            write_txt_cube(txt_path, tst_x, n_test, (int)npix);
            snprintf(txt_path, sizeof(txt_path), "%s/%s_test_Y.txt", outdir, prefix);
            write_txt_cube(txt_path, tst_y, n_test, (int)npix);
            snprintf(txt_path, sizeof(txt_path), "%s/%s_test_Z.txt", outdir, prefix);
            write_txt_cube(txt_path, tst_z, n_test, (int)npix);
        }
    }

    free(all_x);
    free(all_y);
    free(all_z);

    printf("Done! Datasets written to %s/\n", outdir);
    return 0;
}
