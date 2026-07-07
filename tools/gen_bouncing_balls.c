/**
 * @file gen_bouncing_balls.c
 * @brief Generate a 3D FITS cube of bouncing balls
 *        for GRIC clustering benchmarks.
 *
 * Simulates N circular disks bouncing elastically
 * off the edges of a 2D box.  Balls do not interact
 * with each other; overlapping pixel values add.
 *
 * Usage:
 *   gen_bouncing_balls [options] output.fits
 *
 * Options:
 *   -n <int>     Number of balls   (default: 1)
 *   -r <float>   Ball radius       (default: 5.0)
 *   -W <int>     Image width       (default: 32)
 *   -H <int>     Image height      (default: 32)
 *   -f <int>     Number of frames  (default: 10000)
 *   -s <int>     Random seed       (default: 42)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fitsio.h>

typedef struct
{
    double x, y;    /* centre position */
    double vx, vy;  /* velocity (pixels/frame) */
    double radius;
} Ball;

/**
 * init_balls() - Place N balls at random positions
 *     with random velocities.
 * @balls:   Output array of N balls.
 * @nballs:  Number of balls.
 * @radius:  Disk radius in pixels.
 * @W:       Image width.
 * @H:       Image height.
 */
static void init_balls(
    Ball   *balls,
    int     nballs,
    double  radius,
    int     W,
    int     H)
{
    for (int i = 0; i < nballs; i++)
    {
        int attempts = 0;
        int overlap = 0;
        do
        {
            overlap = 0;
            /* Random position inside bounds */
            balls[i].x = radius
                + (W - 2.0 * radius)
                  * ((double) rand() / RAND_MAX);
            balls[i].y = radius
                + (H - 2.0 * radius)
                  * ((double) rand() / RAND_MAX);
            balls[i].radius = radius;

            /* Check against all previously placed balls */
            for (int j = 0; j < i; j++)
            {
                double dx = balls[i].x - balls[j].x;
                double dy = balls[i].y - balls[j].y;
                double dist = sqrt(dx * dx + dy * dy);
                if (dist < balls[i].radius + balls[j].radius)
                {
                    overlap = 1;
                    break;
                }
            }
            attempts++;
        } while (overlap && attempts < 1000);

        if (overlap)
        {
            fprintf(stderr,
                    "WARNING: Could not place ball %d "
                    "without overlap after 1000 attempts\n",
                    i);
        }

        /* Random velocity: 0.5 to 2.0 px/frame */
        double speed = 0.5
            + 1.5 * ((double) rand() / RAND_MAX);
        double angle = 2.0 * M_PI
            * ((double) rand() / RAND_MAX);

        balls[i].vx = speed * cos(angle);
        balls[i].vy = speed * sin(angle);
    }
}

/**
 * step_balls() - Advance all balls by one time step
 *     and bounce off edges.
 * @balls:   Array of balls.
 * @nballs:  Number of balls.
 * @W:       Image width.
 * @H:       Image height.
 */
static void step_balls(
    Ball *balls,
    int   nballs,
    int   W,
    int   H,
    int   ball_collisions)
{
    /* 1. Move all balls by velocity */
    for (int i = 0; i < nballs; i++)
    {
        balls[i].x += balls[i].vx;
        balls[i].y += balls[i].vy;
    }

    /* 2. Resolve ball-to-ball collisions if enabled */
    if (ball_collisions && nballs > 1)
    {
        for (int pass = 0; pass < 2; pass++)
        {
            for (int i = 0; i < nballs; i++)
            {
                for (int j = i + 1; j < nballs; j++)
                {
                    double dx = balls[j].x - balls[i].x;
                    double dy = balls[j].y - balls[i].y;
                    double dist = sqrt(dx * dx + dy * dy);
                    double min_dist = balls[i].radius + balls[j].radius;

                    if (dist < min_dist)
                    {
                        if (dist == 0.0)
                        {
                            dx = 0.1;
                            dy = 0.0;
                            dist = 0.1;
                        }

                        double nx = dx / dist;
                        double ny = dy / dist;

                        /* Push apart to resolve overlap */
                        double overlap = min_dist - dist;
                        balls[i].x -= 0.5 * overlap * nx;
                        balls[i].y -= 0.5 * overlap * ny;
                        balls[j].x += 0.5 * overlap * nx;
                        balls[j].y += 0.5 * overlap * ny;

                        /* Relative velocity along normal */
                        double rvx = balls[j].vx - balls[i].vx;
                        double rvy = balls[j].vy - balls[i].vy;
                        double vel_n = rvx * nx + rvy * ny;

                        if (vel_n < 0.0)
                        {
                            /* Equal mass elastic collision */
                            balls[i].vx += vel_n * nx;
                            balls[i].vy += vel_n * ny;
                            balls[j].vx -= vel_n * nx;
                            balls[j].vy -= vel_n * ny;
                        }
                    }
                }
            }
        }
    }

    /* 3. Bounce off walls and clamp to boundary */
    for (int i = 0; i < nballs; i++)
    {
        double r = balls[i].radius;

        if (balls[i].x < r)
        {
            balls[i].x = r;
            balls[i].vx = -balls[i].vx;
        }
        else if (balls[i].x > W - r)
        {
            balls[i].x = W - r;
            balls[i].vx = -balls[i].vx;
        }

        if (balls[i].y < r)
        {
            balls[i].y = r;
            balls[i].vy = -balls[i].vy;
        }
        else if (balls[i].y > H - r)
        {
            balls[i].y = H - r;
            balls[i].vy = -balls[i].vy;
        }
    }
}

/**
 * render_frame() - Render all balls into a pixel
 *     buffer.  Overlapping disks add their values.
 * @buf:     Output pixel buffer [H × W], zeroed.
 * @balls:   Array of balls.
 * @nballs:  Number of balls.
 * @W:       Image width.
 * @H:       Image height.
 */
static void render_frame(
    float  *buf,
    Ball   *balls,
    int     nballs,
    int     W,
    int     H)
{
    memset(buf, 0, (size_t)(W * H) * sizeof(float));

    for (int b = 0; b < nballs; b++)
    {
        double cx = balls[b].x;
        double cy = balls[b].y;
        double r  = balls[b].radius;
        double r_core = 2.0 * r / 3.0;
        double r_soft = r / 3.0;

        int y0 = (int) floor(cy - r);
        int y1 = (int) ceil(cy + r);
        int x0 = (int) floor(cx - r);
        int x1 = (int) ceil(cx + r);

        if (y0 < 0)  { y0 = 0; }
        if (y1 >= H) { y1 = H - 1; }
        if (x0 < 0)  { x0 = 0; }
        if (x1 >= W) { x1 = W - 1; }

        for (int yy = y0; yy <= y1; yy++)
        {
            double dy = (double) yy - cy;
            for (int xx = x0; xx <= x1; xx++)
            {
                double dx = (double) xx - cx;
                double dist = sqrt(dx * dx + dy * dy);
                if (dist <= r_core)
                {
                    buf[yy * W + xx] += 1.0f;
                }
                else if (dist <= r)
                {
                    double factor = 1.0 - (dist - r_core) / r_soft;
                    buf[yy * W + xx] += (float) factor;
                }
            }
        } // for yy
    } // for b
}


int main(
    int   argc,
    char *argv[])
{
    int    nballs          = 1;
    double radius          = 5.0;
    int    W               = 32;
    int    H               = 32;
    int    nframes         = 10000;
    int    seed            = 42;
    int    ball_collisions = 1;
    char  *outfile         = NULL;

    /* Parse command-line arguments */
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
        {
            nballs = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-r") == 0
                 && i + 1 < argc)
        {
            radius = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-W") == 0
                 && i + 1 < argc)
        {
            W = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-H") == 0
                 && i + 1 < argc)
        {
            H = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-f") == 0
                 && i + 1 < argc)
        {
            nframes = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-s") == 0
                 && i + 1 < argc)
        {
            seed = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-no-ball-collision") == 0
                 || strcmp(argv[i], "-b") == 0)
        {
            ball_collisions = 0;
        }
        else if (argv[i][0] != '-')
        {
            outfile = argv[i];
        }
    }

    if (outfile == NULL)
    {
        fprintf(stderr,
                "Usage: gen_bouncing_balls [opts] "
                "output.fits\n");
        fprintf(stderr,
                "  -n <int>    balls (1)\n"
                "  -r <float>  radius (5.0)\n"
                "  -W <int>    width (32)\n"
                "  -H <int>    height (32)\n"
                "  -f <int>    frames (10000)\n"
                "  -s <int>    seed (42)\n"
                "  -b / -no-ball-collision\n"
                "              Disable ball-to-ball collisions\n");
        return 1;
    }

    printf("Bouncing balls: %d balls, r=%.1f, "
           "%dx%d, %d frames, seed=%d, collisions=%d\n",
           nballs, radius, W, H, nframes, seed, ball_collisions);

    srand((unsigned int) seed);

    Ball *balls = malloc(
        (size_t) nballs * sizeof(Ball));
    if (balls == NULL)
    {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    init_balls(balls, nballs, radius, W, H);

    /* Allocate frame buffer */
    float *frame = malloc(
        (size_t)(W * H) * sizeof(float));
    if (frame == NULL)
    {
        fprintf(stderr, "malloc failed\n");
        free(balls);
        return 1;
    }

    /* Create FITS output */
    fitsfile *fptr = NULL;
    int status = 0;
    char bangpath[4096];
    snprintf(bangpath, sizeof(bangpath),
             "!%s", outfile);

    fits_create_file(&fptr, bangpath, &status);
    if (status)
    {
        fprintf(stderr, "FITS create error %d\n",
                status);
        free(frame);
        free(balls);
        return 1;
    }

    long naxes[3] = { W, H, nframes };
    fits_create_img(fptr, FLOAT_IMG, 3,
                    naxes, &status);

    /* Write frames one at a time */
    long fpixel[3] = { 1, 1, 1 };
    long npix = (long)(W * H);

    for (int fi = 0; fi < nframes; fi++)
    {
        render_frame(frame, balls, nballs, W, H);

        fpixel[2] = fi + 1;
        fits_write_pix(fptr, TFLOAT, fpixel,
                       npix, frame, &status);
        if (status)
        {
            fprintf(stderr,
                    "FITS write error at frame %d\n",
                    fi);
            break;
        }

        step_balls(balls, nballs, W, H, ball_collisions);

        if ((fi + 1) % 1000 == 0)
        {
            printf("  %d / %d frames\n",
                   fi + 1, nframes);
        }
    }

    fits_close_file(fptr, &status);
    if (status)
    {
        fprintf(stderr, "FITS close error %d\n",
                status);
    }
    else
    {
        printf("Written %s\n", outfile);
    }

    free(frame);
    free(balls);
    return 0;
}
