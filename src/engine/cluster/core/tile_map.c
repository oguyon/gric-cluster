/**
 * @file tile_map.c
 * @brief Implementation of tile map creation, loading, and cleanup.
 */

#include "tile_map.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef USE_CFITSIO
#include <fitsio.h>
#endif


/* --------------------------------------------------------
 *  qsort comparator for int arrays
 * -------------------------------------------------------- */
static int cmp_int(
    const void *a,
    const void *b)
{
    int ia = *(const int *) a;
    int ib = *(const int *) b;

    return (ia > ib) - (ia < ib);
}


/**
 * tilemap_create_single() - Create a single tile covering
 *                           the entire image.
 * @img_width:  Image width in pixels.
 * @img_height: Image height in pixels.
 *
 * Allocates a TileMap with one tile whose pixel_indices
 * array contains every pixel index from 0 to w*h-1.
 *
 * Return: Pointer to the new TileMap, or NULL on failure.
 */
TileMap *tilemap_create_single(
    long img_width,
    long img_height)
{
    if (img_width <= 0 || img_height <= 0)
    {
        fprintf(stderr,
                "tilemap_create_single: invalid size "
                "%ld x %ld\n",
                img_width, img_height);
        return NULL;
    }

    TileMap *tm = calloc(1, sizeof(*tm));
    if (tm == NULL)
    {
        return NULL;
    }

    tm->num_tiles  = 1;
    tm->img_width  = img_width;
    tm->img_height = img_height;

    tm->tiles = calloc(1, sizeof(TileDef));
    if (tm->tiles == NULL)
    {
        free(tm);
        return NULL;
    }

    uint64_t npix = (uint64_t) img_width * img_height;

    tm->tiles[0].num_pixels = npix;
    tm->tiles[0].pixel_indices = malloc(
        npix * sizeof(int));
    if (tm->tiles[0].pixel_indices == NULL)
    {
        free(tm->tiles);
        free(tm);
        return NULL;
    }

    for (uint64_t ii = 0; ii < npix; ii++)
    {
        tm->tiles[0].pixel_indices[ii] = (int) ii;
    }

    return tm;
}


/**
 * tilemap_create_grid() - Divide the image into a regular
 *                         rectangular grid of tiles.
 * @img_width:  Image width in pixels.
 * @img_height: Image height in pixels.
 * @grid_x:     Number of tile columns.
 * @grid_y:     Number of tile rows.
 *
 * Edge tiles absorb any remainder pixels so that no pixel
 * is lost.  Pixel indices within each tile are sorted for
 * cache locality.
 *
 * Return: Pointer to the new TileMap, or NULL on failure.
 */
TileMap *tilemap_create_grid(
    long img_width,
    long img_height,
    int  grid_x,
    int  grid_y)
{
    if (img_width <= 0 || img_height <= 0)
    {
        fprintf(stderr,
                "tilemap_create_grid: invalid size "
                "%ld x %ld\n",
                img_width, img_height);
        return NULL;
    }
    if (grid_x <= 0 || grid_y <= 0)
    {
        fprintf(stderr,
                "tilemap_create_grid: invalid grid "
                "%d x %d\n",
                grid_x, grid_y);
        return NULL;
    }

    TileMap *tm = calloc(1, sizeof(*tm));
    if (tm == NULL)
    {
        return NULL;
    }

    int num_tiles  = grid_x * grid_y;

    tm->num_tiles  = num_tiles;
    tm->img_width  = img_width;
    tm->img_height = img_height;

    tm->tiles = calloc(num_tiles, sizeof(TileDef));
    if (tm->tiles == NULL)
    {
        free(tm);
        return NULL;
    }

    long base_tw = img_width  / grid_x;
    long base_th = img_height / grid_y;

    /* Fill each tile with its pixel indices. */
    for (int ty = 0; ty < grid_y; ty++)
    {
        long row0 = ty * base_th;
        long row1 = (ty == grid_y - 1)
                         ? img_height
                         : row0 + base_th;

        for (int tx = 0; tx < grid_x; tx++)
        {
            long col0 = tx * base_tw;
            long col1 = (tx == grid_x - 1)
                             ? img_width
                             : col0 + base_tw;

            int tidx = ty * grid_x + tx;

            uint64_t tile_npix =
                (uint64_t)(row1 - row0)
                * (uint64_t)(col1 - col0);

            tm->tiles[tidx].num_pixels = tile_npix;
            tm->tiles[tidx].pixel_indices = malloc(
                tile_npix * sizeof(int));
            if (tm->tiles[tidx].pixel_indices == NULL)
            {
                goto cleanup;
            }

            uint64_t pi = 0;
            for (long rr = row0; rr < row1; rr++)
            {
                for (long cc = col0; cc < col1; cc++)
                {
                    tm->tiles[tidx].pixel_indices[pi] =
                        (int)(rr * img_width + cc);
                    pi++;
                }
            } // for rr

            /* Indices are already in row-major order,
             * so they are sorted.  qsort is a no-op
             * here but included for correctness. */
            qsort(tm->tiles[tidx].pixel_indices,
                   tile_npix,
                   sizeof(int),
                   cmp_int);
        } // for tx
    } // for ty

    return tm;

cleanup:
    tilemap_free(tm);
    return NULL;
}


#ifdef USE_CFITSIO
/**
 * tilemap_load_fits() - Load a tile map from an integer
 *                       FITS image.
 * @path:       Path to the FITS file.
 * @img_width:  Expected image width.
 * @img_height: Expected image height.
 *
 * Each pixel value in the FITS image is a zero-based tile
 * index.  Negative values are rejected.  The number of
 * tiles equals max_value + 1.
 *
 * Return: Pointer to the new TileMap, or NULL on failure.
 */
TileMap *tilemap_load_fits(
    const char *path,
    long        img_width,
    long        img_height)
{
    if (path == NULL)
    {
        fprintf(stderr,
                "tilemap_load_fits: NULL path\n");
        return NULL;
    }
    if (img_width <= 0 || img_height <= 0)
    {
        fprintf(stderr,
                "tilemap_load_fits: invalid size "
                "%ld x %ld\n",
                img_width, img_height);
        return NULL;
    }

    fitsfile *fptr = NULL;
    int       status = 0;

    fits_open_file(&fptr, path, READONLY, &status);
    if (status)
    {
        fprintf(stderr,
                "tilemap_load_fits: cannot open %s "
                "(CFITSIO status %d)\n",
                path, status);
        return NULL;
    }

    /* Validate NAXIS == 2. */
    int naxis = 0;
    fits_get_img_dim(fptr, &naxis, &status);
    if (status || naxis != 2)
    {
        fprintf(stderr,
                "tilemap_load_fits: expected NAXIS=2, "
                "got %d\n",
                naxis);
        fits_close_file(fptr, &status);
        return NULL;
    }

    /* Validate image dimensions. */
    long naxes[2];
    fits_get_img_size(fptr, 2, naxes, &status);
    if (status)
    {
        fits_close_file(fptr, &status);
        return NULL;
    }
    if (naxes[0] != img_width || naxes[1] != img_height)
    {
        fprintf(stderr,
                "tilemap_load_fits: dimension mismatch: "
                "FITS %ld x %ld vs expected %ld x %ld\n",
                naxes[0], naxes[1],
                img_width, img_height);
        fits_close_file(fptr, &status);
        return NULL;
    }

    uint64_t npix = (uint64_t) img_width * img_height;

    int *data = malloc(npix * sizeof(int));
    if (data == NULL)
    {
        fits_close_file(fptr, &status);
        return NULL;
    }

    /* Read entire image as int. */
    {
        long fpixel[2] = {1, 1};
        int  anynul    = 0;

        fits_read_pix(fptr, TINT, fpixel,
                       (LONGLONG) npix,
                       NULL, data, &anynul, &status);
    }
    fits_close_file(fptr, &status);

    if (status)
    {
        fprintf(stderr,
                "tilemap_load_fits: read error "
                "(CFITSIO status %d)\n",
                status);
        free(data);
        return NULL;
    }

    /* Scan for max value and reject negatives. */
    int max_val = 0;
    for (uint64_t ii = 0; ii < npix; ii++)
    {
        if (data[ii] < 0)
        {
            fprintf(stderr,
                    "tilemap_load_fits: negative tile "
                    "index %d at pixel %lu\n",
                    data[ii],
                    (unsigned long) ii);
            free(data);
            return NULL;
        }
        if (data[ii] > max_val)
        {
            max_val = data[ii];
        }
    } // scan loop

    int num_tiles = max_val + 1;

    /* Count pixels per tile. */
    uint64_t *counts = calloc(num_tiles, sizeof(uint64_t));
    if (counts == NULL)
    {
        free(data);
        return NULL;
    }
    for (uint64_t ii = 0; ii < npix; ii++)
    {
        counts[data[ii]]++;
    }

    /* Allocate TileMap. */
    TileMap *tm = calloc(1, sizeof(*tm));
    if (tm == NULL)
    {
        free(counts);
        free(data);
        return NULL;
    }
    tm->num_tiles  = num_tiles;
    tm->img_width  = img_width;
    tm->img_height = img_height;

    tm->tiles = calloc(num_tiles, sizeof(TileDef));
    if (tm->tiles == NULL)
    {
        free(tm);
        free(counts);
        free(data);
        return NULL;
    }

    for (int tt = 0; tt < num_tiles; tt++)
    {
        tm->tiles[tt].num_pixels = counts[tt];
        if (counts[tt] > 0)
        {
            tm->tiles[tt].pixel_indices = malloc(
                counts[tt] * sizeof(int));
            if (tm->tiles[tt].pixel_indices == NULL)
            {
                free(counts);
                free(data);
                tilemap_free(tm);
                return NULL;
            }
        }
    } // allocate per-tile arrays

    /* Fill pixel indices. */
    uint64_t *fill_pos = calloc(num_tiles,
                                 sizeof(uint64_t));
    if (fill_pos == NULL)
    {
        free(counts);
        free(data);
        tilemap_free(tm);
        return NULL;
    }

    for (uint64_t ii = 0; ii < npix; ii++)
    {
        int tid = data[ii];
        uint64_t pos = fill_pos[tid];
        tm->tiles[tid].pixel_indices[pos] = (int) ii;
        fill_pos[tid]++;
    }

    free(fill_pos);
    free(counts);
    free(data);

    /* Sort pixel indices within each tile. */
    for (int tt = 0; tt < num_tiles; tt++)
    {
        if (tm->tiles[tt].num_pixels > 1)
        {
            qsort(tm->tiles[tt].pixel_indices,
                   tm->tiles[tt].num_pixels,
                   sizeof(int),
                   cmp_int);
        }
    } // sort loop

    return tm;
}

#else /* !USE_CFITSIO */

TileMap *tilemap_load_fits(
    const char *path,
    long        img_width,
    long        img_height)
{
    (void) path;
    (void) img_width;
    (void) img_height;

    fprintf(stderr,
            "tilemap_load_fits: built without CFITSIO "
            "support\n");
    return NULL;
}

#endif /* USE_CFITSIO */


/**
 * tilemap_free() - Free all memory owned by a TileMap.
 * @tm: Pointer to the TileMap to free (may be NULL).
 *
 * Frees every tile's pixel_indices array, the tiles array,
 * and the TileMap struct itself.
 */
void tilemap_free(TileMap *tm)
{
    if (tm == NULL)
    {
        return;
    }

    if (tm->tiles != NULL)
    {
        for (int tt = 0; tt < tm->num_tiles; tt++)
        {
            free(tm->tiles[tt].pixel_indices);
        }
        free(tm->tiles);
    }

    free(tm);
}
