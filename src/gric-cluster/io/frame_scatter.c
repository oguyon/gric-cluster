#include "frame_scatter.h"

#include <stdlib.h>
#include <string.h>

/**
 * frame_scatter_alloc() - Pre-allocate data buffers for tile sub-frames.
 * @tm:          Tile map describing M tiles and their pixel counts.
 * @tile_frames: Caller-allocated array of M Frame structs to initialise.
 *
 * For each tile m, allocates a contiguous double buffer large enough
 * for tile_frames[m].data and sets width = num_pixels, height = 1.
 * The buffers are reused across frames via frame_scatter().
 */
void frame_scatter_alloc(
    const TileMap *tm,
    Frame         *tile_frames)
{
    for (int m = 0; m < tm->num_tiles; m++)
    {
        uint64_t npix = tm->tiles[m].num_pixels;

        tile_frames[m].data =
            (double *) malloc(npix * sizeof(double));
        tile_frames[m].width  = (long) npix;
        tile_frames[m].height = 1;
    }
}

/**
 * frame_scatter() - Scatter a full-image frame into per-tile sub-frames.
 * @src:         Source frame covering the full image.
 * @tm:          Tile map with pixel-index lists for each tile.
 * @tile_frames: Pre-allocated array of M tile sub-frames (from
 *               frame_scatter_alloc).
 *
 * Copies metadata (id, cnt0, atime) from @src into every tile frame,
 * then gathers pixel values from @src->data into each tile's
 * contiguous buffer using the tile's sorted pixel-index list.
 *
 * The data buffers in @tile_frames must already be allocated by
 * frame_scatter_alloc(); this function only writes pixel data and
 * metadata.
 */
void frame_scatter(
    const Frame   *restrict src,
    const TileMap *restrict tm,
    Frame         *restrict tile_frames)
{
    for (int m = 0; m < tm->num_tiles; m++)
    {
        const TileDef *td = &tm->tiles[m];
        Frame *tf = &tile_frames[m];

        /* Copy metadata from source frame */
        tf->id    = src->id;
        tf->cnt0  = src->cnt0;
        tf->atime = src->atime;

        /* Gather pixels into contiguous tile buffer */
        const int      *restrict idx = td->pixel_indices;
        const double   *restrict sd  = src->data;
        double         *restrict dd  = tf->data;
        const uint64_t  npix         = td->num_pixels;

        for (uint64_t i = 0; i < npix; i++)
        {
            dd[i] = sd[idx[i]];
        }
    }
}

/**
 * frame_scatter_free() - Free data buffers of tile sub-frames.
 * @tile_frames: Array of Frame structs whose data buffers are freed.
 * @num_tiles:   Number of tile sub-frames in the array.
 *
 * Frees each tile_frames[m].data and sets the pointer to NULL.
 * Does not free the tile_frames array itself (caller-owned).
 */
void frame_scatter_free(
    Frame *tile_frames,
    int    num_tiles)
{
    for (int m = 0; m < num_tiles; m++)
    {
        free(tile_frames[m].data);
        tile_frames[m].data = NULL;
    }
}
