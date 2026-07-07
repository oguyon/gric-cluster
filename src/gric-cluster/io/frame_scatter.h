#ifndef FRAME_SCATTER_H
#define FRAME_SCATTER_H

#include "common.h"
#include "tile_map.h"

/**
 * @brief Scatter a full-image Frame into per-tile sub-frames.
 */
void frame_scatter(
    const Frame   *src,
    const TileMap *tm,
    Frame         *tile_frames);

/**
 * @brief Allocate data buffers for M tile sub-frames.
 */
void frame_scatter_alloc(
    const TileMap *tm,
    Frame         *tile_frames);

/**
 * @brief Free data buffers of M tile sub-frames.
 */
void frame_scatter_free(
    Frame *tile_frames,
    int    num_tiles);

#endif // FRAME_SCATTER_H
