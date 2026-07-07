#ifndef TILE_MAP_H
#define TILE_MAP_H

/**
 * @file tile_map.h
 * @brief Tile map data structures and API for
 *        partitioning an image into spatial tiles.
 */

#include <stdint.h>

/** Single tile: a list of pixel indices into the full image. */
typedef struct
{
    int      *pixel_indices; /**< Sorted 1D pixel offsets */
    uint64_t  num_pixels;    /**< Number of pixels in tile */
} TileDef;

/** Complete tile map for the image. */
typedef struct
{
    TileDef *tiles;      /**< Array of M tile definitions */
    int      num_tiles;  /**< Number of tiles (M) */
    long     img_width;  /**< Full image width */
    long     img_height; /**< Full image height */
} TileMap;

/** Create a regular gx × gy grid of non-overlapping tiles. */
TileMap *tilemap_create_grid(
    long img_width,
    long img_height,
    int  grid_x,
    int  grid_y);

/** Load tile map from an integer FITS file. */
TileMap *tilemap_load_fits(
    const char *path,
    long        img_width,
    long        img_height);

/** Create a single tile covering the entire image (M=1). */
TileMap *tilemap_create_single(
    long img_width,
    long img_height);

/** Free all memory owned by a TileMap. */
void tilemap_free(TileMap *tm);

#endif // TILE_MAP_H
