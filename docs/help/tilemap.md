# tilemap

## ROLE
Custom Tile Map

## FUNCTION
Loads a tile map from an integer FITS image.
Each pixel value specifies which tile that
pixel belongs to (0-indexed). This allows
irregular (non-rectangular) tile partitions.

## FORMAT
A 2D integer FITS image with the same
width and height as the input frames.
Pixel values are zero-based tile indices.
Number of tiles = max(pixel value) + 1.

## USE
-tilemap my_tiles.fits

## SEE ALSO
- `-tiles`: Regular NxM grid partitioning
