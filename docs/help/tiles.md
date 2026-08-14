# tiles

## ROLE
Image Partitioning

## FUNCTION
Splits the input image into a regular NxM
grid of tiles. Each tile runs its own
independent GRIC clustering instance in
parallel via OpenMP.

## RATIONALE
Tiling provides three benefits:
  1. Arithmetic speedup: smaller sub-frames
     make distance calls cheaper.
  2. Parallelization: tiles dispatch across
     OpenMP threads.
  3. Memory reduction: each tile's cluster
     set is smaller, shrinking DCC matrices.

Avoid partitioning too finely (e.g. 4x4 on a
32x32 image). As grid size M increases, the
joint state combinations grow exponentially
(k^M). Recommend 2x2 for small/medium
sensors.

## USE
-tiles 2x2

## SEE ALSO
- `-tilemap`: Load custom tile map
- `-tileconf`: Per-tile configuration overrides
- `-retrieval_window`: Tuple lookback horizon
- `tiling`: Tiling topic overview
