# retrieval_window

## ROLE
Trajectory Fusion Lookback

## FUNCTION
Sets the lookback horizon (in frames) for
Joint Trajectory Fusion in multi-tile mode
(Default: 1000).

## RATIONALE
Trajectory fusion corrects noisy per-tile
assignments by comparing the current joint
tuple against recent history. The window
controls how far back to look:
  - Too small (<200): weak statistics, poor error correction.
  - Optimal (1000-10000): robust evidence, filters boundary fluctuations.
  - Too large (>20000): stale memory from drifted/recycled clusters acts as noise.

## USE
-retrieval_window 5000

## REQUIRES
-tiles NxM (only active in multi-tile mode)

## SEE ALSO
- `-tiles`: Enable tiling
- `tiling`: Tiling topic overview
- `compression`: State space compression
