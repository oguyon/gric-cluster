# jtf

## ROLE
Joint Trajectory Fusion

## FUNCTION
Enables Joint Trajectory Fusion (Pass 2) in multi-tile mode.
By default, JTF is disabled. Enabling JTF allows correcting
tile-boundary noise by comparing raw assignments against
recent spatial-temporal assignment history.

## USE
-jtf

## REQUIRES
-tiles NxM (only meaningful in multi-tile mode)

## SEE ALSO
- `tiling`: Tiling topic overview
- `-retrieval_window`: Fusion lookback horizon
