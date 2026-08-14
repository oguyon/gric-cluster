# no_xtile

## ROLE
Trajectory Fusion Control

## FUNCTION
Disables cross-tile trajectory correction
in multi-tile mode (this is the default behavior
unless -jtf is set).

## RATIONALE
Useful for debugging tile boundary effects
or when tiles are fully independent and
cross-tile correction is not desired.

## USE
-no_xtile

## REQUIRES
-tiles NxM (only meaningful in multi-tile
 mode)

## SEE ALSO
- `-retrieval_window`: Fusion lookback horizon
- `tiling`: Tiling topic overview
