# tileconf

## ROLE
Per-Tile Configuration

## FUNCTION
Loads per-tile overrides for rlim and maxcl
from an ASCII file. Only rlim and maxcl can
be overridden per tile; all other options
use the global configuration.

## FORMAT
One line per tile:
  tile_id  rlim  maxcl
Lines starting with '#' are comments.

## RATIONALE
Different regions of an image may have
different noise levels or feature densities.
Per-tile rlim adapts the distance threshold
to each region's characteristics.

## USE
-tileconf tile_params.txt

## SEE ALSO
- `-tiles`: Regular NxM grid partitioning
