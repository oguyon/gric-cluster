# xtile

## ROLE
Live Cross-Tile Prior Injection

## FUNCTION
Enables dynamic constraint propagation between tiles
during the solving of a frame. Once one tile resolves
its assignment, its state is broadcast to neighbor tiles,
shifting their target priors.

Modes:
1: Pure spatial CPT prior injection.
2: Hybrid spatial-temporal prior injection (Strategy C, default).

## RATIONALE
Drastically reduces redundant distance calculations
and enforces spatial/temporal coherency on moving bodies.

## USE
-xtile [1|2] (default: 2)

## REQUIRES
-tiles NxM (only meaningful in multi-tile mode)

## SEE ALSO
- `-xtile_decay`: CPT decay parameter
- `tiling`: Tiling topic overview
