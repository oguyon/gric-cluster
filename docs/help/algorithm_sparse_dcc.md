# algorithm/sparse_dcc

## SPARSE DCC
Sparse DCC avoids the quadratic `O(K^2)` cost of maintaining a dense
cluster-to-cluster distance matrix (DCC). Instead of keeping exact
distances, it tracks dynamic interval bounds for each cluster pair.

## BOUNDS MAINTENANCE
1. Interval Bounds
   Stores lower bounds in `dcc_min` and upper bounds in `dcc_max`.

2. On-demand Updates
   Distance bounds are updated lazily. If a bound is too loose, additional
   DCC evaluations are executed to refine the interval.

3. Consistency Mask
   The `recompute_consistency_mask()` function constructs the bitmask using
   interval overlaps, ensuring correctness even with sparse bounds.

## SOURCE IMPLEMENTATION
Implemented in `recompute_consistency_mask()` inside
src/gric-cluster/steps/update_consistency_mask.c.

## SEE ALSO
- `algorithm/pruning`: Details on triangle inequality pruning (TE4/TE5)
- `algorithm`: Overview of the GRIC algorithm
