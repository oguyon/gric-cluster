# algorithm/pruning

## TRIANGLE INEQUALITY PRUNING
Pruning eliminates distant cluster candidates to avoid computing their
full Euclidean distances. GRIC implements 3-point, 4-point, and 5-point
pruning modes.

## PRUNING MODES
1. 3-Point Pruning (Standard)
   After measuring `d(frame, A)`, eliminate candidate `B` if:
     |d(frame, A) - d(A, B)| > rlim

2. 4-Point Pruning (`-te4` mode)
   Uses two previously measured reference clusters to project points into
   a local coordinate system, deriving tighter pruning bounds via
   `calc_min_dist_4pt()`.

3. 5-Point Pruning (`-te5` mode)
   Uses three reference clusters for multi-dimensional bound refinement
   via `prune_candidates_te5()`.

## RATIONALE
Standard 3-point pruning constrains distance along
one dimension. Each additional reference point
constrains an extra dimension, exponentially
shrinking the volume of possible positions for the
candidate. In high-dimensional spaces, this
tightening compensates for the looseness of simple
triangle inequality bounds.

TE4 and TE5 compute lower bounds on the true
distance by embedding points into 2D or 3D
coordinate systems via distance geometry. Since
the reconstructed coordinates use non-negative
components, the result is always a valid lower
bound, safe for pruning.

## SOURCE IMPLEMENTATION
Implemented in `update_probabilities_and_pruning()` inside
src/gric-cluster/steps/update_probabilities_and_pruning.c.

## SEE ALSO
- `algorithm/sparse_dcc`: Details on sparse cluster distance matrix bounds
- `algorithm`: Overview of the GRIC algorithm
