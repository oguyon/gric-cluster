# cpt

## WHAT IS THE CPT?
CPT stands for Conditional Probability Table.

In GRIC's multi-tile mode, the CPT is a shared co-occurrence table
that learns spatial dependencies between tiles in real-time. It tracks
the joint assignments of recently solved frames. If tile A was assigned
to cluster c_A and tile B was assigned to cluster c_B, this co-occurrence
is recorded in the table.

## CROSS-TILE PRIOR INJECTION (-xtile)
During the clustering of a frame, once any tile resolves its local assignment,
it writes its cluster ID to a shared board. Subsequent tiles query the CPT
using the known assignments of already-resolved tiles to extract target
prediction vectors (conditional probabilities).

These probabilities are injected as live priors for unresolved tiles, biasing
their candidate search order and dynamically pruning distant clusters.

## CPT TRAJECTORY DECAY (-xtile_decay)
By default, co-occurrence statistics are accumulated with equal weight
(full memory). By setting -xtile_decay <val> (e.g. 0.999), historical counts
are exponentially decayed over time. This helps the CPT adapt to transient
trajectories or non-stationary patterns (e.g. moving targets) while discounting
stale historical evidence.

## SEE ALSO
- `-xtile`: Cross-tile prior injection
- `-xtile_decay`: CPT history decay coefficient
- `tiling`: Image partitioning and multi-tile processing
