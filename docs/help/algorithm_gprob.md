# algorithm/gprob

## GEOMETRIC PROBABILITY
Geometric probability learns spatial transition patterns dynamically
from measurement history. When a distance measurement to cluster
anchor `C_j` is taken, the scheduler refines the probabilities of all
other active candidate clusters based on past transition correlation.

## VISITOR HISTORY & MATCHING
1. Retrieve History
   Look up past frames (visitors) that measured
   distance to `C_j`.

2. Distance Correlation
   For each visitor, get its assigned cluster `target_cl` and distance
   to `C_j` (`dist_k`). Compare it to the current distance `dfc`.

3. Matching Function
   Scale the candidate's posterior probability and geometric
   probability score using the fmatch() linear ramp
   based on distance similarity:
     dr = |d_current - d_visitor| / rlim
   Close match (dr~0) boosts probability, large
   mismatch (dr>2) kills it. See -h fmatcha.

## RATIONALE
If frame F has a similar distance to anchor A as
a past visitor V of cluster C, then F and V occupy
a similar region of the original space — so F is
likely near C. This geometric correlation transfers
knowledge from past measurements to reduce future
ones.

## SOURCE IMPLEMENTATION
Updates are performed in `update_geometric_probabilities()` inside
src/gric-cluster/steps/update_geometric_probabilities.c.

## SEE ALSO
- `algorithm/gating`: Details on adaptive entropy gating
- `algorithm`: Overview of the GRIC algorithm
