# pred

## ROLE
Time-Series Prediction

## FUNCTION
Predicts next cluster based on sequence history.
Two modes are available: binary (fast, exact) and fuzzy (slow, geometric).

## FORMAT
-pred[len,h,n]   Binary prediction (default)
-predf[len,h,n]  Fuzzy prediction

  len: Length of recent sequence to match (Default: 10).
  h  : History size to search (Default: 1000).
  n  : Number of predicted candidates to test first (Default: 2).

## BINARY MODE (-pred)
Exact pattern match on cluster IDs. Scans the last h assignments for subsequences matching the most recent len cluster IDs. If the sequence [A, B, C] appears in history followed by D, then D is predicted.

Cost: O(h) integer comparisons per frame. Very fast.

## FUZZY MODE (-predf)
Geometric sequence similarity using distances and inter-cluster bounds. For every position in the lookback window, computes a continuous similarity metric between the current trajectory and the historical trajectory using frame-to-cluster distances and DCC triangle inequality bounds.

Produces a full probability distribution over all clusters, handling soft matches where trajectories pass through nearby but different clusters.

Cost: O(h x len) with distance lookups, pow(), exp() per entry. Much slower than binary mode.

## USE
-pred[5,500,1]   (Fast, repeating patterns)
-predf[5,500,1]  (Noisy/drifting trajectories)

## INTERACTS WITH
- -gprob: Both contribute to cluster probability distribution
- -tm: Transition matrix complements pattern detection

## SEE ALSO
- `-gprob`: Use geometrical probability
- `-tm`: Transition matrix mixing
- `tiling`: Tiling topic overview
