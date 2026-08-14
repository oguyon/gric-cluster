# clustering

## OVERVIEW
These options control how frames are assigned
to clusters and how the search is accelerated.

## CORE PARAMETERS
-dprob <val>   (default: 0.01)
  Recency bias: amount added to a cluster's
  probability when a frame is assigned to it.
  Higher = faster adaptation, more volatile.

-maxcl <val>   (default: 1000)
  Maximum number of clusters. Controls memory
  usage (DCC matrix is O(maxcl^2)).

-maxim <val>   (default: 100000)
  Maximum number of frames to process.

-ncpu <val>    (default: 1)
  Number of CPUs for parallel pruning.
  Effective when K >= 256.

## PROBABILITY AND PRUNING
-gprob
  Use distance history to estimate match
  probability (geometric probability).

-soft_bayesian
  Gaussian-like soft pruning instead of hard
  binary elimination.

-te4 / -te5
  4-point or 5-point triangle inequality
  pruning. Tighter bounds, higher per-step
  cost. Best for high-dimensional data.

-entropy
  Information-theoretic target selection.
  Picks the measurement that maximizes
  expected entropy reduction.

-sparse_dcc
  Sparse cluster-to-cluster distance matrix.
  Avoids O(K^2) cost at cluster creation.

## PREDICTION
-pred[l,h,n]   (default: 10,1000,2)
  Binary: exact pattern match on
  cluster IDs. Fast.

-predf[l,h,n]  (default: 10,1000,2)
  Fuzzy: geometric sequence similarity
  using distances and DCC bounds.
  Much slower. Use for noisy data.

-tm <coeff>    (0.0 to 1.0)
  Transition matrix mixing. Blends prior
  probability with transition history.

## RESOURCE LIMITS
-maxcl_strategy <stop|discard|merge>
  What to do when maxcl is reached:
  stop:    exit (default, batch mode)
  discard: evict least-visited cluster
  merge:   merge two closest clusters

-discard_frac <val>  (default: 0.5)
  Fraction of oldest clusters to consider
  for discard eviction.

## CONFIGURATION FILES
-conf <file>
  Read options from a configuration file.

-confw <file>
  Write current options to a file.
