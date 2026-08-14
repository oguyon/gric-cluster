# algorithm

## GRIC: GEOMETRIC REAL-TIME IMAGE CLUSTERING
GRIC is a single-pass, sequential, distance-based
clustering algorithm. It processes frames one at a
time and assigns each to a cluster whose anchor is
within a maximum Euclidean distance rlim. If no
cluster matches, a new one is created.

## WHY GRIC IS FAST
Naively assigning a frame to one of K clusters
requires computing K distances -- O(K) work per
frame. GRIC uses active learning to identify the
matching cluster in far fewer measurements:

  Each distance measurement eliminates multiple
  candidates via triangle inequality. The next
  target is chosen to maximally reduce remaining
  ambiguity (entropy-based selection). Together,
  these dramatically reduce the number of
  measurements per frame — often to just a few,
  regardless of how many clusters exist.

Key insight: not every cluster needs to be measured
to identify the match. One measurement against
cluster A reveals information about clusters B, C, D
through their known inter-cluster distances.

## WHAT IS A CLUSTER
A cluster is represented by a single anchor frame:
the first frame that created the cluster.

  distance-to-cluster = distance-to-anchor

This is different from k-means (which uses centroids)
or DBSCAN (which uses density neighborhoods).
The anchor representation means:
  - No centroid recomputation as members are added
  - Cluster identity is a concrete data sample
  - Distance computations are always frame-to-frame

## DISTANCE METRIC
Euclidean (L2) distance between frame vectors:
  d(a, b) = sqrt( sum( (a[i] - b[i])^2 ) )
Uses AVX2/FMA SIMD intrinsics when the CPU
supports them, with a scalar fallback.

## PER-FRAME PIPELINE
Frame
    |
    v
  [1] Predict priors
    |
    v
  [2] Select target  <-----+
    |                      |
    v                      |
  [3] Measure distance     |
    |                      |
    v                      |
  [4] Update & prune  -----+
    |                  (no match)
    | (match or
    |  all exhausted)
    v
  [5] Assign or create new cluster

1. PREDICT PRIORS
   Build the initial probability distribution
   over clusters (see POSTERIOR PROBABILITIES).
   Frequency prior is always used. If -pred or
   -tm is active, temporal information is mixed
   in to bias the search toward likely clusters.

2. SELECT TARGET
   Pick the cluster to measure next:
   - Greedy: highest posterior probability
   - Entropy (-entropy): min expected posterior entropy after measurement

3. MEASURE DISTANCE
   Compute d(frame, anchor). This is the
   expensive operation the algorithm minimizes.

4. UPDATE & PRUNE
   Use the measured distance to:
   - Eliminate incompatible clusters (pruning)
   - Update geometric probabilities (-gprob)
   - Fade unlikely candidates (-soft_bayesian)
   If d < rlim: go to step 5 (match found).
   Otherwise: go to step 2 (try next cluster).
   If all clusters exhausted: go to step 5.

5. ASSIGN OR CREATE
   If a match was found: assign frame to that
   cluster.
   If all clusters were exhausted without a
   match: the frame becomes the anchor of a
   new cluster.

## POSTERIOR PROBABILITIES
The search order depends on the posterior
probability P(cluster i | frame). This section
explains how P is constructed.

BASE CASE (no -pred, no -tm, no -gprob):
  Each cluster has a frequency score that
  starts at 1.0 when the cluster is created.
  Each time a frame matches cluster i, its
  score increases by dprob (default 0.01).
  Before each new frame, scores are normalized
  to sum to 1:

    P(i) = score(i) / sum(scores)

  This is a recency-weighted frequency prior:
  recently matched clusters accumulate more
  score and are searched first. The parameter
  -dprob controls how much weight recent
  activity gets relative to the baseline.

WITH -tm (transition matrix mixing):
  Blends the frequency prior with the row of
  the transition matrix for the previous frame's
  cluster:

    P(i) = (1-c) * freq(i) + c * trans(prev,i)

  where c = -tm coefficient (0.0 to 1.0).
  trans(prev,i) = fraction of times the system
  went from cluster 'prev' to cluster i.

WITH -pred (sequence prediction):
  Multiplies the frequency prior by a sequence
  match score from pattern detection:

    P(i) = freq(i) * seq_match(i) / Z

  where Z is a normalization constant ensuring the sum of probabilities
  equals 1.0, and seq_match(i) measures how well the
  recent assignment history matches past
  patterns that led to cluster i.

WITH -gprob (geometric probability):
  During the search loop, each measurement
  further refines the posterior by updating
  geometric probabilities based on spatial
  correlations learned from visitor history.
  See '-h algorithm/gprob' for details.

PROBABILITY LAYERING:
    The probability distribution P is constructed and refined in layers:
    1. Baseline: Frequency prior (always active; recency-weighted)
    2. Temporal: + -pred OR -tm (blends in temporal sequence patterns at frame start)
    3. Spatial:  + -gprob (refines dynamically during the search loop using visitor history)

  TARGET SELECTION (using the probability distribution):
    The target selection strategy determines how P is used to pick the next candidate:
    - Greedy (default): Pick the candidate with the highest posterior probability P(i).
    - Entropy (-entropy): Pick the candidate minimizing expected Shannon entropy of P after the measurement, maximizing information gain.

## KEY OPTIMIZATIONS
Triangle inequality (always active)
  After measuring d(frame, A), eliminate cluster B
  if |d(frame,A) - d(A,B)| > rlim. Extends to
  4-point (-te4) and 5-point (-te5) pruning for
  tighter bounds using coordinate reconstruction.

Geometric probability (-gprob)
  Learns spatial relationships from measurement
  history. When cluster A is measured, examine
  past visitors of A and boost/penalize their
  assigned clusters based on distance similarity.

Entropy-based selection (-entropy)
  Information-theoretic target selection. Picks
  the measurement that maximally reduces Shannon
  entropy of the posterior distribution. Uses a
  precomputed consistency mask for efficiency.

Sparse DCC (-sparse_dcc)
  Maintains upper/lower bounds on inter-cluster
  distances instead of computing all O(K^2) pairs.
  Critical for large numbers of clusters.

Soft Bayesian (-soft_bayesian)
  Replaces hard binary pruning with smooth
  Gaussian-like likelihood updates. Candidates
  fade out gradually instead of being eliminated.

Recency bias (-dprob)
  Recently active clusters rise in the search
  priority, reducing average search depth.

## OPTION INTERACTIONS
Options feed into each other along the pipeline:

  -pred / -tm ----> Priors
                      |
                      v
  -entropy -------> Target Selection
                      |
                      v
  -te4/-te5 ------> Pruning
  -sparse_dcc --/     |
                      v
  -gprob ---------> Probability Update
                      |
                      v
  -soft_bayesian --> Likelihood Fading

Synergies:
  - -gprob + -entropy: gprob builds the posterior, entropy schedules measurements to resolve it.
  - -sparse_dcc + large -maxcl: avoids the O(K^2) cost of dense cluster-to-cluster distances.
  - -pred + -tm: pattern detection for multi-step sequences; transition matrix for pairwise.

## COMPLEXITY
Measurements per frame (K = number of clusters):
  - Naive (no pruning): O(K), measure every cluster
  - Greedy + pruning: substantially fewer; data-dependent
  - Entropy + pruning: fewer still; data-dependent
  - Entropy + gprob: often just a few measurements

The actual count depends on data structure: highly
structured data (well-separated clusters) can
require as few as 1-2 measurements per frame.
Worst case (overlapping clusters) approaches O(K).

Memory:
  - DCC matrix (dense): O(K^2)
  - DCC matrix (sparse): O(K)
  - Assignment history: O(N), N = frames seen
  - Gprob visitor lists: O(K x maxvis)

## SEE ALSO
- `rlim`: Distance threshold (the key parameter)
- `algorithm/gating`: Adaptive entropy gating
- `algorithm/gprob`: Geometric probability learning
- `algorithm/entropy`: Shannon entropy target selection
- `algorithm/pruning`: Triangle inequality pruning (TE4/TE5)
- `algorithm/sparse_dcc`: Sparse cluster distance matrix bounds
- `algorithm/soft_bayesian`: Soft Bayesian likelihood updates
- `performance`: How to pick options for best speed
