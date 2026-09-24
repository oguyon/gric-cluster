# entropy

## ROLE
Entropy-Based Target Selection

## OVERVIEW
In standard (greedy) mode, GRIC picks the next cluster to measure by
choosing the one with the highest posterior probability.  This is fast but
ignores the information value of each measurement: measuring a cluster
that is already very likely yields little information, whereas measuring a cluster
that would eliminate many alternatives can resolve ambiguity faster.

With -entropy, GRIC selects the target that minimizes the expected
Shannon entropy of the posterior distribution after measurement.  This
maximizes expected information gain per distance evaluation.

## POSTERIOR DISTRIBUTION
Each frame maintains a probability vector p(c) over active clusters.
This vector is initialized from the mixed priors (gprob + static
priors) and updated after each failed measurement via Bayesian
updates (binary pruning or soft Bayesian likelihood weighting if
-soft_bayesian is enabled).  The entropy H = -sum(p * log2 p) quantifies
how spread out this distribution is: H=0 means certainty, H=log2(K)
means uniform over K clusters.

## MULTI-STAGE PIPELINE
The entropy evaluation runs a 4-stage pipeline to balance
quality against cost:

1. GATING
   Shannon entropy H of the current posterior is computed.  If H is
   below the gate threshold (-entropy_gate at depth >= 1, or
   -entropy_first_gate at depth 0), the distribution is already
   concentrated and greedy argmax is used.  This skips all
   downstream stages.

2. POPCOUNT SCORING
   For each candidate target, a fast heuristic score is computed
   using bitwise AND + popcount on the consistency mask bitfield.
   This score approximates the expected support size reduction:
   low score = measuring this target eliminates many hypotheses.
   Candidates are ranked by this score and the top ones proceed.

3. CANDIDATE FILTERING
   The top candidates are selected by interleaving two lists:
   probability leaders (high p) and popcount leaders (low score).
   The number of candidates is capped at -entropy_max_targets,
   dynamically reduced based on the entropy level.

4. SHANNON EVALUATION
   For each candidate target c_i, compute the expected posterior
   entropy after measuring c_i.  Each hypothesis c_j
   (with p(c_j) > -entropy_min_prob) contributes
   p(c_j) * H(posterior | measure c_i, true cluster = c_j).
   The candidate with the lowest expected entropy wins.
   Early exit: if the running sum exceeds the current best,
   remaining hypotheses are skipped.  Hypotheses are evaluated in
   descending probability order to maximize early exit.

## FAST SURROGATE MODE
With -entropy_fast, stage 4 (Shannon evaluation) is skipped
entirely and the candidate with the lowest popcount score from
stage 2 is returned directly.  This is a first-order approximation
of entropy minimization that uses only bitwise operations.
Benchmarks show near-identical clustering quality at a fraction
of the CPU cost.

## DIAGNOSTICS
When -entropy is active, the final summary includes an
"Entropy Diagnostics" block reporting:
  - Avg/max initial entropy (uncertainty at frame start)
  - Effective candidate count (2^H)
  - Gate ratio (fraction of frames where gating returned greedy)
  - Contextual guidance (warns if rlim may need adjustment)

These metrics are also exported to the SHM status struct for
real-time monitoring via gric-status.

## WHEN TO USE
Entropy mode is most valuable when:
  - Clusters overlap geometrically (high rlim relative to spacing)
  - Frames are randomly ordered (no temporal coherence)
  - The distance function is expensive (reducing measurements matters more than the entropy computation overhead)

Entropy mode adds little benefit when:
  - Clusters are well-separated (gate catches most frames)
  - Frames follow a smooth trajectory (prediction + gprob already narrow the candidates)
  - The distance function is trivially cheap

## WORKS BEST WITH
- -gprob: Provides the probability distribution
- -soft_bayesian: Smoother Bayesian updates between measurements
- -te4 / -te5: Tighter triangle inequality bounds

## SEE ALSO
- `-entropy_fast`: Popcount-only surrogate mode
- `-entropy_gate`: Gating threshold (depth >= 1)
- `-entropy_first_gate`: Gating threshold (depth 0)
- `-entropy_max_targets`: Max targets for evaluation
- `-entropy_min_prob`: Min hypothesis probability
- `-soft_bayesian`: Soft Bayesian update
- `-gprob`: Geometrical probability
