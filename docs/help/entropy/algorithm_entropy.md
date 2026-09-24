# algorithm/entropy

## SHANNON ENTROPY TARGET SELECTION
Entropy-based target selection schedules measurements to maximize
information gain by choosing the target cluster that minimizes expected
Shannon entropy in the next step.

## MATHEMATICAL MECHANISM
For each candidate target `T`, the scheduler computes:
  E[H(T)] = sum_cj( p_current[cj] * H(T | cj) )

Where `H(T | cj)` is the hypothetical entropy if `cj` is the true
cluster. The scheduler uses the precomputed `consistency_mask` to identify
which clusters survive triangle inequality pruning. The target `T` that
minimizes `E[H(T)]` is selected.

Options include target capping (`entropy_max_targets`), skip thresholds
(`entropy_min_prob`), and popcount-only surrogate mode (`entropy_fast`).

Each hypothesis considers two outcomes: match (frame
assigned, search ends) or miss (posterior updated,
search continues). The conditional posterior after a
miss is computed from the consistency mask which
encodes which clusters survive triangle inequality
pruning.

## RATIONALE
Greedy selection always measures the likeliest cluster.
When probability is spread across many candidates
(e.g. 20 clusters each near 5%%), greedy picks one
with a low chance of an early match. Entropy instead
picks the measurement whose outcome — whether match
or miss — eliminates the most alternatives, resolving
ambiguity faster. This minimizes the average number
of distance evaluations needed to find the correct
cluster.

## SOURCE IMPLEMENTATION
Implemented in `select_next_measurement_target_entropy()` inside
src/gric-cluster/steps/select_next_measurement_target.c.

## SEE ALSO
- `algorithm/gating`: Details on adaptive entropy gating
- `algorithm`: Overview of the GRIC algorithm
