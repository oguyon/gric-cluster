# algorithm/soft_bayesian

## SOFT BAYESIAN LIKELIHOOD UPDATES
Soft Bayesian mode replaces binary candidate pruning with soft probability
likelihood updates. Candidates fade out gradually over multiple steps
instead of being eliminated instantly.

## GAUSSIAN LIKELIHOOD
The posterior probabilities are scaled by a
Gaussian likelihood factor:
  likelihood = exp(-(d_measured - d_anchor)^2
                   / (2 * sigma^2))

where sigma = rlim * sigma_coeff (default 1.0,
tunable via -soft_bayesian_sigma).

The exponential is approximated using a minimax
polynomial on the interval [0, 2], avoiding slow
library exp() calls. Returns 0.0 for large
deviations (hard cutoff).

## RATIONALE
Hard pruning is binary: a candidate is either
alive or dead. When clusters are close together,
a small measurement error can wrongly eliminate
the true cluster. Soft Bayesian gradually fades
candidates, making the algorithm robust to
near-boundary measurements.

Most beneficial when:
  - Clusters overlap geometrically
  - Distance measurements are noisy
  - rlim is close to inter-cluster spacing

Less useful when clusters are well-separated.

## SOURCE IMPLEMENTATION
Implemented in `update_probabilities_and_pruning()` inside
src/gric-cluster/steps/update_probabilities_and_pruning.c.

## SEE ALSO
- `algorithm/entropy`: Details on Shannon entropy target selection
- `algorithm`: Overview of the GRIC algorithm
