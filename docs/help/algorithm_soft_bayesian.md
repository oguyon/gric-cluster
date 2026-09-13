# algorithm/soft_bayesian

## SOFT BAYESIAN LIKELIHOOD UPDATES
Soft Bayesian mode applies smooth Gaussian likelihood updates to the
posterior probabilities of candidate clusters surviving triangle
inequality pruning, rather than treating all surviving candidates equally.

## GAUSSIAN LIKELIHOOD
For each surviving candidate cluster i after measuring anchor cj:
  likelihood = exp(-(dfc - dcc)^2 / (2 * sigma^2))

where:
  dfc   = distance from current frame to measured anchor cj
  dcc   = inter-cluster distance between anchor cj and candidate i
  sigma = rlim * sigma_coeff (default 1.0, tunable via -soft_bayesian_sigma)

The exponential is approximated using a minimax polynomial on the
interval [0, 2], avoiding slow library exp() calls. Returns 0.0 for large
deviations (hard cutoff).

Candidates pruned by the triangle inequality (|dfc - dcc| > rlim) are
hard-set to 0.0, while surviving candidates have their posterior
probability scaled by the likelihood and renormalized.

## RATIONALE
Standard pruning is binary: surviving candidates retain their prior
probabilities regardless of whether they were close to the pruning
boundary. Soft Bayesian weights surviving candidates by how closely their
inter-cluster distance matches the observed measurement, accelerating
entropy and dynamic greedy convergence.

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
