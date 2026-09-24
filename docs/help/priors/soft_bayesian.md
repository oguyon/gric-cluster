# soft_bayesian

## ROLE
Target Selection Option

## FUNCTION
Performs smooth Gaussian likelihood fading on candidate probabilities
alongside metric pruning on distance threshold failure.

## ALGORITHM
When a distance evaluation fails (dfc > rlim), candidate clusters that
survive triangle inequality pruning have their posterior probabilities
updated by a Gaussian likelihood factor:

  likelihood = exp(-(dfc - dcc)^2 / (2 * sigma^2))

where:
  dfc   = measured distance to current anchor cj
  dcc   = inter-cluster distance between cj and candidate i
  sigma = rlim * sigma_coeff (default 1.0, tunable via -soft_bayesian_sigma)

The exponential is approximated using a minimax polynomial on [0, 2] for
speed, returning 0.0 for large deviations.

Candidates pruned by triangle inequality are set to 0.0, while surviving
candidates are faded smoothly according to their distance consistency.

## RATIONALE
Surviving candidates near the triangle inequality boundary are less
plausible than candidates whose expected distance closely matches dfc.
Soft Bayesian smoothly suppresses these marginal candidates, refining
posterior Shannon entropy for subsequent target scheduling.

Most beneficial when:
  - Clusters overlap geometrically
  - Distance measurements are noisy
  - rlim is close to inter-cluster spacing

Less useful when clusters are well-separated.

## INTERACTS WITH
-entropy or -gprob (modulates posterior probabilities used in target selection)

## SEE ALSO
- `-entropy`: Entropy-based target selection
- `-entropy_gate`: Entropy gating threshold
