# soft_bayesian

## ROLE
Target Selection Option

## FUNCTION
Performs smooth Bayesian updates on candidates rather than hard pruning on
distance threshold failure.

## ALGORITHM
When a distance evaluation fails (dist > rlim),
the target's probability is updated by multiplying
it with a Gaussian-like likelihood function:

  likelihood = exp( -(d_measured - d_anchor)^2
                    / (2 * sigma^2) )

where sigma = rlim * sigma_coeff (default 1.0,
tunable via -soft_bayesian_sigma).

The exponential is approximated using a minimax
polynomial on [0, 2] for speed, returning 0.0 for
large deviations.

This retains near-miss candidates that hard binary
pruning would discard prematurely, leading to faster
information gain convergence.

## RATIONALE
Hard pruning is binary: a candidate is either alive
or dead. When clusters are close together, a small
measurement error can wrongly eliminate the true
cluster. Soft Bayesian gradually fades candidates,
making the algorithm robust to near-boundary
measurements.

Most beneficial when:
  - Clusters overlap geometrically
  - Distance measurements are noisy
  - rlim is close to inter-cluster spacing

Less useful when clusters are well-separated.

## REQUIRES
-gprob (has no effect without it)

## SEE ALSO
- `-entropy`: Entropy-based target selection
- `-entropy_gate`: Entropy gating threshold
