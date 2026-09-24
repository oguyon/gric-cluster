# entropy_gate

## ROLE
Entropy Adaptive Gating

## FUNCTION
Shannon entropy threshold (in bits) below which entropy mode falls back to
greedy selection (default: 2.0).

## RATIONALE
Full entropy evaluation is O(T*H*W) per target selection. When gprob has
already concentrated the probability on a few clusters, greedy selection is
near-optimal and the expensive evaluation can be skipped. A threshold of
2.0 means roughly <=4 effective candidates.

## USE
-entropy_gate 1.5 (more aggressive gating)

## REQUIRES
-entropy (has no effect without it)

## SEE ALSO
- `-entropy`: Entropy-based target selection
- `-soft_bayesian`: Soft Bayesian update
