# entropy_min_prob

## ROLE
Entropy Hypothesis Filter

## FUNCTION
Minimum probability for a cluster to be considered as a hypothesis
in the entropy evaluation loop (default: 0.001).  Clusters below
this threshold are skipped.  A dynamic floor of 1% of the leader's
probability is also applied.

## USE
-entropy_min_prob 0.01

## REQUIRES
-entropy

## SEE ALSO
- `-entropy`: Entropy-based target selection
