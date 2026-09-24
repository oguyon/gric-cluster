# entropy_first_gate

## ROLE
Entropy Gate at Depth 0

## FUNCTION
Entropy gating threshold in bits for the first measurement
attempt (depth 0) of each frame (default: 4.0).  At depth 0, the
posterior is dominated by the static prior and greedy argmax is
near-optimal.  After at least one failed measurement, -entropy_gate
is used instead.

## USE
-entropy_first_gate 3.0 (more aggressive gating at depth 0)

## REQUIRES
-entropy

## SEE ALSO
- `-entropy_gate`: Gate threshold after depth 0
- `-entropy`: Entropy-based target selection
