# entropy_fast

## ROLE
Popcount-Only Surrogate Mode

## WHAT IS POPCOUNT
GRIC maintains a consistency mask: a bitfield where bit k is set if
cluster k is geometrically consistent with the current frame
(i.e., not yet ruled out by triangle inequality bounds).

For a candidate target c_i, the popcount score estimates how many
clusters would survive after measuring c_i.  For each hypothesis c_j
("what if c_j is the true cluster?"), a bitwise AND of
consistency_mask[c_i][c_j] with the active cluster mask gives
the set of clusters still consistent under that scenario.  The CPU
instruction popcount counts those set bits in a single cycle.

Summing over a sample of hypotheses yields the popcount score:
  Score(c_i) = sum_j popcount(mask[c_i][c_j] & active_mask)

Low score = measuring c_i leaves few survivors = high discriminative
power.  Since Shannon entropy is roughly log2(support size),
minimizing support size is a first-order approximation of
minimizing entropy, but computed entirely with fast bitwise
operations instead of floating-point logarithms.

## FUNCTION
Skips Shannon entropy evaluation (stage 4 of the entropy pipeline)
and returns the candidate with the lowest popcount score from
stage 2 directly.  See -h entropy for the full pipeline description.

## RATIONALE
Shannon eval is O(T*H*W) and dominates Step 3b cost.  The popcount
heuristic provides near-identical target selection quality at a
fraction of the CPU cost.

## USE
-entropy -entropy_fast

## REQUIRES
-entropy

## SEE ALSO
- `-entropy`: Full pipeline description
- `-entropy_gate`: Gating threshold in bits
