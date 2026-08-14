# algorithm/gating

## ADAPTIVE ENTROPY GATING
Adaptive entropy gating dynamically decides whether to execute the
computationally intensive expected Shannon entropy calculation or
greedily select the highest-probability candidate. This is controlled
by comparing the current Shannon entropy `H_current` against a
depth-dependent threshold.

## REQUIRES
-entropy (gating is a sub-mechanism of entropy-based
target selection; has no effect without it)

## THRESHOLD LOGIC
The gating threshold depends on the measurement depth `meas_idx`:

1. First Measurement (`meas_idx == 0`)
   Uses `entropy_first_gate_bits` (default 4.0 bits).
   Since no measurements have been attempted yet, the distribution
   is dominated by static priors. The greedy `argmax_p` is
   near-optimal, and entropy calculation is bypassed.

2. Subsequent Measurements (`meas_idx >= 1`)
   Uses `entropy_gate_bits` (default 2.0 bits).
   If uncertainty drops below this limit (fewer than ~4 effective
   candidates), the scheduler falls back to the greedy target.

## RATIONALE
Full entropy evaluation is expensive but only
valuable when uncertainty is high. When the
posterior is already concentrated (e.g., gprob has
identified a strong match), greedy selection is
near-optimal. Gating avoids paying the entropy
cost in these easy cases, typically 60-80%% of all
target selections.

## SOURCE IMPLEMENTATION
Implemented in `select_next_measurement_target_entropy()` inside
src/gric-cluster/steps/select_next_measurement_target.c.

## SEE ALSO
- `algorithm/entropy`: Details on Shannon entropy target selection
- `algorithm`: Overview of the GRIC algorithm
