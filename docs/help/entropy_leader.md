# entropy_leader

## ROLE
Entropy Target Selection Optimization

## FUNCTION
`-entropy_leader` enables a short-circuit bypass in the entropy target selection engine.
When active, if a single candidate cluster's posterior probability meets or exceeds the
cutoff threshold (configured via `-entropy_leader_cutoff`, default: `0.50`), the engine
immediately measures that dominant leader without performing the expensive expected
Shannon entropy minimization or popcount hypothesis evaluations.

## PARAMETERS
- `-entropy_leader`: Enable dominant leader shortcut mode (default: disabled).
- `-entropy_leader_cutoff <val>`: Probability threshold (float between 0.0 and 1.0,
  default: `0.50`).

## RATIONALE
When a single candidate has a very high posterior probability (P >= 0.50), it is
overwhelmingly likely to be the matching cluster or provide an immediate match.
Calculating expected entropy across all other hypotheses in this scenario yields
negligible information gain while consuming CPU cycles.

## USE
```bash
# Enable leader bypass with default 0.50 cutoff
gric-cluster 0.5 input.txt -entropy -entropy_leader

# Custom leader cutoff at 70% confidence
gric-cluster 0.5 input.txt -entropy -entropy_leader -entropy_leader_cutoff 0.70
```

## REQUIRES
`-entropy`

## SEE ALSO
- `-entropy`: Shannon entropy target selection
- `-entropy_gate`: Adaptive entropy gating threshold
- `-entropy_fast`: Popcount-only fast surrogate mode
