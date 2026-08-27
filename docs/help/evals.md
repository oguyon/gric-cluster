# evals

## ROLE
Output and Trace Telemetry Control

## FUNCTION
`-evals` (alias: `-trace`) controls whether distance evaluation history and per-frame
candidate traces are recorded and written to `frame_evals.txt` in the output directory
(Default: Enabled).

To disable evaluation tracing and reduce memory/disk overhead on long runs, pass `-no_evals`.

## OUTPUT FORMAT
When enabled, `frame_evals.txt` logs per-frame distance measurements:
  FrameIndex  TargetClusterIndex  ComputedDistance  DecisionFlag

## USE
```bash
# Explicitly enable evaluation logging (default)
gric-cluster 0.5 input.txt -evals

# Disable evaluation logging to save memory
gric-cluster 0.5 input.txt -no_evals
```

## SEE ALSO
- `-membership`: Frame membership output
- `-outdir`: Output directory specification
- `-no_evals`: Suppress evaluation history output
