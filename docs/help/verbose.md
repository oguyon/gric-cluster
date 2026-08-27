# verbose

## ROLE
Diagnostics and Logging Level

## FUNCTION
Controls the verbosity of console output and logging during clustering execution:
- `-verbose`: Enables level 1 verbose output, printing per-step diagnostic summaries.
- `-veryverbose`: Enables level 2 detailed debugging output, including candidate probabilities
  and geometric pruning details for every frame.

## USE
```bash
# Standard verbose output
gric-cluster 0.5 input.txt -verbose

# High-detail debugging output
gric-cluster 0.5 input.txt -veryverbose
```

## SEE ALSO
- `-progress`: Progress reporting interval
