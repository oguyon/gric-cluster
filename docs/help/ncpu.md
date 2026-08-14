# ncpu

## ROLE
Parallel Processing

## FUNCTION
Sets the number of OpenMP threads (Default: 1).

## IMPLEMENTATION
Used to parallelize the 'pruning' loops. When checking if a candidate cluster
is valid, the algorithm checks triangle inequalities against all other clusters.
This loop is split across 'ncpu' threads. Also used in batch distance
calculations.

## USE
-ncpu 4

## SEE ALSO
- `-te4`: Use 4-point triangle inequality pruning
- `-te5`: Use 5-point triangle inequality pruning
