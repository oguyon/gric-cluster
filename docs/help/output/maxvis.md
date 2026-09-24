# maxvis

## ROLE
gprob History Limit

## FUNCTION
Max number of recent visitors to track per cluster (Default: 1000).

## DETAILS
To compute gprob, the algorithm scans past frames ('visitors') of candidate clusters.
This limits how many past frames are stored/scanned to maintain performance.

## REQUIRES
-gprob (has no effect without it)

## SEE ALSO
- `-gprob`: Use geometrical probability
