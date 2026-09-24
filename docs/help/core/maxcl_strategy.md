# maxcl_strategy

## ROLE
Memory Management Strategy

## FUNCTION
Determines behavior when the 'maxcl' limit is reached.

## OPTIONS
stop    : (Default) Exit program. Ensures dataset integrity.
discard : 'Cache Eviction'. Scans the oldest 'discard_frac' clusters and removes
          the one with the fewest visits. Useful for continuous monitoring.
merge   : Merges the two geometrically closest clusters (min d(c_i, c_j)).
          Computationally expensive (O(N^2) scan) but preserves information.

## USE
-maxcl 100 -maxcl_strategy discard

## ACTIVE WHEN
Cluster count reaches -maxcl.
Has no effect before that.

## SEE ALSO
- `-maxcl`: Max number of clusters
- `-discard_frac`: Fraction of clusters to discard
