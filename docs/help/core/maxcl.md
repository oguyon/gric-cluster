# maxcl

## ROLE
Resource Limiting

## FUNCTION
Sets the maximum number of clusters allowed (Default: 1000).

## IMPLEMENTATION
Defines the size of static arrays (clusters, visitors) and the N*N distance
cache (dccarray). Affects memory usage (O(N^2) for dccarray).
When this limit is reached, the behavior is controlled by -maxcl_strategy.

## USE
-maxcl 5000

## INTERACTS WITH
- -maxcl_strategy: What happens at the limit
- -sparse_dcc: Avoids O(maxcl^2) DCC

## SEE ALSO
- `-maxcl_strategy`: Strategy when maxcl reached
- `-discard_frac`: Fraction of clusters to discard
