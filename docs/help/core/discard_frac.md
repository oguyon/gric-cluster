# discard_frac

## ROLE
Discard Strategy Parameter

## FUNCTION
Fraction of clusters to consider for discarding (Default: 0.5).

## IMPLEMENTATION
Discarding avoids killing a brand new cluster that
hasn't had time to accumulate visitors. This limits the search
to the first N * discard_frac clusters by index (i.e. the
oldest by creation order). Among those, the one with the
fewest total visitors is removed.

## USE
-discard_frac 0.2 (Only consider oldest 20%)

## REQUIRES
-maxcl_strategy discard
  Has no effect with other strategies.

## SEE ALSO
- `-maxcl`: Max number of clusters
- `-maxcl_strategy`: Strategy when maxcl reached
