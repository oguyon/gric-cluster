# fmatchb

## ROLE
Geometric Matching Parameter B

## FUNCTION
Factor at the pruning limit for gprob (Default: 0.5).

## EQUATION
See -h fmatcha for the full equation. When delta_dist
reaches 2*rlim, factor = b (default 0.5 = halve
probability). Beyond 2*rlim, factor drops to 0.

## REQUIRES
-gprob (has no effect without it)

## SEE ALSO
- `-gprob`: Use geometrical probability
- `-fmatcha`: Set fmatch parameter a
