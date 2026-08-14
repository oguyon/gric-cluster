# fmatcha

## ROLE
Geometric Matching Parameter A

## FUNCTION
Reward factor for exact geometric matches in gprob (Default: 2.0).

## EQUATION
ratio  = |delta_dist| / rlim
factor = a - (a - b) * min(ratio, 2) / 2
         Returns 0.0 if ratio > 2.0 (hard cutoff).

This is a multiplicative scaling factor (not a probability):
  ratio=0 (perfect match)   -> factor = a (2.0 = boost)
  ratio=2 (max separation)  -> factor = b (0.5 = penalty)
  ratio>2                   -> factor = 0 (kills candidate)

## REQUIRES
-gprob (has no effect without it)

## SEE ALSO
- `-gprob`: Use geometrical probability
- `-fmatchb`: Set fmatch parameter b
