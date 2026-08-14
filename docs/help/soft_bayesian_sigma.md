# soft_bayesian_sigma

## ROLE
Soft Bayesian Parameter

## FUNCTION
Sets the sigma coefficient for the Gaussian
likelihood in soft Bayesian mode
(Default: 1.0).

## EQUATION
sigma = rlim * sigma_coeff

Larger sigma_coeff = wider Gaussian =
slower probability decay = more tolerant
of distance mismatches.

Smaller sigma_coeff = narrower Gaussian =
faster elimination of non-matching
candidates, closer to hard pruning.

## USE
-soft_bayesian_sigma 0.5 (narrower)

## REQUIRES
-soft_bayesian (has no effect without it)

## SEE ALSO
- `-soft_bayesian`: Enable Soft Bayesian update
- `algorithm/soft_bayesian`: Soft Bayesian deep dive
