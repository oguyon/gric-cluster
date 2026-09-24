# gprob

## ROLE
Geometric Probability (Trajectory Learning)

## FUNCTION
Uses historical distance patterns to predict cluster membership.

## ALGORITHM
For a new frame 'm', the algorithm looks at recent frames 'k' that share distance
measurements to common clusters. It computes a 'Geometrical Match Coefficient'
based on how similar the distance vector of 'm' is to 'k'.
If 'm' looks like 'k' geometrically, the probability of 'm' belonging to the same
cluster as 'k' is boosted.

## USE
-gprob (Highly recommended for continuous drift/trajectory data)

## TUNED BY
- -fmatcha <val>: Match reward (default: 2.0)
- -fmatchb <val>: Pruning factor (default: 0.5)
- -maxvis <val>: History depth (default: 1000)

## ENHANCED BY
- -entropy: Optimal measurement scheduling
- -soft_bayesian: Smoother probability updates

## SEE ALSO
- `-pred`: Prediction with pattern detection
- `-tm`: Transition matrix mixing
