# tm

## ROLE
Transition Matrix Mixing

## FUNCTION
Uses transition history to predict next cluster.

## USE
-tm <coeff> (0.0 to 1.0)

## ALGORITHM
Mixes the standard probability with the transition probability:
  P_final = (1-coeff)*P_standard + coeff * P(next|prev)
where P(next|prev) is derived from the count of transitions prev->next.

## SEE ALSO
- `-gprob`: Use geometrical probability
- `-pred`: Prediction with pattern detection
- `-tm_out`: Enable transition_matrix.txt output
