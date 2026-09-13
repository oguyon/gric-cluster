# dprob

## ROLE
Cluster Probability Update (Recency Bias)

## FUNCTION
Amount added to a cluster's probability when a frame is assigned
to it (Default: 0.01).

## ALGORITHM
The algorithm maintains a probability distribution P(c) over all clusters.
When frame 'f' is assigned to cluster 'c_k':
  P(c_k) = P(c_k) + dprob
Then all probabilities are re-normalized to sum to 1.0.
This creates a 'recency bias': active clusters rise to the top of the
search list, minimizing the number of distance calculations needed to find
a match.

When sequence prediction (-pred or -predf) is active, this standard
update is replaced by +0.3 on the assigned cluster followed by adding a
uniform floor of 0.2/K across all clusters.

## USE
-dprob 0.05 (Stronger bias, faster adaptation to changing scenes)

## SEE ALSO
- `-gprob`: Use geometrical probability
- `-pred`: Sequence prediction
- `-maxcl`: Max number of clusters
