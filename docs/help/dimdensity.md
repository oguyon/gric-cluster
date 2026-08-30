# dimdensity

## OVERVIEW
`gric-dimdensity` estimates the local intrinsic dimension (LID) and probability
density around each sample from k-nearest neighbor (k-NN) distance matrices.

## ALGORITHMS
- **Local Intrinsic Dimension**: Levina & Bickel Maximum Likelihood Estimation (MLE)
  with finite-sample unbiased correction:
  d_hat = (k - 2) / sum_{j=1}^{k-1} [ln(r_k) - ln(r_j)]
- **Mack-Rosenblatt Density**: Variable-bandwidth k-NN density estimation corrected
  for local manifold geometry:
  f_hat(x) = (k - 1) / [N * V_{d_hat} * (r_k)^{d_hat}]
  where V_d is the volume of the d-dimensional unit Euclidean ball.

## OPTIONS
- `-k <int>`: Number of nearest neighbors to evaluate (default: all available).
- `-kmin <int>`, `-kmax <int>`, `-range`: Enable multi-scale range averaging.
- `-classic`: Use classic (k-1) MLE instead of (k-2) unbiased.
- `-kernel <uniform|epanechnikov|gaussian>`: Select density kernel weighting.
- `-o <path>`: Output destination path (.txt, .bin, or .fits).
- `-json`: Output structured JSON report.

## EXAMPLES
```bash
# Basic dimension and density estimation
gric-dimdensity cluster_out/ -k 15

# Multi-scale smoothed estimation with Epanechnikov kernel
gric-dimdensity knn_distances.bin -kmin 5 -kmax 20 -range -kernel epanechnikov

# Output JSON summary
gric-dimdensity cluster_out/ -k 15 -json
```
