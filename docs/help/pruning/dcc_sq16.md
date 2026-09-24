# dcc_sq16

## ROLE
16-Bit Quantized Inter-Cluster Distance Matrix

## FUNCTION
Stores the pairwise inter-cluster distance matrix (`dcc`) in 16-bit scalar quantized format
(`dcc.sq16.bin`), cutting memory footprint in half.

## DETAILS
- For datasets generating large numbers of clusters ($K \ge 10,000$), storing the full $K \times K$
  matrix in 32-bit float requires $4 K^2$ bytes (400 MB for $K=10k$, 1.6 GB for $K=20k$).
- `-dcc-sq16` compresses each entry into a `uint16_t` relative to the maximum observed diameter,
  halving matrix storage while maintaining sufficient precision for triangular pruning bounds.

## OPTIONS
- `-dcc-sq16`: Enable 16-bit scalar quantized DCC matrix storage
- `-no-dcc-sq16`: Store DCC matrix in standard 32-bit floating point format (default)

## SEE ALSO
- `-sparse_dcc`: Dynamic interval bounding to eliminate $O(K^2)$ matrix memory entirely
- `-sq16`: 16-bit scalar quantization filtering
