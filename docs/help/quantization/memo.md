# memo

## ROLE
Quantized Distance Memoization Cache

## FUNCTION
Caches previously evaluated quantized distance metrics during candidate traversal to avoid
recomputing redundant inner products.

## DETAILS
- When processing dense clusters or multi-candidate test sequences, candidate clusters often share
  common anchor pivots or intermediate quantized dot products.
- The memoization table provides $O(1)$ lookup for recently evaluated candidates within the current
  frame's traversal lifecycle.

## OPTIONS
- `-memo`: Enable quantized distance memoization cache (default: enabled with SQ16/EQ16)
- `-no-memo`: Disable memoization cache

## SEE ALSO
- `-sq16`: 16-bit scalar quantization filtering
- `-eq16`: 16-bit E8 lattice quantization filtering
