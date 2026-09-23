# batch_dist

## ROLE
Multi-Vector SIMD Batch Distance Computation

## FUNCTION
Evaluates Euclidean distance between a query vector and up to 8 candidate vectors concurrently in
a single unrolled SIMD pass.

## DETAILS
- Traditional distance functions evaluate one candidate vector at a time ($1 \times 1$).
- Batched distance evaluation loads query vector components into broadcast registers and accumulates
  squared differences across 8 candidate pointers in parallel ($1 \times 8$).
- Improves L1 instruction cache efficiency and maximizes FMA (Fused Multiply-Add) throughput on AVX2
  and AVX-512 hardware.

## OPTIONS
- `-batch-dist`: Enable multi-vector SIMD batch distance evaluation (default: enabled)
- `-no-batch-dist`: Disable batch distance evaluation (use single-vector kernels)

## SEE ALSO
- `-ncpu`: Number of OpenMP worker threads
- `-double`: Run computations in 64-bit double precision
