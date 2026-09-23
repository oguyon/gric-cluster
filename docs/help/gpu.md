# gpu

## ROLE
CUDA GPU Hardware Acceleration

## FUNCTION
Dispatches distance evaluations and batched GEMM operations to NVIDIA CUDA GPUs via cuBLAS and
Tensor Core hardware pipelines.

## DETAILS
- Offloads intensive metric calculations to GPU memory when built with `-DENABLE_CUDA=ON`.
- Supports micro-batching (`-gpu-batch-size <size>`) to balance throughput vs. latency.
- Fuses metric threshold checking directly on the GPU without intermediate host synchronization.

## OPTIONS
- `-gpu`: Enable CUDA GPU acceleration
- `-cpu`: Force CPU execution (disable GPU acceleration)
- `-gpu-device <id>`: Select CUDA GPU device index (default: 0)
- `-gpu-batch-size <N>`: Set micro-batch query size for GPU execution (default: auto)
- `-gpu-micro-batch <N>`: Alias for `-gpu-batch-size`
- `-gpu-pass1`: Enable GPU acceleration during Pass 1 ingestion
- `-gpu-pass2`: Enable GPU acceleration during Pass 2 / JTF reassignment

## SEE ALSO
- `-ncpu`: Number of OpenMP CPU worker threads
- `-batch-dist`: Multi-vector CPU SIMD batch distance evaluation
