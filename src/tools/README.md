# Developer Tools & Utilities (Level 3)

## Purpose
Auxiliary developer tools, synthetic dataset generators, standalone compression benchmarks,
and data converters.

## Architectural Layer
- **Level**: Level 3 (Tools)
- **Subdirectories**:
  - `tools/benchmarks/`: Standalone micro-benchmarks (`bench_simd_scaling`,
    `bench_sq16_compression`).
  - `tools/generators/`: Synthetic test data generators (`gen_bouncing_balls`, `gen_asteroid`).
  - `tools/converters/`: Data conversion tools (`ascii2bin`, `bin2ascii`).
- **Permitted Dependencies**:
  - Level 0 (`base/*`), Level 1 (`quant/*`), Level 2 (`engine/*`), Level 3 (`ui/*`).
