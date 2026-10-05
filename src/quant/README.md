# Quantization Codecs (Level 1)

## Purpose
Vector compression, quantization algorithms, lookup tables, and distance microkernels for
high-throughput dimensional reduction and accelerated candidate scoring.

## Architectural Layer
- **Level**: Level 1 (Algorithms & Codecs)
- **Public Headers**:
  - `quant/scalar_quant.h`: Scalar Quantization (SQ8, SQ16) encoders, decoders, and fastscan.
  - `quant/eq16_quant.h`: Exponential/enhanced 16-bit quantization and asymmetric filters.
  - `quant/residual_quant.h`: Residual Quantization (RQ8) multi-stage encoders.
  - `quant/product_quant.h`: Product Quantization (PQ) codebooks and ADC kernels.
  - `quant/rabit_quant.h`: RaBitQ 1-bit / multi-bit randomized binary quantization.
  - `quant/e8_lattice.h`: 8-dimensional Gosset lattice \(E_8\) quantizers and fast decoders.
  - `quant/quant_memo.h`: Distance memoization and lookup table caches.
- **Permitted Dependencies**:
  - Level 0 (`base/*`).
  - Standard C and math libraries (`math.h`).
  - May not depend on `accel/`, `engine/`, `ui/`, `apps/`, or `adapters/`.
