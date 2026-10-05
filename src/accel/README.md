# Hardware Acceleration (Level 1)

## Purpose
Hardware-specific acceleration primitives and GPU memory abstractions, including CUDA memory
managers, anchor storage, and inverted file (IVF) index primitives.

## Architectural Layer
- **Level**: Level 1 (Hardware Acceleration)
- **Public Headers**:
  - `accel/gpu/cuda_common.h`: CUDA runtime detection, device allocation, and error handling.
  - `accel/gpu/cuda_anchor_store.h`: GPU-resident anchor matrix stores.
  - `accel/gpu/cuda_ivf_index.h`: GPU inverted file index structures.
- **Permitted Dependencies**:
  - Level 0 (`base/*`).
  - CUDA Runtime / Driver libraries (`cuda_runtime.h`, `cuda.h`).
  - May not depend on `engine/`, `ui/`, `apps/`, or `adapters/`.
