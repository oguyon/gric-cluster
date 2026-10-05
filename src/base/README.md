# Base Layer (Level 0)

## Purpose
Foundational primitives, system utilities, memory allocation wrappers, OpenMP threading
thresholds, binary serialization headers, and stream layouts.

## Architectural Layer
- **Level**: Level 0 (Foundation)
- **Public Headers**:
  - `base/sys/gric_mem.h`: Aligned buffer allocations and hugepages.
  - `base/sys/gric_omp.h`: OpenMP thread controls and scheduling constants.
  - `base/sys/gric_simd.h`: SIMD architecture detection and CPU feature queries.
  - `base/format/gric_bin_header.h`: Binary file header specifications.
  - `base/format/gric_bin_io.h`: Cluster and frame binary file serialization.
  - `base/format/gric_stream_layout.h`: Streaming shared memory buffer layout descriptors.
- **Permitted Dependencies**:
  - POSIX / C standard library headers only (`stdlib.h`, `stdio.h`, `string.h`, `stdint.h`, etc.).
  - No higher-level layers (`quant`, `accel`, `engine`, `ui`, `apps`, `adapters`).
