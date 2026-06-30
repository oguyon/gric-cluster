---
description: Perform compile verification on gric-cluster.
---

# Compile Test

To verify the build:
1. Navigate to the build directory (create if it doesn't exist):
   ```bash
   mkdir -p build && cd build
   ```
2. Configure using CMake:
   ```bash
   cmake ..
   ```
3. Compile with multiple jobs:
   ```bash
   make -j$(nproc)
   ```
4. Verify there are no warnings or errors.
