---
description: Inspect compiler assembly outputs.
---

# Inspect Machine Code

To inspect the generated assembly for hot loop vectorization:
1. Compile the target file to assembly:
   ```bash
   gcc -S -O3 -march=native -fverbose-asm -Isrc src/framedistance.c -o framedistance.s
   ```
2. Inspect the output `.s` file for vector instructions (e.g. `ymm` or `zmm` registers, `vmulpd`, `vaddpd`).
3. Confirm that auto-vectorization did not fall back to scalar instructions.
