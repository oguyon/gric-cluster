---
name: optimize-compute-function
description: Optimization checklist for hot clustering and math loops.
---

# Optimize Compute Functions

Use this methodology to audit and optimize performance-critical loops.

## Optimization Checklist
- [ ] **No dynamic allocations:** Verify that no `malloc` or `free` calls occur inside the hot loop.
- [ ] **Pointer restrict:** Add `restrict` (or `__restrict__`) to pointers in the hot path to assist vectorization.
- [ ] **Float promotions:** Avoid double promotions; use correct float suffixes (`f`) and functions (`sqrtf` vs `sqrt`).
- [ ] **Check loop indexes:** Ensure loop variables and bounds use matching types.
- [ ] **Avoid transcendentals:** Inline small calculations and replace `pow(x, 2)` with `x * x`.
