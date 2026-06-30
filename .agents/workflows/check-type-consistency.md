---
description: Audit type consistency in math operations and loops.
---

# Check Type Consistency

Verify that:
- [ ] Loop index types match the bound types.
- [ ] Math functions match the datatypes (e.g. `sqrt` for double, `sqrtf` for float).
- [ ] Implicit casting doesn't lead to precision loss or overflow.
- [ ] Multiplications that can overflow are cast to wider types (e.g. `(uint64_t)width * height`).
