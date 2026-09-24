# Code Style & Quality Guide

The **GRIC** codebase follows strict C17 coding conventions inspired by the Linux kernel
guidelines, supplemented by explicit safety rules for real-time SIMD execution.

---

## 1. Line Length & Formatting

* **Maximum Line Length**: Strictly **100 characters** for C source files (`.c`, `.h`),
  documentation (`.md`), and build scripts.
* **No Artificial Fragmentation**: If an expression or function call fits comfortably within
  100 columns, write it on a single line. Do not split short expressions across multiple lines.
* **Brace Style: Allman**: Opening braces must always be placed on a new line for function
  definitions and control flow statements (`if`, `for`, `while`, `do`, `switch`).

```c
/* Standard Allman control flow */
if (ptr == NULL)
{
    return -1;
}

for (uint64_t ii = 0; ii < nelement; ii++)
{
    out[ii] = in[ii] * 2.0f;
}
```

---

## 2. Function Parameter Alignment

Multi-line function prototypes and definitions must use **column-aligned** parameter names:

1. Each parameter on its own line, indented by 4 spaces.
2. The type token (including any `*` prefix or `const`/`restrict` qualifier) is padded with
   spaces so all parameter names start at the exact same column within that prototype.
3. Determine alignment column by the longest type prefix.

### Prototype Alignment Example

```c
/* Column-aligned multi-line definition */
int cluster_core_run_pass1_loop(
    ClusterConfig *config,
    ClusterState  *state,
    InputReader   *reader,
    long           maxnbfr,
    int           *prev_assigned)
{
    // ...
}
```

---

## 3. Kernel-Doc Documentation Standards

Every public and non-trivial static function must be documented using Kernel-Doc formatting.

### Required Structure

* **Summary Line**: `function_name() - One-line action summary.`
* **Parameter List**: `@param_name: Detailed role and valid range.`
* **Purpose & Context**: An explicit **`Purpose & Context ("What is this used for?"):`**
  section explaining why the function exists, which callers invoke it, and what problem it solves.
* **Return Section**: `Return: Return value specification and error codes.`

### Complete Kernel-Doc Example

```c
/**
 * cluster_step_search_loop() - Evaluate candidates against current frame.
 * @config:     Clustering configuration parameters.
 * @state:      Active clustering runtime state and clusters.
 * @frame:      Current input coordinate frame.
 * @candidates: Output array of candidate clusters to evaluate.
 * @num_cand:   Number of candidates identified for evaluation.
 *
 * Purpose & Context ("What is this used for?"):
 * Executes the core metric-pruned distance evaluation loop for a newly ingested
 * frame. Evaluates Euclidean distances against candidate cluster anchors while
 * applying TE4, TE5, and DCC lower bounds to prune redundant distance calculations.
 *
 * Return: Index of closest cluster within rlim threshold, or -1 if no cluster qualifies.
 */
int cluster_step_search_loop(
    const ClusterConfig *config,
    ClusterState        *state,
    const Frame         *frame,
    Candidate           *candidates,
    int                  num_cand)
{
    // ...
}
```

---

## 4. Control Flow & Scope Management

* **Early Exit**: Prefer early return over deeply nested indentation blocks.
* **RAII & Cleanup**: Allocate resources in forward order; release in reverse order using the
  canonical `goto cleanup` pattern.
* **Scoping Blocks**: Use local `{ }` blocks to limit variable lifecycles. If a scope exceeds
  10 lines, append a closing comment:

```c
{
    ClusterScratch *s = &state->scratch;
    memset(s->dcc_min, 0, (size_t)N * N * sizeof(double));
    // ... computation ...
} // ClusterScratch scope
```

* **Loop Index Scope**: Always declare loop variables inside the loop statement
  (`for (int ii = 0; ...)`), unless the final index value is required after the loop.

---

## 5. Memory & Performance Rules

* **Zero Inner-Loop Allocation**: Never call `malloc()`, `calloc()`, or `realloc()` inside
  inner clustering loops (`run_clustering`, `cluster_frame`) or distance calculation functions.
* **SIMD Loop Bounds**: Loop indices must match the integer type of the bound (e.g. `uint64_t`
  index for `uint64_t` count) to enable SIMD auto-vectorization.
* **Pointer Non-Aliasing**: Use `restrict` on non-aliasing pointer arguments in compute loops.
* **Strict Float Math**: Use `sqrtf()`, `fabsf()`, and float literals (`0.5f`) in single-precision
  paths to prevent implicit double promotions.
