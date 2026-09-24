# C API Reference

The C API is declared in `<gric/gric.h>` and provided by `libgric.so` (shared) and `libgric.a`
(static). It serves as the official ABI and foreign function interface (FFI) for all language
bindings.

---

## 1. Header & Linkage

Include the public header in your C source files:

```c
#include <gric/gric.h>
```

Compile and link against `libgric` using `pkg-config` or standard compiler flags:

```bash
gcc main.c -I/usr/local/include -L/usr/local/lib -lgric -lm -o my_app
```

---

## 2. Core Types & Return Codes

### `gric_status_t`
All mutating library functions return an explicit status code:

| Status Code | Value | Description |
| :--- | :--- | :--- |
| `GRIC_SUCCESS` | `0` | Operation completed successfully |
| `GRIC_ERR_INVALID_PARAM` | `-1` | Null pointer or invalid argument passed |
| `GRIC_ERR_OUT_OF_MEMORY` | `-2` | Failed memory allocation |
| `GRIC_ERR_DIM_MISMATCH` | `-3` | Vector dimensions do not match session `ndim` |
| `GRIC_ERR_CAPACITY` | `-4` | Maximum cluster limit reached in stop mode |
| `GRIC_ERR_GENERIC` | `-5` | Internal processing error |

### `gric_cluster_config_t`
Structure defining clustering and pruning parameters:

```c
typedef struct
{
    double rlim;                   /**< Clustering radius limit */
    int    maxnbclust;             /**< Maximum cluster capacity (0 = dynamic) */
    long   maxnbfr;                /**< Max frames expected (buffer sizing) */
    double tm_mixing_coeff;        /**< Temporal transition mixing weight */
    int    use_double;             /**< 1 = double precision, 0 = single */
    int    use_sq16;               /**< 1 = enable 16-bit scalar quantization */
    int    use_eq16;               /**< 1 = enable 16-bit E8 lattice quantization */
    int    te4_mode;               /**< 1 = 4-point triangle inequality pruning */
    int    te5_mode;               /**< 1 = 5-point inequality pruning */
    int    entropy_mode;           /**< 1 = entropy-guided target evaluation */
    double entropy_gate_bits;      /**< Entropy information gain threshold */
    int    pred_mode;              /**< Temporal Markov prediction mode */
    int    gprob_mode;             /**< Geometric probability guidance mode */
    int    soft_bayesian_mode;     /**< Soft Bayesian likelihood weighting */
    int    sparse_dcc_mode;        /**< Sparse DCC matrix mode */
    int    sparse_dcc_extra_evals; /**< Extra DCC evals in sparse mode */
    int    maxcl_strategy;         /**< 0=stop, 1=discard lowest count */
    double discard_fraction;       /**< Fraction to discard on eviction */
    int    ncpu;                   /**< Thread count (0 = default/OpenMP) */
} gric_cluster_config_t;
```

---

## 3. Function Reference

### Configuration & Initialization

#### `gric_cluster_config_default`
```c
gric_status_t gric_cluster_config_default(
    gric_cluster_config_t *cfg);
```
Populates `cfg` with standard recommended parameter defaults (`rlim = 0.5`, `maxnbclust = 256`,
`te4_mode = 1`, `use_double = 1`).

#### `gric_cluster_create`
```c
gric_cluster_t *gric_cluster_create(
    const gric_cluster_config_t *cfg,
    size_t                       ndim);
```
Allocates and initializes an in-memory clustering session. Returns an opaque pointer `gric_cluster_t*`
or `NULL` on allocation failure.

#### `gric_cluster_create_simple`
```c
gric_cluster_t *gric_cluster_create_simple(
    size_t ndim,
    double rlim);
```
Convenience constructor with standard defaults and a custom radius limit.

---

### Frame Ingestion

#### `gric_cluster_feed_frame`
```c
gric_status_t gric_cluster_feed_frame(
    gric_cluster_t *ctx,
    const double   *coords,
    int64_t        *out_cluster_id);
```
Streams a single coordinate vector of length `ndim`. Writes the assigned cluster ID ($\ge 0$)
to `*out_cluster_id`. If distance to all existing anchors exceeds $r_{\text{lim}}$, a new cluster
anchor is established.

#### `gric_cluster_feed_batch`
```c
gric_status_t gric_cluster_feed_batch(
    gric_cluster_t *ctx,
    const double   *coords_flat,
    size_t          num_frames,
    int64_t        *out_cluster_ids);
```
Processes a contiguous row-major buffer of `num_frames` vectors (total length `num_frames * ndim`).
Writes assigned cluster IDs to `out_cluster_ids`.

---

### Querying Results

#### `gric_cluster_get_num_clusters`
```c
int64_t gric_cluster_get_num_clusters(
    const gric_cluster_t *ctx);
```
Returns the number of active clusters discovered so far.

#### `gric_cluster_get_anchors`
```c
int64_t gric_cluster_get_anchors(
    const gric_cluster_t *ctx,
    double               *out_coords,
    int                  *out_members,
    size_t                max_anchors);
```
Exports discovered centroid vectors into `out_coords` (capacity: `max_anchors * ndim`) and
member counts into `out_members`. Returns the actual number of anchors copied.

#### `gric_cluster_get_dcc`
```c
gric_status_t gric_cluster_get_dcc(
    const gric_cluster_t *ctx,
    double               *out_dcc,
    size_t                K);
```
Copies the square $K \times K$ distance-between-cluster-centers matrix into `out_dcc`.

---

### Cleanup & Reset

#### `gric_cluster_reset`
```c
gric_status_t gric_cluster_reset(
    gric_cluster_t *ctx);
```
Clears all discovered clusters and resets counters while keeping scratch buffer allocations.

#### `gric_cluster_destroy`
```c
void gric_cluster_destroy(
    gric_cluster_t *ctx);
```
Releases all heap allocations, distance matrices, and the session context. Safe to pass `NULL`.

---

## 4. Complete C Example

```c
#include <gric/gric.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    printf("GRIC C Library Version: %s\n", gric_version());

    // 1. Initialize configuration
    gric_cluster_config_t cfg;
    gric_cluster_config_default(&cfg);
    cfg.rlim = 0.5;

    // 2. Create clustering session (3-dimensional vectors)
    gric_cluster_t *ctx = gric_cluster_create(&cfg, 3);
    if (!ctx)
    {
        fprintf(stderr, "Failed to allocate clusterer\n");
        return 1;
    }

    // 3. Ingest frames
    double sample1[3] = {0.0, 0.0, 0.0};
    int64_t cid1 = -1;
    gric_cluster_feed_frame(ctx, sample1, &cid1);
    printf("Sample 1 -> Cluster %ld\n", cid1);

    double sample2[3] = {0.1, 0.0, 0.0};
    int64_t cid2 = -1;
    gric_cluster_feed_frame(ctx, sample2, &cid2);
    printf("Sample 2 -> Cluster %ld\n", cid2);

    printf("Total clusters: %ld\n", gric_cluster_get_num_clusters(ctx));

    // 4. Free session
    gric_cluster_destroy(ctx);
    return 0;
}
```
