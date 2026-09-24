# Rust Crate Reference

The `gric` Rust crate provides safe, zero-cost idiomatic bindings to `libgric`. It is published as
a dual-crate workspace:
* **`gric-sys`**: Unsafe raw FFI declarations generated from `<gric/gric.h>`.
* **`gric`**: Safe, high-level Rust wrapper providing RAII via Rust's `Drop` trait and slice ergonomics.

---

## 1. Adding to `Cargo.toml`

Add the dependency to your Rust project:

```toml
[dependencies]
gric = { path = "path/to/gric-cluster/bindings/rust/gric" }
```

Ensure `libgric.so` is discoverable via `LD_LIBRARY_PATH` or system library paths:

```bash
export LD_LIBRARY_PATH="/path/to/gric-cluster/build:$LD_LIBRARY_PATH"
```

---

## 2. Quick Start

```rust
use gric::{Clusterer, Config};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize with radius threshold 0.5
    let mut clusterer = Clusterer::new(3, Config::default().rlim(0.5))?;

    // 2. Stream incoming coordinate frames
    let frame1 = [0.0, 0.0, 0.0];
    let cid1 = clusterer.feed(&frame1)?;
    println!("Frame 1 -> Cluster {}", cid1);

    let frame2 = [0.1, 0.0, 0.0];
    let cid2 = clusterer.feed(&frame2)?;
    println!("Frame 2 -> Cluster {}", cid2);

    println!("Total clusters: {}", clusterer.num_clusters());

    // 3. Inspect discovered centroids
    let anchors = clusterer.anchors();
    for (i, anchor) in anchors.iter().enumerate() {
        println!("  Anchor {}: {:?}", i, anchor);
    }

    Ok(())
} // <-- Memory is automatically freed here via Rust's Drop trait!
```

---

## 3. Safety & Concurrency

### Guaranteed Leak-Freedom via `Drop`
The `Clusterer` struct wraps `*mut gric_cluster_t`. When `Clusterer` goes out of scope, its `Drop`
implementation calls `gric_cluster_destroy()`:

```rust
impl Drop for Clusterer {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { gric_sys::gric_cluster_destroy(self.raw); }
        }
    }
}
```

### Thread Safety (`Send`)
`Clusterer` implements `Send`, allowing instances to be moved across threads or spawned inside
worker thread pipelines.

---

## 4. Configuration Builder (`gric::Config`)

```rust
let config = gric::Config::default()
    .rlim(0.25)
    .max_clusters(500)
    .sq16(true)
    .ncpu(4);

let mut clusterer = gric::Clusterer::new(64, config)?;
```

---

## 5. API Reference

### `Clusterer::new(ndim: usize, config: Config) -> Result<Self, Error>`
Creates a clusterer with custom configuration.

### `Clusterer::simple(ndim: usize, rlim: f64) -> Result<Self, Error>`
Creates a clusterer with default parameters and specified radius.

### `Clusterer::feed(&mut self, coords: &[f64]) -> Result<i64, Error>`
Ingests a single vector slice. Throws `Error::DimMismatch` if `coords.len() != ndim`.

### `Clusterer::feed_batch(&mut self, flat_coords: &[f64], num_frames: usize) -> Result<Vec<i64>, Error>`
Ingests a contiguous flat slice of `num_frames * ndim` elements. Returns `Vec<i64>` with assigned
cluster labels.

### `Clusterer::anchors(&self) -> Vec<Vec<f64>>`
Returns a vector of cluster centroid vectors.

### `Clusterer::num_clusters(&self) -> usize`
Returns the active cluster count.

### `Clusterer::reset(&mut self) -> Result<(), Error>`
Resets clustering state.
