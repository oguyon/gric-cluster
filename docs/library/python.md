# Python Package Reference

The `gric` Python package provides fast, zero-copy, in-memory clustering powered by `libgric`.
It supports both single-sample online streaming (for telemetry and camera feeds) and batch
processing compatible with the **scikit-learn** API.

---

## 1. Installation

Install the package directly from source:

```bash
cd gric-cluster
pip install python/
```

Or make the library directly importable in development:

```bash
export PYTHONPATH="/path/to/gric-cluster/python:$PYTHONPATH"
```

---

## 2. Quick Start

```python
import gric
import numpy as np

# 1. Create a 3-dimensional clusterer with radius limit 0.5
clusterer = gric.Clusterer(ndim=3, rlim=0.5)

# 2. Feed a single streaming frame
point = np.array([0.1, 0.2, 0.0])
cid = clusterer.feed(point)
print(f"Sample assigned to cluster: {cid}")

# 3. Batch processing (scikit-learn style)
X = np.random.randn(1000, 3)
labels = clusterer.fit_predict(X)
print(f"Discovered {clusterer.num_clusters} clusters across 1000 frames.")

# 4. Access centroids and member counts
centroids = clusterer.anchors        # shape: (K, 3)
counts = clusterer.member_counts     # shape: (K,)
```

---

## 3. Class: `gric.Clusterer`

```python
Clusterer(
    ndim: int,
    rlim: float = 0.5,
    max_clusters: int = 256,
    use_sq16: bool = False,
    use_eq16: bool = False,
    te4_mode: bool = True,
    entropy_mode: bool = False,
    ncpu: int = 1,
    lib_path: Optional[str] = None
)
```

### Parameters
* **`ndim`** (*int*): Feature vector dimensionality (number of columns).
* **`rlim`** (*float*, default=`0.5`): Distance clustering threshold.
* **`max_clusters`** (*int*, default=`256`): Maximum cluster table capacity. Set to `0` for dynamic
  growth.
* **`use_sq16`** (*bool*, default=`False`): Enable 16-bit scalar quantization filtering.
* **`use_eq16`** (*bool*, default=`False`): Enable 16-bit $E_8$ lattice quantization filtering.
* **`te4_mode`** (*bool*, default=`True`): Enable 4-point triangle inequality metric pruning.
* **`entropy_mode`** (*bool*, default=`False`): Enable entropy-guided target ranking.
* **`ncpu`** (*int*, default=`1`): Number of worker threads for parallel evaluation.

---

## 4. Methods & Properties

### `feed(frame: Union[np.ndarray, List[float]]) -> int`
Processes a single coordinate frame. Returns the assigned cluster ID.
* **Throws**: `ValueError` if the array length does not match `ndim`.

### `feed_batch(X: np.ndarray) -> np.ndarray`
Processes a 2D matrix of shape `(n_samples, ndim)` using zero-copy memory pointer passing directly
into C. Returns an array of cluster labels of shape `(n_samples,)`.

### `fit_predict(X: np.ndarray) -> np.ndarray`
Alias for `feed_batch(X)` conforming to the scikit-learn clusterer interface.

### `anchors -> np.ndarray`
Returns a 2D array of shape `(K, ndim)` containing all cluster centroid coordinate vectors.

### `member_counts -> np.ndarray`
Returns a 1D array of shape `(K,)` containing the member counts for each discovered cluster.

### `dcc -> np.ndarray`
Returns a square $K \times K$ distance matrix between all cluster centroids.

### `num_clusters -> int`
The number of clusters discovered so far.

### `reset()`
Resets clustering state while keeping buffer allocations hot for the next run.

### `close()` / Context Manager
Releases native C engine memory. Can also be managed automatically with a `with` statement:

```python
with gric.Clusterer(ndim=4, rlim=0.3) as cl:
    labels = cl.fit_predict(dataset)
```
