# Python Wrapper for GRIC Cluster

A Python wrapper around the `gric-cluster` C executable.

## Usage

```python
from image_cluster import ImageCluster

# Initialize
# Automatically resolves ./build/gric-cluster or system PATH
ic = ImageCluster(rlim=0.5, maxcl=10, dprob=0.05)

# Run on file
res = ic.run("data.txt")
print(f"Clusters: {res['total_clusters']}")

# Run on in-memory data
data = [[0, 0], [0.1, 0.1], [1, 1], [1.1, 1.1]]
res = ic.run_sequence(data)
print(f"Assignments: {res['assignments']}")
```

## Options

Pass command-line options as keyword arguments to the constructor:
- `gprob=True` -> `-gprob`
- `pred="10,1000,2"` -> `-pred 10,1000,2`
- `maxvis=500` -> `-maxvis 500`
- `eq16=True` -> `-eq16`
