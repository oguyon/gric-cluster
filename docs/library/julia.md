# Julia Package Reference

The `GRIC.jl` package connects Julia directly to `libgric` with zero compilation glue, leveraging
Julia's native `ccall` mechanism. It integrates seamlessly with Julia's scientific ecosystem,
standard arrays, and automatic garbage collection.

---

## 1. Quick Start

Include the package in your Julia session:

```julia
push!(LOAD_PATH, "path/to/gric-cluster/bindings/julia/src")
using GRIC

# 1. Initialize Clusterer with 4 dimensions and radius 1.0
clusterer = Clusterer(4; rlim=1.0)

# 2. Feed individual vectors
c0 = feed!(clusterer, [0.0, 0.0, 0.0, 0.0])
println("Sample 0 -> Cluster ", c0)

c1 = feed!(clusterer, [0.1, 0.0, 0.0, 0.0])
println("Sample 1 -> Cluster ", c1)

println("Discovered clusters: ", num_clusters(clusterer))

# 3. Retrieve centroids as a Julia Matrix
centroids = anchors(clusterer)
display(centroids)
```

---

## 2. Automatic Memory Management (GC Finalizers)

In Julia, the `Clusterer` struct registers a **finalizer** with the garbage collector:

```julia
finalizer(obj) do c
    if c.handle != C_NULL
        ccall((:gric_cluster_destroy, LIBGRIC[]), Cvoid, (Ptr{Cvoid},), c.handle)
        c.handle = C_NULL
    end
end
```

When the Julia object goes out of scope and is reclaimed, the underlying C memory is automatically
freed.

---

## 3. Function Reference

### `Clusterer(ndim::Integer; rlim::Real = 0.5)`
Constructs a new clustering session for vectors with `ndim` features.

### `feed!(c::Clusterer, frame::AbstractVector{<:Real}) -> Int64`
Feeds a 1D vector into the engine. Returns the assigned cluster index ($\ge 0$).
* Throws `DimensionMismatch` if the vector length does not match `c.ndim`.

### `feed_batch!(c::Clusterer, X::AbstractMatrix{<:Real}) -> Vector{Int64}`
Feeds a 2D matrix of shape `n_samples × ndim`. Returns a vector of cluster indices.

### `num_clusters(c::Clusterer) -> Int`
Returns the count of active clusters discovered so far.

### `anchors(c::Clusterer) -> Matrix{Float64}`
Exports discovered cluster centroids as a Julia matrix of dimensions `K × ndim`.

### `version() -> String`
Returns the `libgric` semantic version string.
