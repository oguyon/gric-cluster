# C++ API Reference

The C++ API is provided as a modern, header-only RAII wrapper in `<gric/gric.hpp>`. It wraps the
underlying C functions in idiomatic C++ classes with automatic resource management, move semantics,
and standard library container interoperability.

---

## 1. Quick Start

Include `<gric/gric.hpp>` and compile with C++17 or higher:

```cpp
#include <gric/gric.hpp>
#include <iostream>
#include <vector>

int main()
{
    // Automatically freed upon leaving scope!
    gric::Clusterer clusterer(3, 0.5);

    std::vector<double> point = {0.1, 0.2, 0.0};
    int64_t cluster_id = clusterer.feed(point);

    std::cout << "Assigned cluster: " << cluster_id << std::endl;
    std::cout << "Total clusters: " << clusterer.num_clusters() << std::endl;

    return 0;
}
```

Compile with:
```bash
g++ -std=c++17 main.cpp -lgric -o main
```

---

## 2. RAII & Exception Safety

### What is RAII?
**Resource Acquisition Is Initialization (RAII)** ties the lifecycle of the underlying C context
`gric_cluster_t` to the lifetime of the stack-allocated `gric::Clusterer` object:
* **Constructor**: Calls `gric_cluster_create()` to allocate cluster buffers.
* **Destructor**: Automatically invokes `gric_cluster_destroy()` when the object leaves scope.
* **Exceptions**: If an exception occurs, stack unwinding ensures that resources are safely reclaimed
  without memory leaks.

### Move Semantics
`gric::Clusterer` is a move-only type. Accidental shallow copies that could cause double-free errors
are rejected at compile time:

```cpp
gric::Clusterer c1(4, 0.5);
// gric::Clusterer c2 = c1;       // Compile error: copy constructor deleted!
gric::Clusterer c2 = std::move(c1); // OK: ownership transferred to c2
```

---

## 3. Configuration Builder (`gric::Config`)

The `gric::Config` helper uses method chaining to customize clustering sessions:

```cpp
gric::Config cfg;
cfg.with_rlim(0.35)
   .with_max_clusters(512)
   .with_sq16(true)
   .with_threads(4);

gric::Clusterer clusterer(64, cfg);
```

### Available Builder Methods
* `.with_rlim(double r)`: Set distance clustering radius threshold.
* `.with_max_clusters(int max_cl)`: Set cluster capacity limit (or `0` for dynamic growth).
* `.with_sq16(bool enabled)`: Toggle 16-bit scalar quantization pruning.
* `.with_eq16(bool enabled)`: Toggle 16-bit E8 lattice quantization.
* `.with_threads(int threads)`: Set number of OpenMP worker threads.

---

## 4. Class Methods (`gric::Clusterer`)

### `feed(const double *coords)` / `feed(const std::vector<double> &frame)`
Process a single sample vector.
```cpp
std::vector<double> frame = {0.0, 1.0, 2.0};
int64_t id = clusterer.feed(frame);
```

### `feed_batch(const double *flat_coords, size_t num_frames)` / `feed_batch(const std::vector<double> &flat, size_t num_frames)`
Ingest a contiguous batch of coordinate frames.
```cpp
std::vector<double> batch = {
    0.0, 0.0, 0.0,
    0.1, 0.1, 0.0,
    5.0, 5.0, 5.0
};
std::vector<int64_t> labels = clusterer.feed_batch(batch, 3);
// labels: [0, 0, 1]
```

### `num_clusters()`
```cpp
size_t k = clusterer.num_clusters();
```

### `reset()`
Clears all clusters and state while preserving allocated memory.

### `c_handle()`
Provides access to the underlying `gric_cluster_t*` pointer for calling low-level C functions.

---

## 5. Error Handling (`gric::Error`)

Failed operations throw `gric::Error`, which derives from `std::runtime_error`:

```cpp
try
{
    gric::Clusterer clusterer(4, 0.5);
    std::vector<double> wrong_dim = {1.0, 2.0}; // Expecting 4, got 2
    clusterer.feed(wrong_dim);
}
catch (const std::invalid_argument &e)
{
    std::cerr << "Invalid dimension: " << e.what() << std::endl;
}
catch (const gric::Error &e)
{
    std::cerr << "GRIC failure (code " << e.status() << "): " << e.what() << std::endl;
}
```
