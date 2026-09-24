# Build & Integration Guide

`libgric` is built using CMake and can be integrated into downstream projects using modern CMake
targets or traditional `pkg-config`.

---

## 1. Building and Installing `libgric`

### Build from Source
```bash
git clone https://github.com/oguyon/gric-cluster.git
cd gric-cluster
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

This generates:
* `build/libgric.so`: Dynamic shared library
* `build/libgric.a`: Static archive
* `build/gric-cluster`: Command-line executable

### Run Unit Tests
```bash
ctest --output-on-failure
```

---

## 2. Modern CMake Integration (`find_package`)

To use `libgric` in your own CMake project, install the targets or point `CMAKE_PREFIX_PATH`:

```cmake
cmake_minimum_required(VERSION 3.15)
project(my_pipeline C CXX)

find_package(gric REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE gric::gric)
```

### Static vs. Shared Linking
* Link against the shared library:
  ```cmake
  target_link_libraries(my_app PRIVATE gric::gric)
  ```
* Link against the static library:
  ```cmake
  target_link_libraries(my_app PRIVATE gric::gric_static)
  ```

---

## 3. Pkg-Config Integration

If using Makefiles or custom build scripts, you can locate headers and libraries via `pkg-config`:

```makefile
CFLAGS += $(shell pkg-config --cflags gric)
LDFLAGS += $(shell pkg-config --libs gric)

my_app: main.c
	$(CC) $(CFLAGS) main.c $(LDFLAGS) -o my_app
```

---

## 4. Hardware Acceleration Flags

When compiling `libgric`, optional hardware acceleration modules can be toggled:

| CMake Flag | Default | Description |
| :--- | :--- | :--- |
| `-DENABLE_CUDA=ON` | `OFF` | Enables NVIDIA GPU acceleration (CUDA kernels & cuBLAS) |
| `-DUSE_BLAS=ON` | `OFF` | Enables external BLAS library acceleration (OpenBLAS/MKL) |
| `-DIMAGESTREAMIO=ON`| Auto | Enables shared-memory streaming buffers (`ImageStreamIO`) |
