---
name: diagnose-build-failure
description: Troubleshooting CMake and compilation failures.
---

# Diagnose Build Failures

A guide for troubleshooting build and link errors in `gric-cluster`.

## 1. CMake Configuration Failures
- **Missing dependencies:** Look for pkg-config errors: e.g. "cfitsio not found", "libpng not found", "libavcodec not found".
- **Fix:** Install missing packages (`libcfitsio-dev`, `libpng-dev`, `libavformat-dev`, etc.) or verify they are registered in PKG_CONFIG_PATH.

## 2. Compiler Errors
- **Undefined references / symbols:** Check if headers are missing or if the source file was not added to the `CLUSTER_SRCS` list in `CMakeLists.txt`.
- **Implicit declaration warnings:** Occur when a function is called without a matching header include. Fix by explicitly adding the `#include` of the header defining the function.

## 3. Linker Errors
- **Missing library linkage:** Ensure that the target executables link against the library variables in `CMakeLists.txt` (e.g., `${PNG_LIBRARIES}`, `m`).
