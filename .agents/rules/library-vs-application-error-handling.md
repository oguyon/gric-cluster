---
description: Design principles for error boundaries between core library and application files.
---

# Library vs Application Error Handling

Maintain a strict boundary between library functions (e.g., `cluster_core.c`, `framedistance.c`) and CLI/application drivers (e.g., `main.c`, `gric-info.c`).

## 1. Core Library Functions
- Must **never** call `exit()` or terminate the process.
- Must return error codes (`-1`, non-zero, or `NULL` pointers) to the caller.
- Should log warnings or errors to `stderr` only on severe, unrecoverable states.

## 2. CLI and Application Driver Layers
- Handle the error codes returned by the core library functions.
- Present clean user-facing error messages.
- Decide whether to abort execution, log details, or exit using `exit(EXIT_FAILURE)`.
