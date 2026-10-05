# Framework & Runtime Adapters (Level 3)

## Purpose
Integration bridges connecting GRIC compute engines to external runtimes, frameworks, and
platforms (Milk/CACAO adaptive optics runtime, WebAssembly, etc.).

## Architectural Layer
- **Level**: Level 3 (Adapters)
- **Adapter Targets**:
  - `adapters/milk/`: Milk Function Parameter Structures (FPS) streaming clustering module
    (`libmilkgric.so`, `milk-fpsexec-gric-cluster`).
  - `adapters/wasm/`: WebAssembly Emscripten bindings for web-based exploration.
- **Permitted Dependencies**:
  - Level 0 (`base/*`), Level 1 (`quant/*`), Level 2 (`engine/*`).
  - External framework headers (`ImageStreamIO.h`, `fps.h`, `CommandLineInterface/CLIcore.h`,
    `emscripten.h`).
