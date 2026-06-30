# Always compile after editing source code

After modifying any C/CMake source file in the `gric-cluster` tree, you **must** run the compile-test workflow to verify the build still succeeds before considering the task complete.

If the build fails, fix the errors and rebuild until it passes.

Verification steps:
```bash
mkdir -p build && cd build && cmake .. && make -j$(nproc)
```
