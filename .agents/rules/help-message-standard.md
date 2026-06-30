---
description: Unified help message format for gric-cluster binaries.
---

# Help Message Standard

Every executable in `gric-cluster` MUST support a help screen via the `-h` or `--help` flags.

## Help Message Layout
The help output must follow this section order:
1. **BANNER** — program name and brief description
2. **USAGE** — command-line syntax synopsis
3. **DESCRIPTION** — brief description of functionality and options
4. **OPTIONS** — detailed list of supported flags, defaults, and arguments
5. **EXAMPLES** — 1-3 concrete usage examples

## Options Formatting
Align option flags and descriptions cleanly:
```
  -c, --conf <file>    Load configuration file
  -v, --verbose        Increase output logging verbosity (default: off)
```

## Error Handling on Arguments
If required command-line arguments are missing or invalid:
1. Print a clear error message to `stderr`.
2. Print the usage synopsis.
3. Exit with a non-zero code (e.g., `1`).
