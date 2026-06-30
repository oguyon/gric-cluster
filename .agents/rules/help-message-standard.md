---
description: Unified help message format for gric-cluster binaries.
---

# Help Message Standard

Every executable in `gric-cluster` MUST support a help screen via the `-h` or `--help` flags.

## Help Message Layout
The help output must follow this section order:
1. **NAME** — program name and brief description
2. **USAGE** — command-line syntax synopsis
3. **DESCRIPTION** — brief description of functionality and options
4. **OPTIONS** — detailed list of supported flags, defaults, and arguments
5. **EXAMPLES** — 1-3 concrete usage examples
6. **COLOR MODE** — current color status and NO_COLOR instructions

## Options Formatting
Align option flags and descriptions cleanly:
```
  -c, --conf <file>    Load configuration file
  -v, --verbose        Increase output logging verbosity (default: off)
```

## Color Formatting and Suppression
To keep help displays highly readable, apply standard ANSI color coding:
* **Section Headings** (`NAME`, `USAGE`, etc.): **Bold** (`\x1b[1m`) or **Bold Cyan** (`\x1b[1;36m`)
* **Commands / Executables**: **Bold Green** (`\x1b[1;32m`)
* **Option Flags**: **Regular Green** (`\x1b[32m`)
* **Variables / Placeholders**: **Regular Magenta** (`\x1b[35m`)
* **Warnings / Overrides**: **Yellow** (`\x1b[33m`)
* **Errors**: **Bold Red** (`\x1b[1;31m`)
* **Shell Operators / Prompts**: **Dim/Grey** (`\x1b[90m`)

### Suppression Standard
All binaries must support the [no-color.org](https://no-color.org/) standard. Check for the presence of the `NO_COLOR` environment variable on startup. If it is present, suppress all ANSI escape sequence outputs.

## Error Handling on Arguments
If required command-line arguments are missing or invalid:
1. Print a clear error message to `stderr`.
2. Print the usage synopsis.
3. Exit with a non-zero code (e.g., `1`).
