# gric-mcp: Native C17 Model Context Protocol (MCP) Server

`gric-mcp` is a native C17 server implementing the open Model Context Protocol (MCP). It exposes
high-speed C-native diagnostic, auditing, and dataset introspection tools to AI pair programming
agents (such as Google Antigravity, Claude Desktop, Cursor, and IDE extensions) via standard input
and output JSON-RPC 2.0.

---

## 1. Overview & Architecture

Unlike typical Python or TypeScript MCP servers, `gric-mcp` is written directly in standard C17
with zero runtime dependencies:

- **Ultra-Low Latency**: Sub-millisecond tool execution directly compiled against GRIC libraries.
- **Zero Dependencies**: Requires no Node.js, Python virtual environments, or extra packages.
- **Standard Protocol**: Speaks standard JSON-RPC 2.0 over standard I/O (`stdio`).
- **Domain Verification**: Gives AI coding agents native tools to enforce GRIC style guides, verify
  SIMD vectorization, inspect runtime logs, and profile datasets.

---

## 2. Tools Provided

| MCP Tool Name | Description | Key Capabilities |
| :--- | :--- | :--- |
| **`gric_audit_code_style`** | Style audit | Checks line limits, Allman braces, comments. |
| **`gric_align_parameters`** | Prototype align | Column-aligns multi-line function parameters. |
| **`gric_inspect_simd`** | SIMD inspect | Verifies AVX2/AVX-512 attributes and loop bounds. |
| **`gric_verify_invariants`** | Safety checker | Verifies memory invariants and triangle bounds. |
| **`gric_inspect_run`** | Run diagnostics | Analyzes `cluster_run.log` and telemetry logs. |
| **`gric_probe_dataset`** | Dataset probe | Generates dataset statistics and radius presets. |
| **`gric_probe_shm`** | SHM inspection | Queries live `ImageStreamIO` buffer and ring counters. |

---

## 3. Configuration & IDE Setup

To connect `gric-mcp` to your AI coding environment, add it to your client's MCP configuration:

### Claude Desktop / Google Antigravity
Add to your client configuration file (`mcp_settings.json` or `config.json`):

```json
{
  "mcpServers": {
    "gric": {
      "command": "/absolute/path/to/gric-cluster/build/gric-mcp",
      "args": []
    }
  }
}
```

---

## 4. Standalone CLI Usage

`gric-mcp` can also be queried from the terminal for diagnostics and schema inspection:

```bash
# Print tool schemas in JSON-RPC format
gric-mcp --list

# Print version and protocol details
gric-mcp -v

# Interactive stdio loop (for automated pipe integration)
gric-mcp
```
