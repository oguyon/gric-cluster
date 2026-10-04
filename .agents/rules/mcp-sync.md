---
trigger: model_decision
description: Keep gric-mcp in sync when changing CLI flags, FPS parameters, output
  formats, stream layouts, docs/help, docs/recipes, or installed programs.
---

# MCP Synchronization Rules

To prevent documentation and tooling drift between the core compute engines and `gric-mcp`,
always keep the following source-of-truth files and their dependent MCP interfaces synchronized:

## 1. Source-of-Truth Mappings

- **CLI Flags**:
  - Source: `config_utils.c`, `knn_cli.c`, `gric-probe/main.c`
  - Downstream: `docs/help/**/*.md`, `tools/gen_help_c.py`
  - Verification: `test_doc_flag_drift`
- **FPS Parameters**:
  - Source: `src/gric-fps/*_params.h`
  - Downstream: `docs/help/milk/*.md`
  - Verification: `test_fps_param_docs`
- **Stream Layout**:
  - Source: `src/shared/gric_stream_layout.h`
  - Downstream: `docs/help/milk/milk_streams.md`, `tool_fps_ops.c`
  - Verification: `test_stream_layout`
- **Suite Catalog**:
  - Source: Installed binaries in `CMakeLists.txt`
  - Downstream: `docs/suite/programs.json`
  - Verification: `test_mcp_suite_catalog`
- **Cookbook Recipes**:
  - Source: `docs/recipes/*.md`
  - Downstream: Auto-embedded into `mcp_content_gen.c`
  - Verification: `test_mcp_protocol`
- **Tool Schemas**:
  - Source: `src/gric-mcp/tools/tool_*.c`
  - Downstream: `tests/mcp/tools_snapshot.json`
  - Verification: `test_mcp_tools_snapshot`
- **Output Contracts**:
  - Source: Core output formats (.clusterdat, logs)
  - Downstream: `tool_inspect_run.c`, `tool_verify_invariants.c`
  - Verification: `test_mcp_contract`

## 2. Maintenance Workflows

1. **Adding or Modifying CLI Flags**:
   - Document new flags in the corresponding Markdown files under `docs/help/`.
   - If an experimental flag cannot be documented yet, add it to `tests/mcp/doc_flags_allowlist.txt`
     with a rationale comment.
   - Run `ctest -R test_doc_flag_drift --output-on-failure`.

2. **Adding or Modifying FPS Parameters**:
   - Add parameter definition to the X-macro in `src/gric-fps/*_params.h`.
   - Document the parameter token in `docs/help/milk/*.md`.
   - Run `ctest -R test_fps_param_docs --output-on-failure`.

3. **Altering Stream Layout**:
   - Update `GRIC_ASSIGN_FIELDS` in `src/shared/gric_stream_layout.h`.
   - Update field descriptions in `docs/help/milk/milk_streams.md`.
   - Run `ctest -R test_stream_layout --output-on-failure`.

4. **Adding New Executables or Suite Programs**:
   - Register program metadata in `docs/suite/programs.json`.
   - Run `ctest -R test_mcp_suite_catalog --output-on-failure`.

5. **Updating Tool Capabilities or Schemas**:
   - When tools or parameter schemas in `src/gric-mcp/tools/tool_*.c` change intentionally,
     update the baseline snapshot:
     ```bash
     make -C build mcp-snapshot
     ```
   - Commit the updated `tests/mcp/tools_snapshot.json`.
