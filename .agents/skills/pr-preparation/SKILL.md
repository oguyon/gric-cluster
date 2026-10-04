---
name: pr-preparation
description: Standard check before preparing a Pull Request.
---

# PR Preparation

Before opening a PR or merging code:

## Checklist
1. **Clean compilation:** Check that the code builds with zero warnings or errors.
2. **Format compliance:** Ensure code adheres to the project C style (Allman braces, short lines).
3. **Tests passing:** Run local dataset tests to verify no regressions
   occur in clustering outputs.
4. **Clean Git tree:** Keep commit history structured and focus each
   commit on a single logical change.
5. **Agentic tool disclosure:** If agentic tools were used, disclose in the PR description with:
   - Notice: `Implemented by <model name>. Reviewed and signed off by O. Guyon`
     (e.g., `Implemented by gemini 3.8 flash (high). Reviewed and signed off by O. Guyon`).
   - A short description/summary of what the prompts asked the agent to do.
6. **MCP synchronization:** If CLI flags, output formats, FPS parameters, stream layout,
   or installed programs changed: MCP snapshot updated, drift tests pass (`test_mcp_*`,
   `test_doc_flag_drift`, `test_fps_param_docs`, `test_mcp_suite_catalog`, `test_stream_layout`).
