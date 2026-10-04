#!/usr/bin/env python3
"""
check_snapshot.py - Verify or update the gric-mcp tools schema snapshot.
Usage: check_snapshot.py <path_to_gric_mcp> <path_to_snapshot_json> [--update]
"""

import difflib
import json
import subprocess
import sys


def normalize_tools(tools):
    """Sort tools array by name and recursively sort keys."""
    sorted_tools = sorted(tools, key=lambda t: t.get("name", ""))
    return sorted_tools


def main():
    if len(sys.argv) < 3:
        print("Usage: check_snapshot.py <gric_mcp_bin> <snapshot_json> [--update]")
        sys.exit(2)

    bin_path = sys.argv[1]
    snapshot_path = sys.argv[2]
    update_mode = "--update" in sys.argv

    # Run gric-mcp --list --toolsets=all
    proc = subprocess.run(
        [bin_path, "--list", "--toolsets=all"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        print(f"Error: {bin_path} exited with code {proc.returncode}")
        print(proc.stderr)
        sys.exit(1)

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        print(f"Error: Failed to parse JSON from {bin_path} --list: {exc}")
        print("Raw stdout:")
        print(proc.stdout[:500])
        sys.exit(1)

    tools = data.get("result", {}).get("tools", [])
    normalized = normalize_tools(tools)
    actual_text = json.dumps(normalized, indent=2, sort_keys=True) + "\n"

    if update_mode:
        with open(snapshot_path, "w", encoding="utf-8") as fp:
            fp.write(actual_text)
        print(f"Updated MCP tool snapshot: {snapshot_path}")
        sys.exit(0)

    try:
        with open(snapshot_path, "r", encoding="utf-8") as fp:
            expected_text = fp.read()
    except FileNotFoundError:
        print(f"Error: Snapshot file {snapshot_path} not found.")
        print("Run 'make mcp-snapshot' to generate the initial snapshot.")
        sys.exit(1)

    if actual_text != expected_text:
        diff = difflib.unified_diff(
            expected_text.splitlines(keepends=True),
            actual_text.splitlines(keepends=True),
            fromfile="tests/mcp/tools_snapshot.json",
            tofile="actual_tools",
        )
        sys.stdout.writelines(diff)
        print("\nError: MCP tools snapshot mismatch.")
        print("Run 'make mcp-snapshot' to update and commit the result.")
        sys.exit(1)

    print("MCP tools snapshot matches tests/mcp/tools_snapshot.json")
    sys.exit(0)


if __name__ == "__main__":
    main()
