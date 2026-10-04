#!/usr/bin/env python3
"""
Verify that all Milk FPS parameters from X-macro schemas are documented in docs/help/milk/.
"""

import json
import os
import subprocess
import sys

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <path_to_gric-mcp> <docs_milk_dir>")
        sys.exit(1)

    mcp_bin = sys.argv[1]
    docs_dir = sys.argv[2]

    if not os.path.isfile(mcp_bin):
        print(f"Error: binary {mcp_bin} not found", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(docs_dir):
        print(f"Error: directory {docs_dir} not found", file=sys.stderr)
        sys.exit(1)

    # Read all documentation markdown content
    doc_text = ""
    for fname in sorted(os.listdir(docs_dir)):
        if fname.endswith(".md"):
            with open(os.path.join(docs_dir, fname), "r", encoding="utf-8") as f:
                doc_text += "\n" + f.read()

    modules = ["cluster", "knn", "recon"]
    missing = []
    total_checked = 0

    for mod in modules:
        uri = f"gric://milk/fps_params/{mod}"
        proc = subprocess.run(
            [mcp_bin, "--resource", uri],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if proc.returncode != 0:
            print(f"Error querying resource {uri}: {proc.stderr}", file=sys.stderr)
            sys.exit(1)

        data = json.loads(proc.stdout)
        contents = data.get("result", {}).get("contents", [])
        if not contents:
            print(f"Error: empty contents for {uri}", file=sys.stderr)
            sys.exit(1)

        payload = json.loads(contents[0]["text"])
        params = payload.get("parameters", [])

        for p in params:
            key = p["key"]
            total_checked += 1
            # Check key (e.g. .rlim) or clean key (e.g. rlim) in documentation
            clean_key = key.lstrip(".")
            if key not in doc_text and clean_key not in doc_text:
                missing.append(f"Module '{mod}': parameter '{key}' not found in {docs_dir}/*.md")

    if missing:
        print("Undocumented FPS parameters detected:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        sys.exit(1)

    print(f"PASS: All {total_checked} FPS parameters across {len(modules)} modules are documented.")

if __name__ == "__main__":
    main()
