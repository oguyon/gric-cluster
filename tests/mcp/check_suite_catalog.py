#!/usr/bin/env python3
"""
Verify suite catalog manifest (docs/suite/programs.json) matches installed programs.
"""

import json
import os
import sys

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <programs.json> <installed_programs.txt>")
        sys.exit(1)

    programs_json_path = sys.argv[1]
    installed_txt_path = sys.argv[2]

    if not os.path.isfile(programs_json_path):
        print(f"Error: {programs_json_path} not found", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(installed_txt_path):
        print(f"Error: {installed_txt_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(programs_json_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    with open(installed_txt_path, "r", encoding="utf-8") as f:
        installed = set(line.strip() for line in f if line.strip())

    catalog_by_name = {entry["name"]: entry for entry in catalog}
    errors = []

    # 1. Every installed program must be in catalog
    for prog in sorted(installed):
        if prog not in catalog_by_name:
            errors.append(
                f"Installed program '{prog}' is missing from catalog ({programs_json_path})"
            )

    # 2. Check catalog entries
    for name, entry in sorted(catalog_by_name.items()):
        alias_of = entry.get("alias_of")
        if alias_of:
            if alias_of not in catalog_by_name:
                errors.append(
                    f"Catalog entry '{name}' specifies alias_of '{alias_of}', "
                    f"but '{alias_of}' is not in catalog"
                )

        requires = entry.get("requires")
        if not requires:
            if name not in installed:
                errors.append(
                    f"Catalog lists unconditional program '{name}', but it was not installed"
                )
        else:
            if requires == "milk" and "milk-fpsexec-gric-cluster" in installed:
                if name not in installed:
                    errors.append(
                        f"Milk is enabled, but catalog program '{name}' was not installed"
                    )
            elif requires == "imagestreamio" and "gric-txt2stream" in installed:
                if name not in installed:
                    errors.append(
                        f"ImageStreamIO is enabled, but catalog program '{name}' was not installed"
                    )
            elif requires == "cfitsio" and "gric-gen-balls" in installed:
                if name not in installed:
                    errors.append(
                        f"CFITSIO is enabled, but catalog program '{name}' was not installed"
                    )

    if errors:
        print("Suite catalog verification failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    print(
        f"PASS: Suite catalog ({len(catalog_by_name)} entries) matches "
        f"installed programs ({len(installed)} active)."
    )

if __name__ == "__main__":
    main()
